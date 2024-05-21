import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp, lax
from jax._src.typing import SupportsDType

from dsa2000_cal.common import wgridder
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.check_utils import check_fft_predict_inputs
from dsa2000_cal.predict.vec_utils import kron_product
from dsa2000_cal.source_models.corr_translation import stokes_to_linear, flatten_coherencies


class FFTStokesIModelData(NamedTuple):
    """
    Data for predicting with FFT.
    """
    image: jax.Array  # [[chan,] Nx, Ny] in Stokes I
    gains: jax.Array  # [time, ant, [chan,] 2, 2]
    l0: jax.Array  # [[chan,]]
    m0: jax.Array  # [[chan,]]
    dl: jax.Array  # [[chan,]]
    dm: jax.Array  # [[chan,]]


@dataclasses.dataclass(eq=False)
class FFTStokesIPredict:
    num_threads: int = 1
    epsilon: float = 1e-4
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64

    def predict(self, freqs: jax.Array, faint_model_data: FFTStokesIModelData,
                visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Predict visibilities from faint model data contained in FITS Stokes-I images.
        Predict takes into account frequency depending on the image provided.

        If the image has a frequency dimension then it must match freqs, and we feed in image by image.

        If the image doesn't have a frequency dimension then it must be shaped (Nx, Ny), and we replicate the image
        for all frequencies.

        Similarly, for gains we replicate the gains for all frequencies if they don't have a frequency dimension.

        In all cases, the output frequencies are determined by the freqs argument.


        Args:
            freqs: [chan] frequencies in Hz.
            faint_model_data: data, see above for shape info.
            visibility_coords: visibility coordinates.

        Returns:
            visibilities: [row, chan, 4]
        """

        direction_dependent_gains, image_has_chan, gains_have_chan, stokes_I_image = check_fft_predict_inputs(
            freqs=freqs,
            image=faint_model_data.image,
            gains=faint_model_data.gains,
            l0=faint_model_data.l0,
            m0=faint_model_data.m0,
            dl=faint_model_data.dl,
            dm=faint_model_data.dm
        )
        if not stokes_I_image:
            raise ValueError("Image must be in Stokes I format.")

        if direction_dependent_gains:
            raise ValueError("Direction dependent gains are not supported for FFT predict.")

        if image_has_chan:
            if gains_have_chan:
                print("Gains map 1-to-1 to image channels.")
            else:
                print("A single gain maps to n to image channels.")
        else:
            if gains_have_chan:
                print("n Gains map to single image channel.")
            else:
                print("Single gain maps to single image.")

        g1 = faint_model_data.gains[
            visibility_coords.time_idx, visibility_coords.antenna_1, ...
        ]  # [row, [chan,] 2, 2]
        g2 = faint_model_data.gains[
            visibility_coords.time_idx, visibility_coords.antenna_2, ...
        ]  # [row, [chan,] 2, 2]

        # Data will be sharded over frequency so don't reduce over these dimensions, or else communication happens.
        # We want the outer broadcast to be over chan, so we'll do this order:
        # Over chan

        # g1, g2: [row, [chan,] 2, 2]
        # freqs: [chan]
        # image: [[chan',] Nx, Ny]
        # dl, dm, l0, m0: [[chan']]
        # uvw: [row, 3]
        def compute_visibility(g1: jax.Array, g2: jax.Array,
                               freqs: jax.Array,
                               image: jax.Array, dl: jax.Array, dm: jax.Array,
                               l0: jax.Array, m0: jax.Array,
                               uvw: jax.Array):
            vis = self._single_predict_jax(g1, g2,
                                           freqs,
                                           image,
                                           dl, dm, l0, m0,
                                           uvw
                                           )  # [row, chan, 2, 2]
            return vis  # [row, chan, 4]

        visibilities = compute_visibility(
            g1, g2,
            freqs, faint_model_data.image,
            faint_model_data.dl, faint_model_data.dm,
            faint_model_data.l0, faint_model_data.m0,
            visibility_coords.uvw
        )  # [row, chan, 2, 2]
        return visibilities

    def _single_predict_jax(self, g1: jax.Array, g2: jax.Array, freqs: jax.Array,
                            image: jax.Array, dl: jax.Array, dm: jax.Array, l0: jax.Array, m0: jax.Array,
                            uvw: jax.Array):
        """
        Predict visibilities for a single frequency.

        Args:
            g1: [row, [num_freqs,] 2, 2]
            g2: [row, [num_freqs,] 2, 2]
            freqs: [num_freqs]
            image: [[num_freqs,] Nx, Ny]
            dl: [[num_freqs,]]
            dm: [[num_freqs,]]
            l0: [[num_freqs,]]
            m0: [[num_freqs,]]
            uvw: [row, 3]

        Returns:
            vis: [row, num_freqs, 2, 2]
        """
        if self.convention == 'casa':
            uvw = jnp.negative(uvw)

        # image: [[num_freqs,] Nx, Ny]
        # freqs: [num_freqs]
        image_has_chans = len(np.shape(image)) == 3

        if image_has_chans:
            @partial(jax.vmap, in_axes=(0,  # freqs
                                        0,  # image
                                        0,  # dl
                                        0,  # dm
                                        0,  # l0
                                        0))  # m0
            def predict(freqs: jax.Array, image: jax.Array, dl: jax.Array, dm: jax.Array, l0: jax.Array,
                        m0: jax.Array) -> jax.Array:
                if len(np.shape(image)) != 2:
                    raise ValueError(f"Expected image to have shape [Nx, Ny], got {np.shape(image)}")
                if np.shape(dl) != () or np.shape(dm) != () or np.shape(l0) != () or np.shape(m0) != ():
                    raise ValueError("If image doesn't have a channel then l0, m0, dl, and dm must be scalars.")
                vis = wgridder.dirty2vis(
                    uvw=uvw,
                    freqs=freqs[None],
                    dirty=image,
                    pixsize_x=-dl,
                    pixsize_y=dm,
                    center_x=l0,
                    center_y=m0,
                    epsilon=self.epsilon
                )  # [num_rows, 1]
                return vis[:, 0]  # [num_rows]

            vis_I = predict(freqs, image, dl, dm, l0, m0)  # [num_freqs, num_rows]
            vis_I = lax.transpose(vis_I, (1, 0))  # [num_rows, num_freqs]
        else:
            vis_I = wgridder.dirty2vis(
                uvw=uvw,
                freqs=freqs,
                dirty=image,
                pixsize_x=-dl,
                pixsize_y=dm,
                center_x=l0,
                center_y=m0,
                epsilon=self.epsilon
            )  # [num_rows, num_freqs]

        return self._translate_to_linear(g1, g2, vis_I)  # [num_rows, num_freqs, 2, 2]

    def _translate_to_linear(self, g1: jax.Array, g2: jax.Array, vis_I: jax.Array):
        """
        Translate visibilities to linear basis and apply gains.

        Args:
            g1: [num_rows, [num_freqs,] 2, 2]
            g2: [num_rows, [num_freqs,] 2, 2]
            vis_I: [num_rows, num_freqs]

        Returns:
            vis_linear: [num_rows, num_freqs, 2, 2]
        """
        zero = jnp.zeros_like(vis_I)
        vis_stokes = jnp.stack([vis_I, zero, zero, zero], axis=-1)  # [num_rows, num_freqs, 4]
        shape = np.shape(vis_stokes)
        if np.shape(g1)[0] != shape[0] or np.shape(g2)[0] != shape[0]:
            raise ValueError(f"Expected gains to have shape [{shape[0]}, ...], got {np.shape(g1)} and {np.shape(g2)}")
        gains_have_freqs = len(np.shape(g1)) == 4

        # Over chan->row
        # g1, g2: [num_rows, [num_freqs,] 2, 2]
        # vis: [num_rows, num_freqs, 4]
        @partial(jax.vmap, in_axes=(
                1 if gains_have_freqs else None,
                1 if gains_have_freqs else None,
                1
        ))
        # g1, g2: [num_rows, 2, 2]
        # vis: [num_rows, 4]
        @partial(jax.vmap, in_axes=(0, 0, 0))
        # g1, g2: [2, 2]
        # vis: [4]
        def transform(g1, g2, vis_stokes):
            vis_linear = stokes_to_linear(vis_stokes, flat_output=False)  # [2, 2]
            vis_linear = kron_product(g1, vis_linear, g2.T.conj()) # [2, 2]
            return vis_linear  # [2, 2]


        vis_linear = transform(g1, g2, vis_stokes)  # [num_freqs, num_rows, 2, 2]
        vis_linear = lax.transpose(vis_linear, (1, 0, 2, 3))  # [num_rows, num_freqs, 2, 2]
        return vis_linear  # [num_rows, num_freqs, 2, 2]
