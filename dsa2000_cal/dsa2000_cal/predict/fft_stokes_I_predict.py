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
from dsa2000_cal.source_models.corr_translation import stokes_to_linear


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

        g1 = faint_model_data.gains[visibility_coords.time_idx, visibility_coords.antenna_1, ...
        ]  # [row, [chan,] 2, 2]
        g2 = faint_model_data.gains[visibility_coords.time_idx, visibility_coords.antenna_2, ...
        ]  # [row, [chan,] 2, 2]

        # Data will be sharded over frequency so don't reduce over these dimensions, or else communication happens.
        # We want the outer broadcast to be over chan, so we'll do this order:

        if image_has_chan:
            # g1, g2, [row, [chan,] 2, 2]
            # freqs, [chan]
            # image, [chan, Nx, Ny]
            # dl, dm, l0, m0, [chan]
            # uvw [row, 3]
            @partial(
                jax.vmap, in_axes=[
                    1 if gains_have_chan else None, 1 if gains_have_chan else None,
                    0,
                    0,
                    0, 0, 0, 0,
                    None
                ],
                out_axes=1  # -> [row, chan, 2, 2]
            )
            # g1, g2, [row, 2, 2]
            # freqs, []
            # image, [Nx, Ny]
            # dl, dm, l0, m0, []
            # uvw [row, 3]
            def compute_visibility(g1: jax.Array, g2: jax.Array, freqs: jax.Array,
                                   image: jax.Array, dl: jax.Array, dm: jax.Array,
                                   l0: jax.Array, m0: jax.Array,
                                   uvw: jax.Array):
                vis = self._single_predict_jax(g1[:, None, :, :], g2[:, None, :, :],
                                               freqs[None],
                                               image,
                                               dl, dm, l0, m0,
                                               uvw
                                               )  # [row, 1, 4]
                return vis[:, 0, :]  # [row, 4]
        else:
            if not gains_have_chan:
                # Add chan
                g1 = jnp.repeat(g1[:, None, :, :], np.shape(freqs)[0], axis=1)  # [row, chan, 2, 2]
                g2 = jnp.repeat(g2[:, None, :, :], np.shape(freqs)[0], axis=1)  # [row, chan, 2, 2]

            # g1, g2, [row, chan, 2, 2]
            # freqs, [chan]
            # image, [Nx, Ny]
            # dl, dm, l0, m0, []
            # uvw [row, 3]
            def compute_visibility(*args):
                return self._single_predict_jax(*args)  # [row, chan, 4]

        visibilities = compute_visibility(
            g1, g2,
            freqs, faint_model_data.image,
            faint_model_data.dl, faint_model_data.dm,
            faint_model_data.l0, faint_model_data.m0,
            visibility_coords.uvw
        )  # [row, chan, 4]
        return visibilities

    def _single_predict_jax(self, g1: jax.Array, g2: jax.Array, freqs: jax.Array,
                            image: jax.Array, dl: jax.Array, dm: jax.Array, l0: jax.Array, m0: jax.Array,
                            uvw: jax.Array):
        if self.convention == 'casa':
            uvw = jnp.negative(uvw)

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

        return self._translate_to_linear(g1, g2, vis_I)  # [num_rows, num_freqs, 4]

    def _translate_to_linear(self, g1: jax.Array, g2: jax.Array, vis_I: jax.Array):
        zero = jnp.zeros_like(vis_I)
        vis = jnp.stack([vis_I, zero, zero, zero], axis=-1)  # [num_rows, num_freqs, 4]
        shape = np.shape(vis)
        if np.shape(g1)[:2] != shape[:2] or np.shape(g2)[:2] != shape[:2]:
            raise ValueError("g1 and g2 must have the same number of rows as vis_I.")
        vis = lax.reshape(vis, (shape[0] * shape[1], 4))  # [num_rows * num_freqs, 4]
        g1 = lax.reshape(g1, (shape[0] * shape[1], 2, 2))
        g2 = lax.reshape(g2, (shape[0] * shape[1], 2, 2))

        def transform(g1, g2, vis):
            vis_linear = stokes_to_linear(vis, flat_output=False)  # [2, 2]
            vis_linear = kron_product(g1, vis_linear, g2.T.conj())  # [2, 2]
            return vis_linear

        vis_linear = jax.vmap(transform)(g1, g2, vis)  # [num_rows * num_freqs, 2, 2]
        vis_linear = lax.reshape(vis_linear, shape[:-1] + (2, 2))  # [num_rows, num_freqs, 2, 2]
        return vis_linear
