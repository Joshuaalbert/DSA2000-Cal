import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp, lax
from jax._src.typing import SupportsDType

from dsa2000_cal.common.wgridder import dirty2vis
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.vec_utils import kron_product
from dsa2000_cal.source_models.corr_translation import stokes_to_linear, flatten_coherencies


class FaintModelData(NamedTuple):
    """
    Data for predict.
    """
    image: jax.Array  # [[chan,] Nx, Ny] in Stokes I
    gains: jax.Array  # [time, ant, [chan,] 2, 2]
    l0: jax.Array  # [[chan,]]
    m0: jax.Array  # [[chan,]]
    dl: jax.Array  # [[chan,]]
    dm: jax.Array  # [[chan,]]


@dataclasses.dataclass(eq=False)
class FaintPredict:
    num_threads: int = 1
    epsilon: float = 1e-4
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64

    def predict(self, freqs: jax.Array, faint_model_data: FaintModelData,
                visibility_coords: VisibilityCoords) -> jax.Array:
        g1 = faint_model_data.gains[visibility_coords.time_idx, visibility_coords.antenna_1, ...
        ]  # [row, [chan,] 2, 2]
        g2 = faint_model_data.gains[visibility_coords.time_idx, visibility_coords.antenna_2, ...
        ]  # [row, [chan,] 2, 2]
        gain_has_chan = len(np.shape(g1)) == 4

        # If the image has a channel then it must match freqs, and we feed in image by image
        image_has_chan = len(np.shape(faint_model_data.image)) == 3
        if image_has_chan:
            if np.shape(faint_model_data.image)[0] != np.shape(freqs)[0]:
                raise ValueError("If image has a channel then it must match freqs.")
            for x in [faint_model_data.l0, faint_model_data.m0, faint_model_data.dl, faint_model_data.dm]:
                if np.shape(x) != np.shape(freqs):
                    raise ValueError("If image has a channel then l0, m0, dl, and dm must be shaped (freqs,).")
        else:
            for x in [faint_model_data.l0, faint_model_data.m0, faint_model_data.dl, faint_model_data.dm]:
                if np.shape(x) != ():
                    raise ValueError("If image doesn't have a channel then l0, m0, dl, and dm must be shaped ().")

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
                    1 if gain_has_chan else None, 1 if gain_has_chan else None,
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
                                   image: jax.Array, dl: jax.Array, dm: jax.Array, l0: jax.Array, m0: jax.Array,
                                   uvw: jax.Array):
                vis = self._single_predict_jax(g1[:, None, :, :], g2[:, None, :, :],
                                               freqs[None],
                                               image,
                                               dl, dm, l0, m0,
                                               uvw
                                               )  # [row, 1, 4]
                return vis[:, 0, :]  # [row, 4]
        else:
            if not gain_has_chan:
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

        vis_I = dirty2vis(
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
            return flatten_coherencies(vis_linear)  # [4]

        vis_linear = jax.vmap(transform)(g1, g2, vis)  # [num_rows * num_freqs, 4]
        vis_linear = lax.reshape(vis_linear, shape)  # [num_rows, num_freqs, 4]
        return vis_linear
