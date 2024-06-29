import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
from astropy import constants
from jax import numpy as jnp, lax
from jax._src.typing import SupportsDType

from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.check_utils import check_dft_predict_inputs
from dsa2000_cal.predict.vec_utils import kron_product


class PointModelData(NamedTuple):
    """
    Data for predict.
    """
    image: jax.Array  # [source, chan, 2, 2] in [[xx, xy], [yx, yy]] format
    gains: jax.Array  # [[source,] time, ant, chan, 2, 2]
    lmn: jax.Array  # [source, 3]


@dataclasses.dataclass(eq=False)
class PointPredict:
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64

    def predict(self, freqs: jax.Array, dft_model_data: PointModelData,
                visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Predict visibilities from DFT model data.

        Args:
            freqs: [chan] frequencies in Hz.
            dft_model_data: data, see above for shape info.
            visibility_coords: visibility coordinates.

        Returns:
            visibilities: [row, chan, 2, 2] in linear correlation basis.
        """

        direction_dependent_gains = check_dft_predict_inputs(
            freqs=freqs,
            image=dft_model_data.image,
            gains=dft_model_data.gains,
            lmn=dft_model_data.lmn
        )

        if direction_dependent_gains:
            print(f"Point prediction with unique gains per source.")
        else:
            print(f"Point prediction with shared gains across sources.")

        g1 = dft_model_data.gains[
             ..., visibility_coords.time_idx, visibility_coords.antenna_1, :, :, :
             ]  # [[source,] row, chan, 2, 2]
        g2 = dft_model_data.gains[
             ..., visibility_coords.time_idx, visibility_coords.antenna_2, :, :, :
             ]  # [[source,] row, chan, 2, 2]

        if direction_dependent_gains:
            g_mapping = "[s,r,c,2,2]"
        else:
            g_mapping = "[r,c,2,2]"

        # Data will be sharded over frequency so don't reduce over these dimensions, or else communication happens.
        # We want the outer broadcast to be over chan, so we'll do this order.

        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: [[source,] row, chan, 2, 2]
        # freq: [chan]
        # image: [source, chan, 2, 2]
        @partial(multi_vmap, in_mapping=f"[s,3],[r,3],{g_mapping},{g_mapping},[c],[s,c,2,2]",
                 out_mapping="[r,c]", verbose=True)
        def compute_visibility(lmn, uvw, g1, g2, freq, image):
            """
            Compute visibilities for a single row, channel, accumulating over sources.

            Args:
                lmn: [source, 3]
                uvw: [3]
                g1: [[source,] 2, 2]
                g2: [[source,] 2, 2]
                freq: []
                image: [source, 2, 2]

            Returns:
                vis_accumulation: [2, 2] visibility for given baseline, accumulated over all provided directions.
            """
            # TODO: Can use associative_scan here.
            if direction_dependent_gains:
                def body_fn(accumulate, x):
                    (lmn, g1, g2, image) = x
                    delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image)  # [2, 2]
                    accumulate += delta
                    return accumulate, ()

                xs = (lmn, g1, g2, image)
            else:
                def body_fn(accumulate, x):
                    (lmn, image) = x
                    delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image)  # [2, 2]
                    accumulate += delta
                    return accumulate, ()

                xs = (lmn, image)
            init_accumulate = jnp.zeros((2, 2), dtype=self.dtype)
            vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs, unroll=1)
            return vis_accumulation  # [2, 2]

        visibilities = compute_visibility(
            dft_model_data.lmn,
            visibility_coords.uvw,
            g1,
            g2,
            freqs,
            dft_model_data.image
        )  # [row, chan, 2, 2]
        return visibilities

    def _single_compute_visibilty(self, lmn, uvw, g1, g2, freq, image):
        """
        Compute the visibility from a single direction for a single baseline.

        Args:
            lmn: [3]
            uvw: [3]
            g1: [2, 2]
            g2: [2, 2]
            freq: []
            image: [2, 2]

        Returns:
            [2, 2] visibility in given direction for given baseline.
        """
        wavelength = quantity_to_jnp(constants.c) / freq

        if self.convention == 'casa':
            uvw = jnp.negative(uvw)

        uvw /= wavelength

        u, v, w = uvw  # scalar

        l, m, n = lmn  # scalar

        # -2*pi*freq/c*(l*u + m*v + (n-1)*w)
        delay = l * u + m * v + (n - 1.) * w  # scalar

        phi = jnp.asarray(
            (-2j * np.pi) * delay,
            dtype=self.dtype
        )  # scalar
        fringe = (jnp.exp(phi) / n)
        return fringe * kron_product(g1, image, g2.T.conj())
