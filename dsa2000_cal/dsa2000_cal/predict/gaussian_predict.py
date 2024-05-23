import dataclasses
import warnings
from functools import partial
from typing import NamedTuple

import jax
from astropy import constants
from jax import numpy as jnp, lax
from jax._src.typing import SupportsDType

from dsa2000_cal.common.ellipse_utils import Gaussian
from dsa2000_cal.common.jvp_linear_op import JVPLinearOp
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.check_utils import check_dft_predict_inputs
from dsa2000_cal.predict.vec_utils import kron_product
from dsa2000_cal.source_models.corr_translation import linear_to_stokes, stokes_to_linear


class GaussianModelData(NamedTuple):
    """
    Data for predict.
    """
    image: jax.Array  # [source, chan, 2, 2] in [[xx, xy], [yx, yy]] format
    gains: jax.Array  # [[source,] time, ant, chan, 2, 2]
    lmn: jax.Array  # [source, 3]
    ellipse_params: jax.Array  # [source, 3] (major, minor, theta)


@dataclasses.dataclass(eq=False)
class GaussianPredict:
    order_approx: int = 0
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64

    def predict(self, freqs: jax.Array, gaussian_model_data: GaussianModelData,
                visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Predict visibilities from Gaussian model data.

        Args:
            freqs: [chan] frequencies in Hz.
            gaussian_model_data: data, see above for shape info.
            visibility_coords: visibility coordinates.

        Returns:
            visibilities: [row, chan, 2, 2] in linear correlation basis.
        """

        direction_dependent_gains = check_dft_predict_inputs(
            freqs=freqs,
            image=gaussian_model_data.image,
            gains=gaussian_model_data.gains,
            lmn=gaussian_model_data.lmn
        )

        if direction_dependent_gains:
            print(f"Gaussian prediction with unique gains per source.")
        else:
            print(f"Gaussian prediction with shared gains across sources.")

        g1 = gaussian_model_data.gains[
             ..., visibility_coords.time_idx, visibility_coords.antenna_1, :, :, :
             ]  # [[source,] row, chan, 2, 2]
        g2 = gaussian_model_data.gains[
             ..., visibility_coords.time_idx, visibility_coords.antenna_2, :, :, :
             ]  # [[source,] row, chan, 2, 2]

        # Data will be sharded over frequency so don't reduce over these dimensions, or else communication happens.
        # We want the outer broadcast to be over chan, so we'll do this order:
        # chan -> row -> source
        # vmap(vmap(scan)) is the preferred approach for this, Which avoids setting up vmap overhead.

        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: [[source,] row, chan, 2, 2]
        # freq: [chan]
        # image: [source, chan, 2, 2]
        # ellipse_params: [source, 3]
        @partial(jax.vmap, in_axes=[None, None, -3, -3, 0, -3, None])  # -> chan
        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: [[source,] row, 2, 2]
        # freq: []
        # image: [source, 2, 2]
        # ellipse_params: [source, 3]
        @partial(jax.vmap, in_axes=[None, 0, -3, -3, None, None, None])  # -> row
        # lmn: [source, 3]
        # uvw: [3]
        # g1, g2: [[source,] 2, 2]
        # freq: []
        # image: [source, 2, 2]
        # ellipse_params: [source, 3]
        def compute_visibility(lmn, uvw, g1, g2, freq, image, ellipse_params):
            # TODO: Can use associative_scan here.
            if direction_dependent_gains:
                def body_fn(accumulate, x):
                    (lmn, g1, g2, image, ellipse_params) = x
                    delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image, ellipse_params)  # [2, 2]
                    accumulate += delta
                    return accumulate, ()

                xs = (lmn, g1, g2, image, ellipse_params)
            else:
                def body_fn(accumulate, x):
                    (lmn, image, ellipse_params) = x
                    delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image, ellipse_params)  # [2, 2]
                    accumulate += delta
                    return accumulate, ()

                xs = (lmn, image, ellipse_params)

            init_accumulate = jnp.zeros((2, 2), dtype=self.dtype)
            vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs, unroll=1)
            return vis_accumulation  # [2, 2]

        visibilities = compute_visibility(
            gaussian_model_data.lmn,
            visibility_coords.uvw,
            g1,
            g2,
            freqs,
            gaussian_model_data.image,
            gaussian_model_data.ellipse_params
        )  # [chan, row, 2, 2]
        # make sure the output is [row, chan, 2, 2]
        return lax.transpose(visibilities, (1, 0, 2, 3))  # [row, chan, 2, 2]

    def _single_compute_visibilty(self, lmn, uvw, g1, g2, freq, image, ellipse_params):
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

        l0, m0, n0 = lmn  # scalar
        major, minor, theta = ellipse_params

        stokes_I = linear_to_stokes(image, flat_output=True)[0]

        vis_I = self._single_predict(
            u, v, w,
            A=stokes_I,
            l0=l0,
            m0=m0,
            n0=n0,
            major=major,
            minor=minor,
            theta=theta
        )
        zero = jnp.zeros_like(vis_I)
        vis = jnp.asarray([vis_I, zero, zero, zero])
        vis_linear = stokes_to_linear(vis, flat_output=False)  # [2, 2]
        return kron_product(g1, vis_linear, g2.T.conj())  # [2, 2]

    def _gaussian_fourier(self, u, v, A, l0, m0, major, minor, theta):
        """
        Computes the Fourier transform of the Gaussian source, over given u, v coordinates.

        Args:
            u: scalar
            v: scalar
            A: scalar
            l0: scalar
            m0: scalar
            major: scalar
            minor: scalar
            theta: scalar

        Returns:
            Fourier transformed Gaussian source evaluated at uvw
        """
        gaussian = Gaussian(
            x0=jnp.asarray([l0, m0]),
            major_fwhm=major,
            minor_fwhm=minor,
            pos_angle=theta,
            total_flux=A
        )
        return gaussian.fourier(jnp.asarray([u, v]))

    def _single_predict(self, u, v, w,
                        A,
                        l0, m0, n0, major, minor, theta):
        F = lambda u, v: self._gaussian_fourier(
            u, v,
            A=A,
            l0=l0,
            m0=m0,
            major=major,
            minor=minor,
            theta=theta
        )

        w_term = jnp.exp(-2j * jnp.pi * w * (n0 - 1)) / n0

        C = w_term

        if self.order_approx == 0:
            vis = F(u, v) * C
        elif self.order_approx == 1:

            warnings.warn("Order 1 approximation is not tested.")

            # F[I(l,m) * (C + (l - l0) * A + (m - m0) * B)]
            # = F[I(l,m)] * (C - l0 * A - m0 * B) + A * i / (2pi) * d/du F[I(l,m)] + B * i / (2pi) * d/dv F[I(l,m)]

            def wkernel(l, m):
                n = jnp.sqrt(1. - l ** 2 - m ** 2)
                return jnp.exp(-2j * jnp.pi * w * (n - 1)) / n

            C = wkernel(l0, m0)

            wkernel_grad = jax.grad(wkernel, (0, 1), holomorphic=True)

            _grad = wkernel_grad(l0 + 0j, m0 + 0j)
            A = _grad[0]
            B = _grad[1]

            F_jvp = JVPLinearOp(F, promote_dtypes=True)
            vec = (
                A * (1j / (2 * jnp.pi)),
                B * (1j / (2 * jnp.pi))
            )
            # promote_dtypes=True so we don't need to cast the primals here. Otherwise:
            # primals = (u.astype(vec[0].dtype), v.astype(vec[1].dtype))
            primals = (u, v)
            F_jvp = F_jvp(*primals)

            vis = F(u, v) * (C - l0 * A - m0 * B) + F_jvp.matvec(*vec)
        else:
            raise ValueError("order_approx must be 0 or 1")
        return vis
