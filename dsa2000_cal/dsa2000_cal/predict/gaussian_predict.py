import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
from astropy import constants
from jax import numpy as jnp, lax
from jax._src.typing import SupportsDType

from dsa2000_cal.common.jvp_linear_op import JVPLinearOp
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.vec_utils import kron_product
from dsa2000_cal.source_models.corr_translation import linear_to_stokes, stokes_to_linear


class GaussianModelData(NamedTuple):
    """
    Data for predict.
    """
    image: jax.Array  # [source, chan, 2, 2] in [[xx, xy], [yx, yy]] format
    gains: jax.Array  # [[source,] time, ant, chan, 2, 2]
    lmn: jax.Array  # [source, 3]
    ellipse_params: jax.Array  # [source, 3]


@dataclasses.dataclass(eq=False)
class GaussianPredict:
    order_approx: int = 0
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64

    def predict(self, freqs: jax.Array, gaussian_model_data: GaussianModelData,
                visibility_coords: VisibilityCoords) -> jax.Array:
        g1 = gaussian_model_data.gains[
             ..., visibility_coords.time_idx, visibility_coords.antenna_1, :, :, :
             ]  # [[source,] row, chan, 2, 2]
        g2 = gaussian_model_data.gains[
             ..., visibility_coords.time_idx, visibility_coords.antenna_2, :, :, :
             ]  # [[source,] row, chan, 2, 2]

        direction_dependent_gains = len(np.shape(g1)) == 5

        # Data will be sharded over frequency so don't reduce over these dimensions, or else communication happens.
        # We want the outer broadcast to be over chan, so we'll do this order:
        # chan -> source -> row
        # TODO: explore other orders

        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: [[source,] row, chan, 2, 2]
        # freq: [chan]
        # image: [source, chan, 2, 2]
        # ellipse_params: [source, 3]
        @partial(jax.vmap, in_axes=[None, None, -3, -3, 0, 1, None])
        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: [[source,] row, 2, 2]
        # freq: []
        # image: [source, 2, 2]
        # ellipse_params: [source, 3]
        @partial(jax.vmap, in_axes=[0, None,
                                    0 if direction_dependent_gains else None,
                                    0 if direction_dependent_gains else None,
                                    None, 0, 0])
        # lmn: [3]
        # uvw: [rows, 3]
        # g1, g2: [row, 2, 2]
        # freq: []
        # image: [2, 2]
        # ellipse_params: [3]
        @partial(jax.vmap, in_axes=[None, 0, 0, 0, None, None, None])
        # lmn: [3]
        # uvw: [3]
        # g1, g2: [2, 2]
        # freq: []
        # image: [2, 2]
        # ellipse_params: [3]
        def compute_visibility(*args):
            return self._single_compute_visibilty(*args)

        visibilities = compute_visibility(
            gaussian_model_data.lmn,
            visibility_coords.uvw,
            g1,
            g2,
            freqs,
            gaussian_model_data.image,
            gaussian_model_data.ellipse_params
        )  # [chan, source, row, 2, 2]
        visibilities = jnp.sum(visibilities, axis=1)  # [chan, row, 2, 2]
        # make sure the output is [row, chan, 2, 2]
        return lax.transpose(visibilities, (1, 0, 2, 3))

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
            wavelength=wavelength,
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

    def _gaussian_fourier(self, u, v, wavelength, A, l0, m0, major, minor, theta):
        """
        Computes the Fourier transform of the Gaussian source, over given u, v coordinates.

        Args:
            u: scalar
            v: scalar
            wavelength: scalar
            A: scalar
            l0: scalar
            m0: scalar
            major: scalar
            minor: scalar
            theta: scalar

        Returns:
            Fourier transformed Gaussian source evaluated at uvw
        """
        # l = l0 + R @ r @ l_circ
        # Gaussian source: e^(- l_circ(l)^2 / (2 * FWHM^2)) such that 1/2 = e^(-1/(2 * FWHM^2))
        # FWHM = 1/sqrt(2*log(2))
        # FT(l -> u) = FT(l_circ -> u')

        fwhm = 1. / np.sqrt(2.0 * np.log(2.0))

        # Scale uvw by wavelength
        u /= wavelength
        v /= wavelength

        # Spectral shape, Convert to l-projection, m-projection: u' = (r @ R @ u)
        u_prime = u * minor * jnp.cos(theta) - v * minor * jnp.sin(theta)
        v_prime = u * major * jnp.sin(theta) + v * major * jnp.cos(theta)

        # phase shift
        phase_shift = -2 * jnp.pi * (u * l0 + v * m0)

        # Calculate spectral decay: FT[exp(-a x^2)] = sqrt(pi/a) * exp(-pi^2 u^2 / a)
        # With a = 1 / (2 FWHM**2)
        # 2D norm_factor: pi / a
        norm_factor = np.pi * 2.0 * fwhm ** 2
        spectral_decay_factor = (-np.pi ** 2 * 2. * fwhm ** 2) * (u_prime ** 2 + v_prime ** 2)
        spectral_decay = norm_factor * jnp.exp(spectral_decay_factor)

        # Calculate the fourier transform
        return (A * spectral_decay) * (jnp.cos(phase_shift) + jnp.sin(phase_shift) * 1j)

    def _single_predict(self, u, v, w,
                        wavelength, A,
                        l0, m0, n0, major, minor, theta):
        F = lambda u, v: self._gaussian_fourier(
            u, v,
            wavelength=wavelength,
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
            # F[I(l,m) * (C + (l - l0) * A + (m - m0) * B)]
            # = F[I(l,m)] * (C - l0 * A - m0 * B) + A * i / (2pi) * d/du F[I(l,m)] + B * i / (2pi) * d/dv F[I(l,m)]
            A = l0 * (1. + 2j * jnp.pi * w * n0) * w_term / n0 ** 2
            B = m0 * (1. + 2j * jnp.pi * w * n0) * w_term / n0 ** 2

            F_jvp = JVPLinearOp(F, promote_dtypes=True)
            vec = (
                (A / (2 * jnp.pi)) * 1j,
                (B / (2 * jnp.pi)) * 1j
            )
            # promote_dtypes=True so we don't need to cast the primals here. Otherwise:
            # primals = (u.astype(vec[0].dtype), v.astype(vec[1].dtype))
            primals = (u, v)
            F_jvp = F_jvp(*primals)

            vis = F(u, v) * (C - l0 * A - m0 * B) + F_jvp.matvec(*vec)
        else:
            raise ValueError("order_approx must be 0 or 1")
        return vis
