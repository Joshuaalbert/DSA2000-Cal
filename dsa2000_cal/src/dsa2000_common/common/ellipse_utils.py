import dataclasses

import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_common.common.array_types import FloatArray


@dataclasses.dataclass(eq=False)
class Gaussian:
    """
    Represents a Gaussian with definition:

        f(x) = 1/sigma_major/sigma_minor/(2 pi) e^(-1/2 (x - x0)^T R^-T D^-T D^-1 R^-1 (x - x0))

        F(s) = e^(-2pi i s^T x0) e^(-1/2 (4 pi^2) s^T R D D^T R^T s)
             = e^(-2pi i s^T x0) e^(-1/2 s^T R^-T E^-T E^-1 R^-1 s)
    where:
        sigma_minor is the semi-minor axis
        sigma_major is the semi-major axis
        x0 is the center of the Gaussian
        R is the rotation matrix
        D = diag(sigma_minor, sigma_major) / (2 sqrt(2 log(2)))
        E = diag((2pi)/sigma_minor, (2pi)/sigma_major) * / (2 sqrt(2 log(2)))
    """
    x0: FloatArray  # [2]
    major_fwhm: FloatArray  # []
    minor_fwhm: FloatArray  # []
    pos_angle: FloatArray  # []
    total_flux: FloatArray  # []

    def __post_init__(self):
        if np.shape(self.x0) != (2,):
            raise ValueError(f"x0 must have shape (2,), got {np.shape(self.x0)}")
        if np.shape(self.major_fwhm) != ():
            raise ValueError(f"major_fwhm must have shape (), got {np.shape(self.major_fwhm)}")
        if np.shape(self.minor_fwhm) != ():
            raise ValueError(f"minor_fwhm must have shape (), got {np.shape(self.minor_fwhm)}")
        if np.shape(self.pos_angle) != ():
            raise ValueError(f"pos_angle must have shape (), got {np.shape(self.pos_angle)}")

    def beam_area(self) -> jax.Array:
        """
        Calculate the area of the ellipse at half-power.
        """
        return (0.25 * np.pi) * self.major_fwhm * self.minor_fwhm

    def beam_solid_angle(self) -> jax.Array:
        """
        Calculate the integral of the Gaussian over the solid angle.
        """
        return np.pi / (4. * np.log(2.)) * self.major_fwhm * self.minor_fwhm

    def fourier(self, k: jax.Array) -> jax.Array:
        """
        Compute the Fourier transform of the Gaussian at the given k coordinates.

            F(k) = int_R^2 f(x) e^(-2pi i k^T x) dx
                 = e^(-2pi i k^T x0) e^(-2 pi^2 k^T Sigma k)

                 Sigma^-1 = R^-T D^-T D^-1 R^-1
                 Sigma = R D D^T R^T

        Note: Beware that this gives the exact Fourier transform, so if you combine with Point DFT
        (where you divide by adjoint normalising factor), you'll need to scale this by that too.
        Args:
            k: [2] the k coordinates

        Returns:
            the Fourier transform of the Gaussian at the given k coordinates
        """
        if np.shape(k) != (2,):
            raise ValueError(f"k must have shape (2,), got {np.shape(k)}")
        RT = ellipse_rotation(-self.pos_angle)
        fwhm = 2. * np.sqrt(2. * np.log(2))
        sigma_major = self.major_fwhm / fwhm
        sigma_minor = self.minor_fwhm / fwhm
        D_diag = jnp.asarray([sigma_minor, sigma_major])
        dk = D_diag * (RT @ k)
        return self.total_flux * jnp.exp(-2j * jnp.pi * jnp.sum(k * self.x0)) * jnp.exp(
            -2. * jnp.pi ** 2 * jnp.sum(jnp.square(dk)))

    def compute_flux_density(self, x: jax.Array) -> jax.Array:
        """
        Compute the flux of the Gaussian at the given x coordinates.

        f(x) = 1/sigma_major/sigma_minor/(2 pi) e^(-1/2 (x - x0)^T R^-T D^-T D^-1 R^-1 (x - x0))

        Args:
            x: the x coordinates

        Returns:
            the flux of the Gaussian at the given x coordinates
        """
        if np.shape(x) != (2,):
            raise ValueError(f"x must have shape (2,), got {np.shape(x)}")
        R_inv = ellipse_rotation(-self.pos_angle)
        fwhm = 2. * np.sqrt(2. * np.log(2))
        sigma_major = self.major_fwhm / fwhm
        sigma_minor = self.minor_fwhm / fwhm
        D_diag = jnp.asarray([sigma_minor, sigma_major])
        diff = x - self.x0
        diff = R_inv @ diff
        diff = diff / D_diag
        dist2 = jnp.sum(jnp.square(diff))
        norm_inv = jnp.reciprocal(sigma_major * sigma_minor * 2 * jnp.pi)
        return norm_inv * self.total_flux * jnp.exp(-0.5 * dist2)

    def peak_flux_density(self) -> jax.Array:
        """
        Calculate the peak flux of the ellipse. The F s.t.:

        total_flux = int_R^2 F * f(x) dx
        """
        fwhm = 2. * np.sqrt(2. * np.log(2))
        sigma_major = self.major_fwhm / fwhm
        sigma_minor = self.minor_fwhm / fwhm
        return self.total_flux * jnp.reciprocal(sigma_major * sigma_minor * 2 * jnp.pi)

    @staticmethod
    def total_flux_from_peak( peak_flux: jax.Array, major_fwhm: jax.Array, minor_fwhm: jax.Array) -> jax.Array:
        """
        Calculate the total flux of the ellipse from the peak flux. The F s.t.:

        total_flux = int_R^2 F * f(x) dx
        """
        fwhm = 2. * np.sqrt(2. * np.log(2))
        sigma_major = major_fwhm / fwhm
        sigma_minor = minor_fwhm / fwhm
        return peak_flux * (sigma_major * sigma_minor * 2 * jnp.pi)


def ellipse_rotation(pos_angle):
    return jnp.asarray([[jnp.cos(pos_angle), jnp.sin(pos_angle)], [-jnp.sin(pos_angle), jnp.cos(pos_angle)]])


def ellipse_eval(A, b_major, b_minor, pos_angle, l, m, l0, m0):
    """
    Evaluate the elliptical Gaussian at the given l, m coordinates.

    Args:
        b_major: the major axis
        b_minor: the minor axis
        pos_angle: the position angle
        l: the l coordinate
        m: the m coordinate

    Returns:
        the value of the Gaussian at the given l, m coordinates
    """
    gaussian = Gaussian(
        x0=jnp.asarray([l0, m0]),
        major_fwhm=b_major,
        minor_fwhm=b_minor,
        pos_angle=pos_angle,
        total_flux=A
    )
    return gaussian.compute_flux_density(jnp.asarray([l, m]))
