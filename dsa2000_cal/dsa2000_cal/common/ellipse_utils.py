import dataclasses

import jax
import numpy as np
from jax import numpy as jnp


@dataclasses.dataclass(eq=False)
class Gaussian:
    x0: jax.Array  # [2]
    major_fwhm: jax.Array  # []
    minor_fwhm: jax.Array  # []
    pos_angle: jax.Array  # []
    total_flux: jax.Array  # []

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

    def fourier(self, k: jax.Array) -> jax.Array:
        """
        Compute the Fourier transform of the Gaussian at the given k coordinates.

            F(k) = int_R^2 f(x) e^(-2pi i k^T x) dx
                 = e^(-2pi i k^T x0) e^(-2 pi^2 k^T Sigma k)

                 Sigma^-1 = R^-T D^-T D^-1 R^-1
                 Sigma = R D D^T R^T
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
