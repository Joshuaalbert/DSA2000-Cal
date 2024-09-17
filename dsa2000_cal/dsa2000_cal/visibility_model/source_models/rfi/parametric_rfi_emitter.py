import dataclasses

import jax
import numpy as np
from astropy import units as au
from jax import numpy as jnp

from dsa2000_cal.common.interp_utils import left_broadcast_multiply
from dsa2000_cal.common.types import FloatArray


@dataclasses.dataclass(eq=False)
class ParametricDelayACF:
    mu: FloatArray  # [E]
    fwhp: FloatArray  # [E]
    spectral_power: FloatArray  # [E[,2,2]] in Jy*m^2/Hz
    channel_lower: FloatArray  # [chan]
    channel_upper: FloatArray  # [chan]
    resolution: int = 32  # Should be chosen so that channel width / resolution ~ PFB kernel resolution
    convention: str = 'physical'  # Doesn't matter for the ACF

    def __post_init__(self):
        if len(np.shape(self.mu)) != 1:
            raise ValueError(f"mu must be 1D, got {np.shape(self.mu)}")

        if np.shape(self.mu) != np.shape(self.fwhp):
            raise ValueError("mu and fwhp must have the same shape")

        if np.shape(self.spectral_power) not in [(len(self.mu), 2, 2), (len(self.mu),)]:
            raise ValueError(f"spectral_power must have the shape [E[,2,2]] got {np.shape(self.spectral_power)}")

        if len(np.shape(self.channel_lower)) != 1:
            raise ValueError(f"channel_lower must be 1D, got {np.shape(self.channel_lower)}")

        if np.shape(self.channel_lower) != np.shape(self.channel_upper):
            raise ValueError("channel_lower and channel_upper must have the same shape")

    @property
    def shape(self):
        shape = list(np.shape(self.spectral_power))
        shape.insert(1, np.shape(self.channel_lower)[0])
        return tuple(shape)

    @staticmethod
    def get_resolution(channel_width: au.Quantity) -> int:
        if not channel_width.unit.is_equivalent("Hz"):
            raise ValueError("channel_width must be in Hz")
        resolution = int(2 * channel_width / (10 * au.kHz))
        return resolution

    def __call__(self, tau: jax.Array) -> jax.Array:
        """
        Compute the auto-correlation function of the RFI signal, using sinc parametrisation of power spectrum.

        Args:
            tau: delay time

        Returns:
            Delay ACF of the RFI signal
        """

        def sinc(x: jax.Array):
            # sinc((fwhm/2) / sigma) = 1/2 ==> (fwhm/2) / sigma = 1.89549 ==> sigma = fwhm / (2 * 1.89549)
            sigma = self.fwhp / jnp.asarray(2. * 1.89549, dtype=self.fwhp.dtype)
            return jnp.sinc((x - self.mu) / sigma)

        def single_channel_acf(lower, upper):
            channel_abscissa = jnp.linspace(lower, upper, self.resolution)  # [R]
            dnu = channel_abscissa[1] - channel_abscissa[0]  # []
            # memory changes fastest over the last axis better for DFT
            spectrum = jax.vmap(sinc, in_axes=0, out_axes=1)(channel_abscissa)  # [E, R]
            # Compute the ACF at tau using inverse Fourier transform
            if self.convention == 'physical':  # +2pi
                acf = jnp.sum(spectrum * jnp.exp(2j * jnp.pi * channel_abscissa * tau), axis=1) * dnu  # [E]
            elif self.convention == 'engineering':
                acf = jnp.sum(spectrum * jnp.exp(-2j * jnp.pi * channel_abscissa * tau), axis=1) * dnu  # [E]
            else:
                raise ValueError(f"Invalid convention {self.convention}")
            return left_broadcast_multiply(self.spectral_power, acf)  # [E[,2,2]]

        return jax.vmap(single_channel_acf, in_axes=0, out_axes=1)(
            self.channel_lower, self.channel_upper)  # [E, chan[,2,2]]


# Define how the object is flattened (converted to a list of leaves and a context tuple)
def parametric_delay_acf_flatten(parametric_delay_acf: ParametricDelayACF):
    return (
        [parametric_delay_acf.mu, parametric_delay_acf.fwhp, parametric_delay_acf.spectral_power,
         parametric_delay_acf.channel_lower, parametric_delay_acf.channel_upper], (
            parametric_delay_acf.resolution, parametric_delay_acf.convention))


# Define how the object is unflattened (reconstructed from leaves and context)
def parametric_delay_acf_unflatten(aux_data, children):
    resolution, convention = aux_data
    mu, fwhp, spectral_power, channel_lower, channel_upper = children
    return ParametricDelayACF(mu=mu, fwhp=fwhp, spectral_power=spectral_power,
                              channel_lower=channel_lower, channel_upper=channel_upper,
                              resolution=resolution, convention=convention)


# Register the custom pytree
jax.tree_util.register_pytree_node(
    ParametricDelayACF,
    parametric_delay_acf_flatten,
    parametric_delay_acf_unflatten
)
