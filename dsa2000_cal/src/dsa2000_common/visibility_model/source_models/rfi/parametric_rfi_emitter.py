import dataclasses
import pickle
import warnings
from typing import Tuple, Any, List

import jax
import numpy as np
from astropy import units as au
from jax import numpy as jnp

from dsa2000_common.common.array_types import FloatArray, ComplexArray
from dsa2000_common.common.interp_utils import left_broadcast_multiply
from dsa2000_common.visibility_model.source_models.rfi.abc import AbstractRFIAutoCorrelationFunction


@dataclasses.dataclass(eq=False)
class ParametricDelayACF(AbstractRFIAutoCorrelationFunction):
    mu: FloatArray  # [E]
    fwhp: FloatArray  # [E]
    spectral_power: FloatArray  # [E[,2,2]] in Jy*m^2/Hz
    channel_lower: FloatArray  # [chan]
    channel_upper: FloatArray  # [chan]
    resolution: int = 32  # Should be chosen so that channel width / resolution ~ PFB kernel resolution
    convention: str = 'physical'  # Doesn't matter for the ACF
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return
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
        # 10 kHz is a typical PFB kernel resolution, we double sample it.
        resolution = int(2 * channel_width / (10 * au.kHz))
        return resolution

    def eval(self, freq: FloatArray, tau: FloatArray) -> ComplexArray:

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

        # TODO: Need to filer for freq
        return jax.vmap(single_channel_acf, in_axes=0, out_axes=1)(
            self.channel_lower, self.channel_upper)  # [E, chan[,2,2]]

    def save(self, filename: str):
        """
        Serialise the model to file.

        Args:
            filename: the filename
        """
        if not filename.endswith('.pkl'):
            warnings.warn(f"Filename {filename} does not end with .pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        """
        Load the model from file.

        Args:
            filename: the filename

        Returns:
            the model
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        children, aux_data = self.flatten(self)
        children_np = jax.tree.map(np.asarray, children)
        serialised = (aux_data, children_np)
        return (self._deserialise, (serialised,))

    @classmethod
    def _deserialise(cls, serialised):
        # Create a new instance, bypassing __init__ and setting the actor directly
        (aux_data, children_np) = serialised
        children_jax = jax.tree.map(jnp.asarray, children_np)
        return cls.unflatten(aux_data, children_jax)

    @classmethod
    def register_pytree(cls):
        jax.tree_util.register_pytree_node(cls, cls.flatten, cls.unflatten)

    # an abstract classmethod
    @classmethod
    def flatten(cls, this: "ParametricDelayACF") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        return (
            [this.mu, this.fwhp, this.spectral_power,
             this.channel_lower, this.channel_upper], (
                this.resolution, this.convention))

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "ParametricDelayACF":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        resolution, convention = aux_data
        mu, fwhp, spectral_power, channel_lower, channel_upper = children
        return ParametricDelayACF(mu=mu, fwhp=fwhp, spectral_power=spectral_power,
                                  channel_lower=channel_lower, channel_upper=channel_upper,
                                  resolution=resolution, convention=convention,
                                  skip_post_init=True)


ParametricDelayACF.register_pytree()
