from abc import ABC, abstractmethod
from functools import partial
from typing import List

import jax
import numpy as np
from astropy import coordinates as ac
from astropy import time as at
from astropy import units as au


class GainModel(ABC):
    """
    An abstract class for a gain model.
    """

    @abstractmethod
    def compute_gain(self, freqs: au.Quantity, sources: ac.ICRS, phase_tracking: ac.ICRS,
                     array_location: ac.EarthLocation, time: at.Time,
                     **kwargs):
        """
        Compute the beam gain at the given pointing direction.

        Args:
            freqs: the frequency values
            sources: (source_shape) the source coordinates
            phase_tracking: the pointing direction
            array_location: the location of the array reference location
            time: the time of the observation
            kwargs: additional keyword arguments

        Returns:
            (source_shape) + [num_ant, num_freq, 2, 2] The beam gain at the given source coordinates.
        """
        ...

    def __matmul__(self, other: 'GainModel'):
        if not isinstance(other, GainModel):
            raise ValueError("Can only multiply by another GainModel.")
        return ProductGainModel([self, other])

    def __rmatmul__(self, other):
        if not isinstance(other, GainModel):
            raise ValueError("Can only multiply by another GainModel.")
        return ProductGainModel([other, self])


class ProductGainModel(GainModel):
    """
    A product of gain models.
    """

    def __init__(self, gain_models: List[GainModel]):
        self.gain_models = []
        for gain_model in gain_models:
            if isinstance(gain_model, ProductGainModel):
                self.gain_models.extend(gain_model.gain_models)
            else:
                self.gain_models.append(gain_model)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_gain_jax(self, gains: List[jax.Array]) -> jax.Array:
        for gain in gains:
            if np.shape(gain) != np.shape(gains[0]):
                raise ValueError("All gains must have the same shape.")

        # perform A @ B @ ... on last two dims
        # matrix mul
        output = gains[0]
        for gain in gains[1:]:
            output = output @ gain
        return output

    def compute_gain(self, freqs: au.Quantity, sources: ac.ICRS, phase_tracking: ac.ICRS,
                     array_location: ac.EarthLocation, time: at.Time,
                     **kwargs):
        gains = [
            gain_model.compute_gain(freqs=freqs, sources=sources, phase_tracking=phase_tracking,
                                    array_location=array_location, time=time,
                                    **kwargs) for gain_model in self.gain_models
        ]

        return self._compute_gain_jax(gains)
