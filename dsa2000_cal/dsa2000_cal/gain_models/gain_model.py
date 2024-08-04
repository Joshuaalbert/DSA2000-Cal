import dataclasses
from abc import ABC, abstractmethod
from typing import List

import jax
import numpy as np
from astropy import coordinates as ac


@dataclasses.dataclass(eq=False)
class GainModel(ABC):
    """
    An abstract class for an antenna-based gain model.

    An antenna-based gain model is one where gains are indexed functions g_i:: (Freq, Source, Time) -> C^2x2

    The antenna positions are stored in the `antennas` attribute, and can be accessed by the methods if needed.

    Args:
        antennas: [num_ant] antenna positions
    """
    antennas: ac.EarthLocation  # [num_ant]
    tile_antennas: bool

    @abstractmethod
    def is_full_stokes(self) -> bool:
        """
        Check if the gain model is full Stokes.

        Returns:
            bool: True if full Stokes, False otherwise
        """
        ...

    @abstractmethod
    def compute_gain(self, freqs: jax.Array, times: jax.Array, geodesics: jax.Array) -> jax.Array:
        """
        Compute the beam gain at the given pointing direction.

        Args:
            freqs: [num_freqs] the frequency values
            times: [num_times] the time values
            geodesics: [num_sources, num_time, num_ant, 3] the lmn coordinates of the source in frame of the antennas.

        Returns:
            [num_sources, num_time, num_ant, num_freq[, 2, 2]] The beam gain at the given source coordinates.
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
        full_stokes = []
        for gain_model in self.gain_models:
            full_stokes.append(gain_model.is_full_stokes())
        if len(set(full_stokes)) == 1:
            self._is_full_stokes = full_stokes[0]
        else:
            raise ValueError("All gain models must be the same type.")

    def is_full_stokes(self) -> bool:
        return self._is_full_stokes

    def compute_gain(self, freqs: jax.Array, times: jax.Array, geodesics: jax.Array) -> jax.Array:
        gains = [
            gain_model.compute_gain(freqs=freqs, times=times, geodesics=geodesics) for gain_model in self.gain_models
        ]

        for gain in gains:
            if np.shape(gain) != np.shape(gains[0]):
                raise ValueError("All gains must have the same shape.")

        # perform A @ B @ ... on last two dims
        # matrix mul
        # TODO: could use associative scan for parallel computation
        output = gains[0]
        for gain in gains[1:]:
            output = output @ gain
        return output
