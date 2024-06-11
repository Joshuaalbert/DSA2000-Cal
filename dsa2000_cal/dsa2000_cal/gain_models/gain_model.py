import dataclasses
from abc import ABC, abstractmethod
from functools import partial
from typing import List

import jax
import numpy as np
from astropy import coordinates as ac
from astropy import time as at
from astropy import units as au
from tomographic_kernel.frames import ENU


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

    def check_inputs(self, freqs: au.Quantity, sources: ac.ICRS | ENU, pointing: ac.ICRS | None,
                     array_location: ac.EarthLocation,
                     time: at.Time, **kwargs):
        if not isinstance(freqs, au.Quantity):
            raise ValueError("freqs must be a Quantity.")
        if not freqs.unit.is_equivalent(au.Hz):
            raise ValueError("freqs must be in Hz.")
        if freqs.isscalar:
            raise ValueError("freqs must be an array.")
        if len(freqs.shape) != 1:
            raise ValueError("freqs must be a 1D array.")
        if not isinstance(sources, (ac.ICRS, ENU)):
            raise ValueError("sources must be an ICRS or ENU object.")
        if pointing is not None:
            if not isinstance(pointing, ac.ICRS):
                raise ValueError("pointing must be an ICRS or ENU object.")
            try:
                np.broadcast_shapes(pointing.shape, self.antennas.shape)
            except ValueError:
                raise ValueError("pointing must have broadcastable shape to the antennas.")
        if not isinstance(array_location, ac.EarthLocation):
            raise ValueError("array_location must be an EarthLocation object.")
        if not isinstance(time, at.Time):
            raise ValueError("time must be a Time object.")

    @abstractmethod
    def compute_gain(self, freqs: au.Quantity, sources: ac.ICRS | ENU, pointing: ac.ICRS | None,
                     array_location: ac.EarthLocation,
                     time: at.Time, **kwargs):
        """
        Compute the beam gain at the given pointing direction.

        Args:
            freqs: [num_freqs] the frequency values
            sources: (source_shape) the source coordinates, ENU then assumed to be the location of the sources in near
                field, and the location of antennas is used to compute gains in direction of sources.
            pointing: [[num_ant]] scalar or 1D array of the pointing direction of each antenna,
                optionally None == zenith pointing, calculated per antenna.
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

    def compute_gain(self, freqs: au.Quantity, sources: ac.ICRS | ENU, pointing: ac.ICRS | None,
                     array_location: ac.EarthLocation,
                     time: at.Time, **kwargs):
        gains = [
            gain_model.compute_gain(freqs=freqs, sources=sources, pointing=pointing, array_location=array_location,
                                    time=time, **kwargs) for gain_model in self.gain_models
        ]

        return self._compute_gain_jax(gains)
