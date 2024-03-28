from abc import ABC, abstractmethod
from typing import List

import numpy as np
from astropy import coordinates as ac
from astropy import time as at


class GainModel(ABC):
    """
    An abstract class for a gain model.
    """
    @abstractmethod
    def compute_beam(self, sources: ac.ICRS, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time,
                     **kwargs):
        """
        Compute the beam gain at the given pointing direction.

        Args:
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


class ProductGainModel(GainModel):
    """
    A product of gain models.
    """
    def __init__(self, gain_models: List[GainModel]):
        self.gain_models = gain_models

    def compute_beam(self, sources: ac.ICRS, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time):
        gains = [
            gain_model.compute_beam(sources, phase_tracking, array_location, time) for gain_model in self.gain_models
        ]
        for gain in gains:
            if np.shape(gain) != np.shape(gains[0]):
                raise ValueError("All gains must have the same shape.")

        # perform A @ B @ ... on last two dims
        # matrix mul
        output = gains[0]
        for gain in gains[1:]:
            output = np.matmul(output, gain)
        return output


def get_interp_indices_and_weights(x, xp) -> tuple[tuple[int, float], tuple[int, float]]:
    """
    One-dimensional linear interpolation. Outside bounds is also linear from nearest two points.

    Args:
        x: the x-coordinates at which to evaluate the interpolated values
        xp: the x-coordinates of the data points, must be increasing

    Returns:
        the interpolated values, same shape as `x`
    """

    x = np.asarray(x, dtype=np.float_)
    xp = np.asarray(xp, dtype=np.float_)

    # xp_arr = np.concatenate([xp[:1], xp, xp[-1:]])
    xp_arr = xp

    i = np.clip(np.searchsorted(xp_arr, x, side='right'), 1, len(xp_arr) - 1)
    dx = xp_arr[i] - xp_arr[i - 1]
    delta = x - xp_arr[i - 1]

    epsilon = np.spacing(np.finfo(xp_arr.dtype).eps)
    dx0 = np.abs(dx) <= epsilon  # Prevent NaN gradients when `dx` is small.
    # f = jnp.where(dx0, fp_arr[i - 1], fp_arr[i - 1] + (delta / jnp.where(dx0, 1, dx)) * df)
    dx = np.where(dx0, 1, dx)
    alpha = delta / dx
    return (i - 1, (1. - alpha)), (i, alpha)
