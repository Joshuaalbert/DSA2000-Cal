import dataclasses
from abc import ABC, abstractmethod
from typing import List

import jax
import numpy as np


@dataclasses.dataclass(eq=False)
class GainModel(ABC):
    """
    An abstract class for an antenna-based gain model.

    An antenna-based gain model is one where gains are indexed functions g_i:: (Freq, Source, Time) -> C^2x2

    The antenna positions are stored in the `antennas` attribute, and can be accessed by the methods if needed.
    """

    @abstractmethod
    def is_full_stokes(self) -> bool:
        """
        Check if the gain model is full Stokes.

        Returns:
            bool: True if full Stokes, False otherwise
        """
        ...

    @abstractmethod
    def compute_gain(self, freqs: jax.Array, times: jax.Array, lmn_geodesic: jax.Array) -> jax.Array:
        """
        Compute the beam gain at the given pointing direction.

        Args:
            freqs: [num_freqs] the frequency values
            times: [num_times] the time values
            lmn_geodesic: [num_sources, num_time, num_ant, 3] the lmn coordinates of the source in frame of the antennas.

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


@dataclasses.dataclass(eq=False)
class ProductGainModel(GainModel):
    """
    A product of gain models.
    """

    gain_models: List[GainModel]
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return
        gain_models = []
        for gain_model in self.gain_models:
            if isinstance(gain_model, ProductGainModel):
                gain_models.extend(gain_model.gain_models)
            else:
                gain_models.append(gain_model)
        self.gain_models = gain_models
        full_stokes = []
        for gain_model in self.gain_models:
            full_stokes.append(gain_model.is_full_stokes())
        if len(set(full_stokes)) == 1:
            self._is_full_stokes = full_stokes[0]
        else:
            raise ValueError("All gain models must be the same type.")

    def is_full_stokes(self) -> bool:
        return self._is_full_stokes

    def compute_gain(self, freqs: jax.Array, times: jax.Array, lmn_geodesic: jax.Array) -> jax.Array:
        gains = [
            gain_model.compute_gain(freqs=freqs, times=times, lmn_geodesic=lmn_geodesic) for gain_model in
            self.gain_models
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


def product_gain_model_flatten(product_gain_model: ProductGainModel):
    return (
        [product_gain_model.gain_models],
        ()
    )


def product_gain_model_unflatten(aux_data, children) -> ProductGainModel:
    gain_models = children
    return ProductGainModel(gain_models, skip_post_init=True)


jax.tree_util.register_pytree_node(
    ProductGainModel,
    product_gain_model_flatten,
    product_gain_model_unflatten
)
