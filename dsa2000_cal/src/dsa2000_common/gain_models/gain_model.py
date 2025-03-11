import dataclasses
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_common.common.array_types import IntArray, FloatArray


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
    def compute_gain(self, freqs: FloatArray, times: FloatArray, lmn_geodesic: FloatArray,
                     antenna_indices: IntArray | None = None) -> jax.Array:
        """
        Compute the beam gain at the given pointing direction.

        Args:
            freqs: [num_freqs] the frequency values
            times: [num_times] the time values
            lmn_geodesic: [num_time, num_ant, num_sources, 3] the lmn coordinates of the source in frame of the antennas.
            antenna_indices: [num_ant] the antenna indices to compute the gain for. If None, compute for all antennas.

        Returns:
            [num_time, num_ant, num_freq, num_sources,[, 2, 2]] The beam gain at the given source coordinates.
        """
        ...

    def __matmul__(self, other: 'GainModel'):
        if not isinstance(other, GainModel):
            raise ValueError("Can only multiply by another GainModel.")
        return ProductGainModel([self, other])


def test_matmul():
    class A:
        def __init__(self, name: str):
            self.name = name

        def __matmul__(self, other):
            print('self', self.name, 'other', other.name)

    A('a') @ A('b')


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

    def compute_gain(self, freqs: jax.Array, times: jax.Array, lmn_geodesic: jax.Array,
                     antenna_indices: IntArray | None = None) -> jax.Array:
        gains = [
            gain_model.compute_gain(freqs=freqs, times=times, lmn_geodesic=lmn_geodesic,
                                    antenna_indices=antenna_indices)
            for gain_model in self.gain_models
        ]

        for gain in gains:
            if np.shape(gain) != np.shape(gains[0]):
                raise ValueError("All gains must have the same shape.")

        # perform A @ B @ ... on last two dims
        # matrix mul
        # TODO: could use associative scan for parallel computation
        output = gains[0]
        for gain in gains[1:]:
            if self.is_full_stokes():
                output = output @ gain
            else:
                output = output * gain
        return output

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
    def flatten(cls, this: "ProductGainModel") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        return (
            [this.gain_models],
            ()
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "ProductGainModel":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        [gain_models] = children
        return ProductGainModel(gain_models, skip_post_init=True)


ProductGainModel.register_pytree()
