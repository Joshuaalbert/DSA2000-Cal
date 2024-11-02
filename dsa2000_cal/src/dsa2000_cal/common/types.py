from typing import Union, NamedTuple

import jax
import numpy as np

PRNGKey = jax.Array

Array = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
]
ComplexArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    complex
]
FloatArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    float,  # valid scalars
]
IntArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    int,  # valid scalars
]
BoolArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_, bool,  # valid scalars
]

Array.__doc__ = "Type annotation for JAX array-like objects, with no scalar types."

ComplexArray.__doc__ = "Type annotation for JAX array-like objects, with complex scalar types."

FloatArray.__doc__ = "Type annotation for JAX array-like objects, with float scalar types."

IntArray.__doc__ = "Type annotation for JAX array-like objects, with int scalar types."

BoolArray.__doc__ = "Type annotation for JAX array-like objects, with bool scalar types."


class VisibilityCoords(NamedTuple):
    """
    Coordinates for a single visibility.
    """
    uvw: FloatArray  # [rows, 3] the uvw coordinates
    time_obs: FloatArray  # [rows] the time relative to the reference time (observation start)
    antenna_1: IntArray  # [rows] the first antenna
    antenna_2: IntArray  # [rows] the second antenna
    time_idx: IntArray  # [rows] the time index
