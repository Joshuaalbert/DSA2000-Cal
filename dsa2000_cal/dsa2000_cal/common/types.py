from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

float_type = jnp.result_type(float)
int_type = jnp.result_type(int)
complex_type = jnp.result_type(complex)

# Create a float scalar to lock in dtype choices.
_ = jnp.array(1., float_type)

PRNGKey = jax.Array

Array = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
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

FloatArray.__doc__ = "Type annotation for JAX array-like objects, with float scalar types."

IntArray.__doc__ = "Type annotation for JAX array-like objects, with int scalar types."

BoolArray.__doc__ = "Type annotation for JAX array-like objects, with bool scalar types."

vis_dtype = jnp.complex64
gain_dtype = jnp.complex64
index_dtype = jnp.int32
freq_dtype = jnp.float32
time_dtype = jnp.float64
position_dtype = jnp.float64


def a_(x):
    return jnp.asarray(x)
