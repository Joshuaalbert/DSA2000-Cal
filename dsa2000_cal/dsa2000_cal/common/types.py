import dataclasses
import warnings
from typing import Union, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_cal.common.alert_utils import get_grandparent_info

if not jax.config.read('jax_enable_x64'):
    warnings.warn("JAX x64 is not enabled. Setting it now, but check for errors.")
    jax.config.update('jax_enable_x64', True)

# Create a float scalar to lock in dtype choices.
if jnp.array(1., jnp.float64).dtype != jnp.float64:
    raise RuntimeError("Failed to set float64 as default dtype.")

float_type = jnp.result_type(float)
int_type = jnp.result_type(int)
complex_type = jnp.result_type(complex)

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

T = TypeVar("T")


def _cast_floating_to(tree: T, dtype: jnp.dtype) -> T:
    def conditional_cast(x):
        if isinstance(x, float):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not jnp.issubdtype(x.dtype, jnp.floating):
                warnings.warn(f"Expected float type, got {x.dtype}, {get_grandparent_info(6)}.")
            return x.astype(dtype)
        except AttributeError:
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_complex_to(tree: T, dtype: jnp.dtype) -> T:
    def conditional_cast(x):
        if isinstance(x, complex):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not jnp.issubdtype(x.dtype, jnp.complexfloating):
                warnings.warn(f"Expected complex type, got {x.dtype}, {get_grandparent_info(6)}.")
            return x.astype(dtype)
        except AttributeError:
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_integer_to(tree: T, dtype: jnp.dtype) -> T:
    def conditional_cast(x):
        if isinstance(x, int):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not jnp.issubdtype(x.dtype, jnp.integer):
                warnings.warn(f"Expected integer type, got {x.dtype}, {get_grandparent_info(6)}.")
            return x.astype(dtype)
        except AttributeError:
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_bool_to(tree: T, dtype: jnp.dtype) -> T:
    def conditional_cast(x):
        if isinstance(x, bool):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not jnp.issubdtype(x.dtype, jnp.bool_):
                warnings.warn(f"Expected bool type, got {x.dtype}, {get_grandparent_info(6)}.")
            return x.astype(dtype)
        except AttributeError:
            return x

    return jax.tree.map(conditional_cast, tree)


X = TypeVar("X")


@dataclasses.dataclass(frozen=True)
class Policy:
    """Encapsulates casting for inputs, outputs and parameters."""
    vis_dtype: jnp.dtype = jnp.complex64
    weight_dtype: jnp.dtype = jnp.float32
    flag_dtype: jnp.dtype = jnp.bool_
    image_dtype: jnp.dtype = jnp.float32
    gain_dtype: jnp.dtype = jnp.complex64
    index_dtype: jnp.dtype = jnp.int64
    freq_dtype: jnp.dtype = jnp.float64
    time_dtype: jnp.dtype = jnp.float64
    length_dtype: jnp.dtype = jnp.float64
    angle_dtype: jnp.dtype = jnp.float64

    def cast_to_vis(self, x: X) -> X:
        """Converts visibility values to the visibility dtype."""
        return _cast_complex_to(x, self.vis_dtype)

    def cast_to_weight(self, x: X) -> X:
        """Converts weight values to the weight dtype."""
        return _cast_floating_to(x, self.weight_dtype)

    def cast_to_flag(self, x: X) -> X:
        """Converts flag values to the flag dtype."""
        return _cast_bool_to(x, self.flag_dtype)

    def cast_to_image(self, x: X) -> X:
        """Converts image values to the image dtype."""
        return _cast_floating_to(x, self.image_dtype)

    def cast_to_gain(self, x: X) -> X:
        """Converts gain values to the gain dtype."""
        return _cast_complex_to(x, self.gain_dtype)

    def cast_to_index(self, x: X) -> X:
        """Converts lookup index values to the index dtype."""
        return _cast_integer_to(x, self.index_dtype)

    def cast_to_freq(self, x: X) -> X:
        """Converts frequency values to the frequency dtype."""
        return _cast_floating_to(x, self.freq_dtype)

    def cast_to_time(self, x: X) -> X:
        """Converts time values to the time dtype."""
        return _cast_floating_to(x, self.time_dtype)

    def cast_to_length(self, x: X) -> X:
        """Converts length values to the position dtype."""
        return _cast_floating_to(x, self.length_dtype)

    def cast_to_angle(self, x: X) -> X:
        """Converts lmn values to the lmn dtype."""
        return _cast_floating_to(x, self.angle_dtype)


mp_policy = Policy()
