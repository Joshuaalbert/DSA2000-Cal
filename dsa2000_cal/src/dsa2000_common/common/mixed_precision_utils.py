import dataclasses
import warnings
from typing import TypeVar

import jax
import numpy as np
from jax import numpy as jnp

if not jax.config.read('jax_enable_x64'):
    warnings.warn("JAX x64 is not enabled. Setting it now, but check for errors.")
    jax.config.update('jax_enable_x64', True)

# Create a float scalar to lock in dtype choices.
if jnp.array(1., jnp.float64).dtype != jnp.float64:
    raise RuntimeError("Failed to set float64 as default dtype.")

from dsa2000_common.common.alert_utils import get_grandparent_info

float_type = jnp.result_type(float)
int_type = jnp.result_type(int)
complex_type = jnp.result_type(complex)

T = TypeVar("T")


def _cast_floating_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, float):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.floating):
                warnings.warn(f"Expected float type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_complex_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, complex):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.complexfloating):
                warnings.warn(f"Expected complex type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_integer_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, int):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.integer):
                warnings.warn(f"Expected integer type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_bool_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, bool):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.bool_):
                warnings.warn(f"Expected bool type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


X = TypeVar("X")


@dataclasses.dataclass(frozen=True)
class Policy:
    """Encapsulates casting for inputs, outputs and parameters."""
    vis_dtype: jnp.dtype = np.complex64
    weight_dtype: jnp.dtype = np.float32
    flag_dtype: jnp.dtype = np.bool_
    image_dtype: jnp.dtype = np.float32
    gain_dtype: jnp.dtype = np.complex64
    index_dtype: jnp.dtype = np.int64
    freq_dtype: jnp.dtype = np.float64
    time_dtype: jnp.dtype = np.float64
    length_dtype: jnp.dtype = np.float64
    angle_dtype: jnp.dtype = np.float64

    def cast_to_vis(self, x: X, quiet: bool = False) -> X:
        """Converts visibility values to the visibility dtype."""
        return _cast_complex_to(x, self.vis_dtype, quiet=quiet)

    def cast_to_weight(self, x: X, quiet: bool = False) -> X:
        """Converts weight values to the weight dtype."""
        return _cast_floating_to(x, self.weight_dtype, quiet=quiet)

    def cast_to_flag(self, x: X, quiet: bool = False) -> X:
        """Converts flag values to the flag dtype."""
        return _cast_bool_to(x, self.flag_dtype, quiet=quiet)

    def cast_to_image(self, x: X, quiet: bool = False) -> X:
        """Converts image values to the image dtype."""
        return _cast_floating_to(x, self.image_dtype, quiet=quiet)

    def cast_to_gain(self, x: X, quiet: bool = False) -> X:
        """Converts gain values to the gain dtype."""
        return _cast_complex_to(x, self.gain_dtype, quiet=quiet)

    def cast_to_index(self, x: X, quiet: bool = False) -> X:
        """Converts lookup index values to the index dtype."""
        return _cast_integer_to(x, self.index_dtype, quiet=quiet)

    def cast_to_freq(self, x: X, quiet: bool = False) -> X:
        """Converts frequency values to the frequency dtype."""
        return _cast_floating_to(x, self.freq_dtype, quiet=quiet)

    def cast_to_time(self, x: X, quiet: bool = False) -> X:
        """Converts time values to the time dtype."""
        return _cast_floating_to(x, self.time_dtype, quiet=quiet)

    def cast_to_length(self, x: X, quiet: bool = False) -> X:
        """Converts length values to the position dtype."""
        return _cast_floating_to(x, self.length_dtype, quiet=quiet)

    def cast_to_angle(self, x: X, quiet: bool = False) -> X:
        """Converts lmn values to the lmn dtype."""
        return _cast_floating_to(x, self.angle_dtype, quiet=quiet)


mp_policy = Policy()
