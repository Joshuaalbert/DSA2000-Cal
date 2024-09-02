import jax
from jax import numpy as jnp
from jax._src import dtypes

original_jnp_array = jnp.array
original_jnp_asarray = jnp.asarray


def is_int_dtype(dtype):
    return dtype in [jnp.int32, jnp.int64, int]


def is_float_dtype(dtype):
    return dtype in [jnp.float32, jnp.float64, float]


def is_bool_type(dtype):
    return dtype in [jnp.bool_, bool]


def is_complex_dtype(dtype):
    return dtype in [jnp.complex64, jnp.complex128, complex]


class BitContext:
    current_context = None

    def __init__(self, bits: int):
        if bits not in [32, 64]:
            raise ValueError("Bits must be either 32 or 64.")
        self.bits = bits
        if not jax.config.read('jax_enable_x64'):
            raise ValueError("JAX x64 must be enabled for this context.")

    def __enter__(self):
        self.previous_context = BitContext.current_context
        BitContext.current_context = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        BitContext.current_context = self.previous_context


def get_default_dtype(object):
    """Determine the default dtype based on the current bit context."""
    if BitContext.current_context is not None:
        dtype = dtypes.canonicalize_dtype(jnp.result_type(object))
        if BitContext.current_context.bits == 64:
            if is_int_dtype(dtype):
                return jnp.int64
            elif is_float_dtype(dtype):
                return jnp.float64
            elif is_bool_type(dtype):
                return jnp.bool_
            elif is_complex_dtype(dtype):
                return jnp.complex128
        else:
            if is_int_dtype(dtype):
                return jnp.int32
            elif is_float_dtype(dtype):
                return jnp.float32
            elif is_bool_type(dtype):
                return jnp.bool_
            elif is_complex_dtype(dtype):
                return jnp.complex64
    return None


def patched_array(object, dtype=None, **kwargs):
    if dtype is None:  # Only override if dtype is not explicitly provided
        dtype = get_default_dtype(object)
    return original_jnp_array(object, dtype=dtype, **kwargs)


def patched_asarray(object, dtype=None, **kwargs):
    if dtype is None:  # Only override if dtype is not explicitly provided
        dtype = get_default_dtype(object)
    return original_jnp_asarray(object, dtype=dtype, **kwargs)


# Patch jnp.array and jnp.asarray with our custom versions
jnp.array = patched_array
jnp.asarray = patched_asarray
