# Context manager class
from contextlib import ContextDecorator


import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from jax._src.dtypes import _canonicalize_dtype

class BitContext(ContextDecorator):
    def __init__(self, bits: int):
        self.bits = bits
        self.prev_flag = None

    def __enter__(self):
        # Store the current state of the jax_enable_x64 flag
        self.prev_flag = jax.config.read('jax_enable_x64')

        # Update the jax_enable_x64 flag based on the bits argument
        jax.config.update('jax_enable_x64', self.bits == 64)

        # Clear the _canonicalize_dtype cache
        _canonicalize_dtype.cache_clear()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original jax_enable_x64 flag
        jax.config.update('jax_enable_x64', self.prev_flag)

        # Clear the _canonicalize_dtype cache again
        _canonicalize_dtype.cache_clear()

def test_bit_context():
    @jax.jit
    def run():
        with BitContext(32):
            a = jnp.asarray(1.)
            b = a + 1.
        with BitContext(64):
            c = jnp.asarray(1.)
            d = c + 1.
        return a, b, c, d

    a, b, c, d = run()
    assert a.dtype == jnp.float32
    assert b.dtype == jnp.float32
    assert c.dtype == jnp.float64
    assert d.dtype == jnp.float64