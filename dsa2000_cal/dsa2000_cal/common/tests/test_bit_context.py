import jax
import jax.numpy as jnp
import pytest

from dsa2000_cal.common.bit_context import BitContext


@pytest.mark.requires_64bit
def test_bit_context():
    with BitContext(64):
        a = jnp.array(0.)
        assert a.dtype == jnp.float64

        # Explicitly specifying float32 bypasses context
        b = jnp.array(0., dtype=jnp.float32)
        assert b.dtype == jnp.float32

    with BitContext(32):
        a = jnp.array(0.)
        assert a.dtype == jnp.float32

        # Explicitly specifying float64 bypasses context
        b = jnp.array(0., dtype=jnp.float64)
        assert b.dtype == jnp.float64

    # test with int32 and int64
    with BitContext(64):
        a = jnp.array(0)
        assert a.dtype == jnp.int64

        b = jnp.array(0, dtype=jnp.int32)
        assert b.dtype == jnp.int32

    with BitContext(32):
        a = jnp.array(0)
        assert a.dtype == jnp.int32

        b = jnp.array(0, dtype=jnp.int64)
        assert b.dtype == jnp.int64


@pytest.mark.requires_64bit
def test_bit_context_under_jit():
    @jax.jit
    def add64():
        with BitContext(64):
            x = jnp.array(1.)
            y = jnp.array(2.)
            return x + y

    @jax.jit
    def add32():
        with BitContext(32):
            x = jnp.array(1.)
            y = jnp.array(2.)
            return x + y

    assert add64().dtype == jnp.float64

    assert add32().dtype == jnp.float32

    @jax.jit
    def nested64():
        with BitContext(32):
            with BitContext(64):
                x = jnp.array(1.)
                y = jnp.array(2.)
                return x + y

    assert nested64().dtype == jnp.float64

    @jax.jit
    def nested32():
        with BitContext(64):
            with BitContext(32):
                x = jnp.array(1.)
                y = jnp.array(2.)
                return x + y

    assert nested32().dtype == jnp.float32

    # Test for int32 and int64 too

    @jax.jit
    def add64_int():
        with BitContext(64):
            x = jnp.array(1)
            y = jnp.array(2)
            return x + y

    @jax.jit
    def add32_int():
        with BitContext(32):
            x = jnp.array(1)
            y = jnp.array(2)
            return x + y

    assert add64_int().dtype == jnp.int64
    assert add32_int().dtype == jnp.int32

    @jax.jit
    def nested64_int():
        with BitContext(32):
            with BitContext(64):
                x = jnp.array(1)
                y = jnp.array(2)
                return x + y

    assert nested64_int().dtype == jnp.int64

    @jax.jit
    def nested32_int():
        with BitContext(64):
            with BitContext(32):
                x = jnp.array(1)
                y = jnp.array(2)
                return x + y

    assert nested32_int().dtype == jnp.int32
