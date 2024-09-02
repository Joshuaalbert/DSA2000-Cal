from jax import numpy as jnp

from dsa2000_cal.common.bit_context import BitContext64, BitContext32


def test_bit_context():
    with BitContext64():
        a = jnp.asarray(0.)
        assert a.dtype == jnp.float64

        # Explicitly specifying float32 bypasses context
        b = jnp.asarray(0., jnp.float32)
        assert b.dtype == jnp.float32

        # Nesting
        with BitContext32():
            a = jnp.asarray(0.)
            assert a.dtype == jnp.float32

            # Explicitly specifying float64 cannot bypass context
            b = jnp.asarray(0., jnp.float64)
            assert b.dtype == jnp.float32

    with BitContext32():
        a = jnp.asarray(0.)
        assert a.dtype == jnp.float32

        # Explicitly specifying float64 cannot bypass context
        b = jnp.asarray(0., jnp.float64)
        assert b.dtype == jnp.float32

        # Nesting
        with BitContext64():
            a = jnp.asarray(0.)
            assert a.dtype == jnp.float64

            # Explicitly specifying float32 bypasses context
            b = jnp.asarray(0., jnp.float32)
            assert b.dtype == jnp.float32
