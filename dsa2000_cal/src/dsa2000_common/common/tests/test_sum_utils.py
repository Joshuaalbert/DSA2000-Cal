import numpy as np
from jax import numpy as jnp

from dsa2000_common.common.sum_utils import kahan_sum


def test_kahan_sum():
    # p = 609359
    # # terms = [math.sin(2 * math.pi * r * r / p) for r in range(p)]
    # def f(r):
    #     return jnp.sin((2 * jnp.pi) * r * r / p)

    # np.testing.assert_allclose(accumulate64, jnp.sqrt(p))

    def f(r):
        return jnp.sin(r)

    xs = jnp.arange(100).astype(jnp.float64)
    accumulate64, error_accumulate64 = kahan_sum(f, jnp.zeros((), dtype=jnp.float64), xs)

    xs = jnp.arange(100).astype(jnp.float16)
    accumulate16, error_accumulate16 = kahan_sum(f, jnp.zeros((), dtype=jnp.float16), xs)
    np.testing.assert_allclose(accumulate16, accumulate64)  # , atol=0.0046)
    np.testing.assert_allclose(f(xs).sum(), accumulate64, atol=0.0027)

    xs = jnp.arange(100).astype(jnp.float32)
    accumulate32, error_accumulate32 = kahan_sum(f, jnp.zeros((), dtype=jnp.float32), xs)
    np.testing.assert_allclose(accumulate32, accumulate64)
    np.testing.assert_allclose(f(xs).sum(), accumulate64, atol=7.0e-8)
