import jax
import numpy as np
import pytest

from dsa2000_cal.common.msqrt import sqrtm_only


@pytest.mark.parametrize("n", [5, 10, 20, 1000])
def test_sqrtm_only(n):
    a = jax.random.normal(jax.random.PRNGKey(42), (n,n))
    a = a @ a.T
    sqrt_a = sqrtm_only(a)
    np.testing.assert_allclose(sqrt_a @ sqrt_a.T, a, atol=5e-5)

    # low rank
    a = jax.random.normal(jax.random.PRNGKey(42), (n, n - 1))
    a = a @ a.T
    sqrt_a = sqrtm_only(a)
    np.testing.assert_allclose(sqrt_a @ sqrt_a.T, a, atol=5e-5)
