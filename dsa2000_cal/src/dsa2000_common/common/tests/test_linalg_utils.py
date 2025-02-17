import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_common.common.linalg_utils import randomized_pinv, sqrtm_only


def test_randomized_svd_pinv():
    # Step 1: Input Matrix
    A = jax.random.normal(jax.random.PRNGKey(0), (10, 5))

    # Step 9: Compute pseudoinverse
    A_pinv = randomized_pinv(jax.random.PRNGKey(1),
                             A, 3, 2)

    # Step 11: Check
    assert jnp.allclose(A_pinv, jnp.linalg.pinv(A))


@pytest.mark.parametrize("n", [5, 10, 20, 1000])
def test_higham_sqrtm(n):
    a = jax.random.normal(jax.random.PRNGKey(42), (n,n))
    a = a @ a.T
    sqrt_a = sqrtm_only(a)
    np.testing.assert_allclose(sqrt_a @ sqrt_a.T, a, atol=5e-5)

    # low rank
    a = jax.random.normal(jax.random.PRNGKey(42), (n, n - 1))
    a = a @ a.T
    sqrt_a = sqrtm_only(a)
    np.testing.assert_allclose(sqrt_a @ sqrt_a.T, a, atol=5e-5)
