import jax
from jax import numpy as jnp

from dsa2000_cal.common.linalg_utils import randomized_pinv


def test_randomized_svd_pinv():
    # Step 1: Input Matrix
    A = jax.random.normal(jax.random.PRNGKey(0), (10, 5))

    # Step 9: Compute pseudoinverse
    A_pinv = randomized_pinv(jax.random.PRNGKey(1),
                             A, 3, 2)

    # Step 11: Check
    assert jnp.allclose(A_pinv, jnp.linalg.pinv(A))
