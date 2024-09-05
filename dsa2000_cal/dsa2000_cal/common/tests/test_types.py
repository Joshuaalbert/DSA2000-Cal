import numpy as np
import jax.numpy as jnp

from dsa2000_cal.common.types import mp_policy


def test_mp_policy():
    print(mp_policy)
    assert mp_policy.length_dtype == jnp.float64
    assert mp_policy.length_dtype == np.float64
