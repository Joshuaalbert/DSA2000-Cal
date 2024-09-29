import jax.numpy as jnp
import numpy as np

from dsa2000_cal.common.mixed_precision_utils import mp_policy


def test_mp_policy():
    print(mp_policy)
    assert mp_policy.length_dtype == jnp.float64
    assert mp_policy.length_dtype == np.float64
