import jax.numpy as jnp
import numpy as np

from dsa2000_cal.common.mixed_precision_utils import mp_policy


def test_mp_policy():
    print(mp_policy)
    assert mp_policy.length_dtype == jnp.float64
    assert mp_policy.length_dtype == np.float64


def test_mp_policy_np_stays_np():
    x = np.array([0], np.complex64)
    assert isinstance(mp_policy.cast_to_vis(x), np.ndarray)
