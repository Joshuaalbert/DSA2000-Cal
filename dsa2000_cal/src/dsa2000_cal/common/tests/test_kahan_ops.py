import pytest
from jax import numpy as jnp

from dsa2000_cal.common.kahan_ops import Kahan


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_kahan(dtype):
    k = Kahan(dtype=dtype)
    for i in range(100000):
        k + 1e-10
    assert k.value == 1e-5

    normal = 0.
    for i in range(100000):
        normal += 1e-10
    assert abs(normal - 1e-5) < 1e-10
    assert normal != 1e-5
