import jax
import pytest
from astropy import units as au
from jax import numpy as jnp

from dsa2000_common.common.quantity_utils import quantity_to_jnp


def test_quantity_to_jnp():
    q = 1 * au.m
    assert quantity_to_jnp(q) == jnp.array(1.0)
    assert quantity_to_jnp(q, au.km) == jnp.array(0.001)
    assert isinstance(quantity_to_jnp(q), jax.Array)

    with pytest.raises(ValueError, match="Expected astropy quantity"):
        quantity_to_jnp(1.0)

    with pytest.raises(ValueError, match="Expected equivalent unit"):
        quantity_to_jnp(1 * au.m, au.s)


    assert quantity_to_jnp(1j * au.dimensionless_unscaled) == 1j
