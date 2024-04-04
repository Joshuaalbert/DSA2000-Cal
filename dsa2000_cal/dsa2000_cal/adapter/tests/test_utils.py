import numpy as np
from jax import numpy as jnp

from dsa2000_cal.adapter.utils import INV_CASA_CORR_TYPES, from_casa_corrs_to_linear, from_linear_to_casa_corrs
from dsa2000_cal.source_models.corr_translation import linear_to_circular


def test_from_casa_corrs_to_linear():
    # Linear to Linear
    casa_coherencies = jnp.asarray([1, 2, 3, 4])
    corrs = [
        INV_CASA_CORR_TYPES["XX"],
        INV_CASA_CORR_TYPES["XY"],
        INV_CASA_CORR_TYPES["YX"],
        INV_CASA_CORR_TYPES["YY"]
    ]
    assert jnp.alltrue(from_casa_corrs_to_linear(casa_coherencies, corrs) == jnp.asarray([1, 2, 3, 4]))

    # Perm
    corrs = [
        INV_CASA_CORR_TYPES["XX"],
        INV_CASA_CORR_TYPES["YY"],
        INV_CASA_CORR_TYPES["XY"],
        INV_CASA_CORR_TYPES["YX"]
    ]
    assert jnp.alltrue(from_casa_corrs_to_linear(casa_coherencies, corrs) == jnp.asarray([1, 3, 4, 2]))

    # Partial
    corrs = [
        INV_CASA_CORR_TYPES["XX"],
        INV_CASA_CORR_TYPES["XY"],
    ]
    assert jnp.alltrue(from_casa_corrs_to_linear(casa_coherencies[:2], corrs) == jnp.asarray([1, 2, 0, 0]))

    # Circular to Linear
    casa_coherencies = jnp.asarray([1, 2, 3, 4])
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["RL"],
        INV_CASA_CORR_TYPES["LR"],
        INV_CASA_CORR_TYPES["LL"]
    ]
    assert jnp.allclose(from_casa_corrs_to_linear(casa_coherencies, corrs), jnp.asarray([1, 2, 3, 4]))

    # Perm
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["LL"],
        INV_CASA_CORR_TYPES["RL"],
        INV_CASA_CORR_TYPES["LR"]
    ]
    assert jnp.allclose(from_casa_corrs_to_linear(casa_coherencies, corrs), jnp.asarray([1, 3, 4, 2]))

    # Partial
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["RL"],
    ]
    assert jnp.allclose(from_casa_corrs_to_linear(casa_coherencies[:2], corrs), jnp.asarray([1, 2, 0, 0]))


def test_from_linear_to_casa_corrs():
    # Linear to Linear
    linear_coherencies = jnp.asarray([1, 2, 3, 4])
    corrs = [
        INV_CASA_CORR_TYPES["XX"],
        INV_CASA_CORR_TYPES["XY"],
        INV_CASA_CORR_TYPES["YX"],
        INV_CASA_CORR_TYPES["YY"]
    ]
    assert jnp.alltrue(from_linear_to_casa_corrs(linear_coherencies, corrs) == jnp.asarray([1, 2, 3, 4]))

    # Perm
    corrs = [
        INV_CASA_CORR_TYPES["XX"],
        INV_CASA_CORR_TYPES["YY"],
        INV_CASA_CORR_TYPES["XY"],
        INV_CASA_CORR_TYPES["YX"]
    ]
    assert jnp.alltrue(from_linear_to_casa_corrs(linear_coherencies, corrs) == jnp.asarray([1, 4, 2, 3]))

    # Partial
    corrs = [
        INV_CASA_CORR_TYPES["XX"],
        INV_CASA_CORR_TYPES["XY"],
    ]
    assert jnp.alltrue(from_linear_to_casa_corrs(linear_coherencies, corrs) == jnp.asarray([1, 2]))

    # Circular to Linear
    linear_coherencies = jnp.asarray([1, 2, 3, 4])
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["RL"],
        INV_CASA_CORR_TYPES["LR"],
        INV_CASA_CORR_TYPES["LL"]
    ]
    RR, RL, LR, LL = linear_to_circular(linear_coherencies, flat_output=True)
    np.testing.assert_allclose(from_linear_to_casa_corrs(linear_coherencies, corrs), [RR, RL, LR, LL], atol=1e-6)

    # Perm
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["LL"],
        INV_CASA_CORR_TYPES["RL"],
        INV_CASA_CORR_TYPES["LR"]
    ]
    RR, RL, LR, LL = linear_to_circular(linear_coherencies, flat_output=True)
    np.testing.assert_allclose(from_linear_to_casa_corrs(linear_coherencies, corrs), [RR, LL, RL, LR], atol=1e-6)

    # Partial
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["RL"],
    ]
    RR, RL, LR, LL = linear_to_circular(linear_coherencies, flat_output=True)
    np.testing.assert_allclose(from_linear_to_casa_corrs(linear_coherencies, corrs), [RR, RL], atol=1e-6)
