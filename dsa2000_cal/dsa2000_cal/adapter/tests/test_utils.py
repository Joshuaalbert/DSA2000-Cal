import jax
import pytest
from jax import numpy as jnp

from dsa2000_cal.adapter.utils import INV_CASA_CORR_TYPES, translate_corrs


def test_transform_corrs():
    linear_coors = ["XX", "XY", "YX", "YY"]
    # Linear to Linear
    casa_coherencies = jnp.asarray([1, 2, 3, 4])
    corrs = [
        INV_CASA_CORR_TYPES["XX"],
        INV_CASA_CORR_TYPES["XY"],
        INV_CASA_CORR_TYPES["YX"],
        INV_CASA_CORR_TYPES["YY"]
    ]
    assert jnp.all(translate_corrs(casa_coherencies, linear_coors, corrs) == jnp.asarray([1, 2, 3, 4]))

    # Perm
    corrs = [
        INV_CASA_CORR_TYPES["XX"],
        INV_CASA_CORR_TYPES["YY"],
        INV_CASA_CORR_TYPES["XY"],
        INV_CASA_CORR_TYPES["YX"]
    ]
    assert jnp.all(translate_corrs(casa_coherencies, linear_coors, corrs) == jnp.asarray([1, 4, 2, 3]))

    # Partial
    corrs = [
        INV_CASA_CORR_TYPES["XX"],
        INV_CASA_CORR_TYPES["XY"],
    ]
    assert jnp.all(translate_corrs(casa_coherencies[:2], linear_coors[:2], corrs) == jnp.asarray([1, 2]))

    # Circular to Linear
    circ_coors = ["RR", "RL", "LR", "LL"]
    casa_coherencies = jnp.asarray([1, 2, 3, 4])
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["RL"],
        INV_CASA_CORR_TYPES["LR"],
        INV_CASA_CORR_TYPES["LL"]
    ]
    assert jnp.allclose(translate_corrs(casa_coherencies, circ_coors, corrs), jnp.asarray([1, 2, 3, 4]))

    # Perm
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["LL"],
        INV_CASA_CORR_TYPES["RL"],
        INV_CASA_CORR_TYPES["LR"]
    ]
    assert jnp.allclose(translate_corrs(casa_coherencies, circ_coors, corrs), jnp.asarray([1, 4, 2, 3]))

    # Partial
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["RL"],
    ]
    assert jnp.allclose(translate_corrs(casa_coherencies[:2], circ_coors[:2], corrs), jnp.asarray([1, 2]))

    # Mixed
    casa_coherencies = jnp.asarray([1, 2, 3, 4])
    corrs = [
        INV_CASA_CORR_TYPES["RR"],
        INV_CASA_CORR_TYPES["XY"],
        INV_CASA_CORR_TYPES["LR"],
        INV_CASA_CORR_TYPES["YY"]
    ]
    with pytest.raises(ValueError):
        translate_corrs(casa_coherencies, linear_coors, corrs)

    # Test shaped
    coherencies = jnp.asarray([[1, 2], [3, 4]])
    assert jnp.all(
        translate_corrs(coherencies,
                        [['XX', 'XY'], ['YX', 'YY']], [['XX', 'YX'], ['XY', 'YY']]) == jnp.asarray(
            [[1, 3], [2, 4]]))

    coherencies = jnp.asarray([1, 2, 3, 4])
    assert jnp.all(
        translate_corrs(coherencies,
                        ['XX', 'XY', 'YX', 'YY'], [['XX', 'YX'], ['XY', 'YY']]) == jnp.asarray(
            [[1, 3], [2, 4]]))

    coherencies = jnp.asarray([1, 2, 3, 4])
    assert jnp.all(
        translate_corrs(coherencies,
                        ['XX', 'XY', 'YX', 'YY'], [['XX', 'XY'], ['YX', 'YY']]) == jnp.asarray(
            [[1, 2], [3, 4]]))


def test_ravel():
    coors = [['XX', 'XY'], ['YX', 'YY']]
    leaves, treedef = jax.tree.flatten(coors)
    print(leaves)
    data = [jnp.ones(())] * 4
    assert jnp.asarray(jax.tree.unflatten(treedef, data)).shape == (2, 2)
