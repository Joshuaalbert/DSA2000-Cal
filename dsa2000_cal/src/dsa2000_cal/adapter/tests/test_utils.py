import jax
import pytest
from jax import numpy as jnp

from dsa2000_cal.adapter.utils import INV_CASA_CORR_TYPES, translate_corrs, detect_mixed_corrs, \
    broadcast_translate_corrs


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

    # Stokes to/fro
    stokes_coors = ["I", "Q", "U", "V"]
    casa_coherencies = jnp.asarray([1, 2, 3, 4])
    corrs = [
        INV_CASA_CORR_TYPES["I"],
        INV_CASA_CORR_TYPES["Q"],
        INV_CASA_CORR_TYPES["U"],
        INV_CASA_CORR_TYPES["V"]
    ]
    assert jnp.allclose(translate_corrs(casa_coherencies, stokes_coors, corrs), jnp.asarray([1, 2, 3, 4]))

    # Perm
    corrs = [
        INV_CASA_CORR_TYPES["I"],
        INV_CASA_CORR_TYPES["V"],
        INV_CASA_CORR_TYPES["Q"],
        INV_CASA_CORR_TYPES["U"]
    ]
    assert jnp.allclose(translate_corrs(casa_coherencies, stokes_coors, corrs), jnp.asarray([1, 4, 2, 3]))

    # Partial
    corrs = [
        INV_CASA_CORR_TYPES["I"],
        INV_CASA_CORR_TYPES["Q"],
    ]
    assert jnp.allclose(translate_corrs(casa_coherencies[:2], stokes_coors[:2], corrs), jnp.asarray([1, 2]))


def test_ravel():
    coors = [['XX', 'XY'], ['YX', 'YY']]
    leaves, treedef = jax.tree.flatten(coors)
    print(leaves)
    data = [jnp.ones(())] * 4
    assert jnp.asarray(jax.tree.unflatten(treedef, data)).shape == (2, 2)


def test_detect_mixed_corrs():
    assert not detect_mixed_corrs(["XX", "XY", "YX", "YY"])
    assert not detect_mixed_corrs(["RR", "RL", "LR", "LL"])
    assert not detect_mixed_corrs(["I", "Q", "U", "V"])

    assert detect_mixed_corrs(["XX", "XY", "YX", "YY", "RR"])
    assert detect_mixed_corrs(["XX", "XY", "YX", "YY", "I"])
    assert detect_mixed_corrs(["RR", "RL", "LR", "LL", "I"])
    assert detect_mixed_corrs(["I", "Q", "U", "V", "XX"])
    assert detect_mixed_corrs(["I", "Q", "U", "V", "RR"])


def test_broadcast_translate_corrs():
    coherencies = jnp.ones((11, 13, 4))
    from_corrs = ("XX", "XY", "YX", "YY")
    to_corrs = ("RR", "RL", "LR", "LL")
    assert broadcast_translate_corrs(coherencies, from_corrs, to_corrs).shape == (11, 13, 4)

    to_corrs = (("RR", "RL"), ("LR", "LL"))
    assert broadcast_translate_corrs(coherencies, from_corrs, to_corrs).shape == (11, 13, 2, 2)

    coherencies = jnp.ones((11, 13, 2, 2))
    from_corrs = (("XX", "XY"), ("YX", "YY"))
    to_corrs = ("RR", "RL", "LR", "LL")
    assert broadcast_translate_corrs(coherencies, from_corrs, to_corrs).shape == (11, 13, 4)

    to_corrs = (("RR", "RL"), ("LR", "LL"))
    assert broadcast_translate_corrs(coherencies, from_corrs, to_corrs).shape == (11, 13, 2, 2)

    x = jnp.ones((2, 1), dtype=jnp.complex64)
    y = broadcast_translate_corrs(
        x, from_corrs=('I',), to_corrs=('XX', 'XY', 'YX', 'YY')
    )
    x_rec = broadcast_translate_corrs(
        y, from_corrs=('XX', 'XY', 'YX', 'YY'), to_corrs=('I',)
    )
    assert jnp.allclose(x, x_rec)
