from functools import partial

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.jax_utils import vmap_or_scan, extract_shape, extract_shape_tuples, multi_vmap


@pytest.mark.parametrize('use_scan', [True, False])
def test_vmap_or_scan(use_scan: bool):
    def f(x, y):
        return x + y

    x = jnp.ones((2, 3))
    y = jnp.ones((2, 3))
    assert jnp.all(vmap_or_scan(f, 0, use_scan=use_scan)(x, y) == f(x, y))
    assert jnp.all(vmap_or_scan(f, (0, 0), use_scan=use_scan)(x, y) == f(x, y))

    assert np.shape(vmap_or_scan(f, (0, 0), out_axes=0, use_scan=use_scan)(x, y)) == (2, 3)
    assert np.shape(vmap_or_scan(f, (0, None), out_axes=0, use_scan=use_scan)(x, y)) == (2, 2, 3)

    assert np.shape(vmap_or_scan(f, (0, 0), out_axes=-1, use_scan=use_scan)(x, y)) == (3, 2)
    assert np.shape(vmap_or_scan(f, (0, None), out_axes=-1, use_scan=use_scan)(x, y)) == (2, 3, 2)


def test_extract_shape():
    assert extract_shape('[n1,n2,n3]') == ['n1', 'n2', 'n3']
    assert extract_shape('[n1,n2, n3]') == ['n1', 'n2', 'n3']
    assert extract_shape('[n1,n3,n2]') == ['n1', 'n3', 'n2']
    assert extract_shape('[n1,n1]') == ['n1', 'n1']


def test_extract_shape_tuples():
    assert extract_shape_tuples('[n1,n2,n3],[n1,n3]') == ['[n1,n2,n3]', '[n1,n3]']
    assert extract_shape_tuples('[n1,n2,n3]') == ['[n1,n2,n3]']


def test_multi_vmap():
    # Simple
    def f(x, y):
        return x + y

    n1, n2 = 3, 4

    x = jnp.ones((n1,))
    y = jnp.ones((n2,))

    f_multi = multi_vmap(f, in_mapping="[n1],[n2]", out_mapping="[n1,n2]", verbose=True)
    res = f_multi(x, y)

    assert res.shape == (n1, n2)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def f2(x, y):
        return x + y

    res2 = f2(x, y)

    np.testing.assert_allclose(res, res2)

    f_multi = multi_vmap(f, in_mapping="[n1],[n2]", out_mapping="[n2,n1]", verbose=True)
    res = f_multi(x, y)
    assert res.shape == (n2, n1)

    # More complex
    def f(x, y):
        return x + y

    with pytest.raises(ValueError, match="must contain"):
        _ = multi_vmap(f, in_mapping="[n1,n2,n3],[n1,n3]", out_mapping="[n1,n2,n3,n4]", verbose=True)

    n1, n2, n3 = 3, 4, 5

    x = jnp.ones((n1, n2, n3))
    y = jnp.ones((n1, n3))

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3],[n1,n3]", out_mapping="[n1,n2,n3]", verbose=True)
    res = f_multi(x, y)

    assert res.shape == (n1, n2, n3)

    @partial(jax.vmap, in_axes=(0, 0))
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(0, 0))
    def f2(x, y):
        return x + y

    res2 = f2(x, y)

    np.testing.assert_allclose(res, res2)

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3],[n1,n3]", out_mapping="[n1,n2]", verbose=True)
    res = f_multi(x, y)
    assert res.shape == (n1, n2, n3)  # last dim is broadcasted

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3],[n1,n3]", out_mapping="[n2,n1]", verbose=True)
    res = f_multi(x, y)
    assert res.shape == (n2, n1, n3)

    x = jnp.ones((n1, n2, n3, 2, 2))
    y = jnp.ones((n1, n3, 2, 2))

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3,2,2],[n1,n3,2,2]", out_mapping="[n2,n1]", verbose=True)
    res = f_multi(x, y)
    assert res.shape == (n2, n1, n3, 2, 2)

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3,2,2],[n1,n3,2,2]", out_mapping="[n2,n1]", verbose=True,
                         scan_dims={'n1'})
    res = f_multi(x, y)
    assert res.shape == (n2, n1, n3, 2, 2)

    # Test putting output in other spots

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3,2,2],[n1,n3,2,2]", out_mapping="[n2,n1,...]", verbose=True)
    res = f_multi(x, y)
    assert res.shape == (n2, n1, n3, 2, 2)

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3,2,2],[n1,n3,2,2]", out_mapping="[...,n3,n2,n1]", verbose=True)
    res = f_multi(x, y)
    assert res.shape == (2, 2, n3, n2, n1)

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3,2,2],[n1,n3,2,2]", out_mapping="[...,n2,n1]", verbose=True)
    res = f_multi(x, y)
    assert res.shape == (n3, 2, 2, n2, n1)

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3,2,2],[n1,n3,2,2]", out_mapping="[n2,...,n1]", verbose=True)
    res = f_multi(x, y)
    assert res.shape == (n2, n3, 2, 2, n1)

    def g(x, y):
        return x + y, x + y

    f_multi = multi_vmap(g, in_mapping="[n1,n2,n3,2,2],[n1,n3,2,2]", out_mapping="[n2,n1,...],[n1,n2,...]",
                         verbose=True)
    res = f_multi(x, y)
    assert res[0].shape == (n2, n1, n3, 2, 2)
    assert res[1].shape == (n1, n2, n3, 2, 2)

    f_multi = multi_vmap(g, in_mapping="[n1,n2,n3,2,2],[n1,n3,2,2]", out_mapping="[n2,...,n1],[n1,...,n2]",
                         verbose=True)
    res = f_multi(x, y)
    assert res[0].shape == (n2, n3, 2, 2, n1)
    assert res[1].shape == (n1, n3, 2, 2, n2)

    f_multi = multi_vmap(g, in_mapping="[n1,n2,n3,2,2],[n1,n3,2,2]", out_mapping="[...,n2,n1],[...,n1,n2]",
                         verbose=True)
    res = f_multi(x, y)
    assert res[0].shape == (n3, 2, 2, n2, n1)
    assert res[1].shape == (n3, 2, 2, n1, n2)
