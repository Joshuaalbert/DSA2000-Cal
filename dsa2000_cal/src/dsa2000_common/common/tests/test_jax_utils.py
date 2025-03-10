from functools import partial
from typing import List, Tuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_common.common.jax_utils import vmap_or_scan, extract_shape, extract_shape_tuples, multi_vmap, \
    auto_multi_vmap, \
    convert_to_ufunc, _get_permutation, simple_broadcast, chunked_pmap


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
    n1, n2, n3, n4, n5 = 3, 4, 5, 6, 7

    def run_test(in_mapping, out_mapping, in_shapes: List[Tuple[int, ...]], expected_out_shapes: List[Tuple[int, ...]]):
        @partial(
            multi_vmap,
            in_mapping=in_mapping,
            out_mapping=out_mapping,
            verbose=True
        )
        def f(x, y):
            if x is None:
                return y
            return x + y

        x = jnp.ones(in_shapes[0]) if in_shapes[0] is not None else None
        y = jnp.ones(in_shapes[1]) if in_shapes[1] is not None else None
        res = f(x, y)
        assert res.shape == expected_out_shapes[0]

    # Test good cases
    run_test(
        "[n1,n2],[n1,n2]",
        "[n1,n2]",
        [(n1, n2), (n1, n2)],
        [(n1, n2)]
    )

    run_test(
        "[n2,n1],[n1,n2]",
        "[n1,n2]",
        [(n2, n1), (n1, n2)],
        [(n1, n2)]
    )

    run_test(
        "[n1,n2,n3],[n1,n2]",
        "[n1,n2]",
        [(n1, n2, n3), (n1, n2)],
        [(n1, n2, n3)]
    )

    run_test(
        "[n1,n2,n3],[n1,n2]",
        "[n1,n2,~n3]",  # ~ means the dimension is not mapped
        [(n1, n2, n3), (n1, n2)],
        [(n1, n2, n3)]
    )

    run_test(
        "[n3,n1,n2],[n1,n2]",
        "[...,n1,n2]",  # ~ means the dimension is not mapped
        [(n3, n1, n2), (n1, n2)],
        [(n3, n1, n2)]
    )

    run_test(
        "[n3,n1,n2],[n1,n2]",
        "[~n3,n1,n2]",  # ~ means the dimension is not mapped
        [(n3, n1, n2), (n1, n2)],
        [(n3, n1, n2)]
    )

    run_test(
        "[n3,n1,n2],[n1,n2]",
        "[n3,n1,n2]",  # ~ means the dimension is not mapped
        [(n3, n1, n2), (n1, n2)],
        [(n3, n1, n2)]
    )

    # Test None value

    run_test(
        "[],[n1,n2]",
        "[n1,n2]",
        [None, (n1, n2)],
        [(n1, n2)]
    )

    run_test(
        "[n1],[n1,n2]",
        "[n1,n2]",
        [None, (n1, n2)],
        [(n1, n2)]
    )

    with pytest.raises(ValueError, match='must have at least one non-None value in in_axes'):
        run_test(
            "[n1],[n2]",
            "[n1,n2]",
            [None, (n1, n2)],
            [(n1, n2)]
        )

    def run_test(in_mapping, out_mapping, in_shapes: List[Tuple[int, ...]], expected_out_shapes: List[Tuple[int, ...]]):
        @partial(
            multi_vmap,
            in_mapping=in_mapping,
            out_mapping=out_mapping,
            verbose=True
        )
        def f(x, y):
            if x is None:
                return y
            return x + y, x - y

        x = jnp.ones(in_shapes[0]) if in_shapes[0] is not None else None
        y = jnp.ones(in_shapes[1]) if in_shapes[1] is not None else None
        res = f(x, y)
        for i, r in enumerate(res):
            assert r.shape == expected_out_shapes[i]

    # Test multi output

    run_test(
        "[n1,n2],[n1,n2]",
        "[n1,n2],[n2,n1]",
        [(n1, n2), (n1, n2)],
        [(n1, n2), (n2, n1)]
    )

    with pytest.raises(ValueError, match='must contain all mapped dims'):
        # Outputs don't contain all mapped dims
        run_test(
            "[n1,n2],[n1,n2,n3]",
            "[n1,n2],[n2,n1,n3]",
            [(n1, n2), (n1, n2, n3)],
            [(n1, n2), (n2, n1, n3)]
        )

    with pytest.raises(ValueError, match='must be in some input dims'):
        # Outputs don't contain all mapped dims
        run_test(
            "[n1,n2],[n1,n2]",
            "[n1,n2,n3],[n2,n1,n3]",
            [(n1, n2), (n1, n2, n3)],
            [(n1, n2), (n2, n1, n3)]
        )

    run_test(
        "[n1,...,n2],[n1,n2]",
        "[n1,...,n2],[n2,...,n1]",
        [(n1, n3, n2), (n1, n2)],
        [(n1, n3, n2), (n2, n3, n1)]
    )

    run_test(
        "[n1,...,n2,n4],[n1,n2]",
        "[n1,...,n2,~n4],[n2,...,n1,~n4]",
        [(n1, n3, n2, n4), (n1, n2)],
        [(n1, n3, n2, n4), (n2, n3, n1, n4)]
    )

    run_test(
        "[n5,n1,...,n2,n4],[n1,n2]",
        "[~n5,n1,...,n2,~n4],[~n5,n2,...,n1,~n4]",
        [(n5, n1, n3, n2, n4), (n1, n2)],
        [(n5, n1, n3, n2, n4), (n5, n2, n3, n1, n4)]
    )


def test_multi_vmap_scan_performance():
    n1, n2, n3, n4 = 3, 3, 3, 3

    def f(x, y):
        return jnp.sin(x + y) / x

    x = jnp.ones((n1, n2, n3, n4))
    y = jnp.ones((n1, n2, n3, n4))

    f_multi = auto_multi_vmap(f, in_mapping="[n1,n2,n3,n4],[n1,n2,n3,n4]", out_mapping="[n1,n2,n3,n4]",

                              verbose=True)
    res = f_multi(x, y)


def test_multi_vmap_vs_vectorize():
    def f(x, y):
        return x + y, x - y

    x = jnp.ones((3, 4))
    y = jnp.ones((3, 4))

    f_multi = multi_vmap(f, in_mapping="[n1,n2],[n1,n2]", out_mapping="[n1,n2],[n1,n2]", verbose=True)
    res = f_multi(x, y)
    f_vec = jnp.vectorize(f, signature="(),()->(),()")
    res2 = f_vec(x, y)
    assert jnp.all(res[0] == res2[0])
    assert jnp.all(res[1] == res2[1])

    print(jax.jit(f_multi).lower(x, y).compile().cost_analysis())
    print(jax.jit(f_vec).lower(x, y).compile().cost_analysis())


@pytest.mark.parametrize("tile", [True, False])
def test_convert_to_ufunc(tile: bool):
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    @partial(convert_to_ufunc, tile=tile)
    def f(x, y):
        def add(x, y):
            print(x.shape, y.shape)
            if tile:
                assert x.shape == (4, 5)
                assert y.shape == (4, 5)
            else:
                assert x.shape == (4, 1)
                assert y.shape == (1, 5)
            return x + y

        return jax.pure_callback(add, jax.ShapeDtypeStruct(
            shape=jnp.broadcast_shapes(x.shape, y.shape), dtype=x.dtype), x, y, vectorized=True)

    x = jnp.arange(4)
    y = jnp.arange(5)

    res = f(x, y)

    assert np.allclose(res, x[:, None] + y[None, :])


def test_get_permutation():
    assert _get_permutation(5, ['a', 'b', '...', 'c'], ['a', '...', 'b', 'c']) == (0, 2, 3, 1, 4)
    assert _get_permutation(4, ['a', 'b', '...', 'c'], ['a', '...', 'b', 'c']) == (0, 2, 1, 3)
    assert _get_permutation(4, ['...'], ['...']) == (0, 1, 2, 3)
    assert _get_permutation(4, ['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']) == (3, 2, 1, 0)
    assert _get_permutation(4, ['a', 'b', 'c', 'd', '...'], ['d', 'c', 'b', 'a']) == (3, 2, 1, 0)

    with pytest.raises(ValueError, match="must have 4 dimensions."):
        assert _get_permutation(4, ['a', 'b', 'c', 'd', 'e'], ['d', 'c', 'b', 'a']) == (3, 2, 1, 0)

    with pytest.raises(ValueError, match="Dimension e not found in input axes"):
        _get_permutation(4, ['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a', 'e'])


def test_simple_broadcast():
    def f(x, y):
        assert np.shape(x) == (4, 5)
        return x + y

    x = jnp.ones((3, 4, 5))
    y = jnp.ones((3, 4, 5))

    res = simple_broadcast(f, 1)(x, y)

    assert np.all(res == 2)
    assert np.shape(res) == (3, 4, 5)

    def f(x, y):
        assert np.shape(x) == (5,)
        return x + y

    x = jnp.ones((3, 4, 5))
    y = jnp.ones((3, 4, 5))

    res = simple_broadcast(f, 2)(x, y)

    assert np.all(res == 2)
    assert np.shape(res) == (3, 4, 5)

    def f(x, y):
        assert np.shape(x) == ()
        return x + y

    x = jnp.ones((3, 4, 5))
    y = jnp.ones((3, 4, 5))

    res = simple_broadcast(f, 3)(x, y)

    assert np.all(res == 2)
    assert np.shape(res) == (3, 4, 5)


def test_chunked_pmap():
    @chunked_pmap
    def f(x):
        assert x.shape == (2, 3)
        return x.mean(axis=-1)

    x = jnp.ones((4, 2, 3))
    assert jax.block_until_ready(f(x)).shape == (4, 2)
