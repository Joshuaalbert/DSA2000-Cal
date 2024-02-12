import pytest
from jax import numpy as jnp

from dsa2000_cal.jax_utils import add_chunk_dim, cumulative_op_static, cumulative_op_dynamic
from dsa2000_cal.types import float_type, int_type


def test_add_chunk_dim():
    x = jnp.ones((4, 5))
    chunked_x, unchunk_fn = add_chunk_dim(x, chunk_size=2)
    assert chunked_x.shape == (2, 2, 5)
    assert unchunk_fn(chunked_x).shape == x.shape

    x = jnp.ones((4,))
    chunked_x, unchunk_fn = add_chunk_dim(x, chunk_size=2)
    assert chunked_x.shape == (2, 2)
    assert unchunk_fn(chunked_x).shape == x.shape

    x = jnp.ones((3, 5))
    chunked_x, unchunk_fn = add_chunk_dim(x, chunk_size=2)
    assert chunked_x.shape == (2, 2, 5)
    assert unchunk_fn(chunked_x).shape == x.shape

    x = jnp.ones((3,))
    chunked_x, unchunk_fn = add_chunk_dim(x, chunk_size=2)
    assert chunked_x.shape == (2, 2)
    assert unchunk_fn(chunked_x).shape == x.shape

    x = jnp.ones(())
    with pytest.raises(ValueError, match="Leaves must have batch dim."):
        chunked_x, unchunk_fn = add_chunk_dim(x, chunk_size=2)

    x = jnp.ones((0, 1))
    with pytest.raises(ValueError, match="Leaves must have non-zero batch dim."):
        chunked_x, unchunk_fn = add_chunk_dim(x, chunk_size=2)


def test_cumulative_op_static():
    def op(accumulate, y):
        return accumulate + y

    init = jnp.asarray(0, float_type)
    xs = jnp.asarray([1, 2, 3], float_type)
    final_accumulate, result = cumulative_op_static(op=op, init=init, xs=xs)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([1, 3, 6], float_type))

    final_accumulate, result = cumulative_op_static(op=op, init=init, xs=xs, pre_op=True)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([0, 1, 3], float_type))


def test_cumulative_op_dynamic():
    def op(accumulate, y):
        return accumulate + y

    init = jnp.asarray(0, float_type)
    xs = jnp.asarray([1, 2, 3], float_type)
    stop_idx = jnp.asarray(3, int_type)
    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([1, 3, 6], float_type))

    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx, pre_op=True)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([0, 1, 3], float_type))

    stop_idx = jnp.asarray(2, int_type)
    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx)
    assert final_accumulate == 3
    assert all(result == jnp.asarray([1, 3, 0], float_type))

    final_accumulate, result = cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx, pre_op=True)
    assert final_accumulate == 3
    assert all(result == jnp.asarray([0, 1, 0], float_type))
