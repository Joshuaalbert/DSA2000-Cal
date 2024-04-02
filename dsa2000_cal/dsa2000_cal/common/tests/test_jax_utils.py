from jax import numpy as jnp

from dsa2000_cal.common.jax_utils import cumulative_op_static, cumulative_op_dynamic
from dsa2000_cal.common.types import float_type, int_type


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
