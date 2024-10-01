from typing import TypeVar, Callable, Tuple, Optional

import jax
import numpy as np
from jax import lax, numpy as jnp, tree_util
from tensorflow_probability.substrates.jax import math as tfp_math

from dsa2000_cal.common.mixed_precision_utils import int_type

X = TypeVar('X')
V = TypeVar('V')
Y = TypeVar('Y')


def _compute_max_num_levels(batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    # max_num_levels: Python `int`. The `axis` of the tensors in `elems` must have
    # size less than `2**(max_num_levels + 1)`
    # max_num_levels = inf{L : batch_size < 2**(L + 1)}
    # batch_size == 2**(max_num_levels + 1) - 1
    # ==> max_num_levels = log2(batch_size + 1) - 1
    return max(0, int(np.ceil(np.log2(batch_size + 1) - 1)))


def test_compute_max_num_levels():
    for batch_size in range(1, 1000):
        max_num_levels = _compute_max_num_levels(batch_size)
        print(batch_size, max_num_levels)
        assert batch_size < 2 ** (max_num_levels + 1)
        assert batch_size >= 2 ** (max_num_levels)


def scan_associative_cumulative_op(op: Callable[[X, X], X], init: X, xs: X, pre_op: bool = False) -> Tuple[X, X]:
    """
    Compute a cumulative operation on an array of values using scan_associative.

    Args:
        op: the operation to perform, must be associative.
        init: the initial value.
        xs: the array of values.

    Returns:
        the final accumulated value, and the result of the cumulative operation applied on input
    """

    def associative_op(a, b):
        return op(a, b)

    # Prepare the input array by prepending the initial value
    full_input = jax.tree.map(lambda x, y: jnp.concatenate([x[None], y], axis=0), init, xs)

    batch_size = np.shape(jax.tree.leaves(full_input)[0])[0]
    max_num_levels = _compute_max_num_levels(batch_size)

    # Apply the operation to accumulate results using scan_associative
    scanned_results = tfp_math.scan_associative(
        associative_op, full_input,
        max_num_levels=max_num_levels
    )

    # The final accumulated value is the last element in the results
    final_accumulate = jax.tree.map(lambda x: x[-1], scanned_results)

    if pre_op:
        scanned_results = jax.tree.map(lambda x: x[:-1], scanned_results)
    else:
        scanned_results = jax.tree.map(lambda x: x[1:], scanned_results)

    return final_accumulate, scanned_results


def cumulative_op_static(op: Callable[[V, Y], V], init: V, xs: Y, pre_op: bool = False, unroll: int = 1) -> Tuple[
    V, V]:
    """
    Compute a cumulative operation on a list of values.

    Args:
        op: the operation to perform
        init: the initial value
        xs: the list of values
        pre_op: if True, the operation is applied before the accumulation, so the first value is the initial value.
        unroll: how many iterations to unroll the loop at a time

    Returns:
        the final accumulated value, and the result of the cumulative operation applied on input
    """

    def body(accumulate: V, y: Y):
        next_accumulate = op(accumulate, y)
        if pre_op:
            return next_accumulate, accumulate
        return next_accumulate, next_accumulate

    final_accumulate, result = lax.scan(
        f=body,
        init=init,
        xs=xs,
        unroll=unroll
    )

    return final_accumulate, result


def cumulative_op_dynamic(op: Callable[[V, Y], V], init: V, xs: Y, stop_idx: jax.Array, pre_op: bool = False,
                          empty_fill: Optional[V] = None) -> Tuple[
    V, V]:
    """
    Compute a cumulative operation on a list of values with a dynamic stop index.

    Args:
        op: the operation to perform
        init: the initial value
        xs: the list of values
        stop_idx: how many accumulations to perform
        pre_op: if True, the operation is applied before the accumulation, so the first value is the initial value.
        empty_fill: the value to fill the output with if the stop_idx is provided, else uses `init`

    Returns:
        the final accumulated value, and the result of the cumulative operation applied on input
    """

    def cond(carry: Tuple[V, jax.Array, V]):
        (accumulate, i, output) = carry
        return jnp.less(i, stop_idx)

    def body(carry: Tuple[V, jax.Array, V]):
        (accumulate, i, output) = carry
        y = jax.tree.map(lambda x: x[i], xs)
        next_accumulate = op(accumulate, y)
        next_i = i + jnp.ones_like(i)
        if pre_op:
            next_output = jax.tree.map(lambda a, b: a.at[i].set(b), output, accumulate)
            return (next_accumulate, next_i, next_output)
        next_output = jax.tree.map(lambda a, b: a.at[i].set(b), output, next_accumulate)
        return (next_accumulate, next_i, next_output)

    length = tree_util.tree_flatten(xs)[0][0].shape[0]

    output = jax.tree.map(
        lambda x: jnp.tile(x[None], [length] + [1] * len(x.shape)),
        empty_fill if empty_fill is not None else init
    )

    w_init = (init, jnp.asarray(0, int_type), output)

    (final_accumulate, _, final_output) = lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=w_init
    )

    return final_accumulate, final_output
