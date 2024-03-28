from typing import TypeVar, Callable, Optional, Tuple

import jax
import numpy as np
from jax import numpy as jnp, tree_map, pmap, lax, tree_util

__all__ = [
    'chunked_pmap'
]

from dsa2000_cal.src.common.types import IntArray, int_type
from typing import TypeVar, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util, tree_map, pmap, lax
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact

V = TypeVar('V')


def promote_pytree(func_name: str, pytree: V) -> V:
    """
    Promotes a pytree to a common dtype pytree.

    Args:
        func_name: name of the function calling this function
        pytree: pytree to promote

    Returns:
        pytree with promoted dtypes
    """
    leaves, tree_def = tree_util.tree_flatten(pytree)
    check_arraylike(func_name, *leaves)
    leaves = promote_dtypes_inexact(*leaves)
    return tree_util.tree_unflatten(tree_def, leaves)


def tree_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree_map(lambda *xs: list(xs), *list_of_trees)


PT = TypeVar('PT')


def pytree_unravel(example_tree: PT) -> Tuple[Callable[[PT], jax.Array], Callable[[jax.Array], PT]]:
    """
    Returns functions to ravel and unravel a pytree.
    """
    leaf_list, tree_def = tree_util.tree_flatten(example_tree)

    sizes = [leaf.size for leaf in leaf_list]
    shapes = [leaf.shape for leaf in leaf_list]

    def ravel_fun(pytree: PT) -> jax.Array:
        leaf_list, tree_def = tree_util.tree_flatten(pytree)
        return jnp.concatenate([lax.reshape(leaf, (size,)) for leaf, size in zip(leaf_list, sizes)])

    def unravel_fun(flat_array: jax.Array) -> PT:
        leaf_list = []
        start = 0
        for size, shape in zip(sizes, shapes):
            leaf_list.append(lax.reshape(flat_array[start:start + size], shape))
            start += size
        return tree_util.tree_unflatten(tree_def, leaf_list)

    return ravel_fun, unravel_fun


FV = TypeVar('FV')


def chunked_pmap(f: Callable[..., FV], chunk_size: int | None = None, unroll: int = 1) -> Callable[..., FV]:
    """
    A version of pmap which chunks the input into smaller pieces to avoid memory issues.

    Args:
        f: callable
        chunk_size: the size of the chunks. Default is len(devices())
        unroll: the number of times to unroll the computation

    Returns:
        a chunked version of f
    """
    if chunk_size is None:
        chunk_size = len(jax.devices())

    def _f(*args, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = lax.scan(body, (), (args, kwargs), unroll=unroll)
            return result

        if chunk_size > 1:
            # Get from first leaf
            leaves = tree_util.tree_leaves((args, kwargs))
            batch_size = np.shape(leaves[0])[0]
            for leaf in leaves:
                if np.shape(leaf)[0] != batch_size:
                    raise ValueError(f"All leaves must have the same first dimension, got {np.shape(leaf)}.")
            remainder = batch_size % chunk_size
            extra = (chunk_size - remainder) % chunk_size
            if extra > 0:
                (args, kwargs) = tree_map(lambda x: _pad_extra(x, extra), (args, kwargs))
            (args, kwargs) = tree_map(lambda x: jnp.reshape(x, (chunk_size, x.shape[0] // chunk_size) + x.shape[1:]),
                                      (args, kwargs))
            result = pmap(queue)(*args, **kwargs)  # [chunksize, batch_size // chunksize, ...]
            result = tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), result)
            if extra > 0:
                result = tree_map(lambda x: x[:-extra], result)
        else:
            result = queue(*args, **kwargs)
        return result

    return _f


def _pad_extra(x, extra):
    return jnp.concatenate([x, jnp.repeat(x[:1], repeats=extra, axis=0)], axis=0)


T = TypeVar('T')
S = TypeVar('S')


def pad_to_chunksize(py_tree: T, chunk_size: int) -> Tuple[T, Callable[[S], S]]:
    """
    Pad data to a multiple of chunk size

    Args:
        py_tree: pytree to add chunk dimension to
        chunk_size: size of chunk dimension

    Returns:
        pytree with chunk dimension added, and callable to remove extra
    """

    leaves = jax.tree_util.tree_leaves(py_tree)

    if len(leaves) == 0:
        raise ValueError("Leaves must be non-empty to add a chunk dim.")

    if not all(len(np.shape(x)) > 0 for x in leaves):
        raise ValueError("Leaves must have batch dim.")

    if not all(np.shape(x)[0] > 0 for x in leaves):
        raise ValueError("Leaves must have non-zero batch dim.")

    batch_size = np.shape(leaves[0])[0]
    if not all(np.shape(x)[0] == batch_size for x in leaves):
        raise ValueError("Leaves do not have consistent batch dim.")

    remainder = batch_size % chunk_size
    extra = (chunk_size - remainder) % chunk_size
    if extra > 0:
        py_tree = tree_map(lambda x: jnp.concatenate([x, jnp.repeat(x[0:1], extra, 0)]), py_tree)

    def _remove_extra(output_py_tree: S) -> S:
        if extra > 0:
            output_py_tree = jax.tree_map(lambda x: x[:-extra], output_py_tree)
        return output_py_tree

    return py_tree, _remove_extra



V = TypeVar('V')
Y = TypeVar('Y')


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


def cumulative_op_dynamic(op: Callable[[V, Y], V], init: V, xs: Y, stop_idx: IntArray, pre_op: bool = False,
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

    def cond(carry: Tuple[V, IntArray, V]):
        (accumulate, i, output) = carry
        return jnp.less(i, stop_idx)

    def body(carry: Tuple[V, IntArray, V]):
        (accumulate, i, output) = carry
        y = tree_map(lambda x: x[i], xs)
        next_accumulate = op(accumulate, y)
        next_i = i + jnp.ones_like(i)
        if pre_op:
            next_output = tree_map(lambda a, b: a.at[i].set(b), output, accumulate)
            return (next_accumulate, next_i, next_output)
        next_output = tree_map(lambda a, b: a.at[i].set(b), output, next_accumulate)
        return (next_accumulate, next_i, next_output)

    length = tree_util.tree_flatten(xs)[0][0].shape[0]

    output = tree_map(
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
