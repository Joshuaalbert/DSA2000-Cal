import re
import time

__all__ = [
    'chunked_pmap'
]

from functools import partial

from typing import TypeVar, Callable, Tuple, Set, List

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util, tree_map, pmap, lax
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact

from dsa2000_cal.common.jvp_linear_op import isinstance_namedtuple

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


def get_time_jax(*deps, **kwargs):
    def _get_time(*deps, **kwargs):
        return np.asarray(time.time(), np.float32)

    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=(),
        dtype=jnp.float32
    )

    return jax.pure_callback(_get_time, result_shape_dtype, *deps, vectorized=False, **kwargs)


def vmap_or_scan(f, in_axes: Tuple[int | None, ...] | int, out_axes: Tuple[int, ...] | int = 0, use_scan: bool = False):
    """
    Returns a vmap or scan function based on the input axes.

    Args:
        f: function to vmap or scan over, must be jax compatible
        in_axes: input axes, one per input argument, None if argument is passed directly without vectorization
        out_axes: output axes, one per output argument, or single int if all outputs are swapped
        use_scan: whether to use scan instead of vmap
    """
    if not (isinstance(in_axes, tuple) or isinstance(in_axes, int)):
        raise ValueError(f"in_axes must be a tuple, out {in_axes}")
    if not (isinstance(out_axes, tuple) or isinstance(out_axes, int)):
        raise ValueError(f"out_axes must be a tuple, or int, got {out_axes}")
    if not use_scan:
        return jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
    else:
        def _scan(*xs, in_axes=in_axes, out_axes=out_axes):
            in_axes = (in_axes,) * len(xs) if isinstance(in_axes, int) else in_axes

            def body_fn(carry, index):
                args = [
                    (jnp.take(x, index, axis=in_axis) if in_axis is not None else x)
                    for x, in_axis in zip(xs, in_axes)
                ]
                return carry, f(*args)

            # Transpose if necessary
            length = None
            for x, in_axis in zip(xs, in_axes):
                if in_axis is None:
                    continue
                length = np.shape(jax.tree.leaves(x)[0])[in_axis]
                break

            if length is None:
                return f(*xs)

            _, out = lax.scan(body_fn, None, jnp.arange(length))

            # Mapped axis will be dim 0, so need to move if needed

            if isinstance(out_axes, tuple):
                if isinstance_namedtuple(out) or (not isinstance(out, tuple)):
                    raise ValueError("out_axes must match output structure if output is a tuple.")
                else:
                    out = (
                        (jax.tree.map(lambda _o: jnp.moveaxis(_o, 0, out_axis), o) if out_axis != 0 else o)
                        for o, out_axis in zip(out, out_axes)
                    )
            else:
                if isinstance_namedtuple(out) or (not isinstance(out, tuple)):
                    out = (
                        jax.tree.map(lambda _o: jnp.moveaxis(_o, 0, out_axes), out) if out_axes != 0 else out
                    )
                else:
                    raise ValueError("out_axes must match output structure if output is a tuple.")

            return out

        return _scan


C = TypeVar('C')


def extract_shape(s):
    # Define the regex pattern to match elements inside the brackets
    pattern = r'\[([^\]]+)\]'

    # Use re.match to find the whole match
    match = re.match(pattern, s)

    if match:
        # Extract the whole matched part within the brackets
        content = match.group(1)
        # Split the content by commas to get individual items
        groups = content.split(',')
        return list(map(str.strip, groups))
    else:
        return []


def extract_shape_tuples(s):
    # Define the regex pattern to match the entire list of tuples
    pattern = r'\[.*?\](?=,|$)'

    # Use re.findall to find all matching tuples
    matches = re.findall(pattern, s)

    return matches


def multi_vmap(f: C, in_mapping: str | List[str], out_mapping: str | List[str], scan_dims: Set[str] | None = None,
               verbose: bool = False) -> C:
    """
    A version of vmap which maps over multiple arguments.

    Args:
        f: function to map over
        in_mapping: string of input shapes, e.g. "[n1,n2,n3],[n1,n3]", only left most dims need to be represented
        out_mapping: string of output shapes, e.g. "[n1,n2,n3,...]", one per output. All input variables must be mentioned
            once. '...' means the shape of core function output. Assumes end if not given.
        scan_dims: set of dimensions to scan over
        verbose: whether to print the implied function signature

    Returns:
        mapped function


    >>> def f(x, y):
    >>>     return x + y

    >>> n1, n2, n3 = 3, 4, 5

    >>> x = jnp.ones((n1,n2,n3,2,2))
    >>> y = jnp.ones((n1,n2,n3,2,2))

    >>> f_multi = multi_vmap(f, in_mapping="[n1,n2,n3],[n1,n2,n3,2,2]", out_mapping="[..., n1,n2]", verbose=True)
    >>> res = f_multi(x, y)
    >>> assert res.shape == (n3, 2, 2) + (n1, n2) # batch shape (n3, 2, 2) and output shape (n1, n2) with transpose

    """
    if scan_dims is None:
        scan_dims = {}

    if isinstance(in_mapping, list):
        in_mapping = ','.join(in_mapping)
    if isinstance(out_mapping, list):
        out_mapping = ','.join(out_mapping)

    input_shapes = extract_shape_tuples(in_mapping)
    input_dims = [extract_shape(s) for s in input_shapes]
    all_dims = set()
    for dims in input_dims:
        if '...' in dims:
            raise ValueError(f"Input shapes must not contain '...', got {dims}.")
        all_dims.update(set(dims))

    output_shapes = extract_shape_tuples(out_mapping)
    output_dims = [extract_shape(s) for s in output_shapes]
    for dims in output_dims:
        if '...' not in dims:  # Assume it's at the end.
            dims.append('...')
        if set(dims) != set(output_dims[0]):
            raise ValueError(f"Each output shape must contain same dimensions, got {output_dims}.")
        # Ensure all dims contains
        if not all_dims.union({'...'}).issuperset(set(dims)):
            raise ValueError(f"Output shape must contain all input dims, got {dims} not all in {all_dims}.")

    out_perm = []
    out_sig = [dims.copy() for dims in output_dims]
    applicators = []
    for dim in output_dims[0]:
        if dim == '...':
            continue
        out_perm.append(dim)
        in_axes = tuple([in_dims.index(dim) if dim in in_dims else None for in_dims in input_dims])
        if verbose:
            if dim in scan_dims:
                print(f"scan({dim}, in_axes={in_axes})")
            else:
                print(f"vmap({dim}, in_axes={in_axes})")
        applicators.append(partial(vmap_or_scan, in_axes=in_axes, out_axes=0, use_scan=dim in scan_dims))
        # Remove dim from each input if it's there
        for in_dims in input_dims:
            if dim in in_dims:
                in_dims.remove(dim)
        # Remove dim from output
        for dims in out_sig:
            dims.remove(dim)

    out_perm.append('...')

    if verbose:
        input_sig = [f"({','.join(dims)})" for dims in input_dims]
        input_sig = ','.join(input_sig)
        out_sig = [f"({','.join(dims)})" for dims in out_sig]
        out_sig = ','.join(out_sig)

        print(f"Implied function signature: ({f.__module__}) {f.__name__} :: {input_sig} -> {out_sig}")

    multi_f = f
    for applicator in applicators[::-1]:
        multi_f = applicator(multi_f)

    def _permute_output(*args):
        outs = multi_f(*args)
        if isinstance_namedtuple(outs) or not isinstance(outs, tuple):
            outs = (outs,)

        res = []
        for out, dims in zip(outs, output_dims):
            if dims == out_perm:
                res.append(out)
                continue

            # Different arrangements of `dims`
            # a) [..., n1,n2] -- ellipsis at start
            # b) [n1, ..., n2] -- ellipsis at middle
            # c) [n1, n2, ...] -- ellipsis at end

            indices = list(range(len(np.shape(out))))
            num_mapped_out_dims = len(dims) - 1  # remove ...
            out_dim_map = dict(zip(out_perm[:-1], indices[:num_mapped_out_dims]))  # dim -> idx
            ellipsis_map = indices[num_mapped_out_dims:]
            if dims[0] == '...':  # a
                perm = ellipsis_map + [out_dim_map[dim] for dim in dims[1:]]
            elif dims[-1] == '...':  # c
                perm = [out_dim_map[dim] for dim in dims[:-1]] + ellipsis_map
            else:  # b
                num_start_dims = dims.index('...')  # [n1, ..., n2] -> 1
                num_end_dims = num_mapped_out_dims - num_start_dims  # [n1, ..., n2] -> 1
                perm = [out_dim_map[dim] for dim in dims[:num_start_dims]] + ellipsis_map + [out_dim_map[dim] for dim in
                                                                                             dims[-num_end_dims:]]
            res.append(lax.transpose(out, perm))
        if len(res) == 1:
            return res[0]
        return tuple(res)

    return _permute_output
