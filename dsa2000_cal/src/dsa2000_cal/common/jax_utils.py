import itertools
import os
import re
import time

__all__ = [
    'chunked_pmap'
]

import warnings

from functools import partial, wraps

from typing import TypeVar, Callable, Tuple, Set, List, Dict, Union, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import pmap, lax, NamedSharding
from jax._src.mesh import Mesh
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact
from jax._src.partition_spec import PartitionSpec
from jax.experimental.mesh_utils import create_device_mesh

from dsa2000_cal.common.jvp_linear_op import isinstance_namedtuple

V = TypeVar('V')

VERBOSE_MULTI_VMAP = os.environ.get("VERBOSE_MULTI_VMAP", "0") == "1"


def promote_pytree(func_name: str, pytree: V) -> V:
    """
    Promotes a pytree to a common dtype pytree.

    Args:
        func_name: name of the function calling this function
        pytree: pytree to promote

    Returns:
        pytree with promoted dtypes
    """
    leaves, tree_def = jax.tree.flatten(pytree)
    check_arraylike(func_name, *leaves)
    leaves = promote_dtypes_inexact(*leaves)
    return jax.tree.unflatten(tree_def, leaves)


def tree_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree.map(lambda *xs: list(xs), *list_of_trees)


PT = TypeVar('PT')


def pytree_unravel(example_tree: PT) -> Tuple[Callable[[PT], jax.Array], Callable[[jax.Array], PT]]:
    """
    Returns functions to ravel and unravel a pytree.
    """
    leaf_list, tree_def = jax.tree.flatten(example_tree)

    sizes = [leaf.size for leaf in leaf_list]
    shapes = [leaf.shape for leaf in leaf_list]

    def ravel_fun(pytree: PT) -> jax.Array:
        leaf_list, tree_def = jax.tree.flatten(pytree)
        return jnp.concatenate([lax.reshape(leaf, (size,)) for leaf, size in zip(leaf_list, sizes)])

    def unravel_fun(flat_array: jax.Array) -> PT:
        leaf_list = []
        start = 0
        for size, shape in zip(sizes, shapes):
            leaf_list.append(lax.reshape(flat_array[start:start + size], shape))
            start += size
        return jax.tree.unflatten(tree_def, leaf_list)

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
            leaves = jax.tree.leaves((args, kwargs))
            batch_size = np.shape(leaves[0])[0]
            for leaf in leaves:
                if np.shape(leaf)[0] != batch_size:
                    raise ValueError(f"All leaves must have the same first dimension, got {np.shape(leaf)}.")
            remainder = batch_size % chunk_size
            extra = (chunk_size - remainder) % chunk_size
            if extra > 0:
                (args, kwargs) = jax.tree.map(lambda x: _pad_extra(x, extra), (args, kwargs))
            (args, kwargs) = jax.tree.map(
                lambda x: jnp.reshape(x, (chunk_size, x.shape[0] // chunk_size) + x.shape[1:]),
                (args, kwargs))
            result = pmap(queue)(*args, **kwargs)  # [chunksize, batch_size // chunksize, ...]
            result = jax.tree.map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), result)
            if extra > 0:
                result = jax.tree.map(lambda x: x[:-extra], result)
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

    leaves = jax.tree.leaves(py_tree)

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
        py_tree = jax.tree.map(lambda x: jnp.concatenate([x, jnp.repeat(x[0:1], extra, 0)]), py_tree)

    def _remove_extra(output_py_tree: S) -> S:
        if extra > 0:
            output_py_tree = jax.tree.map(lambda x: x[:-extra], output_py_tree)
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
    """
    Extracts the shape from a string.

    Args:
        s: string, e.g. '[n1,n2,n3]'

    Returns:
        list of dimensions, e.g. ['n1', 'n2', 'n3']
    """
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


def auto_multi_vmap(f: C, in_mapping: str | List[str], out_mapping: str | List[str],
                    scan_dims: Set[str] | None = None, compute_bound: bool = False, max_scan_dims: int | None = None,
                    verbose: bool = VERBOSE_MULTI_VMAP):
    """
    Finds best dims to scan over based on input and output shapes.

    Args:
        f: function to map over (see multi_vmap)
        in_mapping: the input shapes
        out_mapping: the output shapes
        scan_dims: set of dimensions to search over
        compute_bound: whether to use compute bound instead of memory bound
        max_scan_dims: maximum combination size of scan dims to search over
        verbose: whether to print the implied function signature

    Returns:
        the mapped function with best scan dim
    """

    warnings.warn("auto_multi_vmap is still experimental, and often fails.")

    if scan_dims is None:
        if isinstance(out_mapping, list):
            out_mapping = ','.join(out_mapping)
        output_shapes = extract_shape_tuples(out_mapping)
        output_dims = [extract_shape(s) for s in output_shapes]
        scan_dims = set()
        for dims in output_dims:
            scan_dims.update(set(dims))
        # remove "..."
        scan_dims = scan_dims.difference({'...'})

    if max_scan_dims is None:
        max_scan_dims = 1

    def _f(*args):
        best_s = {}
        best_memory = np.inf
        best_compute = np.inf
        if verbose:
            print(f"Auto optimising multi_vmap({f.__name__})...")

        for c in range(0, max_scan_dims + 1):
            for _scan_dims in itertools.combinations(scan_dims, c):
                _scan_dims = set(_scan_dims)
                f_multi_candiate = multi_vmap(
                    f,
                    in_mapping=in_mapping,
                    out_mapping=out_mapping,
                    scan_dims=_scan_dims,
                    verbose=False
                )
                g = jax.jit(f_multi_candiate).lower(*args).compile()
                [analysis] = g.cost_analysis()
                # print(analysis)
                compute_cost = 0
                if 'flops' in analysis:
                    compute_cost += analysis["flops"]
                if 'transcendentals' in analysis:
                    compute_cost += 20 * analysis["transcendentals"]  # avg. worth ~20 flops
                memory_cost = 0
                if 'bytes accessed' in analysis:
                    memory_cost += analysis["bytes accessed"]

                if verbose:
                    print(
                        f"Considering: scan_dims={_scan_dims}, memory_cost={memory_cost}, compute_cost={compute_cost}")

                if compute_bound:
                    if compute_cost < best_compute:
                        best_s = _scan_dims
                        best_memory = memory_cost
                        best_compute = compute_cost
                        if verbose:
                            print(f"> Accepted as better!")
                    elif compute_cost <= 1.2 * best_compute and memory_cost < best_memory:
                        best_s = _scan_dims
                        best_memory = memory_cost
                        best_compute = compute_cost
                        if verbose:
                            print(f"> Accepted as better!")
                else:
                    if memory_cost < best_memory:
                        best_s = _scan_dims
                        best_memory = memory_cost
                        best_compute = compute_cost
                        if verbose:
                            print(f"> Accepted as better!")
                    elif memory_cost <= 1.2 * best_memory and compute_cost < best_compute:
                        best_s = _scan_dims
                        best_memory = memory_cost
                        best_compute = compute_cost
                        if verbose:
                            print(f"> Accepted as better!")
        f_multi = multi_vmap(
            f,
            in_mapping=in_mapping,
            out_mapping=out_mapping,
            scan_dims=best_s,
            verbose=verbose
        )
        return f_multi(*args)

    return _f


def multi_vmap(f: C, in_mapping: str | List[str], out_mapping: str | List[str], scan_dims: Set[str] | None = None,
               verbose: bool = VERBOSE_MULTI_VMAP, auto: bool = False, auto_kwargs: Dict | None = None) -> C:
    """
    A version of vmap which maps over multiple arguments according to a signature.

    Args:
        f: function to map over
        in_mapping: string of input shapes, e.g. "[n1,n2,n3],[n1,n3]", only left most dims need to be represented
        out_mapping: string of output shapes, e.g. "[n1,n2,n3,...]", one per output. All input variables must be
            mentioned once. '...' can be used as a placeholder, and ~ prefixes mean that the dimension is not mapped.
        scan_dims: set of dimensions to scan over
        verbose: whether to print the implied function signature
        auto: whether to automatically find best scan dims
        auto_kwargs: kwargs for auto_multi_vmap

    Returns:
        mapped function


    def f(x, y):
        return x + y

    n1, n2, n3 = 3, 4, 5

    x = jnp.ones((n1,n2,n3,2,2))
    y = jnp.ones((n1,n2,n3,2,2))

    f_multi = multi_vmap(f, in_mapping="[n1,n2,n3],[n1,n2,n3,2,2]", out_mapping="[..., n1,n2]", verbose=True)
    res = f_multi(x, y)
    assert res.shape == (n3, 2, 2) + (n1, n2) # batch shape (n3, 2, 2) and output shape (n1, n2) with transpose

    """

    if auto:
        if auto_kwargs is None:
            auto_kwargs = dict()
        return auto_multi_vmap(
            f=f,
            in_mapping=in_mapping,
            out_mapping=out_mapping,
            verbose=verbose,
            **auto_kwargs
        )

    if scan_dims is None:
        scan_dims = {}

    if isinstance(in_mapping, list):
        in_mapping = ','.join(in_mapping)
    if isinstance(out_mapping, list):
        out_mapping = ','.join(out_mapping)

    input_dims = [extract_shape(s) for s in extract_shape_tuples(in_mapping)]
    output_dims = [extract_shape(s) for s in extract_shape_tuples(out_mapping)]
    mapped_dims = []
    for dims in output_dims:
        # Put ... at the end if not present
        if '...' not in dims:
            dims.append('...')
        for dim in dims:
            if dim.startswith("~") or dim == '...':
                # Dims prefixed with ~ are not mapped
                if dims.count(dim) > 1:
                    raise ValueError(f"Unmapped dimension {dim} must be unique in output dims {dims}.")
                continue
            if dim not in mapped_dims:
                mapped_dims.append(dim)
    for dims in input_dims:
        if "..." not in dims:
            dims.append('...')
    # Ensure all output dims contain all the mapped dims
    for dims in output_dims:
        if not all(dim in dims for dim in mapped_dims):
            raise ValueError(f"Output dims {dims} must contain all mapped dims {mapped_dims}.")
    # Ensure all mapped dims are in all input dims
    for dim in mapped_dims:
        if not any(dim in in_dims for in_dims in input_dims):
            raise ValueError(f"Mapped dim {dim} must be in some input dims {input_dims}.")

    # handle dims from left to right, for best efficiency
    in_sig = [dims.copy() for dims in input_dims]
    out_sig = [dims.copy() for dims in output_dims]
    applicators = []

    def fmt_sig(dims):
        dims = [f"[{','.join(dim)}]" for dim in dims]
        return f"({','.join(dims)})"

    if verbose:
        print(f"=== ({f.__module__}) {f.__name__} ===")
        print(f"In: {fmt_sig(in_sig)} Out: {fmt_sig(out_sig)}")

    for dim in mapped_dims:
        in_axes = tuple([in_dims.index(dim) if dim in in_dims else None for in_dims in in_sig])
        # Remove dim from each input if it's there
        for in_dims in in_sig:
            if dim in in_dims:
                in_dims.remove(dim)
        # Remove dim from output
        for dims in out_sig:
            dims.remove(dim)
        if verbose:
            if dim in scan_dims:
                print(f"scan({dim}, in_axes={in_axes}, out_axes=0) # -> In: {fmt_sig(in_sig)} Out: {fmt_sig(out_sig)}")
            else:
                print(f"vmap({dim}, in_axes={in_axes}, out_axes=0)  # -> In: {fmt_sig(in_sig)} Out: {fmt_sig(out_sig)}")
        applicators.append(partial(vmap_or_scan, in_axes=in_axes, out_axes=0, use_scan=dim in scan_dims))

    post_application_output_dims = []
    for out_dims in out_sig:
        post_application_output_dims.append(mapped_dims + out_dims)

    if verbose:
        input_sig = [f"({','.join(dims)})" for dims in in_sig]
        input_sig = ','.join(input_sig)
        out_sig = [f"({','.join(dims)})" for dims in out_sig]
        out_sig = ','.join(out_sig)
        print(f"Implied function signature :: {input_sig} -> {out_sig}")

    multi_f = f
    for applicator in applicators[::-1]:
        multi_f = applicator(multi_f)

    def _compute_and_permute_output(*args):
        # Apply the function, and then permute the output
        outs = multi_f(*args)
        if isinstance_namedtuple(outs) or not isinstance(outs, tuple):
            outs = (outs,)

        if len(outs) != len(output_dims):
            raise ValueError(f"Number of outputs {len(outs)} must match output mapping {out_mapping}.")

        def _apply_perm(x, in_dims, out_dims):
            perm = _get_permutation(len(np.shape(x)), in_dims, out_dims)
            if np.all(np.diff(perm) == 1):
                return x
            return lax.transpose(x, perm)

        res = []
        for out, in_dims, out_dims in zip(outs, post_application_output_dims, output_dims):
            res.append(jax.tree.map(lambda x: _apply_perm(x, in_dims, out_dims), out))

        if len(res) == 1:
            return res[0]
        return tuple(res)

    return _compute_and_permute_output


def _get_permutation(num_dims: int, in_axes: List[str], out_axes: List[str]) -> Tuple[int, ...]:
    """
    Get the permutation to move input axes to output axes.

    Args:
        num_dims: number of dimensions
        in_axes: input axes
        out_axes: output axes

    Returns:
        the permutation to move input axes to output axes

    Examples:
        _get_permutation(5, ['a', 'b', '...', 'c'], ['a', '...', 'b', 'c'])
        (0, 2, 3, 1, 4)
    """
    if len(list(filter(lambda x: x != '...', in_axes))) > num_dims:
        raise ValueError(f"Input axes {in_axes} must have {num_dims} dimensions.")
    # Get indices
    indices = list(range(num_dims))
    # Assign dimensions to indices, and ellipsis to range of indices
    ellipsis_size = num_dims - len(in_axes) + 1 if '...' in in_axes else 0  # 3, ['a', '...'] -> 2
    dim_indices_map: Dict[str, Tuple[int, int]] = dict()  # dim -> (idx_from, idx_to)
    idx = 0
    for dim in in_axes:
        if dim == '...':
            dim_indices_map[dim] = (idx, idx + ellipsis_size)
            idx += ellipsis_size
        else:
            dim_indices_map[dim] = (idx, idx + 1)
            idx += 1
    # Get permutation
    perm = []
    for dim in out_axes:
        if dim not in dim_indices_map:
            raise ValueError(f"Dimension {dim} not found in input axes {in_axes}.")
        idx_from, idx_to = dim_indices_map[dim]
        perm.extend(indices[idx_from:idx_to])
    return tuple(perm)


def create_mesh(shape, axis_names, devices=None):
    """
    Create a mesh from a shape and axis names.

    Args:
        shape: the shape of the mesh, total size must evenly divide number of devices.
        axis_names: the axis names of the mesh.
        devices: the devices to use, if None, uses all devices.

    Returns:
        the mesh
    """
    if len(shape) != len(axis_names):
        raise ValueError(f"Shape {shape} and axis names {axis_names} must have the same length.")
    mesh_size = int(np.prod(shape))
    if devices is None:
        devices = jax.devices()
        if mesh_size < len(devices):
            devices = devices[:mesh_size]
    if mesh_size % len(devices) != 0:
        raise ValueError(f"Mesh size {mesh_size} must evenly divide number of devices {len(devices)}.")
    mesh_devices = create_device_mesh(mesh_shape=shape, devices=devices)
    mesh = Mesh(mesh_devices, axis_names=axis_names)
    return mesh


SPT = TypeVar('SPT')


def tree_device_put(tree: SPT, mesh: Mesh, axis_names: Tuple[str | None, ...]) -> SPT:
    """
    Put a pytree on a device.

    Args:
        tree: the pytree to put on a device.
        mesh: the mesh to put the pytree on.
        axis_names: the axis names of the mesh.

    Returns:
        the pytree on the device.
    """
    sharding = NamedSharding(mesh, PartitionSpec(*axis_names))
    return jax.tree.map(lambda x: jax.device_put(x, sharding), tree)


BUX = TypeVar('BUX', bound=Union[jax.Array, Any])


def block_until_ready(x: BUX) -> BUX:
    return jax.block_until_ready(x)


CUF = TypeVar('CUF', bound=Callable[..., Any])


def convert_to_ufunc(f: CUF, tile: bool = True) -> CUF:
    f = jax.custom_batching.custom_vmap(f)

    @f.def_vmap
    def rule(axis_size, in_batched, *args):
        axis_size = axis_size if tile else 1
        batched_args = jax.tree.map(
            lambda x, b: x if b else jax.lax.broadcast(x, (axis_size,)), args,
            tuple(in_batched))
        out = f(*batched_args)
        out_batched = jax.tree.map(lambda _: True, out)
        return out, out_batched

    return f


def print_nans(args):
    def print_args(args):
        if any(np.any(np.isnan(a)) for a in jax.tree.leaves(args)):
            print(args)
        return args

    return jax.pure_callback(
        print_args,
        jax.tree.map(lambda x: jax.ShapeDtypeStruct(jnp.shape(x), jnp.dtype(x)), args),
        args
    )


U = TypeVar('U')


def simple_broadcast(f: Callable[..., U], leading_dims: int) -> U:
    """
    Simple broadcast function which reshapes the inputs to have the same leading dimensions, and then reshapes the output
    to have the original leading dimensions.

    Args:
        f: function to apply
        leading_dims: the leading dimensions to broadcast over

    Returns:
        the result of the function applied to the arguments with single vmap
    """

    @wraps(f)
    def _f(*args):
        nonlocal leading_dims
        leading_shapes = jax.tree.map(lambda x: np.shape(x)[:leading_dims], jax.tree.leaves(args))
        if len(leading_shapes) == 0:
            raise ValueError("No args given.")

        if len(set(leading_shapes)) != 1:
            raise ValueError(f"Leading dimensions mis-match, got these different ones {leading_shapes}.")

        leading_shape = leading_shapes[0]
        leading_size = int(np.prod(leading_shape))
        args = jax.tree.map(lambda x: jax.lax.reshape(x, (leading_size,) + np.shape(x)[leading_dims:]), args)
        out = jax.vmap(f)(*args)
        # reshape
        out = jax.tree.map(lambda x: jax.lax.reshape(x, leading_shape + np.shape(x)[1:]), out)
        return out

    return _f
