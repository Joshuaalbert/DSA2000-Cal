import atexit
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

import jax
import numpy as np

__all__ = [
    'construct_threaded_pure_callback',
]


class _ThreadPoolSingleton:
    """Singleton wrapper for ThreadPoolExecutor."""
    _instance: ThreadPoolExecutor | None = None
    # Take the same default as threading library, but set it here for clarity
    _num_threads = os.environ.get('JAX_PURE_CALLBACK_NUM_THREADS', min(32, (os.cpu_count() or 1) + 4))

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ThreadPoolExecutor(max_workers=cls._num_threads, thread_name_prefix="jax_pure_callback_")
            atexit.register(cls.shutdown)
        return cls._instance

    @classmethod
    def shutdown(cls):
        if cls._instance:
            # wait True ensures a displaced running threadpool finishes before shutdown.
            cls._instance.shutdown(wait=True)
            cls._instance = None


def _build_callback_from_kernel(cb_kernel: Callable, batch_shape_determiner: Callable, num_threads: int | None):
    def callback(*args):
        # Determine leading dims.
        batch_shape = batch_shape_determiner(*args)
        batch_size = int(np.prod(batch_shape))

        def sliced_kernel(index):
            multi_idx = np.unravel_index(index, batch_shape)

            def _slice(x):
                if x is None:
                    return x
                _multi_idx = []
                for idx, s in (zip(multi_idx, np.shape(x))):
                    if s == 1:
                        _multi_idx.append(0)
                    else:
                        _multi_idx.append(idx)
                return x[tuple(_multi_idx)]

            args_slice = jax.tree.map(_slice, args)
            return cb_kernel(*args_slice)

        if num_threads is not None:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                result_map = executor.map(sliced_kernel, range(batch_size))
        else:
            executor = _ThreadPoolSingleton.get_instance()
            result_map = executor.map(sliced_kernel, range(batch_size))

        results_list = list(result_map)
        # pytree stack
        results = jax.tree.map(lambda *r: np.stack(r, axis=0), *results_list)
        # unflatten
        results = jax.tree.map(lambda x: jax.lax.reshape(x, batch_shape + np.shape(x)[1:]), results)
        return results

    return callback


def _build_batch_shape_determiner(*args_shape_size):
    def batch_shape_determiner(*args):
        if len(args) != len(args_shape_size):
            raise ValueError(f'Expected {len(args_shape_size)} arguments, got {len(args)}.')

        def _determine(x, shape_size):
            if not isinstance(shape_size, int):
                raise ValueError(f'shape_size must be an integer, got {type(shape_size)}.')
            if x is None:
                return None
            if shape_size == 0:
                return np.shape(x)
            return np.shape(x)[:-shape_size]

        batch_shapes = jax.tree.map(_determine, args, args_shape_size, is_leaf=lambda x: x is None)

        def is_leaf(s):
            # if tuple of int then it is a leaf
            return isinstance(s, tuple) and all(isinstance(i, int) for i in s)

        leaves = jax.tree.leaves(batch_shapes, is_leaf=is_leaf)
        shapes = set(leaves)
        # remove None
        shapes.discard(None)
        # broadcast
        try:
            batch_shape = np.broadcast_shapes(*list(shapes))
        except ValueError as e:
            if "shape mismatch" in str(e):
                raise ValueError(f'Inconsistent batch shapes: {shapes}')
            raise e
        return batch_shape

    return batch_shape_determiner


def construct_threaded_pure_callback(cb_kernel: Callable, result_shape_dtypes: Any, *args_shape_size,
                                     num_threads: int | None = None, vmap_method='expand_dims'):
    """
    Construct a pure callback with vmap using threading.

    Args:
        cb_kernel: a callable that takes a consistently shaped set of arguments and returns a consistently shaped
            pytree of results.
        result_shape_dtypes: a pytree of ShapeDtypeStruct objects representing the expected shape and dtype of the
            result of cb_kernel.
        *args_shape_size: the number of (unbatched) dimensions for each argument to cb_kernel.
        num_threads: the number of threads to use. If None, reuses a shared global threadpool, by default using all
            available CPUs, and which can be configured with environment variable `JAX_PURE_CALLBACK_NUM_THREADS`.
        vmap_method: the vmap method to use. Must be one of 'expand_dims' or 'broadcast_all'. Default is 'expand_dims'.
            See jax.pure_callback for more information.

    Returns:
        A pure callback that works with vmap, using threading to parallelize the computation.
    """

    def wrapped_cb_kernel(*args):
        def _check_shape(x, shape_size):
            if x is None:
                return
            if len(np.shape(x)) != shape_size:
                raise ValueError(
                    f'Expected shape of size {shape_size} but got {np.shape(x)}, sized ({len(np.shape(x))}).')

        jax.tree.map(_check_shape, args, args_shape_size, is_leaf=lambda x: x is None)
        return cb_kernel(*args)

    batch_shape_determiner = _build_batch_shape_determiner(*args_shape_size)
    cb = _build_callback_from_kernel(wrapped_cb_kernel, batch_shape_determiner, num_threads=num_threads)

    def callback(*args, vmap_method=vmap_method):
        return jax.pure_callback(cb, result_shape_dtypes, *args, vmap_method=vmap_method)

    return callback


def test_construct_threaded_pure_callback():
    import jax.numpy as jnp

    def cb_kernel(x, y):
        return x + y

    result_shape_dtypes = jax.ShapeDtypeStruct((), np.float32)

    cb = jax.vmap(jax.vmap(construct_threaded_pure_callback(cb_kernel, result_shape_dtypes, 0, 0)))
    x = jnp.ones((2, 3), np.float32)
    y = jnp.ones((2, 3), np.float32)
    res = jax.jit(cb)(x, y)
