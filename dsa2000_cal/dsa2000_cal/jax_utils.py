from typing import TypeVar, Callable

from jax import numpy as jnp, tree_map, pmap
from jax._src.lax.control_flow import scan

FV = TypeVar('FV')


def chunked_pmap(f: Callable[..., FV], chunksize: int, *, batch_size=None) -> Callable[..., FV]:
    """
    A version of pmap which chunks the input into smaller pieces to avoid memory issues.

    Args:
        f: callable
        chunksize: the size of the chunks
        batch_size: the size of the batch, if None, will be inferred from the first argument.

    Returns:
        a chunked version of f
    """

    def _f(*args, batch_size=batch_size, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = scan(body, (), (args, kwargs))
            return result

        if chunksize > 1:
            if batch_size is None:
                batch_size = args[0].shape[0] if len(args) > 0 else None
            assert batch_size is not None, "Couldn't get batch_size, please provide explicitly"
            remainder = batch_size % chunksize
            extra = (chunksize - remainder) % chunksize
            args = tree_map(lambda arg: _pad_extra(arg, chunksize), args)
            kwargs = tree_map(lambda arg: _pad_extra(arg, chunksize), kwargs)
            result = pmap(queue)(*args, **kwargs)
            result = tree_map(lambda arg: jnp.reshape(arg, (-1,) + arg.shape[2:]), result)
            if extra > 0:
                result = tree_map(lambda x: x[:-extra], result)
        else:
            result = queue(*args, **kwargs)
        return result

    _f.__doc__ = f.__doc__
    _f.__annotations__ = f.__annotations__
    return _f


def _pad_extra(arg, chunksize):
    N = arg.shape[0]
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        # only pad if not a zero remainder
        extra = (chunksize - remainder) % chunksize
        arg = jnp.concatenate([arg] + [arg[0:1]] * extra, axis=0)
        N = N + extra
    else:
        extra = 0
    T = N // chunksize
    arg = jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:])
    return arg


def prepad(a, chunksize: int):
    return tree_map(lambda arg: _pad_extra(arg, chunksize), a)
