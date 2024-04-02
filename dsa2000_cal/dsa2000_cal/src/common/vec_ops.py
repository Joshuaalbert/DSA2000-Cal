from typing import Tuple, NamedTuple

import numpy as np
from jax import numpy as jnp, lax
from jax._src.numpy.lax_numpy import ndim, shape

def vec(a: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorize a matrix.

    Args:
        a: [n, m] array

    Returns:
        [n*m] array
    """
    if len(a.shape) != 2:
        raise ValueError(f"a should be a matrix, got shape {a.shape}")
    n, m = a.shape
    # a.T.ravel()
    return lax.reshape(a, (n * m,), (1, 0))


def unvec(a: jnp.ndarray, shape: Tuple[int, ...] | None = None) -> jnp.ndarray:
    """
    Unvectorize a matrix.

    Args:
        a: [n*m] array
        shape: shape of the unvectorized array

    Returns:
        [n, m] array
    """
    if shape is None:
        # assume square
        n = int(np.sqrt(a.shape[-1]))
        if n * n != a.shape[-1]:
            raise ValueError(f"a is not square. Can't infer unvec shape.")
        shape = (n, n)
    if len(shape) != 2:
        raise ValueError(f"shape should be length 2, got {len(shape)}")
    # jnp.reshape(a, shape).T
    return lax.transpose(lax.reshape(a, shape), (1, 0))


def kron(a, b):
    """
    Compute the Kronecker product of two arrays.

    Args:
        a: [n, m]
        b: [p, q]

    Returns:
        [n*p, m*q]
    """
    if ndim(a) < ndim(b):
        a = lax.expand_dims(a, range(ndim(b) - ndim(a)))
    elif ndim(b) < ndim(a):
        b = lax.expand_dims(b, range(ndim(a) - ndim(b)))
    a_reshaped = lax.expand_dims(a, range(1, 2 * ndim(a), 2))
    b_reshaped = lax.expand_dims(b, range(0, 2 * ndim(b), 2))
    out_shape = tuple(np.multiply(shape(a), shape(b)))
    return lax.reshape(lax.mul(a_reshaped, b_reshaped), out_shape)


def kron_product(a, b, c):
    # return unvec(kron(c.T, a) @ vec(b), (a.shape[0], c.shape[1]))
    # Fewer bytes accessed, better utilisation (2x as many flops though -- which is better than memory access)
    return unvec(jnp.sum(kron(c.T, a) * vec(b), axis=-1), (a.shape[0], c.shape[1]))


class VisibilityCoords(NamedTuple):
    """
    Coordinates for a single visibility.
    """
    uvw: jnp.ndarray  # [rows, 3] the uvw coordinates
    time: jnp.ndarray  # [rows] the time
    antenna_1: jnp.ndarray  # [rows] the first antenna
    antenna_2: jnp.ndarray  # [rows] the second antenna
    time_idx: jnp.ndarray  # [rows] the time index
