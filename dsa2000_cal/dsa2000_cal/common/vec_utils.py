from typing import Tuple

import numpy as np
from jax import numpy as jnp, lax


def vec(a: jnp.ndarray, transpose: bool=False) -> jnp.ndarray:
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
    if transpose:
        return lax.reshape(a, (n * m,))
    return lax.reshape(a, (n * m,), (1, 0))


def unvec(a: jnp.ndarray, shape: Tuple[int, ...] | None = None, transpose: bool=False) -> jnp.ndarray:
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
    if transpose:
        # jnp.reshape(a, shape).T
        return lax.reshape(a, shape)
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
    if len(np.shape(a)) < len(np.shape(b)):
        a = lax.expand_dims(a, range(np.ndim(b) - np.ndim(a)))
    elif np.ndim(b) < np.ndim(a):
        b = lax.expand_dims(b, range(np.ndim(a) - np.ndim(b)))
    a_reshaped = lax.expand_dims(a, range(1, 2 * np.ndim(a), 2))
    b_reshaped = lax.expand_dims(b, range(0, 2 * np.ndim(b), 2))
    out_shape = tuple(np.multiply(np.shape(a), np.shape(b)))
    return lax.reshape(lax.mul(a_reshaped, b_reshaped), out_shape)


def kron_product(a, b, c):
    """
    Compute the matrix product of three matrices using Kronecker product.

    a @ b @ c

    Args:
        a: [n, m]
        b: [m, p]
        c: [p, q]

    Returns:
        [n, q]
    """
    # return unvec(kron(c.T, a) @ vec(b), (a.shape[0], c.shape[1]))
    # Fewer bytes accessed, better utilisation (2x as many flops though -- which is better than memory access)
    return unvec(jnp.sum(kron(c.T, a) * vec(b), axis=-1), (a.shape[0], c.shape[1]))
