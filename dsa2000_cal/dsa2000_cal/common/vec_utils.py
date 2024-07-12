from typing import Tuple

import jax
import numpy as np
from jax import numpy as jnp, lax


def vec(a: jnp.ndarray, transpose: bool = False) -> jnp.ndarray:
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


def unvec(a: jnp.ndarray, shape: Tuple[int, ...] | None = None, transpose: bool = False) -> jnp.ndarray:
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


def kron_product(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
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
    # if np.shape(a) == (2, 2) and np.shape(b) == (2, 2) and np.shape(c) == (2, 2):
    #     # Still slower than using kron product
    #     return kron_product_2x2(a, b, c)
    # return a @ b @ c
    # return unvec(kron(c.T, a) @ vec(b), (a.shape[0], c.shape[1]))
    # Fewer bytes accessed, better utilisation (2x as many flops though -- which is better than memory access)
    return unvec(jnp.sum(kron(c.T, a) * vec(b), axis=-1), (a.shape[0], c.shape[1]))


def kron_product_2x2(M0: jax.Array, M1: jax.Array, M2: jax.Array) -> jax.Array:
    # Matrix([[a0*(a1*a2 + b1*c2) + b0*(a2*c1 + c2*d1), a0*(a1*b2 + b1*d2) + b0*(b2*c1 + d1*d2)], [c0*(a1*a2 + b1*c2) + d0*(a2*c1 + c2*d1), c0*(a1*b2 + b1*d2) + d0*(b2*c1 + d1*d2)]])
    # 36
    # ([(x0, a1*a2 + b1*c2), (x1, a2*c1 + c2*d1), (x2, a1*b2 + b1*d2), (x3, b2*c1 + d1*d2)], [Matrix([
    # [a0*x0 + b0*x1, a0*x2 + b0*x3],
    # [c0*x0 + d0*x1, c0*x2 + d0*x3]])])
    a0, b0, c0, d0 = M0[0, 0], M0[0, 1], M0[1, 0], M0[1, 1]
    a1, b1, c1, d1 = M1[0, 0], M1[0, 1], M1[1, 0], M1[1, 1]
    a2, b2, c2, d2 = M2[0, 0], M2[0, 1], M2[1, 0], M2[1, 1]
    x0 = a1 * a2 + b1 * c2
    x1 = a2 * c1 + c2 * d1
    x2 = a1 * b2 + b1 * d2
    x3 = b2 * c1 + d1 * d2

    flat = jnp.stack([a0 * x0 + b0 * x1, c0 * x0 + d0 * x1, a0 * x2 + b0 * x3, c0 * x2 + d0 * x3], axis=-1)
    return unvec(flat, (2, 2))
