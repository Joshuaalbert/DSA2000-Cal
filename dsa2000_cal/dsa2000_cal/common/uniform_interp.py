import jax.numpy as jnp
import numpy as np
from jax import lax


def optimized_interp_jax_safe(x, xp, yp):
    """
    Perform linear interpolation on a set of points, with additional handling
    to avoid NaN gradients when the difference in consecutive xp values is very small.

    This function is specifically optimized for input xp values that are uniformly
    spaced. It uses JAX for JIT compilation to accelerate computation and ensure
    gradient safety for automatic differentiation.

    Parameters:
    - x (jax.numpy.ndarray): The x-coordinates at which to evaluate the interpolated values.
    - xp (jax.numpy.ndarray): The x-coordinates of the data points, must be uniformly spaced.
    - yp (jax.numpy.ndarray): The y-coordinates of the data points, corresponding to xp.

    Returns:
    - jax.numpy.ndarray: The interpolated values at each x.
    """
    x = jnp.where(x < xp[0], xp[0], jnp.where(x > xp[-1], xp[-1], x))
    a = xp[0]  # Start of the xp interval
    b = xp[-1]  # End of the xp interval
    N = xp.shape[0]
    dx = (b - a) / (N - 1)  # Compute spacing between points in xp

    # Calculate indices for each x value, ensuring they are within bounds
    indices = jnp.floor((x - a) / dx).astype(jnp.int32)

    # Get the relevant yp values and compute the difference
    yp0 = yp[indices]
    yp1 = yp[indices + 1]
    xp0 = xp[indices]
    delta = x - xp0
    df = yp1 - yp0

    # Calculate epsilon based on the data type of xp for safe division
    epsilon = np.spacing(jnp.finfo(xp.dtype).eps)
    dx0 = lax.abs(dx) <= epsilon  # Check if dx is too small, indicating potential numerical instability

    # Perform interpolation, adjusting for small dx values to avoid NaN gradients
    f = jnp.where(dx0, yp0, yp0 + (delta / jnp.where(dx0, 1.0, dx)) * df)
    f = yp0 + (delta / dx) * df
    print(delta, dx, df)
    return f


def multilinear_interp_2d(x, y, xp, yp, z):
    """
    Perform 2D multilinear interpolation on a uniform grid.

    Parameters:
    - x (jax.numpy.ndarray): x-coordinates at which to evaluate, shape [M,].
    - y (jax.numpy.ndarray): y-coordinates at which to evaluate, shape [M,].
    - xp (jax.numpy.ndarray): x-coordinates of the data points, shape [Nx,].
    - yp (jax.numpy.ndarray): y-coordinates of the data points, shape [Ny,].
    - z (jax.numpy.ndarray): values at the data points, shape [Nx, Ny].

    Returns:
    - jax.numpy.ndarray: Interpolated values at (x, y) points, shape [M,].

    Notes:
    - Assumes xp and yp are uniformly spaced.
    - For out-of-bounds (x, y), uses the closest edge value.
    """
    if np.shape(x) != np.shape(y):
        raise ValueError("x and y must have the same shape")
    if len(np.shape(xp)) != 1 or len(np.shape(yp)) != 1:
        raise ValueError("xp and yp must be 1D arrays")
    if np.shape(z)[:2] != (np.size(xp), np.size(yp)):
        raise ValueError(f"z must have shape (len(xp), len(yp)), got {np.shape(z)}")

    x = jnp.where(x < xp[0], xp[0], jnp.where(x > xp[-1], xp[-1], x))
    y = jnp.where(y < yp[0], yp[0], jnp.where(y > yp[-1], yp[-1], y))

    # Grid spacing
    dx = xp[1] - xp[0]
    dy = yp[1] - yp[0]

    # Normalize x, y coordinates to grid indices
    xi = (x - xp[0]) / dx
    yi = (y - yp[0]) / dy

    # Calculate lower indices
    xi0 = np.clip(jnp.floor(xi).astype(jnp.int32), 0, len(xp) - 2)
    yi0 = np.clip(jnp.floor(yi).astype(jnp.int32), 0, len(yp) - 2)

    # Calculate weights
    wx = xi - xi0
    wy = yi - yi0

    # Gather corners
    z00 = z[xi0, yi0]
    z01 = z[xi0, yi0 + 1]
    z10 = z[xi0 + 1, yi0]
    z11 = z[xi0 + 1, yi0 + 1]

    # Bilinear interpolation
    z0 = _left_broadcast_multiply(z00, (1 - wx)) + _left_broadcast_multiply(z10, wx)
    z1 = _left_broadcast_multiply(z01, (1 - wx)) + _left_broadcast_multiply(z11, wx)
    z_interp = _left_broadcast_multiply(z0, (1 - wy)) + _left_broadcast_multiply(z1, wy)

    return z_interp


def _left_broadcast_multiply(x, y):
    """
    Left broadcast multiply of two arrays.
    Equivalent to right-padding before multiply

    Args:
        x: [a,b,c,...]
        y: [a, b]

    Returns:
        [a, b, c, ...]
    """
    len_x = len(np.shape(x))
    len_y = len(np.shape(y))
    if len_x == len_y:
        return x * y
    if len_x > len_y:
        y = np.reshape(y, np.shape(y) + (1,) * (len_x - len_y))
        return x * y
    else:
        x = np.reshape(x, np.shape(x) + (1,) * (len_y - len_x))
        return x * y


def test__left_broadcast_multiply():
    assert np.all(_left_broadcast_multiply(np.ones((2, 3)), np.ones((2,))) == np.ones((2, 3)))
    assert np.all(_left_broadcast_multiply(np.ones((2,)), np.ones((2, 3))) == np.ones((2, 3)))
    assert np.all(_left_broadcast_multiply(np.ones((2, 3)), np.ones((2, 3))) == np.ones((2, 3)))
