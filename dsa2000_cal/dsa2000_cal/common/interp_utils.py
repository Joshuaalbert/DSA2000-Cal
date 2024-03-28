import jax
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
    xi0 = jnp.clip(jnp.floor(xi).astype(jnp.int32), 0, len(xp) - 2)
    yi0 = jnp.clip(jnp.floor(yi).astype(jnp.int32), 0, len(yp) - 2)

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
        y = jnp.reshape(y, np.shape(y) + (1,) * (len_x - len_y))
        return x * y
    else:
        x = jnp.reshape(x, np.shape(x) + (1,) * (len_y - len_x))
        return x * y


def get_interp_indices_and_weights(x, xp) -> tuple[
    tuple[int | jax.Array, float | jax.Array], tuple[int | jax.Array, float | jax.Array]]:
    """
    One-dimensional linear interpolation. Outside bounds is also linear from nearest two points.

    Args:
        x: the x-coordinates at which to evaluate the interpolated values
        xp: the x-coordinates of the data points, must be increasing

    Returns:
        the interpolated values, same shape as `x`
    """

    x = jnp.asarray(x, dtype=jnp.float_)
    xp = jnp.asarray(xp, dtype=jnp.float_)

    # xp_arr = np.concatenate([xp[:1], xp, xp[-1:]])
    xp_arr = xp

    i = jnp.clip(jnp.searchsorted(xp_arr, x, side='right'), 1, len(xp_arr) - 1)
    dx = xp_arr[i] - xp_arr[i - 1]
    delta = x - xp_arr[i - 1]

    epsilon = np.spacing(np.finfo(xp_arr.dtype).eps)
    dx0 = jnp.abs(dx) <= epsilon  # Prevent NaN gradients when `dx` is small.
    dx = jnp.where(dx0, 1, dx)
    alpha = delta / dx
    return (i - 1, (1. - alpha)), (i, alpha)


def convolved_interp(x, y, z, k=3, mode='scaled_euclidean'):
    """
    Perform k-nearest neighbour interpolation on a set of points.

    Args:
        x: [num_x, dim] array points to evaluate convoluation at
        y: [num_y, dim] array of points to interpolate from
        z: [num_y] array of values at each point in y
        k: number of nearest neighbours to use
        mode: 'euclidean' or 'dot' for distance metric

    Returns:
        [num_x] array of interpolated values
    """
    if k > np.shape(y)[0]:
        raise ValueError("k must be less than the number of points in y")
    if mode == 'scaled_euclidean':
        y_std = jnp.std(y, axis=0) + 1e-6
        x = x / y_std
        y = y / y_std
        dist = jnp.sqrt(jnp.sum(jnp.square(x[:, None, :] - y[None, :, :]), axis=-1))  # [num_x, num_y]
    elif mode == 'euclidean':
        dist = jnp.sqrt(jnp.sum(jnp.square(x[:, None, :] - y[None, :, :]), axis=-1))  # [num_x, num_y]
    elif mode == 'dot':
        dist = 1. - jnp.sum(x[:, None, :] * y[None, :, :], axis=-1)  # [num_x, num_y]
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Get the indices of the k nearest neighbours
    select_idx = jnp.argsort(dist, axis=-1)[:, :k]  # [num_x, k]
    weights = jnp.take_along_axis(1. / (dist + 1e-6), select_idx, axis=-1)  # [num_x, k]
    weights /= jnp.sum(weights, axis=-1, keepdims=True)  # [num_x, k]
    z_interp = jnp.sum(jnp.take_along_axis(z[None, :], select_idx, axis=-1) * weights, axis=-1)  # [num_x]
    return z_interp


def batched_convolved_interp(x, y, z, k=3, mode='scaled_euclidean', unroll=1):
    """
    Perform k-nearest neighbour interpolation on a set of points.

    Args:
        x: [batch_dim, num_x, dim] array points to evaluate convoluation at
        y: [num_y, dim] array of points to interpolate from
        z: [num_y] array of values at each point in y
        k: number of nearest neighbours to use
        mode: 'euclidean' or 'dot' for distance metric

    Returns:
        [batch_dim, num_x] array of interpolated values
    """

    # Use scan to apply convolved_interp to each batch element
    def body_fn(carry, x):
        return carry, convolved_interp(x, y, z, k, mode)

    _, z_interp_batched = lax.scan(body_fn, (), x, unroll=unroll)
    return z_interp_batched
