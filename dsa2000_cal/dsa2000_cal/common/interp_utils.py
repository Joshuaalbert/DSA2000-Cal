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
    z0 = left_broadcast_multiply(z00, (1 - wx)) + left_broadcast_multiply(z10, wx)
    z1 = left_broadcast_multiply(z01, (1 - wx)) + left_broadcast_multiply(z11, wx)
    z_interp = left_broadcast_multiply(z0, (1 - wy)) + left_broadcast_multiply(z1, wy)

    return z_interp


def apply_interp(x: jax.Array, i0: jax.Array, alpha0: jax.Array, i1: jax.Array, alpha1: jax, axis: int = 0):
    """
    Apply interpolation alpha given axis.

    Args:
        x: nd-array
        i0: [N] or scalar
        alpha0: [N] or scalar
        i1: [N] or scalar
        alpha1: [N] or scalar
        axis: axis to take along

    Returns:
        [N] or scalar interpolated along axis
    """
    return left_broadcast_multiply(jnp.take(x, i0, axis=axis), alpha0, axis=axis) + left_broadcast_multiply(
        jnp.take(x, i1, axis=axis), alpha1, axis=axis)


def left_broadcast_multiply(x, y, axis: int = 0):
    """
    Left broadcast multiply of two arrays.
    Equivalent to right-padding before multiply

    Args:
        x: [..., a,b,c,...]
        y: [a, b]

    Returns:
        [..., a, b, c, ...]
    """
    needed_length = len(np.shape(x)[axis:])
    len_y = len(np.shape(y))
    extra = needed_length - len_y
    if extra < 0:
        raise ValueError(f"Shape mismatch {np.shape(x)} x {np.shape(y)}.")
    y = lax.reshape(y, np.shape(y) + (1,) * extra)
    return x * y


def get_interp_indices_and_weights(x, xp, regular_grid: bool = False) -> tuple[
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
    if len(np.shape(xp)) == 0:
        xp = jnp.reshape(xp, (-1,))
    if np.shape(xp)[0] == 0:
        raise ValueError("xp must be non-empty")
    if np.shape(xp)[0] == 1:
        return (jnp.zeros_like(x, dtype=jnp.int32), jnp.ones_like(x)), (
            jnp.zeros_like(x, dtype=jnp.int32), jnp.zeros_like(x))

    # Find xp[i1-1] < x <= xp[i1]
    if regular_grid:
        # Use faster index determination
        delta_x = xp[1] - xp[0]
        i1 = jnp.clip((jnp.ceil((x - xp[0]) / delta_x)).astype(jnp.int64), 1, len(xp) - 1)
        i0 = i1 - 1
    else:
        i1 = jnp.clip(jnp.searchsorted(xp, x, side='right'), 1, len(xp) - 1)
        i0 = i1 - 1

    dx = xp[i1] - xp[i0]
    delta = x - xp[i0]

    epsilon = np.spacing(np.finfo(xp.dtype).eps)
    dx0 = jnp.abs(dx) <= epsilon  # Prevent NaN gradients when `dx` is small.
    dx = jnp.where(dx0, 1, dx)
    alpha = delta / dx
    return (i0, (1. - alpha)), (i1, alpha)


def get_nn_points(x, y, k=3, mode='euclidean'):
    """
    Perform k-nearest neighbour interpolation on a set of points.

    Args:
        x: [num_x, dim] array points to evaluate convoluation at
        y: [num_y, dim] array of points to interpolate from
        k: number of nearest neighbours to use
        mode: 'euclidean' or 'dot' for distance metric

    Returns:
        [num_x, k] array of indices of the k nearest neighbours, and [num_x, k] array of distances
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
        dist = - jnp.sum(x[:, None, :] * y[None, :, :], axis=-1)  # [num_x, num_y]
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Get the indices of the k nearest neighbours
    neg_dist, select_idx = jax.lax.top_k(lax.neg(dist), k)  # [num_x, k]
    dist = lax.neg(neg_dist)  # [num_x, k]
    return select_idx, dist


def convolved_interp(x, y, z, k=3, mode='euclidean'):
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
    select_idx, dist = get_nn_points(x=x, y=y, k=k, mode=mode) # [num_x, k]
    weights = 1. / (dist + 1e-6)  # [num_x, k]
    weights /= jnp.sum(weights, axis=-1, keepdims=True)  # [num_x, k]
    z_interp = jnp.sum(jnp.take_along_axis(z[None, :], select_idx, axis=-1) * weights, axis=-1)  # [num_x]
    return z_interp


def batched_convolved_interp(x, y, z, k=3, mode='euclidean', unroll=1):
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


def get_centred_insert_index(insert_value: np.ndarray, grid_centres: np.ndarray,
                             ignore_out_of_bounds: bool = False) -> np.ndarray:
    """
    Get the insert_idx to insert the values at. Values are all relative. Since grid_centre represent
    the centre of intervals we need to find the insert indexes that respect this centring.

    Args:
        insert_value: the values to insert
        grid_centres: the centre grids to insert into

    Returns:
        the insert_indsec to insert the values at

    Raises:
        ValueError: if insert values is too far outside range
    """
    # Finds the index such that t[i] <= t_insert < t[i+1],
    # where t[i] = t_centre[i] - 0.5 * dt and t[i+1] = t_centre[i] + 0.5 * dt
    if len(grid_centres) == 0:
        raise ValueError("Grid centres must be non-empty")
    elif len(grid_centres) == 1:
        return np.zeros_like(insert_value, dtype=np.int32)
    dt0 = grid_centres[1] - grid_centres[0]
    edge = grid_centres - 0.5 * np.diff(grid_centres, prepend=grid_centres[0] - dt0)
    edge = np.append(edge, edge[-1] + dt0)
    insert_idx = np.searchsorted(edge, insert_value, side='right') - 1
    if not ignore_out_of_bounds and (np.any(insert_idx < 0) or np.any(insert_idx >= len(grid_centres))):
        raise ValueError("Insert value is too far outside range. "
                         f"{insert_value[insert_idx < 0]} < {edge[0]} or "
                         f"{insert_value[insert_idx >= len(grid_centres)]} > {edge[-1]}")
    elif ignore_out_of_bounds:
        insert_idx = np.clip(insert_idx, 0, len(grid_centres) - 1)
    return insert_idx
