import dataclasses
from typing import Tuple, TypeVar, Generic

import jax
import numpy as np
from jax import lax, numpy as jnp

from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.types import FloatArray, Array, ComplexArray


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


def canonicalise_axis(axis, shape_len):
    if isinstance(axis, int):
        # Convert single integer axis to positive if negative
        return axis if axis >= 0 else axis + shape_len
    else:
        # Convert each axis in a tuple to positive if negative
        return tuple(ax if ax >= 0 else ax + shape_len for ax in axis)


def apply_interp(x: jax.Array, i0: jax.Array, alpha0: jax.Array, i1: jax.Array, alpha1: jax.Array, axis: int = 0):
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
    axis = canonicalise_axis(axis, len(np.shape(x)))

    def take(i):
        # return jnp.take(x, i, axis=axis)
        num_dims = len(np.shape(x))
        # [0] [1] [2 3 4], num_dims=5, axis=1
        slices = [slice(None)] * axis + [i] + [slice(None)] * (num_dims - axis - 1)
        return x[tuple(slices)]

    return left_broadcast_multiply(take(i0), alpha0.astype(x), axis=axis) + left_broadcast_multiply(
        take(i1), alpha1.astype(x), axis=axis)


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


def get_interp_indices_and_weights(x: FloatArray, xp: FloatArray, regular_grid: bool = False,
                                   check_spacing: bool = False, clip_out_of_bounds: bool = False) -> Tuple[
    Array, Array, Array, Array]:
    """
    One-dimensional linear interpolation. Outside bounds is also linear from nearest two points.

    Args:
        x: scalar, the x-coordinate at which to evaluate the interpolated values
        xp: [n] the x-coordinates of the data points, must be increasing
        regular_grid: if True, use faster index determination
        check_spacing: if True, check spacing between points
        clip_out_of_bounds: if True, clip out-of-bounds values to the nearest edge

    Returns:
        i0: the index of the first point
        alpha0: the weight of the first point
        i1: the index of the second point
        alpha1: the weight of the second point
    """
    if not isinstance(x, (jax.Array, np.ndarray)):
        x = jnp.asarray(x)
    if not isinstance(xp, (jax.Array, np.ndarray)):
        xp = jnp.asarray(xp)
    if len(np.shape(xp)) != 1:
        raise ValueError(f"Times must be 1D, got {np.shape(xp)}.")
    if np.size(xp) == 0:
        raise ValueError("xp must be non-empty")
    if np.shape(xp) == (1,):
        return (jnp.zeros_like(x, dtype=mp_policy.index_dtype), jnp.ones_like(x),
                jnp.zeros_like(x, dtype=mp_policy.index_dtype), jnp.zeros_like(x))
    if clip_out_of_bounds:
        x = jax.lax.clamp(xp[0], x, xp[-1])

    # Find xp[i1-1] < x <= xp[i1]
    one = jnp.asarray(1, mp_policy.index_dtype)
    if regular_grid:
        # Use faster index determination
        dx = xp[1] - xp[0]
        _i1 = jnp.ceil((x - xp[0]) / dx).astype(mp_policy.index_dtype)
        i1 = mp_policy.cast_to_index(jnp.clip(_i1, one, len(xp) - 1))
        i0 = i1 - one
    else:
        i1 = mp_policy.cast_to_index(jnp.clip(jnp.searchsorted(xp, x, side='right'), one, len(xp) - 1))
        i0 = i1 - one
        dx = xp[i1] - xp[i0]

    delta = x - xp[i0]
    if check_spacing:
        epsilon = np.spacing(np.finfo(xp.dtype).eps)
        dx0 = jnp.abs(dx) <= epsilon  # Prevent NaN gradients when `dx` is small.
        dx = jnp.where(dx0, 1., dx)
    alpha1 = delta / dx
    alpha0 = 1. - alpha1
    return i0, alpha0.astype(x.dtype), i1, alpha1.astype(x.dtype)


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
    select_idx, dist = get_nn_points(x=x, y=y, k=k, mode=mode)  # [num_x, k]
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


def canonicalize_axis(axis, shape_len):
    if isinstance(axis, int):
        # Convert single integer axis to positive if negative
        return axis if axis >= 0 else axis + shape_len
    else:
        # Convert each axis in a tuple to positive if negative
        return tuple(ax if ax >= 0 else ax + shape_len for ax in axis)


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


VT = TypeVar('VT')


def field_dunder(binary_op, self: 'InterpolatedArray',
                 other: FloatArray | ComplexArray | 'InterpolatedArray') -> 'InterpolatedArray':
    if isinstance(other, InterpolatedArray):
        if other.axis != self.axis:
            raise ValueError("InterpolatedArrays must have the same axis.")
        # assumes x values are the same, only check shapes though
        if np.shape(other.x) != np.shape(self.x):
            raise ValueError("InterpolatedArrays must have the same x values. Only shape checks performed.")
        if np.shape(other.values) != np.shape(self.values):
            raise ValueError("InterpolatedArrays must have the same shape. Broadcast not supported.")
        values_sum = jax.tree_map(lambda x, y: binary_op(x, y), self.values, other.values)
        return InterpolatedArray(
            self.x, values_sum, axis=self.axis, regular_grid=self.regular_grid,
            check_spacing=self.check_spacing, clip_out_of_bounds=self.clip_out_of_bounds,
            normalise=self.normalise, auto_reorder=self.auto_reorder
        )
    else:
        values = jax.tree.map(lambda x: binary_op(x, other), self.values)
        return InterpolatedArray(
            self.x, values, axis=self.axis, regular_grid=self.regular_grid,
            check_spacing=self.check_spacing, clip_out_of_bounds=self.clip_out_of_bounds,
            normalise=self.normalise, auto_reorder=self.auto_reorder
        )


@dataclasses.dataclass(eq=False)
class InterpolatedArray(Generic[VT]):
    x: FloatArray  # [N]
    values: VT  # pytree with per leaf [..., N, ...] `axis` has N elements

    axis: int = 0
    regular_grid: bool = False
    check_spacing: bool = False
    clip_out_of_bounds: bool = False
    normalise: bool = False
    auto_reorder: bool = True

    def __post_init__(self):
        num_dims = len(np.shape(self.values))
        if self.axis == num_dims - 1:
            self.axis = -1  # Prefer it like this for getitem

        if self.auto_reorder and self.axis != -1:
            # Move axis to the last dimension
            self.values = jax.tree.map(lambda x: jnp.moveaxis(x, self.axis, -1), self.values)
            self.axis = -1

        if len(np.shape(self.x)) != 1:
            raise ValueError(f"x must be 1D, got {np.shape(self.x)}.")
        if np.shape(self.x)[0] == 0:
            raise ValueError("x must be non-empty")

        def _assert_shape(x):
            if np.shape(x)[self.axis] != np.shape(self.x)[0]:
                raise ValueError(
                    f"Input values must have size x {np.shape(self.x)[0]} on `axis` dimension ({self.axis}), "
                    f"got value shape {np.shape(x)}."
                )

        jax.tree.map(_assert_shape, self.values)

        def _promote_to_weakest_floating(x):
            x_dtype = jnp.result_type(x)
            if jnp.issubdtype(x_dtype, jnp.floating) or jnp.issubdtype(x_dtype, jnp.complexfloating):
                return x
            common_weak_floating = jnp.promote_types(jnp.float32, x_dtype)
            return x.astype(common_weak_floating)

        self.values = jax.tree.map(_promote_to_weakest_floating, self.values)

        if self.normalise:
            # Use mean and std of values, can help with precision when interpolating
            self.values_mean = jax.tree.map(lambda y: jnp.nanmean(y, axis=self.axis), self.values)
            self.values_std = jax.tree.map(lambda y: jnp.nanstd(y, axis=self.axis) + jnp.asarray(1e-6, y.dtype),
                                           self.values)
            self.values_norm = jax.tree.map(
                lambda m, s, y: (y - jnp.expand_dims(m, self.axis)) / jnp.expand_dims(s, self.axis), self.values_mean,
                self.values_std, self.values)
            # Same for x
            self.x_mean = jnp.nanmean(self.x)
            self.x_std = jnp.nanstd(self.x) + jnp.asarray(1e-6, self.x.dtype)

            self.x_norm = (self.x - self.x_mean) / self.x_std

    # Define dunder methods for field operations
    def __add__(self, other):
        return field_dunder(jnp.add, self, other)

    def __sub__(self, other):
        return field_dunder(jnp.subtract, self, other)

    def __mul__(self, other):
        return field_dunder(jnp.multiply, self, other)

    def __truediv__(self, other):
        return field_dunder(jnp.true_divide, self, other)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of returned value.
        """

        def _shape(x):
            axis = self.axis
            shape = np.shape(x)
            if axis is None:
                # Mean over all elements, resulting in a scalar (empty shape)
                return ()

            # Canonicalize the axis to ensure all are positive
            axis = canonicalize_axis(axis, len(shape))

            if isinstance(axis, int):
                # Remove the single axis
                return shape[:axis] + shape[axis + 1:]
            else:
                # Handle multiple axes, remove each specified dimension
                return tuple(dim for i, dim in enumerate(shape) if i not in axis)

        return jax.tree.map(_shape, self.values)

    def __getitem__(self, item) -> 'InterpolatedArray':
        values = jax.vmap(lambda x: x[item], in_axes=self.axis, out_axes=self.axis)(self.values)

        return InterpolatedArray(
            self.x, values, axis=self.axis, regular_grid=self.regular_grid,
            check_spacing=self.check_spacing, clip_out_of_bounds=self.clip_out_of_bounds,
            normalise=self.normalise, auto_reorder=self.auto_reorder
        )

    def __call__(self, x: FloatArray) -> VT:
        """
        Interpolate at time based on input times.

        Args:
            x: [...] time to evaluate at.

        Returns:
            values at given time with shape [...] + shape, i.e. the batch shape of x is always at the front of the
                resulting shape.
        """
        x_size = np.size(x)
        x_shape = np.shape(x)
        vec = x_shape != ()
        if vec:
            x = jax.lax.reshape(x, (x_size,))

        if self.normalise:
            x = (x - self.x_mean) / self.x_std
            (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(
                x=x,
                xp=self.x_norm,
                regular_grid=self.regular_grid,
                check_spacing=self.check_spacing,
                clip_out_of_bounds=self.clip_out_of_bounds
            )
            values = jax.tree.map(lambda x: apply_interp(x, i0, alpha0, i1, alpha1, axis=self.axis), self.values_norm)
            if np.size(x) > 1:
                values = jax.tree.map(lambda m, s, y: y * jnp.expand_dims(s, self.axis) + jnp.expand_dims(m, self.axis),
                                      self.values_mean, self.values_std, values)
            else:
                values = jax.tree.map(lambda m, s, y: y * s + m, self.values_mean, self.values_std, values)
        else:
            (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(
                x=x,
                xp=self.x,
                regular_grid=self.regular_grid,
                check_spacing=self.check_spacing,
                clip_out_of_bounds=self.clip_out_of_bounds
            )
            values = jax.tree.map(lambda x: apply_interp(x, i0, alpha0, i1, alpha1, axis=self.axis), self.values)
        # Ensure the batch shape is at the front, and the shape is correct
        if vec:
            if self.axis != 0:
                values = jax.tree.map(lambda x: jnp.moveaxis(x, self.axis, 0), values)
            values = jax.tree.map(lambda x: jax.lax.reshape(x, x_shape + np.shape(x)[1:]), values)
        return values


# Define how the object is flattened (converted to a list of leaves and a context tuple)
def interpolated_array_flatten(interpolated_array: InterpolatedArray):
    # Leaves are the arrays (x, values), and auxiliary data is the rest
    return (
        [interpolated_array.x, interpolated_array.values], (
            interpolated_array.axis, interpolated_array.regular_grid, interpolated_array.check_spacing,
            interpolated_array.clip_out_of_bounds, interpolated_array.normalise, interpolated_array.auto_reorder)
    )


# Define how the object is unflattened (reconstructed from leaves and context)
def interpolated_array_unflatten(aux_data, children):
    x, values = children
    axis, regular_grid, check_spacing, clip_out_of_bounds, normalise, auto_reorder = aux_data
    return InterpolatedArray(x=x, values=values, axis=axis, regular_grid=regular_grid, check_spacing=check_spacing,
                             clip_out_of_bounds=clip_out_of_bounds,
                             normalise=normalise, auto_reorder=auto_reorder)


# Register the custom pytree
jax.tree_util.register_pytree_node(
    InterpolatedArray,
    interpolated_array_flatten,
    interpolated_array_unflatten
)


def is_regular_grid(q: np.ndarray):
    """
    Check if the given quantity is a regular grid.

    Args:
        q: The quantity to check.

    Returns:
        bool: True if regular grid, False otherwise.
    """
    if len(q) < 2:
        return True
    return np.allclose(np.diff(q), q[1] - q[0])
