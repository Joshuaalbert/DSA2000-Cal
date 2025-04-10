import dataclasses
import pickle
import warnings
from typing import Tuple, TypeVar, Generic, List, Any, Union

import jax
import numpy as np
from jax import lax, numpy as jnp

index_dtype = jnp.int32
FloatArray = jax.Array | np.ndarray | float
IntArray = jax.Array | np.ndarray | int


def cast_to_index(idx):
    return idx.astype(index_dtype)


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

    def _canonicalise_axis(axis, shape_len):
        if isinstance(axis, int):
            # Convert single integer axis to positive if negative
            return axis if axis >= 0 else axis + shape_len
        else:
            # Convert each axis in a tuple to positive if negative
            return tuple(ax if ax >= 0 else ax + shape_len for ax in axis)

    axis = _canonicalise_axis(axis, len(np.shape(x)))

    def take(i):
        # return jnp.take(x, i, axis=axis)
        num_dims = len(np.shape(x))
        # [0] [1] [2 3 4], num_dims=5, axis=1
        slices = [slice(None)] * axis + [i] + [slice(None)] * (num_dims - axis - 1)
        return x[tuple(slices)]

    return left_broadcast_multiply(take(i0), alpha0.astype(jnp.result_type(x)), axis=axis) + left_broadcast_multiply(
        take(i1), alpha1.astype(jnp.result_type(x)), axis=axis)


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
    FloatArray, FloatArray, FloatArray, FloatArray]:
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
        return (jnp.zeros_like(x, dtype=index_dtype), jnp.ones_like(x),
                jnp.zeros_like(x, dtype=index_dtype), jnp.zeros_like(x))
    if clip_out_of_bounds:
        x = jax.lax.clamp(xp[0], x, xp[-1])

    # Find xp[i1-1] < x <= xp[i1]
    one = jnp.asarray(1, index_dtype)
    if regular_grid:
        # Use faster index determination
        dx = xp[1] - xp[0]
        _i1 = jnp.ceil((x - xp[0]) / dx).astype(index_dtype)
        i1 = cast_to_index(jnp.clip(_i1, one, len(xp) - 1))
        i0 = i1 - one
    else:
        i1 = cast_to_index(jnp.clip(jnp.searchsorted(xp, x, side='right'), one, len(xp) - 1))
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


def _canonicalize_axis(axis, shape_len):
    if isinstance(axis, int):
        # Convert single integer axis to positive if negative
        return axis if axis >= 0 else axis + shape_len
    else:
        # Convert each axis in a tuple to positive if negative
        return tuple(ax if ax >= 0 else ax + shape_len for ax in axis)


VT = TypeVar('VT')


def field_dunder(binary_op, self: 'InterpolatedArray',
                 other: Union[FloatArray, 'InterpolatedArray']) -> 'InterpolatedArray':
    """
    Perform binary operation on two fields.

    Args:
        binary_op: the binary operation to perform
        self: the first field
        other: the second field

    Returns:
        the resulting field
    """
    if isinstance(other, InterpolatedArray):
        values = jax.vmap(
            lambda x, y: binary_op(x, y), in_axes=(self.axis, other.axis), out_axes=self.axis
        )(self.values, other.values)
        return InterpolatedArray(
            self.x, values, axis=self.axis, regular_grid=self.regular_grid,
            check_spacing=self.check_spacing, clip_out_of_bounds=self.clip_out_of_bounds,
            normalise=self.normalise, auto_reorder=self.auto_reorder
        )
    else:
        values = jax.vmap(lambda x: binary_op(x, other), in_axes=self.axis, out_axes=self.axis)(self.values)
        return InterpolatedArray(
            self.x, values, axis=self.axis, regular_grid=self.regular_grid,
            check_spacing=self.check_spacing, clip_out_of_bounds=self.clip_out_of_bounds,
            normalise=self.normalise, auto_reorder=self.auto_reorder
        )


def test_interpolated_array():
    x = jnp.linspace(0., 1., 10)
    y = jnp.linspace(0., 1., 10)

    # Builds the efficient interpolator down axis 0 of `y` (default)
    interp = InterpolatedArray(x, y, axis=0, regular_grid=True, check_spacing=False, clip_out_of_bounds=True,
                               auto_reorder=True)

    x_test = jnp.linspace(-1.1, 1.1, 5)
    y_test = interp(x_test)
    print(y_test)


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
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return

        if self.auto_reorder and self.axis != -1:
            # Move axis to the last dimension
            self.values = jax.tree.map(lambda x: jnp.moveaxis(x, self.axis, -1), self.values)
            self.axis = -1

        if len(np.shape(self.x)) != 1:
            raise ValueError(f"x must be 1D, got {np.shape(self.x)}.")
        if np.shape(self.x)[0] == 0:
            raise ValueError("x must be non-empty")

        def _assert_shape(val):
            if np.shape(val)[self.axis] != np.shape(self.x)[0]:
                raise ValueError(
                    f"Input values must have size {np.shape(self.x)[0]} on `axis` dimension ({self.axis}), "
                    f"got value shape {np.shape(val)}."
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
            axis = _canonicalize_axis(axis, len(shape))

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

    def save(self, filename: str):
        """
        Serialise the model to file.

        Args:
            filename: the filename
        """
        if not filename.endswith('.pkl'):
            warnings.warn(f"Filename {filename} does not end with .pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        """
        Load the model from file.

        Args:
            filename: the filename

        Returns:
            the model
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        children, aux_data = self.flatten(self)
        children_np = jax.tree.map(np.asarray, children)
        serialised = (aux_data, children_np)
        return (self._deserialise, (serialised,))

    @classmethod
    def _deserialise(cls, serialised):
        # Create a new instance, bypassing __init__ and setting the actor directly
        (aux_data, children_np) = serialised
        children_jax = jax.tree.map(jnp.asarray, children_np)
        return cls.unflatten(aux_data, children_jax)

    @classmethod
    def register_pytree(cls):
        jax.tree_util.register_pytree_node(cls, cls.flatten, cls.unflatten)

    # an abstract classmethod
    @classmethod
    def flatten(cls, this: "InterpolatedArray") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        # Leaves are the arrays (x, values), and auxiliary data is the rest
        extra = dict()
        if this.normalise:
            extra['values_mean'] = this.values_mean
            extra['values_std'] = this.values_std
            extra['x_mean'] = this.x_mean
            extra['x_std'] = this.x_std
            extra['values_norm'] = this.values_norm
            extra['x_norm'] = this.x_norm
        return (
            [this.x, this.values, extra], (
                this.axis, this.regular_grid, this.check_spacing,
                this.clip_out_of_bounds, this.normalise, this.auto_reorder)
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "InterpolatedArray":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        x, values, extra = children
        axis, regular_grid, check_spacing, clip_out_of_bounds, normalise, auto_reorder = aux_data
        output = InterpolatedArray(x=x, values=values, axis=axis, regular_grid=regular_grid,
                                   check_spacing=check_spacing,
                                   clip_out_of_bounds=clip_out_of_bounds,
                                   normalise=normalise, auto_reorder=auto_reorder,
                                   skip_post_init=True)
        for key, value in extra.items():
            setattr(output, key, value)
        return output


InterpolatedArray.register_pytree()
