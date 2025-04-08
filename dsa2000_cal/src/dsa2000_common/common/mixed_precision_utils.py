import dataclasses
import warnings
from typing import TypeVar, Tuple, List, Any

import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.pytree import Pytree

if not jax.config.read('jax_enable_x64'):
    warnings.warn("JAX x64 is not enabled. Setting it now, but check for errors.")
    jax.config.update('jax_enable_x64', True)

# Create a float scalar to lock in dtype choices.
if jnp.array(1., jnp.float64).dtype != jnp.float64:
    raise RuntimeError("Failed to set float64 as default dtype.")

from dsa2000_common.common.alert_utils import get_grandparent_info

float_type = jnp.result_type(float)
int_type = jnp.result_type(int)
complex_type = jnp.result_type(complex)

T = TypeVar("T")


def _cast_floating_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, float):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.floating):
                warnings.warn(f"Expected float type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_complex_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, complex):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.complexfloating):
                warnings.warn(f"Expected complex type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_integer_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, int):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.integer):
                warnings.warn(f"Expected integer type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_bool_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, bool):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.bool_):
                warnings.warn(f"Expected bool type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


X = TypeVar("X")


@dataclasses.dataclass(frozen=True)
class Policy:
    """Encapsulates casting for inputs, outputs and parameters."""
    vis_dtype: jnp.dtype = np.complex64
    weight_dtype: jnp.dtype = np.float16
    flag_dtype: jnp.dtype = np.bool_
    image_dtype: jnp.dtype = np.float32
    gain_dtype: jnp.dtype = np.complex64
    index_dtype: jnp.dtype = np.int64
    freq_dtype: jnp.dtype = np.float64
    time_dtype: jnp.dtype = np.float64
    length_dtype: jnp.dtype = np.float64
    angle_dtype: jnp.dtype = np.float64

    def cast_to_vis(self, x: X, quiet: bool = False) -> X:
        """Converts visibility values to the visibility dtype."""
        return _cast_complex_to(x, self.vis_dtype, quiet=quiet)

    def cast_to_weight(self, x: X, quiet: bool = False) -> X:
        """Converts weight values to the weight dtype."""
        return _cast_floating_to(x, self.weight_dtype, quiet=quiet)

    def cast_to_flag(self, x: X, quiet: bool = False) -> X:
        """Converts flag values to the flag dtype."""
        return _cast_bool_to(x, self.flag_dtype, quiet=quiet)

    def cast_to_image(self, x: X, quiet: bool = False) -> X:
        """Converts image values to the image dtype."""
        return _cast_floating_to(x, self.image_dtype, quiet=quiet)

    def cast_to_gain(self, x: X, quiet: bool = False) -> X:
        """Converts gain values to the gain dtype."""
        return _cast_complex_to(x, self.gain_dtype, quiet=quiet)

    def cast_to_index(self, x: X, quiet: bool = False) -> X:
        """Converts lookup index values to the index dtype."""
        return _cast_integer_to(x, self.index_dtype, quiet=quiet)

    def cast_to_freq(self, x: X, quiet: bool = False) -> X:
        """Converts frequency values to the frequency dtype."""
        return _cast_floating_to(x, self.freq_dtype, quiet=quiet)

    def cast_to_time(self, x: X, quiet: bool = False) -> X:
        """Converts time values to the time dtype."""
        return _cast_floating_to(x, self.time_dtype, quiet=quiet)

    def cast_to_length(self, x: X, quiet: bool = False) -> X:
        """Converts length values to the position dtype."""
        return _cast_floating_to(x, self.length_dtype, quiet=quiet)

    def cast_to_angle(self, x: X, quiet: bool = False) -> X:
        """Converts lmn values to the lmn dtype."""
        return _cast_floating_to(x, self.angle_dtype, quiet=quiet)


mp_policy = Policy()


@dataclasses.dataclass(eq=False)
class ComplexMP(Pytree):
    real: FloatArray
    imag: FloatArray
    dtype: jnp.dtype = jnp.float16
    skip_post_init: bool = False

    @classmethod
    def flatten(cls, this) -> Tuple[List[Any], Tuple[Any, ...]]:
        return (
            [
                this.real, this.imag,
            ],
            (
                this.dtype,
            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]):
        real, imag = children
        dtype, = aux_data
        return ComplexMP(real=real, imag=imag, dtype=dtype, skip_post_init=True)

    def __post_init__(self):
        if self.skip_post_init:
            return
        self.real = self.real.real.astype(self.dtype)
        self.imag = self.imag.real.astype(self.dtype)

    def _linear_binary_dunder(self, op, this: "ComplexMP", other):
        if isinstance(other, ComplexMP) or jnp.iscomplexobj(other):
            return ComplexMP(op(this.real, other.real), op(this.imag, other.imag),
                             jnp.promote_types(this.dtype, other.real.dtype))
        elif jnp.issubdtype(jnp.result_type(other), jnp.floating):
            return ComplexMP(op(this.real, other.real), this.imag,
                             jnp.promote_types(this.dtype, other.real.dtype))
        else:
            raise ValueError(f"unsupported operand type: {type(other)}")

    def __neg__(self):
        return ComplexMP(-self.real, -self.imag, self.dtype)

    def __add__(self, other):
        return self._linear_binary_dunder(jnp.add, self, other)

    def __sub__(self, other):
        return self._linear_binary_dunder(jnp.subtract, self, other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, ComplexMP) or jnp.iscomplexobj(other):
            a = other.real
            b = other.imag
            return ComplexMP(self.real * a - self.imag * b, self.real * b + self.imag * a,
                             jnp.promote_types(self.dtype, other.real.dtype))
        elif jnp.issubdtype(jnp.result_type(other), jnp.floating):
            return ComplexMP(self.real * other, self.imag * other,
                             jnp.promote_types(self.dtype, jnp.result_type(other)))
        else:
            raise ValueError(f"unsupported operand type: {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        if isinstance(other, ComplexMP) or jnp.iscomplexobj(other):
            a = other.real
            b = other.imag
            denom = a * a + b * b
            return ComplexMP((self.real * a + self.imag * b) / denom,
                             (self.imag * a - self.real * b) / denom,
                             jnp.promote_types(self.dtype, other.real.dtype))
        elif jnp.issubdtype(jnp.result_type(other), jnp.floating):
            a = other.real
            return ComplexMP(self.real / a, self.imag / a,
                             jnp.promote_types(self.dtype, other.real.dtype))
        else:
            raise ValueError(f"unsupported operand type: {type(other)}")

    def __rtruediv__(self, other):
        if isinstance(other, ComplexMP) or jnp.iscomplexobj(other):
            a = self.real
            b = self.imag
            x = other.real
            y = other.imag
            denom = a * a + b * b
            return ComplexMP((x * a + y * b) / denom, (y * a - x * b) / denom,
                             jnp.promote_types(self.dtype, other.real.dtype))
        elif jnp.issubdtype(jnp.result_type(other), jnp.floating):
            a = self.real
            b = self.imag
            x = other.real
            denom = a * a + b * b
            return ComplexMP((x * a) / denom, (- x * b) / denom, jnp.promote_types(self.dtype, other.real.dtype))
        else:
            raise ValueError(f"unsupported operand type: {type(other)}")

    def real(self):
        return self.real

    def imag(self):
        return self.imag

    def complex(self):
        dtype = jnp.promote_types(self.dtype, jnp.float32)
        return jax.lax.complex(self.real.astype(dtype), self.imag.astype(dtype))

    def __getitem__(self, item):
        return ComplexMP(self.real[item], self.imag[item], self.dtype)

    def conj(self):
        return ComplexMP(self.real, -self.imag, self.dtype)

    def swapaxes(self, a, b):
        return ComplexMP(jnp.swapaxes(self.real, a, b), jnp.swapaxes(self.imag, a, b), self.dtype)

    def take(self, indices,
             axis: int | None = None,
             out: None = None,
             mode: str | None = None,
             unique_indices: bool = False,
             indices_are_sorted: bool = False,
             fill_value: None = None):
        return ComplexMP(
            jnp.take(self.real, indices, axis, out, mode, unique_indices, indices_are_sorted, fill_value),
            jnp.take(self.imag, indices, axis, out, mode, unique_indices, indices_are_sorted, fill_value),
            self.dtype
        )

    def reshape(self, shape):
        return ComplexMP(jax.lax.reshape(self.real, shape), jax.lax.reshape(self.imag, shape), self.dtype)

    @property
    def shape(self):
        return jnp.broadcast_shapes(np.shape(self.real), np.shape(self.imag))

    def astype(self, dtype):
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError(f"Unsupported dtype: {dtype}.")
        return ComplexMP(self.real, self.imag, dtype)

    def abs(self):
        return jnp.sqrt(jnp.square(self.real) + jnp.square(self.imag))


ComplexMP.register_pytree()


def kron_product_2x2_complex_mp(M0: ComplexMP, M1: ComplexMP, M2: ComplexMP) -> ComplexMP:
    # Matrix([[a0*(a1*a2 + b1*c2) + b0*(a2*c1 + c2*d1), a0*(a1*b2 + b1*d2) + b0*(b2*c1 + d1*d2)], [c0*(a1*a2 + b1*c2) + d0*(a2*c1 + c2*d1), c0*(a1*b2 + b1*d2) + d0*(b2*c1 + d1*d2)]])
    # 36
    # ([(x0, a1*a2 + b1*c2), (x1, a2*c1 + c2*d1), (x2, a1*b2 + b1*d2), (x3, b2*c1 + d1*d2)], [Matrix([
    # [a0*x0 + b0*x1, a0*x2 + b0*x3],
    # [c0*x0 + d0*x1, c0*x2 + d0*x3]])])
    a0, b0, c0, d0 = M0[..., 0, 0], M0[..., 0, 1], M0[..., 1, 0], M0[..., 1, 1]
    a1, b1, c1, d1 = M1[..., 0, 0], M1[..., 0, 1], M1[..., 1, 0], M1[..., 1, 1]
    a2, b2, c2, d2 = M2[..., 0, 0], M2[..., 0, 1], M2[..., 1, 0], M2[..., 1, 1]
    x0 = a1 * a2 + b1 * c2
    x1 = a2 * c1 + c2 * d1
    x2 = a1 * b2 + b1 * d2
    x3 = b2 * c1 + d1 * d2

    # flat = jnp.stack([a0 * x0 + b0 * x1, a0 * x2 + b0 * x3, c0 * x0 + d0 * x1, c0 * x2 + d0 * x3], axis=-1)
    # return lax.reshape(flat, np.shape(flat)[:-1] + (2, 2))
    A, B, C, D = a0 * x0 + b0 * x1, a0 * x2 + b0 * x3, c0 * x0 + d0 * x1, c0 * x2 + d0 * x3
    flat_real = jnp.stack([A.real, B.real, C.real, D.real], axis=-1)
    flat_imag = jnp.stack([A.imag, B.imag, C.imag, D.imag], axis=-1)
    flat_real = jax.lax.reshape(flat_real, np.shape(flat_real)[:-1] + (2, 2))
    flat_imag = jax.lax.reshape(flat_imag, np.shape(flat_imag)[:-1] + (2, 2))
    dtype = jnp.promote_types(jnp.promote_types(M0.dtype, M1.dtype), M2.dtype)
    return ComplexMP(real=flat_real, imag=flat_imag, dtype=dtype)
