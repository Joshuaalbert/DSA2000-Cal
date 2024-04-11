import dataclasses
from typing import Callable, Any

import jax
import numpy as np


def isinstance_namedtuple(obj) -> bool:
    """
    Check if object is a namedtuple.

    Args:
        obj: object

    Returns:
        bool
    """
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )

@dataclasses.dataclass(eq=False)
class JVPLinearOp:
    """
    Represents J_ij = d/d x_j f_i(x), where x is the primal value.

    This is a linear operator that represents the Jacobian of a function.
    """
    fn: Callable  # A function R^n -> R^m
    primals: Any | None = None  # The primal value, i.e. where jacobian is evaluated
    more_outputs_than_inputs: bool = False  # If True, the operator is tall, i.e. m > n
    adjoint: bool = False  # If True, the operator is transposed

    def __post_init__(self):
        if not callable(self.fn):
            raise ValueError('`fn` must be a callable.')

        if isinstance_namedtuple(self.primals) or (not isinstance(self.primals, (tuple, list))):
            self.primals = (self.primals,)

    def __call__(self, *primals):
        return JVPLinearOp(fn=self.fn, primals=primals, more_outputs_than_inputs=self.more_outputs_than_inputs,
                           adjoint=self.adjoint)

    def __neg__(self):
        return JVPLinearOp(fn=lambda *args, **kwargs: -self.fn(*args, **kwargs), primals=self.primals,
                           more_outputs_than_inputs=self.more_outputs_than_inputs, adjoint=self.adjoint)

    def __matmul__(self, other):
        if not isinstance(other, (jax.Array, np.ndarray)):
            raise ValueError(
                'Dunder methods currently only defined for operation on arrays. '
                'Use .matmul(...) for general tangents.'
            )
        if len(np.shape(other)) == 1:
            return self.matvec(other, adjoint=self.adjoint)
        return self.matmul(other, adjoint=self.adjoint, left_multiply=True)

    def __rmatmul__(self, other):
        if not isinstance(other, (jax.Array, np.ndarray)):
            raise ValueError(
                'Dunder methods currently only defined for operation on arrays. '
                'Use .matmul(..., left_multiply=False) for general tangents.'
            )
        if len(np.shape(other)) == 1:
            return self.matvec(other, adjoint=not self.adjoint)
        return self.matmul(other, adjoint=not self.adjoint, left_multiply=False)

    @property
    def T(self) -> 'JVPLinearOp':
        return JVPLinearOp(fn=self.fn, primals=self.primals, more_outputs_than_inputs=self.more_outputs_than_inputs,
                           adjoint=not self.adjoint)

    def matmul(self, tangents: Any, adjoint: bool = False, left_multiply: bool = True):
        """
        Implements matrix multiplication from matvec using vmap.

        Args:
            tangents: pytree of the same structure as the primals, but with appropriate more columns for adjoint=False,
                or more rows for adjoint=True.
            adjoint: if True, compute J.T @ v, else compute J @ v
            left_multiply: if True, compute M @ J, else compute J @ M

        Returns:
            pytree of matching either f-space (output) or x-space (primals)
        """
        if left_multiply:
            # J.T @ M or J @ M
            in_axes = -1
            out_axes = -1
        else:
            # M @ J.T or M @ J
            in_axes = 0
            out_axes = 0
        if adjoint:
            return jax.vmap(lambda _tangent: self.matvec(_tangent, adjoint), in_axes=in_axes, out_axes=out_axes)(
                tangents)
        return jax.vmap(lambda _tangent: self.matvec(_tangent, adjoint), in_axes=in_axes, out_axes=out_axes)(tangents)

    def matvec(self, tangents: Any, adjoint: bool = False):
        """
        Compute J @ v = sum_j(J_ij * v_j) using a JVP, if adjoint is False.
        Compute J.T @ v = sum_i(v_i * J_ij) using a VJP, if adjoint is True.

        Args:
            tangents: pytree of the same structure as the primals.
            adjoint: if True, compute v @ J, else compute J @ v

        Returns:
            pytree of matching either f-space (output) or x-space (primals)
        """
        if self.primals is None:
            raise ValueError("The primal value must be set to compute the Jacobian.")

        if isinstance_namedtuple(tangents) or (not isinstance(tangents, (tuple, list))):
            tangents = (tangents,)

        if adjoint:
            # v @ J
            primals_out, f_vjp = jax.vjp(self.fn, *self.primals)
            output = f_vjp(*tangents)
            if len(output) == 1:
                return output[0]
            return output

        primal_out, tangent_out = jax.jvp(self.fn, self.primals, tangents)
        return tangent_out

    def to_dense(self) -> jax.Array:
        """
        Compute the dense Jacobian at a point.

        Args:
            x: [n] array

        Returns:
            [m, n] array
        """
        if self.primals is None:
            raise ValueError("The primal value must be set to compute the Jacobian.")

        if self.more_outputs_than_inputs:
            return jax.jacfwd(self.fn)(*self.primals)
        return jax.jacrev(self.fn)(*self.primals)
