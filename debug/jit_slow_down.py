import dataclasses
import time
import warnings
from typing import Callable, Any
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from dsa2000_cal.common.jax_utils import block_until_ready


class ProblemState(NamedTuple):
    x: Any  # current solution, may be a pytree
    delta_x: Any  # step, may be a pytree
    damping: jax.Array  # damping factor
    F: Any  # residual, may be a pytree


@dataclasses.dataclass(eq=False)
class Problem:
    residual_fn: Callable
    num_steps: int = 0
    more_outputs_than_inputs: bool = False

    def create_initial_state(self, x0) -> ProblemState:
        x = x0
        delta_x = jax.tree.map(jnp.zeros_like, x)
        F = self.residual_fn(x)
        F_norm = pytree_norm(F)
        F_norm_delta = F_norm
        damping = F_norm_delta

        state = ProblemState(
            x=x,
            delta_x=delta_x,
            damping=damping,
            F=F
        )
        return state

    def solve(self, state: ProblemState) -> ProblemState:
        def build_matvec(J: JVPLinearOp, damping: jax.Array):
            def matvec(v):
                JTJv = J.matvec(J.matvec(v), adjoint=True)
                return jax.tree.map(lambda x, y: x + damping * y, JTJv, v)

            return matvec

        J_bare = JVPLinearOp(fn=self.residual_fn, more_outputs_than_inputs=self.more_outputs_than_inputs)

        def body(state: ProblemState, unused) -> Tuple[ProblemState, None]:
            J = J_bare(state.x)
            JTF = J.matvec(state.F, adjoint=True)
            matvec = build_matvec(J, state.damping)
            delta_x, _ = jax.scipy.sparse.linalg.cg(matvec, jax.tree.map(jax.lax.neg, JTF), x0=state.delta_x)

            state = ProblemState(
                x=state.x,
                delta_x=delta_x,
                damping=state.damping,
                F=state.F
            )

            return state, None

        state, _ = jax.lax.scan(body, state, jnp.arange(self.num_steps))

        return state


def pytree_norm(pytree: Any) -> jax.Array:
    square_sum = jax.tree.map(lambda x: jnp.sum(jnp.square(jnp.abs(x))), pytree)
    leaves = jax.tree.leaves(square_sum)
    total_square_sum = sum(leaves[1:], leaves[0])
    return total_square_sum


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
    promote_dtypes: bool = True  # If True, promote dtypes to match primal during JVP, and cotangent to match primal_out during VJP

    def __post_init__(self):
        if not callable(self.fn):
            raise ValueError('`fn` must be a callable.')

        if self.primals is not None:
            if isinstance_namedtuple(self.primals) or (not isinstance(self.primals, tuple)):
                self.primals = (self.primals,)

    def __call__(self, *primals: Any) -> 'JVPLinearOp':
        return JVPLinearOp(fn=self.fn, primals=primals, more_outputs_than_inputs=self.more_outputs_than_inputs,
                           adjoint=self.adjoint, promote_dtypes=self.promote_dtypes)

    def matvec(self, *tangents: Any, adjoint: bool = False):
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

        if adjoint:
            co_tangents = tangents

            def _get_results_type(primal_out: jax.Array):
                return primal_out.dtype

            def _adjoint_promote_dtypes(co_tangent: jax.Array, dtype: jnp.dtype):
                if co_tangent.dtype != dtype:
                    warnings.warn(f"Promoting co-tangent dtype from {co_tangent.dtype} to {dtype}.")
                return co_tangent.astype(dtype)

            # v @ J
            primals_out, f_vjp = jax.vjp(self.fn, *self.primals)

            if isinstance_namedtuple(primals_out) or (not isinstance(primals_out, tuple)):
                # JAX squeezed structure to a single element, as the function only returns one output
                co_tangents = co_tangents[0]

            if self.promote_dtypes:
                result_type = jax.tree.map(_get_results_type, primals_out)
                co_tangents = jax.tree.map(_adjoint_promote_dtypes, co_tangents, result_type)

            del primals_out
            output = f_vjp(co_tangents)
            if len(output) == 1:
                return output[0]
            return output

        def _promote_dtype(primal: jax.Array, dtype: jnp.dtype):
            if primal.dtype != dtype:
                warnings.warn(f"Promoting primal dtype from {primal.dtype} to {dtype}.")
            return primal.astype(dtype)

        def _get_result_type(primal: jax.Array, tangent: jax.Array):
            return jnp.result_type(primal, tangent)

        primals = self.primals
        if self.promote_dtypes:
            result_types = jax.tree.map(_get_result_type, primals, tangents)
            primals = jax.tree.map(_promote_dtype, primals, result_types)
            tangents = jax.tree.map(_promote_dtype, tangents, result_types)
        primal_out, tangent_out = jax.jvp(self.fn, primals, tangents)
        return tangent_out


def main():
    n = 2048
    m = int(n * (n - 1) / 2)
    s = 3



    def forward(params):
        alpha = jnp.ones((m, s))
        idx1 = jax.random.randint(jax.random.PRNGKey(0), (m,), 0, n)
        idx2 = jax.random.randint(jax.random.PRNGKey(1), (m,), 0, n)
        g1 = params[idx1, :]
        g2 = params[idx2, :]
        return jnp.sum(g1 * alpha * g2.conj(), axis=1)

    def residuals(params):
        return forward(params)

    x = jnp.ones((n, s))

    lm = Problem(residual_fn=residuals,
                 num_steps=1)

    state = lm.create_initial_state(x)

    solve = lm.solve

    t0 = time.time()
    block_until_ready(solve(state))
    run_time = time.time() - t0
    print(f"Run time (no JIT): {run_time}")

    # AOT JIT
    solve = jax.jit(lm.solve).lower(state).compile()

    t0 = time.time()
    block_until_ready(solve(state))
    run_time = time.time() - t0
    print(f"Run time (JIT): {run_time}")


if __name__ == '__main__':
    main()
