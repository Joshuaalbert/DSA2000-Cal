from typing import NamedTuple, TypeVar, Tuple, Callable, Union

import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_cal.solvers.cg import tree_vdot_real_part, tree_scalar_mul, tree_add, cg_solve, tree_sub, tree_neg
from dsa2000_common.common.array_types import FloatArray, IntArray, BoolArray
from dsa2000_common.common.jvp_linear_op import JVPLinearOp

DomainType = TypeVar("DomainType")
CoDomainType = TypeVar("CoDomainType")


class LMDiagnostic(NamedTuple):
    iteration: IntArray  # iteration number
    g_norm: FloatArray  # |J^T R(x_k)|
    mu: FloatArray  # step size
    damping: FloatArray  # g_norm / mu
    cg_iters: IntArray  # number of CG iterations used
    Q: FloatArray  # |R(x_k)|^2
    Q_prop: FloatArray  # |R(x_k + dx_k)|^2
    Q_push: FloatArray  # |R(x_k) + J_k dx_k|^2
    delta_Q_convex: FloatArray  # |R(x_k)|^2 - |R(x_k) + J_k dx_k|^2
    delta_Q_actual: FloatArray  # |R(x_k)|^2 - |R(x_k + dx_k)|^2
    gain_ratio: FloatArray  # delta_Q_actual / delta_Q_convex
    accepted: BoolArray  # delta_Q_convex > 0 and gain_ratio > p_accept
    in_trust_region: BoolArray  # delta_Q_convex > 0 and p_lower < gain_ratio < p_upper
    delta_x_norm: FloatArray  # |dx_k|


CT = TypeVar('CT', bound=Union[jax.Array, DomainType])
_CT = TypeVar('_CT', bound=Union[jax.Array, DomainType])


def convert_to_real(x: CT) -> Tuple[_CT, Callable[[_CT], CT]]:
    def should_split_complex(x: jax.Array):
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            return True
        return False

    def maybe_split(x: jax.Array):
        if should_split_complex(x):
            return (x.real, x.imag)
        return x

    x_leaves, treedef = jax.tree.flatten(x)
    x_real_imag_leaves = jax.tree.map(maybe_split, x_leaves)

    def merge(x: _CT) -> CT:
        def maybe_merge(x: jax.Array | Tuple[jax.Array, jax.Array]):
            if isinstance(x, tuple):
                return jax.lax.complex(x[0], x[1])
            return x

        x_leaves = list(map(maybe_merge, x))
        return jax.tree.unflatten(treedef, x_leaves)

    return x_real_imag_leaves, merge


def lm_solver(
        residual_fn: Callable[..., CoDomainType],
        x0: DomainType,
        args: tuple = (),
        maxiter: int = 50,
        maxiter_cg: int = 50,
        gtol: float = 1e-5,
        p_accept: float = 0.01,
        p_lower: float = 0.25,
        p_upper: float = 1.1,
        mu_init: float = 1.,
        mu_min: float = 1e-6,
        approx_grad: bool = False,
        verbose: bool = False
):
    x0, merge_fn = convert_to_real(x0)

    def _residual_fn(x):
        res = residual_fn(merge_fn(x), *args)

        def _make_real(x):
            if not isinstance(x, jax.Array):
                raise RuntimeError(f"Only jax.Array is supported, got {type(x)}")
            if jnp.iscomplexobj(x):
                return [(x.real, x.imag)]  # List to make it arity preserving
            return x

        real_res = jax.tree.map(_make_real, res)
        scale = np.sqrt(sum(jax.tree.leaves(jax.tree.map(np.size, real_res))))
        # scale**2 = num data points
        return jax.tree.map(lambda x: x / scale, real_res)

    # normalise the residual fn
    J_bare = JVPLinearOp(fn=_residual_fn)

    class LMState(NamedTuple):
        x: DomainType  # current parameter estimate
        R: CoDomainType  # current residual R(x)
        Q: FloatArray  # squared residual norm: Q = ||R(x)||²
        g: DomainType  # gradient g = -J^T R(x)
        g_norm: FloatArray  # norm of the gradient
        mu: FloatArray  # current damping parameter
        delta_x_prev: DomainType  # previous step δx⁻¹
        delta_x_prev2: DomainType  # second previous step δx⁻²
        iter: IntArray  # iteration counter

    def create_initial_state(x0: DomainType) -> LMState:
        R = _residual_fn(x0)
        Q = tree_vdot_real_part(R, R)

        # Compute gradient via vector-Jacobian product:
        J = J_bare(x0)
        g = tree_neg(J.matvec(R, adjoint=True))
        g_norm = jnp.sqrt(tree_vdot_real_part(g, g))
        g_unit = tree_scalar_mul(jnp.reciprocal(g_norm + 1e-12), g)

        # Simple line search to adjust μ: reduce μ until the step improves Q.
        def line_search_cond(mu):
            step = tree_scalar_mul(mu, g_unit)
            R_new = _residual_fn(tree_add(x0, step))
            Q_new = tree_vdot_real_part(R_new, R_new)
            return (Q_new >= Q) & (mu > mu_min)

        def line_search_body(mu):
            mu = 0.5 * mu
            return mu

        mu = jax.lax.while_loop(
            line_search_cond, line_search_body, mu_init
        )
        delta_x_prev = delta_x_prev2 = jax.tree.map(jnp.zeros_like, x0)
        return LMState(
            x=x0, R=R, Q=Q, g=g, g_norm=g_norm, mu=mu,
            delta_x_prev=delta_x_prev,
            delta_x_prev2=delta_x_prev2, iter=0
        )

    def cond_fn(carry: Tuple[LMState, LMDiagnostic]):
        state, _ = carry
        return (state.g_norm > gtol) & (state.iter < maxiter)

    def step_fn(state: LMState):
        # Forecast initial CG guess: δx⁰ = 2δx⁻¹ - δx⁻²
        delta_x0 = tree_sub(tree_add(state.delta_x_prev, state.delta_x_prev), state.delta_x_prev2)
        # Define the linear operator A_op for the inner CG:
        # A_op(p) = J^T (J p) + (g_norm/mu) * p.
        J = J_bare(state.x)
        damping = state.g_norm / state.mu

        def A_op(p):
            JTJv = J.matvec(J.matvec(p), adjoint=True)
            return tree_add(JTJv, tree_scalar_mul(damping, p))

        # Solve the linear system A_op(δx) = g using our CG solver.
        delta_x, cg_diag = cg_solve(
            A=A_op,
            b=state.g,
            x0=delta_x0,
            maxiter=maxiter_cg,
            tol=1e-5,
            atol=0.0
        )

        delta_x_norm = jnp.sqrt(tree_vdot_real_part(delta_x, delta_x))

        # Update parameter estimate
        x_prop = tree_add(state.x, delta_x)
        R_prop = _residual_fn(x_prop)
        Q_prop = tree_vdot_real_part(R_prop, R_prop)

        # Convex decrease using the linear model
        R_push = tree_add(state.R, J.matvec(delta_x))
        Q_push = tree_vdot_real_part(R_push, R_push)
        delta_Q_convex = state.Q - Q_push
        delta_Q_actual = state.Q - Q_prop
        gain_ratio = delta_Q_actual / delta_Q_convex

        # Adjust the damping parameter μ if in the trust region
        in_trust_region = (delta_Q_convex > 0) & (delta_Q_actual > p_lower * delta_Q_convex) & (
                delta_Q_actual < p_upper * delta_Q_convex)
        new_mu = jax.lax.select(in_trust_region, 2 * state.mu, 0.5 * state.mu)
        # Accept step if the model predicts a convex decrease and ratio is high.
        accepted = (delta_Q_convex > 0) & (delta_Q_actual > p_accept * delta_Q_convex)

        diag = LMDiagnostic(
            iteration=state.iter,
            g_norm=state.g_norm,
            mu=state.mu,
            damping=damping,
            cg_iters=cg_diag.iterations,
            Q=state.Q,
            Q_prop=Q_prop,
            Q_push=Q_push,
            delta_Q_convex=delta_Q_convex,
            delta_Q_actual=delta_Q_actual,
            delta_x_norm=delta_x_norm,
            gain_ratio=gain_ratio,
            accepted=accepted,
            in_trust_region=in_trust_region
        )
        if verbose:
            diag_dict = diag._asdict()
            s = [f"{k}: {{{k}}}" for k, v in diag_dict.items()]
            # Join in lines of 3 kwargs
            s = [" ".join(s[i:i + 4]) for i in range(0, len(s), 4)]
            s = "\n".join(s)
            s += "\n----"
            jax.debug.print(s, **diag_dict)

        (
            x_new, Q_new, R_new, delta_x_prev_new, delta_x_prev2
        ) = jax.tree.map(
            lambda x, y: jax.lax.select(accepted, x, y),
            (x_prop, Q_prop, R_prop, delta_x, state.delta_x_prev),
            (state.x, state.Q, state.R, state.delta_x_prev, state.delta_x_prev2)
        )
        # Recompute gradient at the (possibly updated) x.
        if not approx_grad:
            J = J_bare(x_new)

        g_new = tree_neg(J.matvec(R_new, adjoint=True))
        g_norm_new = jnp.sqrt(tree_vdot_real_part(g_new, g_new))

        state = LMState(
            x=x_new, R=R_new, Q=Q_new, g=g_new, g_norm=g_norm_new, mu=new_mu,
            delta_x_prev=delta_x, delta_x_prev2=state.delta_x_prev, iter=state.iter + 1
        )

        return state, diag

    def body_fn(carry: Tuple[LMState, LMDiagnostic]) -> Tuple[LMState, LMDiagnostic]:
        state, diagnostic = carry
        iteration = state.iter
        new_state, new_diag_element = step_fn(state)
        new_diagnostic = jax.tree.map(lambda x, y: x.at[iteration].set(y), diagnostic, new_diag_element)
        return new_state, new_diagnostic

    init_state = create_initial_state(x0)

    # Create diagnostic output structure
    diagnostic_aval = jax.eval_shape(lambda state: step_fn(state)[1], init_state)
    init_diagnostics = jax.tree.map(lambda x: jnp.zeros((maxiter,) + x.shape, dtype=x.dtype), diagnostic_aval)

    final_state, final_diagnostics = jax.lax.while_loop(cond_fn, body_fn, (init_state, init_diagnostics))

    return final_state.x, final_diagnostics


def test_lm_solver():
    def residual_fn(x):
        return jnp.exp(x) ** 2

    x0 = jnp.array([1.0 + 1j, 1.0 + 1j])
    x, diag = lm_solver(residual_fn, x0, verbose=True, approx_grad=False)
