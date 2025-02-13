import dataclasses
from typing import NamedTuple, Any, Callable, TypeVar, Generic, Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_cal.common.ad_utils import tree_norm, tree_dot
from dsa2000_cal.common.array_types import FloatArray, IntArray
from dsa2000_cal.common.jvp_linear_op import JVPLinearOp

X = TypeVar('X', bound=Union[jax.Array, Any])
Y = TypeVar('Y', bound=Union[jax.Array, Any])

CT = TypeVar('CT', bound=Union[jax.Array, Any])
_CT = TypeVar('_CT', bound=Union[jax.Array, Any])


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


class MultiStepLevenbergMarquardtState(NamedTuple):
    iteration: IntArray  # iteration number
    x: X  # current solution, may be a pytree
    delta_x: X  # step, may be a pytree
    F: Y  # residual, may be a pytree
    F_norm: FloatArray  # norm of the residual
    mu: FloatArray  # damping factor
    cg_maxiter: IntArray  # maximum number of CG iterations
    error: FloatArray  # |J^T F(x_k)|
    delta_norm: FloatArray  # |dx_k|


class MultiStepLevenbergMarquardtDiagnostic(NamedTuple):
    iteration: IntArray  # iteration number
    exact_step: IntArray  # A single iteration is an exact step followed by inexact steps
    approx_step: IntArray  # An inexact step
    F_norm: FloatArray  # |F(x_k)|^2
    r: FloatArray  # r = (|F(x_k)|^2 - |F(x_{k+1})|^2) / (|F(x_k)|^2 - |F(x_{k}) + J_k dx_k|^2)
    delta_norm: FloatArray  # |dx_k|
    error: FloatArray  # |J^T F(x_k)|
    damping: FloatArray  # damping factor
    mu: FloatArray  # damping factor
    pred: FloatArray  # predicted reduction
    act: FloatArray  # actual reduction
    cg_maxiter: IntArray  # maximum number of CG iterations


@dataclasses.dataclass(eq=False)
class MultiStepLevenbergMarquardt(Generic[X, Y]):
    """
    Multi-step Levenberg-Marquardt algorithm.

    Finds a local minimum to the least squares problem defined by a residual function, F(x)=0.

    Implements the algorithm described in [1]. In addition, it applies CG method to solve the normal equations,
    efficient use of JVP and VJP to avoid computing the Jacobian matrix, adaptive step size, and mixed precision.

    References:
        [1] Fan, J., Huang, J. & Pan, J. An Adaptive Multi-step Levenberg–Marquardt Method.
            J Sci Comput 78, 531–548 (2019). https://doi.org/10.1007/s10915-018-0777-8
    """
    residual_fn: Callable[[X], Y]
    num_approx_steps: int = 2
    num_iterations: int = 2

    # Improvement threshold
    p_any_improvement: FloatArray = 0.01  # p0 > 0
    p_less_newton: FloatArray = 0.25  # p2 -- less than sufficient improvement
    p_more_newton: FloatArray = 0.9  # p3 -- more than sufficient improvement
    p_leave_newton: FloatArray = 1.05  # p4 -- leave Newton step

    # Damping alteration factors 0 < c_more_newton < 1 < c_less_newton
    c_more_newton: FloatArray = 0.16
    c_less_newton: FloatArray = 2.78
    mu_min: FloatArray = 1e-5
    approx_cg: bool = True
    min_cg_maxiter: IntArray = 10
    init_cg_maxiter: IntArray | None = 10

    gtol: FloatArray = 1e-8
    xtol: FloatArray = 1e-6

    verbose: bool = False

    def __post_init__(self):
        if self.num_approx_steps < 0:
            raise ValueError("num_approx_steps must be non-negative")
        if isinstance(self.mu_min, float) and self.mu_min <= 0:
            raise ValueError("mu_min must be positive")
        if all(map(lambda p: isinstance(p, float), (self.p_any_improvement,
                                                    self.p_more_newton,
                                                    self.p_less_newton))) and not (
                (0. <= self.p_any_improvement)
                and (self.p_any_improvement < self.p_less_newton)
                and (self.p_less_newton < self.p_more_newton)
                and (self.p_more_newton <= 1.)
        ):
            raise ValueError(
                "Improvement thresholds must satisfy 0 < p(any) < p(less) < p(more) < 1, "
                f"got {self.p_any_improvement}, {self.p_less_newton}, "
                f"{self.p_more_newton}"
            )

        if isinstance(self.c_more_newton, float) and isinstance(self.c_less_newton, float) and not (
                (0. < self.c_more_newton)
                and (self.c_more_newton < 1.)
                and (1. < self.c_less_newton)
        ):
            raise ValueError(
                "Damping alteration factors must satisfy 0 < c_more_newton < 1 < c_less_newton, "
                f"got {self.c_more_newton}, {self.c_less_newton}"
            )
        self.mu_min = jnp.asarray(self.mu_min, dtype=jnp.float32)
        self.p_any_improvement = jnp.asarray(self.p_any_improvement, dtype=jnp.float32)
        self.p_less_newton = jnp.asarray(self.p_less_newton, dtype=jnp.float32)
        self.p_more_newton = jnp.asarray(self.p_more_newton, dtype=jnp.float32)
        self.c_more_newton = jnp.asarray(self.c_more_newton, dtype=jnp.float32)
        self.c_less_newton = jnp.asarray(self.c_less_newton, dtype=jnp.float32)
        self.gtol = jnp.asarray(self.gtol, dtype=jnp.float32)
        self.xtol = jnp.asarray(self.xtol, dtype=jnp.float32)
        self.min_cg_maxiter = jnp.asarray(self.min_cg_maxiter, dtype=jnp.int32)
        self.init_cg_maxiter = jnp.asarray(self.init_cg_maxiter,
                                           dtype=jnp.int32) if self.init_cg_maxiter is not None else None

        self._residual_fn = self.wrap_residual_fn(self.residual_fn)

    @staticmethod
    def wrap_residual_fn(residual_fn: Callable[[X], Y]) -> Callable[[X], Y]:
        """
        Wrap the residual function to handle complex outputs by treating real and imag separately.
        """

        def wrapped_residual_fn(x: X) -> Y:
            output = residual_fn(x)

            def _make_real(x):
                if not isinstance(x, jax.Array):
                    raise RuntimeError("Only jax.Array is supported")
                if jnp.issubdtype(x.dtype, jnp.complexfloating):
                    return (x.real, x.imag)
                return x

            real_output = [jax.tree.map(_make_real, output)]
            scale = np.sqrt(sum(jax.tree.leaves(jax.tree.map(np.size, real_output))))
            # scale**2 = n
            normalised_real_output = jax.tree.map(lambda x: x / scale, real_output)
            return normalised_real_output

        return wrapped_residual_fn

    def update_initial_state(self, state: MultiStepLevenbergMarquardtState) -> MultiStepLevenbergMarquardtState:
        """
        Update another state into a valid initial state, using the current state as a starting point.

        Args:
            state: state to update

        Returns:
            updated state
        """
        # Note: If the obj_fn has significantly changed from the one used to produce `state`, using a new initial state
        # is advisable.
        init_state = self.create_initial_state(state.x)
        # We update state attributes that help the algorithm converge faster, i.e.
        # 1. help CG converge faster
        # 2. help choose the right damping factor
        return init_state._replace(
            iteration=state.iteration,
            mu=state.mu,
            delta_x=state.delta_x,
            cg_maxiter=state.cg_maxiter
        )

    def _select_initial_mu(self, residual_fn, obj0: FloatArray, x0: X, grad0: X):
        grad_norm = tree_norm(grad0)
        grad_unit = jax.tree.map(lambda x: x / grad_norm, grad0)

        def obj_fn(x: X):
            residuals = residual_fn(x)
            return tree_dot(residuals, residuals)

        def steepest_descent_point(alpha: FloatArray):
            return jax.tree.map(lambda x, y: x - alpha * y, x0, grad_unit)

        def search_cond(carry):
            alpha, obj = carry
            done = obj < obj0
            return jnp.logical_not(done)

        def search_iter(carry):
            alpha, obj = carry
            alpha = alpha * 0.5
            obj = obj_fn(steepest_descent_point(alpha))
            return alpha, obj

        # Use 1 = alpha_init / |grad|
        alpha_init = grad_norm
        alpha, obj = jax.lax.while_loop(
            search_cond, search_iter,
            (alpha_init, obj_fn(steepest_descent_point(alpha_init)))
        )
        # 1/(mu |grad|) = alpha / |grad|
        mu = jnp.reciprocal(alpha)
        return mu

    def create_initial_state(self, x0: X) -> MultiStepLevenbergMarquardtState:
        """
        Create the initial state for the algorithm.

        Returns:
            initial state
        """
        x = x0
        delta_x = jax.tree.map(jnp.zeros_like, x)  # zeros_like copies over sharding
        F = self._residual_fn(x)
        F_norm = tree_dot(F, F)

        # Extract the real and imaginary parts of the complex numbers of input to do Wirtinger calculus
        x_real_imag, merge_fn = convert_to_real(x)
        residual_fn = lambda x: self._residual_fn(merge_fn(x))

        J_bare = JVPLinearOp(fn=residual_fn)
        J = J_bare(x_real_imag)
        JTF = J.matvec(F, adjoint=True)
        error = tree_norm(JTF)

        mu = self._select_initial_mu(residual_fn, F_norm, x_real_imag, JTF)
        if self.init_cg_maxiter is None:
            cg_maxiter = jnp.asarray(sum(jax.tree.leaves(jax.tree.map(np.size, x))), dtype=jnp.int32)
        else:
            cg_maxiter = self.init_cg_maxiter

        state = MultiStepLevenbergMarquardtState(
            iteration=jnp.asarray(0),
            x=x,
            delta_x=delta_x,
            F=F,
            F_norm=F_norm,
            mu=mu,
            cg_maxiter=cg_maxiter,
            error=error,
            delta_norm=error
        )
        return state

    def solve(self, state: MultiStepLevenbergMarquardtState) -> Tuple[
        MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardtDiagnostic]:

        # Convert complex to real
        x_real_imag, merge_fn = convert_to_real(state.x)
        delta_x_real_imag, _ = convert_to_real(state.delta_x)

        state = state._replace(
            x=x_real_imag,
            delta_x=delta_x_real_imag
        )

        # For solving make the inputs purely real.
        residual_fn = lambda x: self._residual_fn(merge_fn(x))
        J_bare = JVPLinearOp(fn=residual_fn)

        def build_matvec(J: JVPLinearOp, damping: jax.Array):
            def matvec(v: X) -> X:
                JTJv = J.matvec(J.matvec(v), adjoint=True)
                return jax.tree.map(lambda x, y: x + damping * y, JTJv, v)

            return matvec

        output_dtypes = jax.tree.map(lambda x: x.dtype, state)

        def body(exact_step: IntArray, approx_step: IntArray, state: MultiStepLevenbergMarquardtState,
                 J: JVPLinearOp) -> Tuple[
            MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardtDiagnostic]:
            # d_k = -(J_k^T J_k + λ_k I)^(-1) J_k^T F(x_k)
            JTF = J.matvec(state.F, adjoint=True)  # [obj]^2/[x]
            # Units of [obj]/[x]^2
            damping = state.mu * state.error  # [obj]^2/[x]^2 -> mu is 1/[x]

            matvec = build_matvec(J, damping)
            delta_x, _ = jax.scipy.sparse.linalg.cg(
                A=matvec,
                b=jax.tree.map(jax.lax.neg, JTF),
                x0=state.delta_x,
                maxiter=state.cg_maxiter,
            )  # Info returned is not used

            # Determine predicted vs actual reduction gain ratio
            x_prop = jax.tree.map(lambda x, dx: x + dx, state.x, delta_x)
            F_prop = residual_fn(x_prop)
            F_pushfwd = jax.tree.map(lambda x, y: x + y, state.F, J.matvec(delta_x))
            # jax.debug.print("F_prop: {F_prop}, F_pushfwd: {F_pushfwd}", F_prop=F_prop, F_pushfwd=F_pushfwd)
            F_prop_norm = tree_dot(F_prop, F_prop)
            F_pushfwd_norm = tree_dot(F_pushfwd, F_pushfwd)
            predicted_reduction = state.F_norm - F_pushfwd_norm
            actual_reduction = state.F_norm - F_prop_norm

            r = jnp.where(
                jnp.logical_or(predicted_reduction == 0., actual_reduction <= 1e-15),
                jnp.zeros_like(state.F_norm),
                actual_reduction / predicted_reduction
            )

            # Apply our improvement thresholds

            any_improvement = r >= self.p_any_improvement
            more_newton = r > self.p_more_newton
            less_newton = r < self.p_less_newton

            # Determine if we accept the step
            # In principle, could use lax.cond.

            (F_norm, x, F) = jax.tree.map(
                lambda x1, x2: jnp.where(any_improvement, x1, x2),
                (F_prop_norm, x_prop, F_prop),
                (state.F_norm, state.x, state.F)
            )

            if self.approx_cg:
                # adjust the number of CG iterations if there is sufficient improvement
                x_size = sum(jax.tree.leaves(jax.tree.map(np.size, x)))
                cg_maxiter = jnp.where(
                    any_improvement,
                    jnp.where(
                        more_newton,
                        jnp.maximum(state.cg_maxiter * 0.5, self.min_cg_maxiter).astype(state.cg_maxiter),
                        state.cg_maxiter
                    ),
                    jnp.where(
                        state.cg_maxiter == self.min_cg_maxiter,
                        jnp.asarray(x_size, state.cg_maxiter.dtype),
                        jnp.minimum(state.cg_maxiter * 2, x_size).astype(state.cg_maxiter)
                    )
                )
            else:
                cg_maxiter = state.cg_maxiter

            # Update mu
            mu = jnp.where(
                less_newton,
                jnp.where(  # If at bottom apply a few extra "less newton jumps"
                    state.mu == self.mu_min,
                    self.mu_min * self.c_less_newton ** 5,
                    self.c_less_newton * state.mu,
                ),
                jnp.where(
                    more_newton,
                    jnp.maximum(self.c_more_newton * state.mu, self.mu_min),
                    state.mu
                )
            )
            mu = jnp.where(r > self.p_leave_newton, state.mu, mu)

            delta_norm = tree_norm(delta_x)
            error = tree_norm(JTF)

            if self.verbose:
                jax.debug.print(
                    "Iter: {iteration}, Exact Step: {exact_step} Approx Step: {approx_step}, "
                    "cg_maxiter: {cg_maxiter}, "
                    "mu: {mu}, damping: {damping}, r: {r}, pred: {predicted_reduction}, act: {actual_reduction}, "
                    "any_improvement: {any_improvement}, "
                    "more_newton: {more_newton}, less_newton: {less_newton}:\n"
                    "\tF_norm -> {F_norm}, delta_norm -> {delta_norm}, error -> {error}",
                    iteration=state.iteration,
                    exact_step=exact_step, approx_step=approx_step,
                    cg_maxiter=state.cg_maxiter,
                    r=r,
                    predicted_reduction=predicted_reduction, actual_reduction=actual_reduction,
                    any_improvement=any_improvement,
                    more_newton=more_newton, less_newton=less_newton, F_norm=F_norm, damping=damping,
                    mu=state.mu, delta_norm=delta_norm, error=error
                )
            diagnostic = MultiStepLevenbergMarquardtDiagnostic(
                iteration=state.iteration,
                exact_step=exact_step,
                approx_step=approx_step,
                F_norm=F_norm,
                r=r,
                delta_norm=delta_norm,
                error=error,
                damping=damping,
                mu=state.mu,
                pred=predicted_reduction,
                act=actual_reduction,
                cg_maxiter=state.cg_maxiter
            )

            state = MultiStepLevenbergMarquardtState(
                iteration=state.iteration + jnp.ones_like(state.iteration),
                x=x,
                delta_x=delta_x,
                F=F,
                F_norm=F_norm,
                mu=mu,
                cg_maxiter=cg_maxiter,
                error=error,
                delta_norm=delta_norm
            )
            # Cast to the original dtype for sanity
            state = jax.tree.map(lambda x, dtype: x.astype(dtype), state, output_dtypes)

            return state, diagnostic

        class CarryType(NamedTuple):
            exact_iteration: IntArray
            state: MultiStepLevenbergMarquardtState
            diagnostics: MultiStepLevenbergMarquardtDiagnostic

        def single_iteration(carry: CarryType):

            state = carry.state
            diagnostics = carry.diagnostics

            # Does one initial exact step using the current jacobian estimate, followed by inexact steps using the same
            # jacobian estimate (which is slightly cheaper).
            J = J_bare(state.x)
            for approx_step in range(self.num_approx_steps + 1):
                state, diagnostic = body(
                    carry.exact_iteration,
                    approx_step,
                    state,
                    J
                )
                update_index = carry.exact_iteration * (self.num_approx_steps + 1) + approx_step
                diagnostics = jax.tree.map(lambda x, y: x.at[update_index].set(y), diagnostics, diagnostic)
            exact_iteration = carry.exact_iteration + jnp.ones_like(carry.exact_iteration)
            return CarryType(exact_iteration, state, diagnostics)

        def term_cond(carry: CarryType):
            done = jnp.logical_or(
                carry.exact_iteration >= self.num_iterations,
                jnp.logical_or(
                    carry.state.error < self.gtol,
                    carry.state.delta_norm < self.xtol)
            )
            return jnp.logical_not(done)

        # Create diagnostic output structure
        def _fake_step(state):
            J = J_bare(state.x)
            _, diagnostic = body(jnp.zeros_like(state.iteration), jnp.zeros_like(state.iteration), state, J)
            return diagnostic

        diagnostic_aval = jax.eval_shape(_fake_step, state)
        max_iters = self.num_iterations * (self.num_approx_steps + 1)
        diagnostics = jax.tree.map(lambda x: jnp.zeros((max_iters,) + x.shape, dtype=x.dtype), diagnostic_aval)

        carry = jax.lax.while_loop(
            term_cond,
            single_iteration,
            CarryType(exact_iteration=jnp.zeros_like(state.iteration), state=state, diagnostics=diagnostics)
        )
        state = carry.state
        diagnostics = carry.diagnostics

        # Convert back to complex
        state = state._replace(
            x=merge_fn(state.x),
            delta_x=merge_fn(state.delta_x)
        )

        return state, diagnostics
