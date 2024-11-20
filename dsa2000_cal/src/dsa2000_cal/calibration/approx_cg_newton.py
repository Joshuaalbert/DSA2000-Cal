import dataclasses
from typing import NamedTuple, Any, Callable, TypeVar, Generic, Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_cal.common.ad_utils import build_hvp, tree_dot, tree_norm
from dsa2000_cal.common.array_types import FloatArray, IntArray

X = TypeVar('X', bound=Union[jax.Array, Any])

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


class ApproxCGNewtonState(NamedTuple):
    iteration: IntArray  # iteration number
    x: X  # current solution, may be a pytree
    delta_x: X  # step, may be a pytree
    obj: FloatArray  # objective value
    grad_obj: X  # gradient of the objective
    mu: FloatArray  # damping factor, units of 1/[x]
    cg_maxiter: IntArray  # maximum number of CG iterations
    error: FloatArray  # |grad|
    delta_norm: FloatArray  # |dx_k|


class ApproxCGNewtonDiagnostic(NamedTuple):
    iteration: IntArray  # iteration number
    exact_step: IntArray  # A single iteration is an exact step followed by inexact steps
    approx_step: IntArray  # An inexact step
    obj: FloatArray  # objective value
    r: FloatArray  # r = (obj(x_k) - obj(x_{k+1})) / (obj(x_k) - (obj(x_{k}) + grad_k dx_k + 1/2 dx_k^T H_k dx_k))
    pred: FloatArray  # predicted reduction
    act: FloatArray  # actual reduction
    delta_norm: FloatArray  # |dx_k|
    error: FloatArray  # |grad|
    damping: FloatArray  # pre-damping factor
    mu: FloatArray  # pre-damping multiplier factor
    cg_maxiter: IntArray  # maximum number of CG iterations


@dataclasses.dataclass(eq=False)
class ApproxCGNewton(Generic[X]):
    """
    Multi-step CG Newton algorithm.

    Finds a local minimum to the least squares problem defined by a residual function, F(x)=0.

    Implements the algorithm described in [1]. In addition, it applies CG method to solve the normal equations,
    efficient use of JVP and VJP to avoid computing the Jacobian matrix, adaptive step size, and mixed precision.

    References:
        [1] Fan, J., Huang, J. & Pan, J. An Adaptive Multi-step Levenberg–Marquardt Method.
            J Sci Comput 78, 531–548 (2019). https://doi.org/10.1007/s10915-018-0777-8
    """
    obj_fn: Callable[[X], FloatArray]
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
    # mu_min > 0
    mu_min: FloatArray = 1e-5  # 1e-3
    approx_cg: bool = True
    min_cg_maxiter: IntArray = 10
    init_cg_maxiter: IntArray | None = 10

    gtol: FloatArray = 1e-6
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

    def update_initial_state(self, state: ApproxCGNewtonState, key: jax.Array | None = None) -> ApproxCGNewtonState:
        """
        Update another state into a valid initial state, using the current state as a starting point.

        Args:
            state: previous state to update

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

    def _select_initial_mu(self, obj_fn, obj0: FloatArray, x0: X, grad0: X):
        grad_norm = tree_norm(grad0)
        grad_unit = jax.tree.map(lambda x: x / grad_norm, grad0)

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

    def create_initial_state(self, x0: X) -> ApproxCGNewtonState:
        """
        Create the initial state for the algorithm.

        Returns:
            initial state
        """

        x = x0
        delta_x = jax.tree.map(jnp.zeros_like, x)  # zeros_like copies over sharding

        # Extract the real and imaginary parts of the complex numbers of input to do Wirtinger calculus
        x_real_imag, merge_fn = convert_to_real(x)
        x_real_imag_size = sum(jax.tree.leaves(jax.tree.map(np.size, x)))
        # For solving make the inputs purely real.
        obj_fn = lambda x: self.obj_fn(merge_fn(x))
        grad_fn = jax.grad(obj_fn)

        obj = obj_fn(x_real_imag)
        result_dtype = jnp.result_type(obj)
        if not jnp.issubdtype(result_dtype, jnp.floating):
            raise ValueError(f"Objective function must return a floating point array, got {result_dtype}.")

        # Linear search to find a suitable damping factor
        grad_obj = grad_fn(x_real_imag)
        mu = self._select_initial_mu(obj_fn, obj, x_real_imag, grad_obj)

        error = tree_norm(grad_obj)

        if self.init_cg_maxiter is None:
            cg_maxiter = jnp.asarray(sum(jax.tree.leaves(jax.tree.map(np.size, x))), dtype=jnp.int32)
        else:
            cg_maxiter = self.init_cg_maxiter

        state = ApproxCGNewtonState(
            iteration=jnp.asarray(0),
            x=x,
            delta_x=delta_x,
            obj=obj,
            grad_obj=grad_obj,
            mu=mu,
            cg_maxiter=cg_maxiter,
            error=error,
            delta_norm=error
        )
        return state

    def solve(self, state: ApproxCGNewtonState) -> Tuple[
        ApproxCGNewtonState, ApproxCGNewtonDiagnostic]:

        # Convert complex to real for Wirtinger calculus
        x_real_imag, merge_fn = convert_to_real(state.x)
        delta_x_real_imag, _ = convert_to_real(state.delta_x)

        state = state._replace(
            x=x_real_imag,
            delta_x=delta_x_real_imag
        )

        # For solving make the inputs purely real.
        obj_fn = lambda x: self.obj_fn(merge_fn(x))
        grad_fn = jax.grad(obj_fn)

        def build_matvec(hvp: Callable[[X], X], damping: FloatArray):
            # replaces J_k^T J_k + λ_k I
            def matvec(v: X) -> X:
                return jax.tree.map(lambda x, y: x + damping * y, hvp(v), v)

            return matvec

        output_dtypes = jax.tree.map(lambda x: x.dtype, state)

        def body(exact_step: IntArray, approx_step: IntArray, state: ApproxCGNewtonState,
                 hvp: Callable[[X], X]) -> Tuple[ApproxCGNewtonState, ApproxCGNewtonDiagnostic]:

            # Units of [obj]/[x]^2
            damping = state.mu * state.error

            matvec = build_matvec(hvp, damping)
            delta_x, _ = jax.scipy.sparse.linalg.cg(
                A=matvec,
                b=jax.tree.map(jax.lax.neg, state.grad_obj),
                x0=state.delta_x,
                maxiter=state.cg_maxiter
            )  # Info returned is not used

            # Determine predicted vs actual reduction gain ratio
            x_prop = jax.tree.map(lambda x, dx: x + dx, state.x, delta_x)
            obj_prop = obj_fn(x_prop)
            grad_obj_prop = grad_fn(x_prop)
            # obj(x0 + dx) ~ obj(x0) + grad(x0).dx + 0.5 dx^T.H(x0).dx
            d1 = tree_dot(state.grad_obj, delta_x)
            d2 = 0.5 * tree_dot(delta_x, hvp(delta_x))
            obj_pushfwd = state.obj + d1 + d2
            # jax.debug.print("F_prop: {F_prop}, F_pushfwd: {F_pushfwd}", F_prop=F_prop, F_pushfwd=F_pushfwd)
            predicted_reduction = state.obj - obj_pushfwd
            actual_reduction = state.obj - obj_prop
            r = jnp.where(
                jnp.logical_or(predicted_reduction == 0., actual_reduction <= 0.),
                jnp.zeros_like(state.obj),
                actual_reduction / predicted_reduction
            )

            # Apply our improvement thresholds
            any_improvement = r >= self.p_any_improvement
            more_newton = r > self.p_more_newton
            less_newton = r < self.p_less_newton

            # Determine if we accept the step
            # In principle, could use lax.cond.

            (obj, x, grad_obj) = jax.tree.map(
                lambda x1, x2: jnp.where(any_improvement, x1, x2),
                (obj_prop, x_prop, grad_obj_prop),
                (state.obj, state.x, state.grad_obj)
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
                    jnp.minimum(state.cg_maxiter * 2, x_size).astype(state.cg_maxiter)
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
            error = tree_norm(grad_obj)

            if self.verbose:
                jax.debug.print(
                    "Iter: {iteration}, Exact Step: {exact_step} Approx Step: {approx_step}, "
                    "cg_maxiter: {cg_maxiter}, "
                    "mu: {mu}, damping: {damping}, r: {r}, pred: {predicted_reduction}, act: {actual_reduction}, "
                    "any_improvement: {any_improvement}, "
                    "more_newton: {more_newton}, less_newton: {less_newton}:\n"
                    "\tobj -> {obj}, delta_norm -> {delta_norm}, error -> {error}",
                    iteration=state.iteration,
                    exact_step=exact_step, approx_step=approx_step,
                    cg_maxiter=state.cg_maxiter,
                    r=r,
                    predicted_reduction=predicted_reduction, actual_reduction=actual_reduction,
                    any_improvement=any_improvement,
                    more_newton=more_newton, less_newton=less_newton, obj=obj, damping=damping,
                    mu=state.mu, delta_norm=delta_norm, error=error
                )
            diagnostic = ApproxCGNewtonDiagnostic(
                iteration=state.iteration,
                exact_step=exact_step,
                approx_step=approx_step,
                obj=obj,
                r=r,
                delta_norm=delta_norm,
                error=error,
                damping=damping,
                mu=state.mu,
                pred=predicted_reduction,
                act=actual_reduction,
                cg_maxiter=state.cg_maxiter
            )
            state = ApproxCGNewtonState(
                iteration=state.iteration + jnp.ones_like(state.iteration),
                x=x,
                delta_x=delta_x,
                grad_obj=grad_obj,
                obj=obj,
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
            state: ApproxCGNewtonState
            diagnostics: ApproxCGNewtonDiagnostic

        def single_iteration(carry: CarryType) -> CarryType:
            # Does one initial exact step using the HVP at the current point, followed by inexact steps using the same
            # HVP estimate (which is slightly cheaper, because they are already computed).
            state = carry.state
            diagnostics = carry.diagnostics
            hvp = build_hvp(obj_fn, state.x, linearise=True)
            for approx_step in range(self.num_approx_steps + 1):
                state, diagnostic = body(
                    carry.exact_iteration,
                    approx_step,
                    state,
                    hvp
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
            hvp = build_hvp(obj_fn, state.x, linearise=True)
            _, diagnostic = body(jnp.zeros_like(state.iteration), jnp.zeros_like(state.iteration), state, hvp)
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


def _sample_leaf(key, vec):
    # if not floating or complex raise error
    if jnp.issubdtype(vec.dtype, jnp.floating):
        return jax.random.normal(key, shape=vec.shape, dtype=vec.dtype)
    elif jnp.issubdtype(vec.dtype, jnp.complexfloating):
        real_dtype = jnp.real(vec).dtype
        return jax.lax.complex(jax.random.normal(key, shape=vec.shape, dtype=real_dtype),
                               jax.random.normal(key, shape=vec.shape, dtype=real_dtype))
    else:
        raise ValueError("Only floating or complex dtypes are supported")


def sample_unit_vector_pytree(key, x):
    leaves, treedef = jax.tree.flatten(x)
    keys = list(jax.random.split(key, len(leaves)))
    v = jax.tree.map(_sample_leaf, jax.tree.unflatten(treedef, keys), x)
    v_norm = tree_norm(v)
    v = jax.tree.map(lambda x: x / v_norm, v)
    return v
