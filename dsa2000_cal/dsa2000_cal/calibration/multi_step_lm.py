import dataclasses
import os
from typing import NamedTuple, Any, Callable, TypeVar, Generic, Union, Tuple

import numpy as np

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import jax
import jax.numpy as jnp

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


def test_convert_to_real():
    x = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([1.0 + 1.0j, 2.0 + 2.0j])}
    x_real_imag, merge = convert_to_real(x)
    print(x_real_imag)
    np.testing.assert_allclose(x_real_imag[0], x['a'])
    np.testing.assert_allclose(x_real_imag[1][0], x['b'].real)
    np.testing.assert_allclose(x_real_imag[1][1], x['b'].imag)
    x_rec = merge(x_real_imag)
    np.testing.assert_allclose(x_rec['a'], x['a'])
    np.testing.assert_allclose(x_rec['b'], x['b'])
    assert x_rec['a'].dtype == x['a'].dtype
    assert x_rec['b'].dtype == x['b'].dtype


class MultiStepLevenbergMarquardtState(NamedTuple):
    iteration: jax.Array  # iteration number
    x: X  # current solution, may be a pytree
    delta_x: X  # step, may be a pytree
    damping: jax.Array  # damping factor
    F: Y  # residual, may be a pytree
    F_norm: jax.Array  # norm of the residual
    mu: jax.Array  # damping factor


class Diagnostics(NamedTuple):
    iteration: jax.Array
    F_norm: jax.Array
    r: jax.Array
    error: jax.Array


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
    num_approx_steps: int = 0
    more_outputs_than_inputs: bool = False

    # Improvement threshold
    p_any_improvement: float = 0.1  # p0 > 0
    p_less_newton: float = 0.25  # p2 -- less than sufficient improvement
    p_sufficient_improvement: float = 0.5  # p1 > p0
    p_more_newton: float = 0.75  # p3 -- more than sufficient improvement

    # Damping alteration factors 0 < c_more_newton < 1 < c_less_newton
    c_more_newton: float = 0.1
    c_less_newton: float = 2.
    # Damping factor = mu1 * ||F(x)||^delta, 1 <= delta <= 2
    delta: int = 2
    # mu1 > mu_min > 0
    mu1: float = 1.
    mu_min: float = 1e-3

    verbose: bool = False

    def __post_init__(self):
        if self.num_approx_steps < 0:
            raise ValueError("num_approx_steps must be non-negative")
        if self.delta < 1 or self.delta > 2:
            raise ValueError("delta must be 1 <= delta <= 2")
        if self.mu1 <= self.mu_min:
            raise ValueError("mu1 must be greater than mu_min")
        if self.mu_min <= 0:
            raise ValueError("mu_min must be positive")
        if not (
                (0. < self.p_any_improvement)
                and (self.p_any_improvement < self.p_less_newton)
                and (self.p_less_newton < self.p_sufficient_improvement)
                and (self.p_sufficient_improvement < self.p_more_newton)
                and (self.p_more_newton < 1.)
        ):
            raise ValueError(
                "Improvement thresholds must satisfy 0 < p(any) < p(less) < p(sufficient) < p(more) < 1, "
                f"got {self.p_any_improvement}, {self.p_less_newton}, {self.p_sufficient_improvement}, "
                f"{self.p_more_newton}"
            )

        if not (
                (0. < self.c_more_newton)
                and (self.c_more_newton < 1.)
                and (1. < self.c_less_newton)
        ):
            raise ValueError(
                "Damping alteration factors must satisfy 0 < c_more_newton < 1 < c_less_newton, "
                f"got {self.c_more_newton}, {self.c_less_newton}"
            )
        self._residual_fn = self.wrap_residual_fn(self.residual_fn)

    @staticmethod
    def wrap_residual_fn(residual_fn: Callable[[X], Y]) -> Callable[[X], Y]:
        """
        Wrap the residual function to handle complex inputs and outputs.
        """

        def wrapped_residual_fn(x: X) -> Y:
            output = residual_fn(x)

            def _make_real(x):
                if not isinstance(x, jax.Array):
                    raise RuntimeError("Only jax.Array is supported")
                if jnp.issubdtype(x.dtype, jnp.complexfloating):
                    return (x.real, x.imag)
                return x

            return [jax.tree_map(_make_real, output)]

        return wrapped_residual_fn

    def create_initial_state(self, x0: X) -> MultiStepLevenbergMarquardtState:
        """
        Create the initial state for the algorithm.

        Returns:
            initial state
        """
        x = x0
        mu = self.mu1
        d = jax.tree.map(jnp.zeros_like, x)
        F = self._residual_fn(x)
        F_norm = pytree_norm_delta(F, power=2)
        if self.delta == 2:
            F_norm_delta = F_norm
        else:
            F_norm_delta = pytree_norm_delta(F, power=self.delta)
        damping = mu * F_norm_delta

        state = MultiStepLevenbergMarquardtState(
            iteration=jnp.asarray(0),
            x=x,
            delta_x=d,
            damping=damping,
            F=F,
            F_norm=F_norm,
            mu=jnp.asarray(self.mu1)
        )
        return state

    def solve(self, state: MultiStepLevenbergMarquardtState) -> MultiStepLevenbergMarquardtState:

        # Convert complex to real
        x_real_imag, merge_fn = convert_to_real(state.x)
        delta_x_real_imag, _ = convert_to_real(state.delta_x)

        state = state._replace(
            x=x_real_imag,
            delta_x=delta_x_real_imag
        )

        residual_fn = lambda x: self._residual_fn(merge_fn(x))

        def build_matvec(J: JVPLinearOp, damping: jax.Array):
            def matvec(v: X) -> X:
                JTJv = J.matvec(J.matvec(v), adjoint=True)
                return jax.tree.map(lambda x, y: x + damping * y, JTJv, v)

            return matvec

        J_bare = JVPLinearOp(fn=residual_fn, more_outputs_than_inputs=self.more_outputs_than_inputs)

        output_dtypes = jax.tree.map(lambda x: x.dtype, state)

        def body(state: MultiStepLevenbergMarquardtState, J: JVPLinearOp) -> Tuple[
            MultiStepLevenbergMarquardtState, None]:
            # d_k = -(J_k^T J_k + λ_k I)^(-1) J_k^T F(x_k)
            JTF = J.matvec(state.F, adjoint=True)
            # jax.debug.print("JTF: {JTF}", JTF=JTF)
            matvec = build_matvec(J, state.damping)
            delta_x, _ = jax.scipy.sparse.linalg.cg(matvec, jax.tree.map(jax.lax.neg, JTF),
                                                    x0=state.delta_x)  # Info returned is not used
            x_prop = jax.tree.map(lambda x, dx: x + dx, state.x, delta_x)

            F_prop = residual_fn(x_prop)
            F_pushfwd = jax.tree.map(lambda x, y: x + y, state.F, J.matvec(delta_x))
            # jax.debug.print("F_prop: {F_prop}, F_pushfwd: {F_pushfwd}", F_prop=F_prop, F_pushfwd=F_pushfwd)
            F_prop_norm = pytree_norm_delta(F_prop, power=2)
            F_pushfwd_norm = pytree_norm_delta(F_pushfwd, power=2)
            r = jnp.where(
                state.F_norm == F_prop_norm,
                jnp.zeros_like(state.F_norm),
                (state.F_norm - F_prop_norm) / (state.F_norm - F_pushfwd_norm)
            )

            any_improvement = r >= self.p_any_improvement
            sufficient_improvement = r >= self.p_sufficient_improvement
            more_newton = r > self.p_more_newton
            less_newton = r < self.p_less_newton

            x = jax.tree.map(lambda x_prop, x: jnp.where(any_improvement, x_prop, x), x_prop, state.x)
            F = jax.tree.map(lambda F_prop, F: jnp.where(any_improvement, F_prop, F), F_prop, state.F)
            F_norm = jnp.where(any_improvement, F_prop_norm, state.F_norm)

            mu = jnp.where(
                less_newton,
                self.c_less_newton * state.mu,
                jnp.where(
                    more_newton,
                    jnp.maximum(self.c_more_newton * state.mu, self.mu_min),
                    state.mu
                )
            )

            if self.delta == 2:
                F_norm_delta = F_norm
            else:
                F_norm_delta = pytree_norm_delta(F, power=self.delta)

            damping = mu * F_norm_delta
            # jnp.where(
            #     sufficient_improvement,
            #     damping,
            #     mu * F_norm_delta
            # )
            state = MultiStepLevenbergMarquardtState(
                iteration=state.iteration + 1,
                x=x,
                delta_x=delta_x,
                damping=damping,
                F=F,
                F_norm=F_norm,
                mu=mu
            )
            # Cast to the original dtype for sanity
            state = jax.tree_map(lambda x, dtype: x.astype(dtype), state, output_dtypes)

            if self.verbose:
                error = pytree_norm_delta(delta_x, power=1)
                jax.debug.print(
                    "Iter: {i}, r: {r}, any_improvement: {any_improvement}, "
                    "sufficient_improvement: {sufficient_improvement}, more_newton: {more_newton}, "
                    "less_newton: {less_newton}:\n"
                    "\t|F|^2 -> {F_norm}, damping -> {damping}, mu -> {mu}, error -> {error}",
                    i=state.iteration, r=r, any_improvement=any_improvement,
                    sufficient_improvement=sufficient_improvement,
                    more_newton=more_newton, less_newton=less_newton, F_norm=F_norm, damping=damping,
                    mu=mu, error=error
                )
            return state, None

        J = J_bare(state.x)
        for i in range(self.num_approx_steps + 1):
            state, _ = body(state, J)
        J = J_bare(state.x)
        state, _ = body(state, J)

        # Convert back to complex
        state = state._replace(
            x=merge_fn(state.x),
            delta_x=merge_fn(state.delta_x)
        )

        return state


def pytree_norm_delta(pytree: Any, power: int | float = 2) -> jax.Array:
    """
    Compute 2-norm raised to the power of a pytree.

    Args:
        pytree: pytree of arrays
        power: power to raise 2-norm to

    Returns:
        2-norm of the pytree raised to the power
    """
    square_sum = jax.tree.map(lambda x: jnp.sum(jnp.square(jnp.abs(x))), pytree)
    leaves = jax.tree_leaves(square_sum)
    total_square_sum = sum(leaves[1:], leaves[0])
    if power == 2:
        return total_square_sum
    if power == 1:
        return jnp.sqrt(total_square_sum)
    return total_square_sum ** (power / 2.)
