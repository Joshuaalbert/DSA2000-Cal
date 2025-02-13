import functools
import math
from typing import Any, Callable
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["sqrtm", "sqrtm_only", "inv_sqrtm_only"]


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def sqrtm(
        x: jnp.ndarray,
        threshold: float = 1e-6,
        min_iterations: int = 0,
        inner_iterations: int = 10,
        max_iterations: int = 1000,
        regularization: float = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Higham algorithm to compute matrix square root of p.d. matrix.

    See :cite:`higham:97`, eq. 2.6b

    Args:
      x: a (batch of) square p.s.d. matrices of the same size.
      threshold: convergence tolerance threshold for Newton-Schulz iterations.
      min_iterations: min number of iterations after which error is computed.
      inner_iterations: error is re-evaluated every inner_iterations iterations.
      max_iterations: max number of iterations.
      regularization: small regularizer added to norm of x, before normalization.

    Returns:
      Square root matrix of x (or x's if batch), its inverse,
      errors along iterates.
    """
    dimension = x.shape[-1]
    norm_x = jnp.linalg.norm(x, axis=(-2, -1)) * (1 + regularization)

    if jnp.ndim(x) > 2:
        norm_x = norm_x[..., jnp.newaxis, jnp.newaxis]

    def cond_fn(iteration, const, state):
        """Stopping criterion. Checking decrease of objective is needed here."""
        _, threshold = const
        errors, _, _ = state
        err = errors[iteration // inner_iterations - 1]

        return jnp.logical_or(
            iteration == 0,
            jnp.logical_and(
                jnp.logical_and(jnp.isfinite(err), err > threshold),
                jnp.all(jnp.diff(errors) <= 0)
            )
        )  # check decreasing obj, else stop

    def body_fn(iteration, const, state, compute_error):
        """Carry out matrix updates on y and z, stores error if requested.

        Args:
          iteration: iteration number
          const: tuple of constant parameters that do not change throughout the
            loop.
          state: state variables currently updated in the loop.
          compute_error: flag to indicate this iteration computes/stores an error

        Returns:
          state variables.
        """
        x, _ = const
        errors, y, z = state
        w = 0.5 * jnp.matmul(z, y)
        y = 1.5 * y - jnp.matmul(y, w)
        z = 1.5 * z - jnp.matmul(w, z)

        err = jnp.where(compute_error, new_err(x, norm_x, y), jnp.inf)

        errors = errors.at[iteration // inner_iterations].set(err)

        return errors, y, z

    def new_err(x, norm_x, y):
        res = x - norm_x * jnp.matmul(y, y)
        norm_fn = functools.partial(jnp.linalg.norm, axis=(-2, -1))
        return jnp.max(norm_fn(res) / norm_fn(x))

    y = x / norm_x
    z = jnp.eye(dimension)
    if jnp.ndim(x) > 2:
        z = jnp.tile(z, list(x.shape[:-2]) + [1, 1])
    errors = -jnp.ones(math.ceil(max_iterations / inner_iterations))
    state = (errors, y, z)
    const = (x, threshold)
    errors, y, z = fixpoint_iter_backprop(
        cond_fn, body_fn, min_iterations, max_iterations, inner_iterations, const,
        state
    )
    sqrt_x = jnp.sqrt(norm_x) * y
    inv_sqrt_x = z / jnp.sqrt(norm_x)

    return sqrt_x, inv_sqrt_x, errors


def solve_sylvester_bartels_stewart(
        a: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
) -> jnp.ndarray:
    """Solve the real Sylvester equation AX - XB = C using Bartels-Stewart."""
    # See https://nhigham.com/2020/09/01/what-is-the-sylvester-equation/ for
    # discussion of the algorithm (but note that in the derivation, the sign on
    # the right hand side is flipped in the equation in which the columns are set
    # to be equal).
    m = a.shape[-1]
    n = b.shape[-1]
    # Cast a and b to complex to ensure we get the complex Schur decomposition
    # (the real Schur decomposition may not give an upper triangular solution).
    # For the decomposition below, a = u r u* and b = v s v*
    r, u = jax.lax.linalg.schur(a + 0j)
    s, v = jax.lax.linalg.schur(b + 0j)
    d = jnp.matmul(
        jnp.conjugate(jnp.swapaxes(u, axis1=-2, axis2=-1)), jnp.matmul(c, v)
    )
    # The solution in the transformed space will in general be complex, too.
    y = jnp.zeros(a.shape[:-2] + (m, n)) + 0j
    idx = jnp.arange(m)
    for j in range(n):
        lhs = r.at[..., idx, idx].add(-s[..., j:j + 1, j])
        rhs = d[..., j] + jnp.matmul(y[..., :j], s[..., :j, j:j + 1])[..., 0]
        y = y.at[..., j].set(jax.scipy.linalg.solve_triangular(lhs, rhs))

    x = jnp.matmul(
        u, jnp.matmul(y, jnp.conjugate(jnp.swapaxes(v, axis1=-2, axis2=-1)))
    )
    # The end result should be real; remove the imaginary part of the solution.
    return jnp.real(x)


def sqrtm_fwd(
        x: jnp.ndarray,
        threshold: float,
        min_iterations: int,
        inner_iterations: int,
        max_iterations: int,
        regularization: float,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray,
jnp.ndarray]]:
    """Forward pass of custom VJP."""
    sqrt_x, inv_sqrt_x, errors = sqrtm(
        x=x,
        threshold=threshold,
        min_iterations=min_iterations,
        inner_iterations=inner_iterations,
        max_iterations=max_iterations,
        regularization=regularization,
    )
    return (sqrt_x, inv_sqrt_x, errors), (sqrt_x, inv_sqrt_x)


def sqrtm_bwd(
        threshold: float,
        min_iterations: int,
        inner_iterations: int,
        max_iterations: int,
        regularization: float,
        residual: Tuple[jnp.ndarray, jnp.ndarray],
        cotangent: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray]:
    """Compute the derivative by solving a Sylvester equation."""
    del threshold, min_iterations, inner_iterations, \
        max_iterations, regularization
    sqrt_x, inv_sqrt_x = residual
    # ignores cotangent associated with errors
    cot_sqrt, cot_inv_sqrt, _ = cotangent

    # Solve for d(X^{1/2}):
    # Start with X^{1/2} X^{1/2} = X
    # Differentiate to obtain
    # d(X^{1/2}) X^{1/2} + X^{1/2} d(X^{1/2}) = dX
    # The above is a Sylvester equation that we can solve using Bartels-Stewart.
    # Below think of cot_sqrt as (dX)^T and vjp_cot_sqrt as d(X^{1/2})^T.
    # See https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    vjp_cot_sqrt = jnp.swapaxes(
        solve_sylvester_bartels_stewart(
            a=sqrt_x, b=-sqrt_x, c=jnp.swapaxes(cot_sqrt, axis1=-1, axis2=-2)
        ),
        axis1=-1,
        axis2=-2
    )

    # Now solve for d(X^{-1/2}):
    # Start with X^{-1/2} X^{-1/2} = X^{-1}
    # Use the product rule and the fact that d(X^{-1}) = -X^{-1} dX X^{-1}
    # to obtain
    # (See The Matrix Cookbook section on derivatives of an inverse
    # https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf )
    # d(X^{-1/2}) X^{-1/2} + X^{-1/2} d(X^{-1/2}) = -X^{-1} dX X^{-1}
    # Again we have a Sylvester equation that we solve as above, and again we
    # think of cot_inv_sqrt as (dX)^T and vjp_cot_inv_sqrt as d(X^{-1/2})^T
    inv_x = jnp.matmul(inv_sqrt_x, inv_sqrt_x)
    vjp_cot_inv_sqrt = jnp.swapaxes(
        solve_sylvester_bartels_stewart(
            a=inv_sqrt_x,
            b=-inv_sqrt_x,
            c=-jnp.matmul(
                inv_x,
                jnp.matmul(jnp.swapaxes(cot_inv_sqrt, axis1=-2, axis2=-1), inv_x)
            )
        ),
        axis1=-1,
        axis2=-2
    )
    return vjp_cot_sqrt + vjp_cot_inv_sqrt,


sqrtm.defvjp(sqrtm_fwd, sqrtm_bwd)


# Specialized versions of sqrtm that compute only the square root or inverse.
# These functions have lower complexity gradients than sqrtm.


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def sqrtm_only(  # noqa: D103
        x: jnp.ndarray,
        threshold: float = 1e-6,
        min_iterations: int = 0,
        inner_iterations: int = 10,
        max_iterations: int = 1000,
        regularization: float = 1e-6
) -> jnp.ndarray:
    return sqrtm(
        x, threshold, min_iterations, inner_iterations, max_iterations,
        regularization
    )[0]


def sqrtm_only_fwd(  # noqa: D103
        x: jnp.ndarray, threshold: float, min_iterations: int,
        inner_iterations: int, max_iterations: int, regularization: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    sqrt_x = sqrtm(
        x, threshold, min_iterations, inner_iterations, max_iterations,
        regularization
    )[0]
    return sqrt_x, sqrt_x


def sqrtm_only_bwd(  # noqa: D103
        threshold: float, min_iterations: int, inner_iterations: int,
        max_iterations: int, regularization: float, sqrt_x: jnp.ndarray,
        cotangent: jnp.ndarray
) -> Tuple[jnp.ndarray]:
    del threshold, min_iterations, inner_iterations, \
        max_iterations, regularization
    vjp = jnp.swapaxes(
        solve_sylvester_bartels_stewart(
            a=sqrt_x, b=-sqrt_x, c=jnp.swapaxes(cotangent, axis1=-2, axis2=-1)
        ),
        axis1=-2,
        axis2=-1
    )
    return vjp,


sqrtm_only.defvjp(sqrtm_only_fwd, sqrtm_only_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def inv_sqrtm_only(  # noqa: D103
        x: jnp.ndarray,
        threshold: float = 1e-6,
        min_iterations: int = 0,
        inner_iterations: int = 10,
        max_iterations: int = 1000,
        regularization: float = 1e-6
) -> jnp.ndarray:
    return sqrtm(
        x, threshold, min_iterations, inner_iterations, max_iterations,
        regularization
    )[1]


def inv_sqrtm_only_fwd(  # noqa: D103
        x: jnp.ndarray,
        threshold: float,
        min_iterations: int,
        inner_iterations: int,
        max_iterations: int,
        regularization: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    inv_sqrt_x = sqrtm(
        x, threshold, min_iterations, inner_iterations, max_iterations,
        regularization
    )[1]
    return inv_sqrt_x, inv_sqrt_x


def inv_sqrtm_only_bwd(  # noqa: D103
        threshold: float, min_iterations: int, inner_iterations: int,
        max_iterations: int, regularization: float, residual: jnp.ndarray,
        cotangent: jnp.ndarray
) -> Tuple[jnp.ndarray]:
    del threshold, min_iterations, inner_iterations, \
        max_iterations, regularization

    inv_sqrt_x = residual
    inv_x = jnp.matmul(inv_sqrt_x, inv_sqrt_x)
    vjp = jnp.swapaxes(
        solve_sylvester_bartels_stewart(
            a=inv_sqrt_x,
            b=-inv_sqrt_x,
            c=-jnp.matmul(
                inv_x,
                jnp.matmul(jnp.swapaxes(cotangent, axis1=-2, axis2=-1), inv_x)
            )
        ),
        axis1=-1,
        axis2=-2
    )
    return vjp,


inv_sqrtm_only.defvjp(inv_sqrtm_only_fwd, inv_sqrtm_only_bwd)


## Fixed point loop


def fixpoint_iter(
        cond_fn: Callable[[int, Any, Any], bool],
        body_fn: Callable[[Any, Any, Any, Any], Any], min_iterations: int,
        max_iterations: int, inner_iterations: int, constants: Any, state: Any
):
    """Implementation of a fixed point loop.
  
    This fixed point loop iterator applies ``body_fn`` to a tuple
    ``(iteration, constants, state, compute_error)`` to output a new state, using
    context provided in iteration and constants.
  
    ``body_fn`` is iterated (inner_iterations -1) times, and one last time with
    the ``compute_error`` flag to ``True``, indicating that additional
    computational effort can be spent on recalculating the latest error
    (``errors`` are stored as the first element of the state tuple).
  
    upon termination of these ``inner_iterations``, the loop is continued if
    iteration is smaller than ``min_iterations``, stopped if equal/larger than
    ``max_iterations``, and interrupted if ``cond_fn`` returns False.
  
    Args:
      cond_fn : termination condition function
      body_fn : body loop instructions
      min_iterations : lower bound on the total amount of fixed point iterations
      max_iterations : upper bound on the total amount of fixed point iterations
      inner_iterations : number of iterations ``body_fn`` will be executed
        successively before calling ``cond_fn``.
      constants : constant (during loop) parameters passed on to body
      state : state variable
  
    Returns:
      outputs state returned by ``body_fn`` upon termination.
    """  # noqa: D401
    # If number of minimal iterations matches maximal number, force a scan instead
    # of a while loop.

    force_scan = (min_iterations == max_iterations)

    compute_error_flags = jnp.arange(inner_iterations) == inner_iterations - 1

    def max_cond_fn(iteration_state):
        iteration, state = iteration_state
        return jnp.logical_and(
            iteration < max_iterations,
            jnp.logical_or(
                iteration < min_iterations, cond_fn(iteration, constants, state)
            )
        )

    def unrolled_body_fn(iteration_state):

        def one_iteration(iteration_state, compute_error):
            iteration, state = iteration_state
            state = body_fn(iteration, constants, state, compute_error)
            iteration += 1
            return (iteration, state), None

        iteration_state, _ = jax.lax.scan(
            one_iteration, iteration_state, compute_error_flags
        )
        return (iteration_state, None) if force_scan else iteration_state

    if force_scan:
        (_, state), _ = jax.lax.scan(
            lambda carry, x: unrolled_body_fn(carry), (0, state),
            None,
            length=max_iterations // inner_iterations
        )
    else:
        _, state = jax.lax.while_loop(max_cond_fn, unrolled_body_fn, (0, state))
    return state


def fixpoint_iter_fwd(
        cond_fn, body_fn, min_iterations, max_iterations, inner_iterations,
        constants, state
):
    """Forward iteration of fixed point iteration to handle backpropagation.

    The main difference with fixpoint_iter is the checkpointing, in variable
    states, of the state variables as they are recorded through iterations, every
    inner_iterations. This sequence of states will be used in the backward loop.

    Args:
      cond_fn : termination condition function
      body_fn : body loop instructions
      min_iterations : lower bound on the total amount of fixed point iterations
      max_iterations : upper bound on the total amount of fixed point iterations
      inner_iterations : number of iterations body_fn will be executed
        successively before calling cond_fn.
      constants : constant (during loop) parameters passed on to body
      state : state variable

    Returns:
      outputs state returned by body_fn upon termination.
    """
    force_scan = min_iterations == max_iterations
    compute_error_flags = jnp.arange(inner_iterations) == inner_iterations - 1
    states = jax.tree_util.tree_map(
        lambda x: jnp.zeros(
            (max_iterations // inner_iterations + 1,) + jnp.shape(x),
            dtype=jax.dtypes.result_type(x)
        ), state
    )

    def max_cond_fn(iteration_states_state):
        iteration, _, state = iteration_states_state
        return jnp.logical_and(
            iteration < max_iterations,
            jnp.logical_or(
                iteration < min_iterations, cond_fn(iteration, constants, state)
            )
        )

    def unrolled_body_fn(iteration_states_state):
        iteration, states, state = iteration_states_state
        states = jax.tree_util.tree_map(
            lambda states, state: jax.lax.dynamic_update_index_in_dim(
                states, state, iteration // inner_iterations, 0
            ), states, state
        )

        def one_iteration(iteration_state, compute_error):
            iteration, state = iteration_state
            state = body_fn(iteration, constants, state, compute_error)
            iteration += 1
            return (iteration, state), None

        iteration_state, _ = jax.lax.scan(
            one_iteration, (iteration, state), compute_error_flags
        )
        iteration, state = iteration_state
        out = (iteration, states, state)
        return (out, None) if force_scan else out

    if force_scan:
        (iteration, states, state), _ = jax.lax.scan(
            lambda carry, x: unrolled_body_fn(carry), (0, states, state),
            None,
            length=max_iterations // inner_iterations
        )
    else:
        iteration, states, state = jax.lax.while_loop(
            max_cond_fn, unrolled_body_fn, (0, states, state)
        )

    return state, (constants, iteration, states)


def fixpoint_iter_bwd(
        cond_fn, body_fn, min_iterations, max_iterations, inner_iterations, res, g
):
    """Backward iteration of fixed point iteration, using checkpointed states."""
    del cond_fn
    force_scan = (min_iterations == max_iterations)
    constants, iteration, states = res
    # The tree may contain some python floats
    g_constants = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x, dtype=x.dtype)
        if isinstance(x, (np.ndarray, jnp.ndarray)) else 0, constants
    )

    def bwd_cond_fn(iteration_g_gconst):
        iteration, _, _ = iteration_g_gconst
        return iteration >= 0

    def unrolled_body_fn_no_errors(iteration, constants, state):
        compute_error_flags = jnp.zeros((inner_iterations,), dtype=bool)

        def one_iteration(iteration_state, compute_error):
            iteration, state = iteration_state
            state = body_fn(iteration, constants, state, compute_error)
            iteration += 1
            return (iteration, state), None

        iteration_state, _ = jax.lax.scan(
            one_iteration, (iteration, state), compute_error_flags
        )
        _, state = iteration_state
        return state

    def unrolled_body_fn(iteration_g_gconst):
        iteration, g, g_constants = iteration_g_gconst
        state = jax.tree_util.tree_map(
            lambda x: x[iteration // inner_iterations], states
        )
        _, pullback = jax.vjp(
            unrolled_body_fn_no_errors, iteration, constants, state
        )
        _, gi_constants, g_state = pullback(g)
        g_constants = jax.tree_util.tree_map(
            lambda x, y: x + y, g_constants, gi_constants
        )
        out = (iteration - inner_iterations, g_state, g_constants)
        return (out, None) if force_scan else out

    if force_scan:
        (_, g_state, g_constants), _ = jax.lax.scan(
            lambda carry, x: unrolled_body_fn(carry), (0, g, g_constants),
            None,
            length=max_iterations // inner_iterations
        )
    else:
        _, g_state, g_constants = jax.lax.while_loop(
            bwd_cond_fn, unrolled_body_fn,
            (iteration - inner_iterations, g, g_constants)
        )

    return g_constants, g_state


# definition of backprop friendly variant of fixpoint_iter.
fixpoint_iter_backprop = jax.custom_vjp(
    fixpoint_iter, nondiff_argnums=(0, 1, 2, 3, 4)
)

fixpoint_iter_backprop.defvjp(fixpoint_iter_fwd, fixpoint_iter_bwd)
