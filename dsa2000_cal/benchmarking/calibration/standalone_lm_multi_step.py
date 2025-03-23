import dataclasses
import inspect
import itertools
import os
import time
import warnings
from functools import partial
from typing import Callable, Any
from typing import NamedTuple, TypeVar, Generic, Union, Tuple

import numpy as np

os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'

import jax
import jax.numpy as jnp

if not jax.config.read('jax_enable_x64'):
    warnings.warn("JAX x64 is not enabled. Setting it now, but check for errors.")
    jax.config.update('jax_enable_x64', True)

# Create a float scalar to lock in dtype choices.
# if jnp.array(1., jnp.float64).dtype != jnp.float32:
#     raise RuntimeError("Failed to set float64 as default dtype.")

PRNGKey = jax.Array
Array = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
]
ComplexArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    complex
]
FloatArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    float,  # valid scalars
]
IntArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    int,  # valid scalars
]
BoolArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_, bool,  # valid scalars
]

Array.__doc__ = "Type annotation for JAX array-like objects, with no scalar types."

ComplexArray.__doc__ = "Type annotation for JAX array-like objects, with complex scalar types."

FloatArray.__doc__ = "Type annotation for JAX array-like objects, with float scalar types."

IntArray.__doc__ = "Type annotation for JAX array-like objects, with int scalar types."

BoolArray.__doc__ = "Type annotation for JAX array-like objects, with bool scalar types."


def get_grandparent_info(relative_depth: int = 7):
    """
    Get the file, line number and function name of the caller of the caller of this function.

    Args:
        relative_depth: the number of frames to go back from the caller of this function. Default is 6. Should be
        enough to get out of a jax.tree.map call.

    Returns:
        str: a string with the file, line number and function name of the caller of the caller of this function.
    """
    # Get the grandparent frame (caller of the caller)
    s = []
    for depth in range(1, min(1 + relative_depth, len(inspect.stack()) - 1) + 1):
        caller_frame = inspect.stack()[depth]
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno
        caller_func = caller_frame.function
        s.append(f"{os.path.basename(caller_file)}:{caller_line} in {caller_func}")
    s = s[::-1]
    s = f"at {' -> '.join(s)}"
    return s


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


def tree_dot(x, y):
    dots = jax.tree.leaves(jax.tree.map(jnp.vdot, x, y))
    return sum(dots[1:], start=dots[0])


def tree_norm(x):
    return jnp.sqrt(tree_dot(x, x).real)


_dot = partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)
_vdot = partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


# aliases for working with pytrees
def _vdot_real_part(x, y):
    """Vector dot-product guaranteed to have a real valued result despite
       possibly complex input. Thus neglects the real-imaginary cross-terms.
       The result is a real float.
    """
    # all our uses of vdot() in CG are for computing an operator of the form
    #  z^H M z
    #  where M is positive definite and Hermitian, so the result is
    # real valued:
    # https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Definitions_for_complex_matrices
    result = _vdot(x.real, y.real)
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
        result += _vdot(x.imag, y.imag)
    return result


def _vdot_real_tree(x, y):
    return sum(jax.tree.leaves(jax.tree.map(_vdot_real_part, x, y)))


def _vdot_tree(x, y):
    return sum(jax.tree.leaves(jax.tree.map(partial(
        jnp.vdot, precision=jax.lax.Precision.HIGHEST), x, y)))


def _norm(x):
    xs = jax.tree.leaves(x)
    return jnp.sqrt(sum(map(_vdot_real_part, xs, xs)))


def _mul(scalar, tree):
    return jax.tree.map(partial(jax.lax.mul, scalar), tree)


_add = partial(jax.tree.map, jax.lax.add)
_sub = partial(jax.tree.map, jax.lax.sub)
_dot_tree = partial(jax.tree.map, _dot)


def astype_single(x):
    def _astype_single(x):
        if jnp.issubdtype(jnp.result_type(x), jnp.complexfloating):
            return x.astype(jnp.complex64)
        elif jnp.issubdtype(jnp.result_type(x), jnp.floating):
            return x.astype(jnp.float32)
        elif jnp.issubdtype(jnp.result_type(x), jnp.integer):
            return x.astype(jnp.int32)
        elif jnp.issubdtype(jnp.result_type(x), jnp.bool_):
            return x.astype(jnp.bool_)
        return x
    return jax.tree.map(_astype_single, x)


def _identity(x):
    return x


def _cg_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):
    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma.real if M is _identity else _vdot_real_tree(r, r)
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / _vdot_real_tree(p, Ap).astype(dtype)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_).astype(dtype)
        beta_ = gamma_ / gamma
        p_ = _add(z_, _mul(beta_, p))
        return x_, r_, gamma_, p_, k + 1

    r0 = _sub(b, A(x0))
    p0 = z0 = M(r0)
    dtype = jnp.result_type(*jax.tree.leaves(p0))
    gamma0 = _vdot_real_tree(r0, z0).astype(dtype)
    initial_value = (x0, r0, gamma0, p0, 0)

    x_final, r, gamma, p, k = jax.lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final, k


def make_linear(f: Callable, *primals0):
    """
    Make a linear function that approximates f around primals0.

    Args:
        f: the function to linearize
        *primals0: the point around which to linearize

    Returns:
        the linearized function
    """
    f0, f_jvp = jax.linearize(f, *primals0)

    def f_linear(*primals):
        diff_primals = jax.tree.map(lambda x, x0: x - x0, primals, primals0)
        df = f_jvp(*diff_primals)
        return jax.tree.map(lambda y0, dy: y0 + dy, f0, df)

    return f_linear


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
    linearize: bool = True  # If True, use linearized function for JVP

    def __post_init__(self):
        if not callable(self.fn):
            raise ValueError('`fn` must be a callable.')

        if self.primals is not None:
            if isinstance_namedtuple(self.primals) or (not isinstance(self.primals, tuple)):
                self.primals = (self.primals,)
            if self.linearize:
                self.linear_fn = make_linear(self.fn, *self.primals)

    def __call__(self, *primals: Any) -> 'JVPLinearOp':
        return JVPLinearOp(
            fn=self.fn,
            primals=primals,
            more_outputs_than_inputs=self.more_outputs_than_inputs,
            adjoint=self.adjoint,
            promote_dtypes=self.promote_dtypes,
            linearize=self.linearize
        )

    def __neg__(self):
        return JVPLinearOp(
            fn=lambda *args, **kwargs: jax.lax.neg(self.fn(*args, **kwargs)),
            primals=self.primals,
            more_outputs_than_inputs=self.more_outputs_than_inputs,
            adjoint=self.adjoint,
            promote_dtypes=self.promote_dtypes,
            linearize=self.linearize
        )

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
        return JVPLinearOp(
            fn=self.fn,
            primals=self.primals,
            more_outputs_than_inputs=self.more_outputs_than_inputs,
            adjoint=not self.adjoint,
            promote_dtypes=self.promote_dtypes,
            linearize=self.linearize
        )

    def matmul(self, *tangents: Any, adjoint: bool = False, left_multiply: bool = True):
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
            return jax.vmap(lambda *_tangent: self.matvec(*_tangent, adjoint=adjoint),
                            in_axes=in_axes, out_axes=out_axes)(*tangents)
        return jax.vmap(lambda *_tangent: self.matvec(*_tangent, adjoint=adjoint),
                        in_axes=in_axes, out_axes=out_axes)(*tangents)

    def matvec(self, *tangents: Any, adjoint: bool = False):
        """
        Compute J @ v = sum_j(J_ij * v_j) using a JVP, if adjoint is False.
        Compute J.T @ v = sum_i(v_i * J_ij) using a VJP, if adjoint is True.

        Args:
            tangents: if adjoint=False, then  pytree of the same structure as the primals, else pytree of the same
                structure as the output.
            adjoint: if True, compute J.T @ v, else compute J @ v

        Returns:
            pytree of matching either f-space (output) if adjoint=False, else x-space (primals)
        """
        if self.primals is None:
            raise ValueError("The primal value must be set to compute the Jacobian.")

        if adjoint:
            co_tangents = tangents

            def _get_results_type(primal_out: jax.Array):
                return primal_out.dtype

            def _adjoint_promote_dtypes(co_tangent: jax.Array, dtype: jnp.dtype):
                if co_tangent.dtype != dtype:
                    warnings.warn(
                        f"Promoting co-tangent dtype from {co_tangent.dtype} to {dtype}, {get_grandparent_info()}."
                    )
                return co_tangent.astype(dtype)

            # v @ J
            if self.linearize:
                f_vjp = jax.linear_transpose(self.linear_fn, *self.primals)
                primals_out = jax.eval_shape(self.linear_fn, *self.primals)
            else:
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
                warnings.warn(f"Promoting primal dtype from {primal.dtype} to {dtype}, at {get_grandparent_info()}.")
            return primal.astype(dtype)

        def _get_result_type(primal: jax.Array):
            return primal.dtype

        primals = self.primals
        if self.promote_dtypes:
            result_types = jax.tree.map(_get_result_type, primals)
            tangents = jax.tree.map(_promote_dtype, tangents, result_types)
        # We use linearised function, so that repeated applications are cheaper.
        if self.linearize:
            primal_out, tangent_out = jax.jvp(self.linear_fn, primals, tangents)
        else:
            primal_out, tangent_out = jax.jvp(self.fn, primals, tangents)
        return tangent_out

    def to_dense(self) -> jax.Array:
        """
        Compute the dense Jacobian at a point.

        Returns:
            [m, n] array
        """
        if self.primals is None:
            raise ValueError("The primal value must be set to compute the Jacobian.")

        if self.more_outputs_than_inputs:
            return jax.jacfwd(self.fn)(*self.primals)
        return jax.jacrev(self.fn)(*self.primals)


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
            damping = astype_single(damping)
            def matvec(v: X) -> X:
                v = astype_single(v)
                JTJv = astype_single(J.matvec(astype_single(J.matvec(v)), adjoint=True))
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
            delta_x, cg_iters_used = _cg_solve(
                A=matvec,
                b=astype_single(jax.tree.map(jax.lax.neg, JTF)),
                x0=astype_single(state.delta_x),
                maxiter=state.cg_maxiter,
            )

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
                    "cg_maxiter: {cg_maxiter}, cg_iters_used: {cg_iters_used}, "
                    "mu: {mu}, damping: {damping}, r: {r}, pred: {predicted_reduction}, act: {actual_reduction}, "
                    "any_improvement: {any_improvement}, "
                    "more_newton: {more_newton}, less_newton: {less_newton}:\n"
                    "\tF_norm -> {F_norm}, delta_norm -> {delta_norm}, error -> {error}",
                    iteration=state.iteration,
                    exact_step=exact_step, approx_step=approx_step,
                    cg_maxiter=state.cg_maxiter, cg_iters_used=cg_iters_used,
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


def kron_product_2x2(M0: jax.Array, M1: jax.Array, M2: jax.Array) -> jax.Array:
    # Matrix([[a0*(a1*a2 + b1*c2) + b0*(a2*c1 + c2*d1), a0*(a1*b2 + b1*d2) + b0*(b2*c1 + d1*d2)], [c0*(a1*a2 + b1*c2) + d0*(a2*c1 + c2*d1), c0*(a1*b2 + b1*d2) + d0*(b2*c1 + d1*d2)]])
    # 36
    # ([(x0, a1*a2 + b1*c2), (x1, a2*c1 + c2*d1), (x2, a1*b2 + b1*d2), (x3, b2*c1 + d1*d2)], [Matrix([
    # [a0*x0 + b0*x1, a0*x2 + b0*x3],
    # [c0*x0 + d0*x1, c0*x2 + d0*x3]])])
    a0, b0, c0, d0 = M0[0, 0], M0[0, 1], M0[1, 0], M0[1, 1]
    a1, b1, c1, d1 = M1[0, 0], M1[0, 1], M1[1, 0], M1[1, 1]
    a2, b2, c2, d2 = M2[0, 0], M2[0, 1], M2[1, 0], M2[1, 1]
    x0 = a1 * a2 + b1 * c2
    x1 = a2 * c1 + c2 * d1
    x2 = a1 * b2 + b1 * d2
    x3 = b2 * c1 + d1 * d2

    # flat = jnp.stack([a0 * x0 + b0 * x1, c0 * x0 + d0 * x1, a0 * x2 + b0 * x3, c0 * x2 + d0 * x3], axis=-1)
    # return unvec(flat, (2, 2))
    flat = jnp.stack([a0 * x0 + b0 * x1, a0 * x2 + b0 * x3, c0 * x0 + d0 * x1, c0 * x2 + d0 * x3], axis=-1)
    return jax.lax.reshape(flat, (2, 2))


def main(num_dir: int, num_ant: int, full_stokes: bool, verbose: bool):
    complex_dtype = jnp.complex64
    antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(num_ant), 2)),
                                     dtype=jnp.int64).T

    def residual_fn(g, vis_model, antenna1, antenna2):
        g1 = g[antenna1, :]
        g2 = g[antenna2, :]

        # Apply gains
        @jax.vmap
        @jax.vmap
        def apply(g1, g2, vis_model):
            if np.shape(g1) == (2, 2):
                return kron_product_2x2(g1, vis_model, g2.conj().T)
            else:
                return g1 * vis_model * g2.conj()

        residual = apply(g1, g2, vis_model).sum(0).astype(complex_dtype)
        return (residual.real, residual.imag)

    B = np.shape(antenna1)[0]
    if full_stokes:
        vis_model = jnp.ones((B, num_dir, 2, 2), dtype=complex_dtype)
        g = jnp.ones((num_ant, num_dir, 2, 2), dtype=complex_dtype)
    else:
        vis_model = jnp.ones((B, num_dir), dtype=complex_dtype)
        g = jnp.ones((num_ant, num_dir), dtype=complex_dtype)
    solver = MultiStepLevenbergMarquardt(
        residual_fn=partial(residual_fn, vis_model=vis_model, antenna1=antenna1, antenna2=antenna2),
        verbose=verbose,
        min_cg_maxiter=1,
        init_cg_maxiter=1,
        num_iterations=1,
        num_approx_steps=0
    )
    state = solver.create_initial_state(g)
    state = state._replace(cg_maxiter=jnp.ones_like(state.cg_maxiter))

    def solve(state, vis_model, antenna1, antenna2):
        solver = MultiStepLevenbergMarquardt(
            residual_fn=partial(residual_fn, vis_model=vis_model, antenna1=antenna1, antenna2=antenna2),
            verbose=verbose,
            min_cg_maxiter=1,
            init_cg_maxiter=1,
            num_iterations=1,
            num_approx_steps=0
        )
        return solver.solve(state)

    t0 = time.time()
    solve_compiled = jax.jit(solve).lower(state, vis_model, antenna1, antenna2).compile()
    t1 = time.time()
    print(f"Compilation time: {t1 - t0}")



    import nvtx

    with nvtx.annotate("first_solve", color="green"):
        t0 = time.time()
        state, diagnostics = jax.block_until_ready(solve_compiled(state, vis_model, antenna1, antenna2))
        t1 = time.time()
        print(f"First Execution time: {t1 - t0}")


    with nvtx.annotate("solve", color="red"):
        t0 = time.time()
        state, diagnostics = jax.block_until_ready(solve_compiled(state, vis_model, antenna1, antenna2))
        t1 = time.time()
        print(f"Second Execution time: {t1 - t0}")


if __name__ == '__main__':
    main(2, 2048, True, False)
