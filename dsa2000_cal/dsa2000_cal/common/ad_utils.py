from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import lax


def tree_dot(x, y):
    dots = jax.tree.leaves(jax.tree.map(jnp.vdot, x, y))
    return sum(dots[1:], start=dots[0])


def tree_norm(x):
    return jnp.sqrt(tree_dot(x, x).real)

def tree_mul(x, y):
    return jax.tree.map(jax.lax.mul, x, y)

def tree_sub(x, y):
    return jax.tree.map(jax.lax.sub, x, y)

def tree_div(x, y):
    return jax.tree.map(jax.lax.div, x, y)


def reorthogonalize(v, vecs, m):
    # Full reorthogonalization using while loop
    def reorthogonalization_cond_fun(carry):
        j, _ = carry
        return j < m

    def reorthogonalization_loop(carry):
        j, v = carry
        tau = jax.tree.map(lambda x: x[j], vecs)
        coeff = tree_dot(v, tau)
        v = jax.tree.map(lambda v, tau: v - coeff * tau, v, tau)
        j = j + jnp.ones_like(j)
        return j, v

    _, v = lax.while_loop(reorthogonalization_cond_fun, reorthogonalization_loop,
                          (jnp.asarray(0), v))
    v_norm = tree_norm(v)
    v = jax.tree.map(lambda x: x / v_norm, v)
    return v


def lanczos_alg(key, matvec, init_vec, order):
    """
    Lanczos algorithm for tridiagonalizing a real symmetric matrix.

    This function applies Lanczos algorithm of a given order.  This function
    does full reorthogonalization using while loops for stability and
    numerical consistency.

    Args:
        key: A PRNG key for random number generation.
        matvec: Function that maps v -> Hv for a real symmetric matrix H.
        init_vec: a pytree, with flatted size (dim)
        order: the maximal order of Krylov subspace to compute.

    Returns:
        tridiag: a nested pytree of pytrees corresponding to the tridiagonal matrix. If the output of matvec is a
            vector then this is a Matrix.
        vecs: a pytree of vectors corresponding to the orthonormal basis of the Krylov subspaces.
    """
    aval = jax.eval_shape(matvec, init_vec)

    # upgrade init_vec to this dtype
    def _upgrade(v, aval):
        source_dtype = jnp.result_type(v)
        target_dtype = aval.dtype
        if not jnp.can_cast(source_dtype, target_dtype, casting='safe'):
            raise ValueError(f"Dtype {source_dtype} cannot be safely cast to {target_dtype}.")
        if np.shape(v) != aval.shape:
            raise ValueError(f"Shape {np.shape(v)} is not compatible with {aval.shape}.")
        return v.astype(aval.dtype)

    init_vec = jax.tree.map(lambda x, aval: _upgrade(x, aval), init_vec, aval)

    dim = sum(jax.tree.map(np.size, jax.tree.leaves(init_vec)))
    if order > dim:
        raise ValueError("Order must be less than or equal to the dimension of the matrix.")

    def sample_leaf(key, vec):
        # if not floating or complex raise error
        if jnp.issubdtype(vec.dtype, jnp.floating):
            return jax.random.normal(key, shape=vec.shape, dtype=vec.dtype)
        elif jnp.issubdtype(vec.dtype, jnp.complexfloating):
            real_dtype = jnp.real(vec).dtype
            return jax.lax.complex(jax.random.normal(key, shape=vec.shape, dtype=real_dtype),
                                   jax.random.normal(key, shape=vec.shape, dtype=real_dtype))
        else:
            raise ValueError("Only floating or complex dtypes are supported")

    def sample_v(key):
        leaves, treedef = jax.tree.flatten(init_vec)
        keys = list(jax.random.split(key, len(leaves)))
        v = jax.tree.map(sample_leaf, jax.tree.unflatten(treedef, keys), init_vec)
        v_norm = tree_norm(v)
        v = jax.tree.map(lambda x: x / v_norm, v)
        return v

    def lanczos_step_cond_fun(carry):
        j, tridiag_d, tridiag_e, vecs, w_jm1, beta_j, v_jm1, key = carry
        not_done = jnp.logical_and(beta_j > 1e-6, j < order)
        return not_done

    def lanczos_step(carry):
        j, tridiag_d, tridiag_e, vecs, w_jm1, beta_j, v_jm1, key = carry

        if np.size(tridiag_e) > 0:
            # handle order=1
            tridiag_e = tridiag_e.at[j - 1].set(beta_j)

        # Beta is > 0 always
        v_j = jax.tree.map(lambda x: x / beta_j, w_jm1)
        vecs = jax.tree.map(lambda x, y: x.at[j].set(y), vecs, v_j)

        w_j = matvec(v_j)
        alpha_j = tree_dot(w_j, v_j)
        tridiag_d = tridiag_d.at[j].set(alpha_j)

        w_j = jax.tree.map(lambda x, y, z: x - alpha_j * y - beta_j * z, w_j, v_j, v_jm1)
        beta_jp1 = tree_norm(w_j)

        j = j + jnp.ones_like(j)

        return (j, tridiag_d, tridiag_e, vecs, w_j, beta_jp1, v_j, key)

    def full_cond(carry):
        j, tridiag_d, tridiag_e, vecs, w_jm1, beta_j, v_jm1, key = carry
        return j < order

    def full_body(carry):
        j, tridiag_d, tridiag_e, vecs, w_jm1, beta_j, v_jm1, key = carry

        # get orthonormal v_j
        sample_key, key = jax.random.split(key, 2)
        v_j = reorthogonalize(sample_v(sample_key), vecs, j)
        vecs = jax.tree.map(lambda x, y: x.at[j].set(y), vecs, v_j)

        w_j = matvec(v_j)
        alpha_j = tree_dot(w_j, v_j)
        tridiag_d = tridiag_d.at[j].set(alpha_j)

        # beta_j = 0 for j==0 to get correct w_1
        w_j = jax.tree.map(lambda x, y, z: x - alpha_j * y - beta_j * z, w_j, v_j, v_jm1)
        beta_jp1 = tree_norm(w_j)

        j = j + jnp.ones_like(j)

        carry = (j, tridiag_d, tridiag_e, vecs, w_j, beta_jp1, v_j, key)
        return jax.lax.while_loop(lanczos_step_cond_fun, lanczos_step, carry)

    # Initialization
    tridiag_d = jnp.zeros((order,))
    tridiag_e = jnp.zeros((order - 1,))

    vecs = jax.tree.map(lambda v: jnp.zeros((order,) + np.shape(v), dtype=v.dtype), init_vec)

    # Lanczos loop using jax.lax.scan
    init_carry = (jnp.asarray(0), tridiag_d, tridiag_e, vecs, init_vec, jnp.asarray(0.), init_vec, key)
    final_carry = jax.lax.while_loop(full_cond, full_body, init_carry)
    tridiag_d, tridiag_e, vecs = final_carry[1], final_carry[2], final_carry[3]
    tridiag = jnp.diag(tridiag_d) + jnp.diag(tridiag_e, k=1) + jnp.diag(tridiag_e, k=-1)
    return tridiag, tridiag_d, tridiag_e, vecs


def eigh_tridiagonal(T_d, T_e):
    """
    Compute the eigenvalues and eigenvectors of a tridiagonal matrix.

    Args:
        T_d: [m] the diagonal of the tridiagonal matrix
        T_e: [m-1] the off-diagonal of the tridiagonal matrix

    Returns:
        eigen_val: [m] the eigenvalues of the tridiagonal matrix
        eigen_vec: [m] the eigenvectors of the tridiagonal matrix, where the i-th eigenvec is v[: , i]
    """

    import scipy

    def _eigh_tridiagonal(T_d, T_e):
        return scipy.linalg.eigh_tridiagonal(T_d, T_e, eigvals_only=False, select='a')

    m = np.shape(T_d)[0]

    eigen_val_result_dtype_shape = jax.ShapeDtypeStruct((m,), T_d.dtype)
    eigen_vec_result_dtype_shape = jax.ShapeDtypeStruct((m, m), T_d.dtype)

    return jax.pure_callback(_eigh_tridiagonal,
                             (eigen_val_result_dtype_shape, eigen_vec_result_dtype_shape),
                             T_d, T_e)


def lanczos_eigen(key, matvec, init_vec, order):
    _, T_d, T_e, V = lanczos_alg(key, matvec, init_vec, order)
    # V is [order, dim]
    e, v = eigh_tridiagonal(T_d, T_e)  # [order], [order, order]
    e_H = e
    vT_H = jax.tree.map(lambda x: v.T @ x, V)  # [m, dim]
    return e_H, vT_H


def hvp_linearized(f, params):
    # Compute the gradient function and linearize it at params
    grad_f = jax.grad(f)
    _, jvp_lin = jax.linearize(grad_f, params)
    # lin_fun is a function that computes the JVP of grad_f at params
    return jvp_lin  # This function computes HVPs for different v


def hvp_forward_over_reverse(f, params):
    def hvp(v):
        return jax.jvp(jax.grad(f), (params,), (v,))[1]

    return hvp


def hvp_reverse_over_reverse(f, params):
    def hvp(v):
        return jax.grad(lambda y: jnp.vdot(jax.grad(f)(y), v))(params)

    return hvp


def hvp_reverse_over_forward(f, params):
    def hvp(v):
        jvp_fun = lambda params: jax.jvp(f, (params,), (v,))[1]
        return jax.grad(jvp_fun)(params)

    return hvp


def grad_and_hvp(f, params, v):
    """
    Compute the gradient and Hessian-vector product of a function.

    Args:
        f: the function to differentiate, should be scalar output
        params: the parameters to differentiate with respect to
        v: the vector to multiply the Hessian with

    Returns:
        the gradient and Hessian-vector product
    """
    return jax.jvp(jax.grad(f), (params,), (v,))


def build_hvp(f, params, linearise: bool = True):
    """
    Build a function that computes the Hessian-vector product of a function.

    Args:
        f: scalar function to differentiate
        params: the parameters to differentiate with respect to
        linearise: whether to linearize the gradient function at params, can be better for reapplying the HVP multiple
            times.

    Returns:
        a function that computes the Hessian-vector product
    """
    if linearise:
        # Compute the gradient function and linearize it at params
        grad_f = jax.grad(f)
        # lin_fun is a function that computes the JVP of grad_f at params
        _, grad_jvp_lin = jax.linearize(grad_f, params)

        def matvec(v):
            return grad_jvp_lin(v)
    else:
        def matvec(v):
            return grad_and_hvp(f, params, v)[1]
    return matvec


def build_lanczos_precond(matvec, init_v, order, key):
    evals, vT = lanczos_eigen(key, matvec, init_v, order)

    # vT is [order, dim] except [dim] is a pytree
    # v = vT.T
    # Hinv = v @ jnp.diag(1. / evals) @ v.conj().T

    def matvec(w):
        # Hinv @ v = v @ jnp.diag(1. / evals) @ v.conj().T @ v
        # handle pytrees properly
        # v.conj().T @ w
        # Note, don't conj, since it's done by the tree_dot.
        v1 = jax.vmap(lambda vT, w: tree_dot(vT, w), in_axes=(0, None))(vT, w)  # [order, dim]
        # jnp.diag(1. / evals) @ v1
        v2 = jax.tree.map(lambda x: x / evals, v1)  # [order, dim]
        # v @ v2
        v3 = jax.vmap(lambda v, v2: jax.tree.map(jax.lax.mul, v, v2))(vT, v2)  # [order, dim]
        v3 = jax.tree.map(lambda v: jnp.sum(v, axis=0), v3)  # [dim]
        return v3

    return matvec


def approx_cg_newton(f, x0, maxiter: int):
    """
    Approximate Newton method using conjugate gradient.

    Args:
        f: the function to minimize
        x0: the initial guess
        maxiter: the maximum number of iterations per CG step

    Returns:
        solution: the solution to the optimization problem
        diagnostics: the diagnostics of the optimization problem
    """

    class CarryType(NamedTuple):
        x_jm1: Any  # current iterate
        p_jm1: Any  # last search direction

    class DiagnosticType(NamedTuple):
        gnorm: Any  # norm of the gradient
        pnorm: Any  # norm of the search direction

    def cg_newton_step(carry: CarryType, x):
        matvec = build_hvp(f, carry.x_jm1)
        b = jax.tree.map(jax.lax.neg, jax.grad(f)(carry.x_jm1))
        # Get approximate newton step
        p_j, _ = jsp.sparse.linalg.cg(matvec, b, x0=carry.p_jm1, maxiter=maxiter)
        x_j = jax.tree.map(jax.lax.add, carry.x_jm1, p_j)
        diagnostic = DiagnosticType(gnorm=tree_norm(b), pnorm=tree_norm(p_j))
        return CarryType(x_jm1=x_j, p_jm1=p_j), diagnostic

    carry, diagnostics = jax.lax.scan(cg_newton_step, CarryType(x_jm1=x0, p_jm1=jax.tree.map(jnp.zeros_like, x0)),
                                      jnp.arange(10))
    return carry.x_jm1, diagnostics
