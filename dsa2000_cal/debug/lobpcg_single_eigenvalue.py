from typing import Callable

import jax
import jax.numpy as jnp
import pytest


def tree_dot(x, y):
    dots = jax.tree.leaves(jax.tree.map(lambda x, y: jnp.sum(x * y), x, y))
    return sum(dots[1:], start=dots[0])


def tree_norm(x):
    x_norm = jnp.sqrt(tree_dot(x, x))
    x = jax.tree_map(lambda x: jnp.where(x_norm > 0., x / x_norm, 0.), x)
    return x


def generalized_eigen(C, D):
    """Solve the generalized eigenvalue problem C v = lambda D v.
    L.L^T = D + jitter * I

    Then C' -> L^-1 C L^-T
    C' v' = lambda v'

    v = L^-T v'
    """
    # Add jitter to D to make it positive definite
    D_inv = jnp.linalg.pinv(D)

    # Solve the generalized eigenvalue problem C v = lambda (D + jitter * I) v
    eigenvalues, eigenvectors = jnp.linalg.eig(D_inv @ C)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    return eigenvalues, eigenvectors


def rayleigh_ritz_three_vectors(A: Callable, B: Callable, x, w, h, minimise: bool = True):
    # Step 1: Compute the inner products to form the 3x3 matrices C and D
    C = jnp.array([
        [jnp.dot(x, A(x)), jnp.dot(x, A(w)), jnp.dot(x, A(h))],
        [jnp.dot(w, A(x)), jnp.dot(w, A(w)), jnp.dot(w, A(h))],
        [jnp.dot(h, A(x)), jnp.dot(h, A(w)), jnp.dot(h, A(h))]
    ])

    D = jnp.array([
        [jnp.dot(x, B(x)), jnp.dot(x, B(w)), jnp.dot(x, B(h))],
        [jnp.dot(w, B(x)), jnp.dot(w, B(w)), jnp.dot(w, B(h))],
        [jnp.dot(h, B(x)), jnp.dot(h, B(w)), jnp.dot(h, B(h))]
    ])

    # Step 2: Solve the generalized eigenvalue problem C v = rho D v
    eigenvalues, eigenvectors = generalized_eigen(C, D)

    # Step 3: Extract the smallest eigenvalue and corresponding eigenvector
    if minimise:
        idx = jnp.argmin(jnp.abs(eigenvalues))
    else:
        idx = jnp.argmax(jnp.abs(eigenvalues))
    rho_opt = eigenvalues[idx]
    v_opt = eigenvectors[:, idx]

    # Step 4: Construct the optimal y = alpha * x + beta * w + gamma * h
    alpha, beta, gamma = v_opt

    y_opt = jax.tree_map(lambda x, w, h: alpha * x + beta * w + gamma * h, x, w, h)
    y_opt = tree_norm(y_opt)
    return rho_opt, y_opt


def rayleigh_ritz_two_vectors(A: Callable, B: Callable, x, w, minimise: bool = True):
    # Step 1: Compute the inner products to form the 2x2 matrices C and D
    C = jnp.array([
        [jnp.dot(x, A(x)), jnp.dot(x, A(w))],
        [jnp.dot(w, A(x)), jnp.dot(w, A(w))]
    ])

    D = jnp.array([
        [jnp.dot(x, B(x)), jnp.dot(x, B(w))],
        [jnp.dot(w, B(x)), jnp.dot(w, B(w))]
    ])

    # Step 2: Solve the generalized eigenvalue problem C v = rho D v
    eigenvalues, eigenvectors = generalized_eigen(C, D)

    # Step 3: Extract the smallest eigenvalue and corresponding eigenvector
    if minimise:
        idx = jnp.argmin(jnp.abs(eigenvalues))
    else:
        idx = jnp.argmax(jnp.abs(eigenvalues))
    rho_opt = eigenvalues[idx]
    v_opt = eigenvectors[:, idx]

    # Step 4: Construct the optimal y = alpha * x + beta * w
    alpha, beta = v_opt
    y_opt = jax.tree_map(lambda x, w: alpha * x + beta * w, x, w)
    y_opt = tree_norm(y_opt)
    return rho_opt, y_opt


def lobpcg_single(matvec, x0, maxiters: int, minimise: bool = True):
    B = lambda x: x
    T = lambda r: r

    def compute_rho(y):
        return tree_dot(y, matvec(y)) / tree_dot(y, B(y))

    x = x0
    x_last = jax.tree.map(jnp.ones_like, x0)
    rho = compute_rho(x)
    for i in range(maxiters):
        Ax = matvec(x)
        Bx = B(x)
        xAx = tree_dot(x, Ax)
        xBx = tree_dot(x, Bx)
        rho = xAx / xBx
        r = Ax - rho * Bx
        r = tree_norm(r)
        w = T(r)
        # argmin_{y in span(x,w)} rho(y)
        rho_min2, x_next = rayleigh_ritz_two_vectors(matvec, B, x, w, minimise=minimise)
        # rho_min3, x_next = rayleigh_ritz_three_vectors(matvec, B, x, w, x_last, minimise=minimise)
        rho = rho_min2
        x_last = x
        x = x_next

        # print(i, rho, x, r)
        # print(f"Iteration {i}: rho_min 2x2 = {rho_min3}, rho_min 3x3 = {rho_min3}")
    return x, rho


def hutchisons_diag_estimator(key, matvec, x0, num_samples: int, rv_type: str = "normal"):
    def single_sample(key):
        leaves, tree_def = jax.tree.flatten(x0)
        sample_keys = jax.random.split(key, len(leaves))
        sample_keys = jax.tree.unflatten(tree_def, sample_keys)

        def sample(key, shape, dtype):
            if rv_type == "normal":
                return jax.random.normal(key, shape=shape, dtype=dtype)
            elif rv_type == "uniform":
                rv_scale = jnp.sqrt(1. / 3.)
                return jax.random.uniform(key, shape=shape, dtype=dtype, minval=-1., maxval=1.) / rv_scale
            elif rv_type == "rademacher":
                return jnp.where(jax.random.uniform(key, shape=shape) < 0.5, -1., 1.)
            else:
                raise ValueError(f"Unknown rv_type: {rv_type}")

        v = jax.tree.map(lambda key, x: sample(key, x.shape, x.dtype), sample_keys, x0)
        Av = matvec(v)
        return jax.tree.map(lambda x, y: x * y, v, Av)

    keys = jax.random.split(key, num_samples)
    results = jax.vmap(single_sample)(keys)
    return jax.tree.map(lambda y: jnp.mean(y, axis=0), results)

@pytest.mark.parametrize("rv_type", ["normal", "uniform", "rademacher"])
@pytest.mark.parametrize("n", [4, 16, 128, 512, 1024])
@pytest.mark.parametrize("offset", [0., 1e-10, 1e-6, 1e-4, 1e-1, 1.])
def test_lobpcg_single(n, offset, rv_type):
    def single_run(key):
        eigen_vectors = jax.random.normal(key, (n, n))
        eigen_values = jnp.arange(n) + offset
        A = eigen_vectors @ jnp.diag(eigen_values) @ eigen_vectors.T
        Aop = lambda x: A @ x
        x0 = jnp.ones((n,)) / n

        diag_est = hutchisons_diag_estimator(jax.random.PRNGKey(0), Aop, x0, num_samples=1000, rv_type=rv_type)

        preconditioner = jnp.diag(1. / diag_est)
        A_p = A @ preconditioner
        cond_A = jnp.linalg.cond(A)
        cond_A_p = jnp.linalg.cond(A_p)
        reduction = cond_A_p / cond_A
        return reduction
    reductions = jax.vmap(single_run)(jax.random.split(jax.random.PRNGKey(0), 10))
    mean_reduction = jnp.mean(reductions)
    stddev_reduction = jnp.std(reductions)
    print(f"n={n:04d}, offset={offset:.1e}, rv_type={rv_type}, reduction={mean_reduction:.3f} +/- {stddev_reduction:.3f}")
    # assert jnp.linalg.cond(A_p) < jnp.linalg.cond(A)
