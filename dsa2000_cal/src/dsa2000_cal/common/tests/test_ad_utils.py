import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.ad_utils import grad_and_hvp, eigh_tridiagonal, lanczos_eigen, build_lanczos_precond, \
    approx_cg_newton, build_hvp


def test_grad_and_hvp():
    def f(params):
        W, b = params
        x = jnp.ones_like(b)
        return jnp.sum(jax.nn.sigmoid(W @ x + b))

    W = jnp.array([[1., 2.], [3., 4.]])
    b = jnp.array([1., 2.])

    params = (W, b)

    v = params

    grad, hvp = grad_and_hvp(f, params, v)

    print(grad)
    print(hvp)


def test_eigh_tridiagonal():
    T_d = jnp.array([1., 2., 3.])
    T_e = jnp.array([4., 5.])

    e, v = eigh_tridiagonal(T_d, T_e)

    np.testing.assert_allclose(v.conj().T @ v, np.eye(3), atol=1e-6)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", ['float', 'complex'])
def test_lanczos_eigen(order, dtype):
    print(f"Testing Lanczos eigen with order {order} and dtype {dtype}")
    if dtype == 'float':
        H = jax.random.normal(jax.random.PRNGKey(0), (6, 6))
    else:
        H = jax.lax.complex(
            jax.random.normal(jax.random.PRNGKey(0), (6, 6)),
            jax.random.normal(jax.random.PRNGKey(1), (6, 6))
        )
    H = H @ H.conj().T
    init_v = jnp.ones((6,))

    e, v = jnp.linalg.eigh(H)
    Hinv = v @ jnp.diag(1. / e) @ v.conj().T

    def matvec(v):
        return H @ v

    evals, vT = lanczos_eigen(jax.random.PRNGKey(0), matvec, init_v, order)
    v = vT.T
    Hinv_approx = v @ jnp.diag(1. / evals) @ v.conj().T

    cond_before = np.linalg.cond(H)
    cond_after = np.linalg.cond(Hinv_approx @ H)
    print(f"Condition number before: {cond_before}")
    print(f"Condition number after: {cond_after}")

    if order == np.size(init_v):
        np.testing.assert_allclose(Hinv @ H, np.eye(6), atol=1e-6)
        np.testing.assert_allclose(Hinv_approx @ H, Hinv @ H, atol=1e-6)
        np.testing.assert_allclose(Hinv_approx @ H, np.eye(6), atol=1e-6)

    else:
        print("Hinv @ H approx")
        print(Hinv_approx @ H)
        print("Hinv @ H error")
        print(Hinv @ H - Hinv_approx @ H)

    Hinv_vp = build_lanczos_precond(matvec, init_v, order, jax.random.PRNGKey(0))
    vec = init_v
    out = Hinv_vp(vec)

    out_exact = Hinv @ vec
    if order == np.size(init_v):
        np.testing.assert_allclose(out, out_exact, atol=1e-6)
    else:
        print("Hinv @ vec approx")
        print(out)
        print("Hinv @ vec exact")
        print(out_exact)


@pytest.mark.parametrize("order, m", [(999, 1000), (998, 1000)])
@pytest.mark.parametrize("dtype", ['float', 'complex'])
def test_lanczos_eigen_big(order, m, dtype):
    print(f"Testing Lanczos eigen with order {order} and dtype {dtype}")
    if dtype == 'float':
        H = jax.random.normal(jax.random.PRNGKey(0), (m, m))
    else:
        H = jax.lax.complex(
            jax.random.normal(jax.random.PRNGKey(0), (m, m)),
            jax.random.normal(jax.random.PRNGKey(1), (m, m))
        )

    # make sparse
    rows = jax.random.randint(jax.random.PRNGKey(0), (int(m ** 2 * 0.75),), 0, m)
    cols = jax.random.randint(jax.random.PRNGKey(1), (int(m ** 2 * 0.75),), 0, m)
    H = H.at[rows, cols].set(0.)

    H = H @ H.conj().T
    init_v = jnp.ones((m,))

    e, v = jnp.linalg.eigh(H)
    Hinv = v @ jnp.diag(1. / e) @ v.conj().T

    def matvec(v):
        return H @ v

    evals, vT = lanczos_eigen(jax.random.PRNGKey(0), matvec, init_v, order)
    v = vT.T
    Hinv_approx = v @ jnp.diag(1. / evals) @ v.conj().T

    cond_before = np.linalg.cond(H)
    cond_after = np.linalg.cond(Hinv_approx @ H)
    print(f"Condition number before: {cond_before}")
    print(f"Condition number after: {cond_after}")

    Hinv_vp = build_lanczos_precond(matvec, init_v, order, jax.random.PRNGKey(0))
    vec = init_v
    out = Hinv_vp(vec)

    out_exact = Hinv @ vec

    print("Hinv @ vec approx")
    print(out)
    print("Hinv @ vec exact")
    print(out_exact)


def test_approx_cg_newton():
    def f(params):
        W, b = params
        x = jnp.ones_like(b)
        return jnp.sum(jax.nn.sigmoid(W @ x + b) ** 2) + jnp.sum(W ** 2) + jnp.sum(b ** 2)

    W = jnp.array([[1., 2.], [3., 4.]])
    b = jnp.array([1., 2.])

    params = (W, b)

    x0 = params

    solution, diagnostics = approx_cg_newton(f, x0, 6)
    print(solution)
    print(diagnostics)


def test_build_hvp():
    def f_crazy(x):
        return jnp.sum(jnp.cos(x) ** 2) + jnp.sum(jnp.sin(x) ** 2)

    x = jnp.ones((10,))
    v = jnp.ones((10,))

    matvec = build_hvp(f_crazy, x, linearise=False)
    matvec_lin = build_hvp(f_crazy, x, linearise=True)

    np.testing.assert_allclose(matvec(v), matvec_lin(v), atol=1e-8)
