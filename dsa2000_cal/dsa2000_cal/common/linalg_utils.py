import jax
from jax import numpy as jnp
from scipy.sparse.linalg import eigs


def randomized_svd(key, A, k, p):
    # Step 2: Parameters
    l = k + p
    m, n = A.shape

    # Step 3: Random Test Matrix
    Omega = jax.random.normal(key, (n, l))

    # Step 4: Sample Matrix
    Y = A @ Omega

    # Step 5: Orthonormalize Y
    Q, _ = jnp.linalg.qr(Y)

    # Step 6: Project A to lower dimension
    B = Q.T @ A

    # Step 7: Compute SVD of B
    tilde_U, Sigma, VT = jnp.linalg.svd(B, full_matrices=False)

    # Step 8: Approximate U
    U = Q @ tilde_U
    return U, Sigma, VT


def randomized_pinv(key, A, k, p):
    U, Sigma, VT = randomized_svd(key, A, k, p)
    # Step 10: Compute pseudoinverse
    Sigma_plus = jnp.diag(jnp.where(Sigma > 1e-10, jnp.reciprocal(Sigma), 0.))
    A_pinv = VT.T @ Sigma_plus @ U.T
    return A_pinv


def test_randomized_svd_pinv():
    # Step 1: Input Matrix
    A = jax.random.normal(jax.random.PRNGKey(0), (10, 5))

    # Step 9: Compute pseudoinverse
    A_pinv = randomized_pinv(jax.random.PRNGKey(1),
                             A, 3, 2)

    # Step 11: Check
    assert jnp.allclose(A_pinv, jnp.linalg.pinv(A))


def msqrt(A):
    """
    Computes the matrix square-root using SVD, which is robust to poorly conditioned covariance matrices.
    Computes, M such that M @ M.T = A

    Args:
    A: [N,N] Square matrix to take square root of.

    Returns: [N,N] matrix.
    """
    U, s, Vh = jnp.linalg.svd(A)
    L = U * jnp.sqrt(s)
    max_eig = jnp.max(s)
    min_eig = jnp.min(s)
    return max_eig, min_eig, L


def randomised_msqrt(key, A, k, p):
    """
    Computes the matrix square-root using SVD, which is robust to poorly conditioned covariance matrices.
    Computes, M such that M @ M.T = A

    Args:
    A: [N,N] Square matrix to take square root of.

    Returns: [N,N] matrix.
    """
    U, s, Vh = randomized_svd(key, A, k, p)
    L = U * jnp.sqrt(s)
    max_eig = jnp.max(s)
    min_eig = jnp.min(s)
    return max_eig, min_eig, L

def test_randomised_msqrt():
    A = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
    A = A @ A.T
    max_eig, min_eig, L = randomised_msqrt(jax.random.PRNGKey(1), A, 3, 2)
    assert jnp.allclose(L @ L.T, A)

