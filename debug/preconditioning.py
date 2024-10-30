import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

tfla = tfp.experimental.linalg
tfp_math = tfp.math


def pivoted_cholesky(matrix,
                     max_rank,
                     diag_rtol=1e-3,
                     return_pivoting_order=False,
                     name=None):
  """Computes the (partial) pivoted cholesky decomposition of `matrix`.

  The pivoted Cholesky is a low rank approximation of the Cholesky decomposition
  of `matrix`, i.e. as described in [(Harbrecht et al., 2012)][1]. The
  currently-worst-approximated diagonal element is selected as the pivot at each
  iteration. This yields from a `[B1...Bn, N, N]` shaped `matrix` a `[B1...Bn,
  N, K]` shaped rank-`K` approximation `lr` such that `lr @ lr.T ~= matrix`.
  Note that, unlike the Cholesky decomposition, `lr` is not triangular even in
  a rectangular-matrix sense. However, under a permutation it could be made
  triangular (it has one more zero in each column as you move to the right).

  Such a matrix can be useful as a preconditioner for conjugate gradient
  optimization, i.e. as in [(Wang et al. 2019)][2], as matmuls and solves can be
  cheaply done via the Woodbury matrix identity, as implemented by
  `tf.linalg.LinearOperatorLowRankUpdate`.

  Args:
    matrix: Floating point `Tensor` batch of symmetric, positive definite
      matrices.
    max_rank: Scalar `int` `Tensor`, the rank at which to truncate the
      approximation.
    diag_rtol: Scalar floating point `Tensor` (same dtype as `matrix`). If the
      errors of all diagonal elements of `lr @ lr.T` are each lower than
      `element * diag_rtol`, iteration is permitted to terminate early.
    return_pivoting_order: If `True`, return an `int` `Tensor` indicating the
      pivoting order used to produce `lr` (in addition to `lr`).
    name: Optional name for the op.

  Returns:
    lr: Low rank pivoted Cholesky approximation of `matrix`.
    perm: (Optional) pivoting order used to produce `lr`.

  #### References

  [1]: H Harbrecht, M Peters, R Schneider. On the low-rank approximation by the
       pivoted Cholesky decomposition. _Applied numerical mathematics_,
       62(4):428-440, 2012.

  [2]: K. A. Wang et al. Exact Gaussian Processes on a Million Data Points.
       _arXiv preprint arXiv:1903.08114_, 2019. https://arxiv.org/abs/1903.08114
  """
  with tf.name_scope(name or 'pivoted_cholesky'):
    dtype = dtype_util.common_dtype([matrix, diag_rtol],
                                    dtype_hint=tf.float32)
    if not isinstance(matrix, tf.linalg.LinearOperator):
      matrix = tf.convert_to_tensor(matrix, name='matrix', dtype=dtype)
    if tensorshape_util.rank(matrix.shape) is None:
      raise NotImplementedError('Rank of `matrix` must be known statically')
    if isinstance(matrix, tf.linalg.LinearOperator):
      matrix_shape = tf.cast(matrix.shape_tensor(), tf.int64)
    else:
      matrix_shape = ps.shape(matrix, out_type=tf.int64)

    max_rank = tf.convert_to_tensor(
        max_rank, name='max_rank', dtype=tf.int64)
    max_rank = tf.minimum(max_rank, matrix_shape[-1])
    diag_rtol = tf.convert_to_tensor(
        diag_rtol, dtype=dtype, name='diag_rtol')
    matrix_diag = tf.linalg.diag_part(matrix)
    # matrix is P.D., therefore all matrix_diag > 0, so we don't need abs.
    orig_error = tf.reduce_max(matrix_diag, axis=-1)

    def cond(m, pchol, perm, matrix_diag):
      """Condition for `tf.while_loop` continuation."""
      del pchol
      del perm
      error = tf.linalg.norm(matrix_diag, ord=1, axis=-1)
      max_err = tf.reduce_max(error / orig_error)
      return (m < max_rank) & (tf.equal(m, 0) | (max_err > diag_rtol))

    batch_dims = tensorshape_util.rank(matrix.shape) - 2
    def batch_gather(params, indices, axis=-1):
      return tf.gather(params, indices, axis=axis, batch_dims=batch_dims)

    def body(m, pchol, perm, matrix_diag):
      """Body of a single `tf.while_loop` iteration."""
      # Here is roughly a numpy, non-batched version of what's going to happen.
      # (See also Algorithm 1 of Harbrecht et al.)
      # 1: maxi = np.argmax(matrix_diag[perm[m:]]) + m
      # 2: maxval = matrix_diag[perm][maxi]
      # 3: perm[m], perm[maxi] = perm[maxi], perm[m]
      # 4: row = matrix[perm[m]][perm[m + 1:]]
      # 5: row -= np.sum(pchol[:m][perm[m + 1:]] * pchol[:m][perm[m]]], axis=-2)
      # 6: pivot = np.sqrt(maxval); row /= pivot
      # 7: row = np.concatenate([[[pivot]], row], -1)
      # 8: matrix_diag[perm[m:]] -= row**2
      # 9: pchol[m, perm[m:]] = row

      # Find the maximal position of the (remaining) permuted diagonal.
      # Steps 1, 2 above.
      permuted_diag = batch_gather(matrix_diag, perm[..., m:])
      maxi = tf.argmax(
          permuted_diag, axis=-1, output_type=tf.int64)[..., tf.newaxis]
      maxval = batch_gather(permuted_diag, maxi)
      maxi = maxi + m
      maxval = maxval[..., 0]
      # Update perm: Swap perm[...,m] with perm[...,maxi]. Step 3 above.
      perm = _swap_m_with_i(perm, m, maxi)
      # Step 4.
      if callable(getattr(matrix, 'row', None)):
        row = matrix.row(perm[..., m])[..., tf.newaxis, :]
      else:
        row = batch_gather(matrix, perm[..., m:m + 1], axis=-2)
      row = batch_gather(row, perm[..., m + 1:])
      # Step 5.
      prev_rows = pchol[..., :m, :]
      prev_rows_perm_m_onward = batch_gather(prev_rows, perm[..., m + 1:])
      prev_rows_pivot_col = batch_gather(prev_rows, perm[..., m:m + 1])
      row -= tf.reduce_sum(
          prev_rows_perm_m_onward * prev_rows_pivot_col,
          axis=-2)[..., tf.newaxis, :]
      # Step 6.
      pivot = tf.sqrt(maxval)[..., tf.newaxis, tf.newaxis]
      # Step 7.
      row = tf.concat([pivot, row / pivot], axis=-1)
      # TODO(b/130899118): Pad grad fails with int64 paddings.
      # Step 8.
      paddings = tf.concat([
          tf.zeros([ps.rank(pchol) - 1, 2], dtype=tf.int32),
          [[tf.cast(m, tf.int32), 0]]], axis=0)
      diag_update = tf.pad(row**2, paddings=paddings)[..., 0, :]
      reverse_perm = _invert_permutation(perm)
      matrix_diag = matrix_diag - batch_gather(diag_update, reverse_perm)
      # Step 9.
      row = tf.pad(row, paddings=paddings)
      # TODO(bjp): Defer the reverse permutation all-at-once at the end?
      row = batch_gather(row, reverse_perm)
      pchol_shape = pchol.shape
      pchol = tf.concat([pchol[..., :m, :], row, pchol[..., m + 1:, :]],
                        axis=-2)
      tensorshape_util.set_shape(pchol, pchol_shape)
      return m + 1, pchol, perm, matrix_diag

    m = np.int64(0)
    pchol = tf.zeros(matrix_shape, dtype=matrix.dtype)[..., :max_rank, :]
    perm = tf.broadcast_to(
        ps.range(matrix_shape[-1]), matrix_shape[:-1])
    _, pchol, _, _ = tf.while_loop(
        cond=cond, body=body, loop_vars=(m, pchol, perm, matrix_diag))
    pchol = tf.linalg.matrix_transpose(pchol)
    tensorshape_util.set_shape(
        pchol, tensorshape_util.concatenate(matrix_diag.shape, [None]))

    if return_pivoting_order:
      return pchol, perm
    else:
      return pchol


def build_pivoted_cholesky_preconditioner(H, max_rank, diag_rtol=1e-6):
    """
    Build a preconditioner using the pivoted Cholesky decomposition.

    Args:
        H: [n, n] Symmetric positive-definite matrix.
        max_rank: Integer, the maximum rank of the approximation.
        diag_rtol: Relative tolerance for the diagonal elements.

    Returns:
        preconditioner: Function that applies the preconditioner to a vector.
    """
    # Compute the pivoted Cholesky decomposition
    lr, perm = pivoted_cholesky(
        matrix=H,
        max_rank=max_rank,
        diag_rtol=diag_rtol,
        return_pivoting_order=True
    )

    # lr: [n, max_rank], low-rank Cholesky factors
    # perm: [n], permutation indices

    n = H.shape[0]

    # Construct the permutation matrix P from perm
    P = jnp.eye(n)[perm, :]

    # Permute lr to match the original indices
    lr_full = P.T @ lr  # Now lr_full corresponds to the original matrix ordering

    # Now, H_approx = lr_full @ lr_full.T approximates H

    # To ensure H_approx is positive-definite and invertible, add a small diagonal term
    epsilon = 1e-6
    diag_operator = tfla.LinearOperatorDiag(jnp.ones(n) * epsilon)

    # Create a LinearOperator for lr_full
    lr_full_op = tfla.LinearOperatorFullMatrix(lr_full)

    # H_approx_op represents H_approx = lr_full @ lr_full.T + epsilon * I
    H_approx_op = tfla.LinearOperatorLowRankUpdate(
        base_operator=diag_operator,
        u=lr_full_op.to_dense(),
        v=lr_full_op.to_dense(),
        is_self_adjoint=True,
        is_positive_definite=True
    )

    # Define the preconditioner function using the 'solve' method
    def preconditioner(b):
        # Solve H_approx_op @ x = b
        x = H_approx_op.solve(b)
        return x

    return preconditioner


def test_pivoted_cholesky_preconditioner():
    # Create a symmetric positive-definite matrix H
    np.random.seed(0)
    n = 100  # Dimension of the matrix
    A = np.random.randn(n, n)
    H = np.dot(A.T, A) + np.eye(n) * 1e-3  # Ensure positive-definiteness
    H = jnp.array(H)

    # Specify the maximum rank for the pivoted Cholesky
    max_rank = 10  # Adjust this parameter as needed

    preconditioner = build_pivoted_cholesky_preconditioner(H, max_rank)

    # Compute the condition number before preconditioning
    e_full = jnp.linalg.eigvalsh(H)
    cond_before = jnp.max(e_full) / jnp.min(e_full)
    print(f"Condition number before preconditioning: {cond_before:.2e}")

    # Estimate the condition number after preconditioning
    # Since H is large, we'll use power iteration to estimate the largest and smallest eigenvalues

    def power_iteration(A_op, num_iters=100):
        b_k = jax.random.normal(jax.random.PRNGKey(0), (n,))
        for _ in range(num_iters):
            b_k1 = A_op(b_k)
            b_k1_norm = jnp.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm
        rayleigh_quotient = jnp.dot(b_k, A_op(b_k))
        return rayleigh_quotient

    def inverse_power_iteration(A_op, num_iters=100):
        b_k = jax.random.normal(jax.random.PRNGKey(1), (n,))
        for _ in range(num_iters):
            b_k1 = preconditioner(b_k)
            b_k1_norm = jnp.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm
        rayleigh_quotient = jnp.dot(b_k, A_op(b_k))
        return rayleigh_quotient

    # Define the preconditioned operator H_precond_op = preconditioner @ H
    def H_precond_op(x):
        return preconditioner(H @ x)

    # Estimate largest eigenvalue of H_precond_op
    lambda_max = power_iteration(H_precond_op)
    # Estimate smallest eigenvalue using inverse power iteration
    lambda_min = inverse_power_iteration(H_precond_op)

    cond_after = jnp.abs(lambda_max) / jnp.abs(lambda_min)
    print(f"Condition number after preconditioning: {cond_after:.2e}")

    # Demonstrate the effect on solving a linear system
    b = jax.random.normal(jax.random.PRNGKey(2), (n,))
    # Solve H x = b without preconditioning
    x = jax.scipy.linalg.solve(H, b, assume_a='pos')
    # Solve H x = b with preconditioning using Conjugate Gradient
    from functools import partial
    def matvec(x):
        return H @ x

    @partial(jax.jit, static_argnums=2)
    def cg_solve(matvec, b, maxiter):
        x0 = jnp.zeros_like(b)
        x, _ = jax.scipy.sparse.linalg.cg(matvec, b, x0=x0, tol=1e-6, maxiter=maxiter)
        return x

    # Without preconditioning
    x_cg = cg_solve(matvec, b, maxiter=500)

    # With preconditioning
    def matvec_precond(x):
        return H @ x

    def preconditioner_cg(x):
        return preconditioner(x)

    x_cg_precond = jax.scipy.sparse.linalg.cg(
        matvec_precond, b, x0=jnp.zeros_like(b), tol=1e-6, maxiter=500, M=preconditioner_cg
    )[0]

    # Compute residuals
    res_norm = jnp.linalg.norm(H @ x_cg - b)
    res_norm_precond = jnp.linalg.norm(H @ x_cg_precond - b)
    print(f"Residual norm without preconditioning: {res_norm:.2e}")
    print(f"Residual norm with preconditioning: {res_norm_precond:.2e}")
