import jax
import jax.numpy as jnp
from jax import lax


def invert_permutation(perm):
    """
    Returns the inverse permutation of perm.

    Args:
        perm: [N], permutation of N elements.

    Returns:
        inverse_perm: [N], inverse permutation.
    """
    inverse_perm = jnp.zeros_like(perm)
    inverse_perm = inverse_perm.at[perm].set(jnp.arange(perm.shape[0]))
    return inverse_perm


def pivoted_cholesky_single(matrix, max_rank, diag_rtol=1e-3, return_pivoting_order=False):
    """
    Computes the (partial) pivoted Cholesky decomposition of `matrix`.

    Args:
        matrix: [N, N] symmetric positive-definite matrix.
        max_rank: int, maximum rank of the approximation.
        diag_rtol: float, relative tolerance for the diagonal elements.
        return_pivoting_order: bool, whether to return the pivoting order.

    Returns:
        pchol: [K, N] lower-rank Cholesky factors.
        perm: [N], pivoting order (optional).
    """
    N = matrix.shape[0]
    max_rank = min(max_rank, N)
    matrix_diag = jnp.diagonal(matrix)
    orig_error = jnp.max(matrix_diag)

    def cond_fun(loop_vars):
        m, pchol, perm, matrix_diag = loop_vars
        error = jnp.linalg.norm(matrix_diag, ord=1)
        max_err = error / orig_error
        continue_loop = (m < max_rank) & ((m == 0) | (max_err > diag_rtol))
        return continue_loop

    def body_fun(loop_vars):
        m, pchol, perm, matrix_diag = loop_vars
        N = matrix.shape[0]
        indices = jnp.arange(N)
        mask = indices >= m  # Mask to select indices from m to N-1

        # Steps 1 and 2: Find the pivot
        permuted_diag = matrix_diag[perm]
        # Apply mask to select only the elements from position m onwards
        masked_diag = jnp.where(mask, permuted_diag, -jnp.inf)
        maxi = jnp.argmax(masked_diag)
        maxval = permuted_diag[maxi]

        # Step 3: Swap perm[m] and perm[maxi]
        perm = perm.at[m].set(perm[maxi]).at[maxi].set(perm[m])

        # Step 4: Compute row
        col_index = perm[m]
        row_indices = perm
        # We need to mask the elements before position m
        row_mask = indices > m  # Only include indices greater than m
        # Compute the full row and then mask
        full_row = matrix[col_index, row_indices]
        row = jnp.where(row_mask, full_row, 0.0)

        # Step 5: Update row
        def compute_row(row):
            prev_rows = pchol[:m, :]
            prev_rows_pivot_col = prev_rows[:, col_index]
            prev_rows_row_indices = prev_rows[:, row_indices]
            # Mask prev_rows_row_indices to exclude columns before m
            prev_rows_row_indices = jnp.where(row_mask, prev_rows_row_indices, 0.0)
            row_update = jnp.sum(prev_rows_row_indices * prev_rows_pivot_col[:, None], axis=0)
            return row - row_update

        row = lax.cond(m > 0, compute_row, lambda row: row, row)

        # Step 6: Normalize row
        pivot = jnp.sqrt(maxval)
        # Avoid division by zero
        pivot_inv = jnp.where(pivot > 0, 1.0 / pivot, 0.0)
        row = row * pivot_inv

        # Step 7: Construct the full row with the pivot at position m
        row_full = jnp.zeros(N, dtype=matrix.dtype)
        row_full = row_full.at[m].set(pivot)
        row_full = row_full.at[m + 1:].set(row[m + 1:])

        # Step 8: Update matrix_diag
        diag_update = row_full ** 2
        reverse_perm = invert_permutation(perm)
        matrix_diag = matrix_diag - diag_update[reverse_perm]

        # Step 9: Update pchol
        pchol = pchol.at[m, :].set(row_full)

        return m + 1, pchol, perm, matrix_diag

    m = 0
    pchol = jnp.zeros((max_rank, N), dtype=matrix.dtype)
    perm = jnp.arange(N)
    loop_vars = (m, pchol, perm, matrix_diag)
    loop_vars = lax.while_loop(cond_fun, body_fun, loop_vars)
    m, pchol, perm, matrix_diag = loop_vars

    # Truncate pchol to the actual rank achieved
    pchol = pchol[:m, :]

    if return_pivoting_order:
        return pchol, perm
    else:
        return pchol


def pivoted_cholesky(matrix, max_rank, diag_rtol=1e-3, return_pivoting_order=False):
    """
    Computes the (partial) pivoted Cholesky decomposition of `matrix`, supports batching.

    Args:
        matrix: [batch_dims..., N, N] symmetric positive-definite matrix.
        max_rank: int, maximum rank of the approximation.
        diag_rtol: float, relative tolerance for the diagonal elements.
        return_pivoting_order: bool, whether to return the pivoting order.

    Returns:
        pchol: [batch_dims..., K, N] lower-rank Cholesky factors.
        perm: [batch_dims..., N], pivoting order (optional).
    """
    # Vectorize over batch dimensions
    batch_pivoted_cholesky = jax.vmap(
        pivoted_cholesky_single,
        in_axes=(0, None, None, None),
        out_axes=(0, 0) if return_pivoting_order else 0
    )
    result = batch_pivoted_cholesky(matrix, max_rank, diag_rtol, return_pivoting_order)
    return result


def test_pivoted_cholesky():
    import numpy as np
    N = 5
    matrix = jnp.array([[6.0, 3.0, 4.0, 8.0, 1.0],
                       [3.0, 6.0, 5.0, 1.0, 2.0],
                       [4.0, 5.0, 10.0, 2.0, 7.0],
                       [8.0, 1.0, 2.0, 12.0, 3.0],
                       [1.0, 2.0, 7.0, 3.0, 14.0]])
    matrix = (matrix + matrix.T) / 2  # Ensure symmetry
    max_rank = 3
    pchol, perm = pivoted_cholesky_single(matrix, max_rank, return_pivoting_order=True)
    print("Pivoted Cholesky factors (pchol):")
    print(pchol)
    print("\nPermutation (perm):")
    print(perm)
    # Reconstruct the approximate matrix
    matrix_approx = pchol.T @ pchol
    # Apply the permutation to match the original matrix ordering
    perm_inv = invert_permutation(perm)
    matrix_approx = matrix_approx[perm_inv][:, perm_inv]
    print("\nApproximated matrix:")
    print(matrix_approx)
    print("\nOriginal matrix:")
    print(matrix)
    # Compute the approximation error
    error = np.linalg.norm(matrix - matrix_approx) / np.linalg.norm(matrix)
    print(f"\nRelative approximation error: {error:.2e}")
