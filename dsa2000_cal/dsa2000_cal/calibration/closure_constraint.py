from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp


def solve_constraints(num_antennas: int) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Solve the constraints for the linear state space model.

    Returns:
        constrained_solution_matrix_free: the matrix for the constraints from free
        constrained_solution_matrix_data: the matrix for the constraints from data
        free_vars: the free variables for the constraints
        constrained_vars: the constrained variables for the constraints
    """

    # Compute the matrix that computes loop sums: phi_ij + phi_jk + phi_ki = 0
    def compute_baseline_index(i, j):
        return i + j * num_antennas

    F = []
    num_baselines = num_antennas ** 2
    baseliase_indices = jnp.arange(num_baselines)
    for i in range(num_antennas):
        for j in range(num_antennas):
            if j == i:
                continue
            for k in range(num_antennas):
                if k == i or k == j:
                    continue
                indices = jnp.asarray([compute_baseline_index(i, j), compute_baseline_index(j, k), compute_baseline_index(k, i)])
                F.append(jnp.isin(baseliase_indices, indices))

    F = jnp.asarray(F)  # [num_loops, 3]
    num_loops = F.shape[0]

    # Make loop-sum variables
    delta = sp.symbols(' '.join([f'delta{i}' for i in range(num_loops)]))
    if num_loops == 1:
        delta = [delta]

    A = F.astype(int).tolist()

    # Create a matrix using sympy
    A = sp.Matrix(A)
    b = sp.Matrix(delta)
    # solve A.x=b with free variables
    sol, params, free_vars = A.gauss_jordan_solve(b, freevar=True)
    constrained_vars = sorted(set(range(num_baselines)) - set(free_vars))

    constrained_solution_matrix_free = sol.jacobian(params)
    constrained_solution_matrix_free = jnp.asarray(np.array(constrained_solution_matrix_free).astype(float),
                                                   jnp.float32)

    constrained_solution_matrix_data = sol.jacobian(b)
    constrained_solution_matrix_data = jnp.asarray(np.array(constrained_solution_matrix_data).astype(float),
                                                   jnp.float32)

    free_vars = jnp.asarray(np.array(free_vars).astype(int), jnp.int32)
    constrained_vars = jnp.asarray(np.array(constrained_vars).astype(int), jnp.int32)

    return constrained_solution_matrix_free, constrained_solution_matrix_data, free_vars, constrained_vars


def test_solve_constraints():
    solve_constraints(num_antennas=5)
