import time
from typing import Tuple

import jax
import numpy as np
import pytest
from jax import random as random, numpy as jnp, lax

from dsa2000_cal.common.nearest_neighbours import ApproximateTreeNN2D, ApproximateTreeNN3D
from dsa2000_cal.common.vec_utils import kron_product, unvec, kron, vec


def brute_force_nearest_neighbors(points: jnp.ndarray, test_point: jnp.ndarray, k: int) -> Tuple[
    jnp.ndarray, jnp.ndarray]:
    """
    A brute-force approach to find the k nearest neighbors to a test point in 3D.

    Parameters:
        points (jax.numpy.ndarray): Array of points with shape (n_points, 3).
        test_point (jax.numpy.ndarray): A point in [0,1]^3 with shape (3,).
        k (int): The number of nearest neighbors to find.

    Returns:
        distances (jax.numpy.ndarray): Distances to the k nearest neighbors.
        indices (jax.numpy.ndarray): Indices of the k nearest neighbors.
    """
    distances = jnp.linalg.norm(points - test_point, axis=1)
    top_k_neg_distances, top_k_indices = jax.lax.top_k(-distances, k)
    return -top_k_neg_distances, top_k_indices


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("m", [1, 100, 1000])
@pytest.mark.parametrize("n", [1000, 100000])
@pytest.mark.parametrize("average_points_per_cell", [9, 16, 25])
@pytest.mark.parametrize("kappa", [1., 5., 10.])
def test_performance_approx_nn_2d(n: int, k: int, m: int, average_points_per_cell: int, kappa: float):
    """
    Performance test comparing ApproximateTree with brute-force approach for varying number of points.
    """

    # Generate uniformly distributed points
    key = random.PRNGKey(0)
    points = random.uniform(key, (n, 2))

    # Test now with vmap test_points
    test_points = random.uniform(jax.random.PRNGKey(1), (m, 2))

    # ApproximateTree method
    approx_tree = ApproximateTreeNN2D(average_points_per_cell=average_points_per_cell, kappa=kappa)
    build_tree = jax.jit(approx_tree.build_tree).lower(points).compile()
    t0 = time.time()
    tree = build_tree(points)
    jax.block_until_ready(tree)
    tree_build_time = time.time() - t0

    query = jax.jit(jax.vmap(lambda test_point: approx_tree.query(tree, test_point, k))).lower(test_points).compile()

    t0 = time.time()
    distances_approx, indices_approx = query(test_points)
    jax.block_until_ready(distances_approx)
    approx_time = time.time() - t0

    # Brute-force method
    brute_force_nearest_neighbors_vmap = jax.jit(
        jax.vmap(lambda test_point: brute_force_nearest_neighbors(points, test_point, k))
    ).lower(test_points).compile()

    t0 = time.time()
    distances_brute, indices_brute = brute_force_nearest_neighbors_vmap(test_points)
    jax.block_until_ready(distances_brute)
    brute_time = time.time() - t0

    speedup = brute_time / approx_time
    index_error_rate = np.mean(indices_brute != indices_approx)
    mean_abs_error = np.mean(np.abs(distances_brute - distances_approx))
    rmse = np.sqrt(np.mean((distances_brute - distances_approx) ** 2))
    print(
        f"n={n}, m={m}, k={k}, avg_points_per_cell={average_points_per_cell}, kappa={kappa}:\n"
        f"\tTree build time: {tree_build_time:.5f}s\n"
        f"\tApproximateTree time: {approx_time:.5f}s\n"
        f"\tBrute-force time: {brute_time:.5f}s\n"
        f"\tSpeedup: {speedup:.2f}\n"
        f"\tIndex error rate: {index_error_rate:.2f}\n"
        f"\tMean absolute error: {mean_abs_error:.5f}\n"
        f"\tRMSE: {rmse:.5f}\n"
    )


@pytest.mark.parametrize("k", [3])
@pytest.mark.parametrize("m", [1000000])
@pytest.mark.parametrize("n", [50 * 50])
@pytest.mark.parametrize("average_points_per_cell", [9, 16, 25])
@pytest.mark.parametrize("kappa", [1., 5., 10.])
def test_performance_approx_nn_usecase_2d(n: int, k: int, m: int, average_points_per_cell: int, kappa: float):
    """
    Performance test comparing ApproximateTree with brute-force approach for varying number of points.
    """

    # Generate uniformly distributed points
    key = random.PRNGKey(0)
    points = random.uniform(key, (n, 2))

    # Test now with vmap test_points
    test_points = random.uniform(jax.random.PRNGKey(1), (m, 2))

    # ApproximateTree method
    approx_tree = ApproximateTreeNN2D(average_points_per_cell=average_points_per_cell, kappa=kappa)
    build_tree = jax.jit(approx_tree.build_tree).lower(points).compile()
    t0 = time.time()
    tree = build_tree(points)
    jax.block_until_ready(tree)
    tree_build_time = time.time() - t0

    query = jax.jit(jax.vmap(lambda test_point: approx_tree.query(tree, test_point, k))).lower(test_points).compile()

    t0 = time.time()
    distances_approx, indices_approx = query(test_points)
    jax.block_until_ready(distances_approx)
    approx_time = time.time() - t0

    print(
        f"n={n}, m={m}, k={k}, avg_points_per_cell={average_points_per_cell}, kappa={kappa}:\n"
        f"\tTree build time: {tree_build_time:.5f}s\n"
        f"\tApproximateTree time: {approx_time:.5f}s\n"
    )


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("m", [1, 100, 1000])
@pytest.mark.parametrize("n", [1000, 100000])
@pytest.mark.parametrize("average_points_per_cell", [9, 16, 25])
@pytest.mark.parametrize("kappa", [1., 5., 10.])
def test_performance_approx_nn_3d(n: int, k: int, m: int, average_points_per_cell: int, kappa: float):
    """
    Performance test comparing ApproximateTree with brute-force approach for varying number of points.
    """

    # Generate uniformly distributed points
    key = random.PRNGKey(0)
    points = random.uniform(key, (n, 3))

    # Test now with vmap test_points
    test_points = random.uniform(jax.random.PRNGKey(1), (m, 3))

    # ApproximateTree method
    approx_tree = ApproximateTreeNN3D(average_points_per_cell=average_points_per_cell, kappa=kappa)
    build_tree = jax.jit(approx_tree.build_tree).lower(points).compile()
    t0 = time.time()
    tree = build_tree(points)
    jax.block_until_ready(tree)
    tree_build_time = time.time() - t0

    query = jax.jit(jax.vmap(lambda test_point: approx_tree.query(tree, test_point, k))).lower(test_points).compile()

    t0 = time.time()
    distances_approx, indices_approx = query(test_points)
    jax.block_until_ready(distances_approx)
    approx_time = time.time() - t0

    # Brute-force method
    brute_force_nearest_neighbors_vmap = jax.jit(
        jax.vmap(lambda test_point: brute_force_nearest_neighbors(points, test_point, k))
    ).lower(test_points).compile()

    t0 = time.time()
    distances_brute, indices_brute = brute_force_nearest_neighbors_vmap(test_points)
    jax.block_until_ready(distances_brute)
    brute_time = time.time() - t0

    speedup = brute_time / approx_time
    index_error_rate = np.mean(indices_brute != indices_approx)
    mean_abs_error = np.mean(np.abs(distances_brute - distances_approx))
    rmse = np.sqrt(np.mean((distances_brute - distances_approx) ** 2))
    print(
        f"n={n}, m={m}, k={k}, avg_points_per_cell={average_points_per_cell}, kappa={kappa}:\n"
        f"\tTree build time: {tree_build_time:.5f}s\n"
        f"\tApproximateTree time: {approx_time:.5f}s\n"
        f"\tBrute-force time: {brute_time:.5f}s\n"
        f"\tSpeedup: {speedup:.2f}\n"
        f"\tIndex error rate: {index_error_rate:.2f}\n"
        f"\tMean absolute error: {mean_abs_error:.5f}\n"
        f"\tRMSE: {rmse:.5f}\n"
    )


@pytest.mark.parametrize("k", [3])
@pytest.mark.parametrize("m", [1000000])
@pytest.mark.parametrize("n", [50 * 50])
@pytest.mark.parametrize("average_points_per_cell", [9, 16, 25])
@pytest.mark.parametrize("kappa", [1., 5., 10.])
def test_performance_approx_nn_usecase_3d(n: int, k: int, m: int, average_points_per_cell: int, kappa: float):
    """
    Performance test comparing ApproximateTree with brute-force approach for varying number of points.
    """

    # Generate uniformly distributed points
    key = random.PRNGKey(0)
    points = random.uniform(key, (n, 3))

    # Test now with vmap test_points
    test_points = random.uniform(jax.random.PRNGKey(1), (m, 3))

    # ApproximateTree method
    approx_tree = ApproximateTreeNN3D(average_points_per_cell=average_points_per_cell, kappa=kappa)
    build_tree = jax.jit(approx_tree.build_tree).lower(points).compile()
    t0 = time.time()
    tree = build_tree(points)
    jax.block_until_ready(tree)
    tree_build_time = time.time() - t0

    query = jax.jit(jax.vmap(lambda test_point: approx_tree.query(tree, test_point, k))).lower(test_points).compile()

    t0 = time.time()
    distances_approx, indices_approx = query(test_points)
    jax.block_until_ready(distances_approx)
    approx_time = time.time() - t0

    print(
        f"n={n}, m={m}, k={k}, avg_points_per_cell={average_points_per_cell}, kappa={kappa}:\n"
        f"\tTree build time: {tree_build_time:.5f}s\n"
        f"\tApproximateTree time: {approx_time:.5f}s\n"
    )


def test_kron_inv_impl_performance():
    def kron_inv_1(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        return kron_product(jnp.linalg.pinv(a), K, jnp.linalg.pinv(c))

    def kron_inv_2(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        return kron_product(jnp.linalg.inv(a), K, jnp.linalg.inv(c))

    def kron_inv_3(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        return unvec(jnp.sum(jnp.linalg.inv(kron(c.T, a)) * vec(K), axis=-1), (a.shape[0], c.shape[1]))

    def kron_inv_4(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        return unvec(jnp.linalg.solve(kron(c.T, a), vec(K)), (a.shape[0], c.shape[1]))

    def kron_inv_2x2(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        # Matrix([[(-c10*(K01*a11 - K11*a01) + c11*(K00*a11 - K10*a01))/((a00*a11 - a01*a10)*(c00*c11 - c01*c10)), (c00*(K01*a11 - K11*a01) - c01*(K00*a11 - K10*a01))/((a00*a11 - a01*a10)*(c00*c11 - c01*c10))], [(c10*(K01*a10 - K11*a00) - c11*(K00*a10 - K10*a00))/((a00*a11 - a01*a10)*(c00*c11 - c01*c10)), (-c00*(K01*a10 - K11*a00) + c01*(K00*a10 - K10*a00))/((a00*a11 - a01*a10)*(c00*c11 - c01*c10))]])
        # CSE:
        # ([(x0, K01*a11 - K11*a01), (x1, K00*a11 - K10*a01), (x2, 1/((a00*a11 - a01*a10)*(c00*c11 - c01*c10))), (x3, K01*a10 - K11*a00), (x4, K00*a10 - K10*a00)], [Matrix([
        # [x2*(-c10*x0 + c11*x1),  x2*(c00*x0 - c01*x1)],
        # [ x2*(c10*x3 - c11*x4), x2*(-c00*x3 + c01*x4)]])])
        a00, a01, a10, a11 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        K00, K01, K10, K11 = K[0, 0], K[0, 1], K[1, 0], K[1, 1]
        c00, c01, c10, c11 = c[0, 0], c[0, 1], c[1, 0], c[1, 1]
        x0 = K01 * a11 - K11 * a01
        x1 = K00 * a11 - K10 * a01
        x2 = jnp.reciprocal((a00 * a11 - a01 * a10) * (c00 * c11 - c01 * c10))
        x3 = K01 * a10 - K11 * a00
        x4 = K00 * a10 - K10 * a00

        # flat = jnp.stack([x2 * (-c10 * x0 + c11 * x1), x2 * (c10 * x3 - c11 * x4), x2 * (c00 * x0 - c01 * x1),  x2 * (-c00 * x3 + c01 * x4)], axis=-1)
        # return unvec(flat, (2, 2))
        flat = jnp.stack([x2 * (-c10 * x0 + c11 * x1), x2 * (c00 * x0 - c01 * x1), x2 * (c10 * x3 - c11 * x4),
                          x2 * (-c00 * x3 + c01 * x4)], axis=-1)
        return lax.reshape(flat, (2, 2))

    fns = [kron_inv_1, kron_inv_2, kron_inv_3, kron_inv_4]

    # for s in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    for s in [2]:
        print(f"s={s}:")
        n, m, p, q = s, s, s, s
        _fns = fns
        if s == 2:
            _fns = _fns + [kron_inv_2x2]
        for fn in _fns:
            a = jax.random.normal(jax.random.PRNGKey(0), (n, m))
            b = jax.random.normal(jax.random.PRNGKey(1), (m, p))
            c = jax.random.normal(jax.random.PRNGKey(2), (p, q))

            K = kron_product(a, b, c)
            b_res = fn(a, K, c)
            # np.testing.assert_allclose(b_res, b, atol=2e-6)

            [analysis] = jax.jit(fn).lower(a, K, c).compile().cost_analysis()
            # print(analysis)

            # Now look at vmap
            B = 1000000
            a = jax.random.normal(jax.random.PRNGKey(0), (B, n, m))
            b = jax.random.normal(jax.random.PRNGKey(1), (B, m, p))
            c = jax.random.normal(jax.random.PRNGKey(2), (B, p, q))

            K = jax.vmap(kron_product)(a, b, c)

            b_res = jax.vmap(fn)(a, K, c)
            max_error = jnp.max(jnp.abs(b_res - b))

            f_jit = jax.jit(jax.vmap(fn)).lower(a, K, c).compile()
            t0 = time.time()
            for _ in range(10):
                f_jit(a, K, c).block_until_ready()
            t1 = time.time()
            run_time = (t1 - t0) / 10.
            [batch_analysis] = f_jit.cost_analysis()
            print(
                f"\t{fn.__name__}: max error={max_error} run_time={run_time} BA={analysis['bytes accessed']}, F={analysis['flops']} BA(B)={batch_analysis['bytes accessed']}, F(B)={batch_analysis['flops']}")


def test_kron_product_impl_performance():
    def kron_product_1(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
        return a @ b @ c

    def kron_product_2(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
        return unvec(kron(c.T, a) @ vec(b), (a.shape[0], c.shape[1]))

    def kron_product_3(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
        return unvec(jnp.sum(kron(c.T, a) * vec(b), axis=-1), (a.shape[0], c.shape[1]))

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
        return lax.reshape(flat, (2, 2))

    fns = [kron_product_1, kron_product_2, kron_product_3]

    # for s in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    for s in [2]:
        print(f"s={s}:")
        n, m, p, q = s, s, s, s
        _fns = fns
        if s == 2:
            _fns = _fns + [kron_product_2x2]

        for fn in _fns:
            a = jax.random.normal(jax.random.PRNGKey(0), (n, m))
            b = jax.random.normal(jax.random.PRNGKey(1), (m, p))
            c = jax.random.normal(jax.random.PRNGKey(2), (p, q))

            [analysis] = jax.jit(fn).lower(a, b, c).compile().cost_analysis()
            # print(analysis)

            # Now look at vmap
            B = 1000000
            a = jax.random.normal(jax.random.PRNGKey(0), (B, n, m))
            b = jax.random.normal(jax.random.PRNGKey(1), (B, m, p))
            c = jax.random.normal(jax.random.PRNGKey(2), (B, p, q))

            K = jax.vmap(lambda a, b, c: a @ b @ c)(a, b, c)

            K_res = jax.vmap(fn)(a, b, c)
            max_error = jnp.max(jnp.abs(K_res - K))
            f_jit = jax.jit(jax.vmap(fn)).lower(a, b, c).compile()
            t0 = time.time()
            for _ in range(10):
                f_jit(a, b, c).block_until_ready()
            t1 = time.time()
            run_time = (t1 - t0) / 10.
            [batch_analysis] = f_jit.cost_analysis()
            print(
                f"\t{fn.__name__}: max error={max_error} run_time={run_time} BA={analysis['bytes accessed']}, F={analysis['flops']} BA(B)={batch_analysis['bytes accessed']}, F(B)={batch_analysis['flops']}")
