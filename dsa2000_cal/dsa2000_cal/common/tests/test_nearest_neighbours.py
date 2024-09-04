from typing import Tuple

import jax
import numpy as np
import pytest
from jax import random as random, numpy as jnp
from jax._src.tree_util import Partial

from dsa2000_cal.common.nearest_neighbours import ApproximateTreeNN, GridTree


@pytest.fixture
def setup_tree():
    approx_tree = ApproximateTreeNN(average_points_per_cell=16, kappa=5.0)
    return approx_tree


def test_build_tree(setup_tree):
    approx_tree = setup_tree
    n_points = 100
    points = random.uniform(random.PRNGKey(0), (n_points, 2))

    tree = approx_tree.build_tree(points)

    assert isinstance(tree, GridTree)
    assert tree.grid.shape == (4, 25 + 5 * np.sqrt(25))
    assert jnp.all(tree.points == points)

    tree = jax.jit(approx_tree.build_tree)(points)

    assert isinstance(tree, GridTree)
    assert tree.grid.shape == (4, 25 + 5 * np.sqrt(25))
    assert jnp.all(tree.points == points)


def test_query_within_single_cell(setup_tree):
    approx_tree = setup_tree
    points = jnp.array([[0.1, 0.1], [0.15, 0.15], [0.2, 0.2]])

    tree = approx_tree.build_tree(points)
    test_point = jnp.array([0.12, 0.12])
    k = 2

    distances, indices = approx_tree.query(tree, test_point, k)

    assert len(distances) == k
    assert len(indices) == k
    assert indices[0] in [0, 1, 2]
    assert jnp.allclose(jnp.sort(distances), jnp.sort(jnp.linalg.norm(points - test_point, axis=1)[:k]))

    distances, indices = jax.jit(Partial(approx_tree.query, k=k))(tree, test_point)

    assert len(distances) == k
    assert len(indices) == k
    assert indices[0] in [0, 1, 2]
    assert jnp.allclose(jnp.sort(distances), jnp.sort(jnp.linalg.norm(points - test_point, axis=1)[:k]))


def test_query_no_points_in_cell(setup_tree):
    approx_tree = setup_tree
    points = jnp.array([[0.8, 0.8], [0.9, 0.9], [0.85, 0.85]])

    tree = approx_tree.build_tree(points)
    tree = tree._replace(grid=tree.grid.at[0, :].set(-1))
    test_point = jnp.array([0.1, 0.1])
    k = 2

    distances, indices = approx_tree.query(tree, test_point, k)

    assert distances.size == 2
    assert indices.size == 2

    np.testing.assert_allclose(distances, jnp.inf)
    np.testing.assert_allclose(indices, -1)


def test_query_with_exactly_k_points(setup_tree):
    approx_tree = setup_tree
    points = jnp.array([[0.1, 0.1], [0.15, 0.15], [0.2, 0.2]])

    tree = approx_tree.build_tree(points)
    test_point = jnp.array([0.12, 0.12])
    k = 3

    distances, indices = approx_tree.query(tree, test_point, k)

    assert len(distances) == k
    assert len(indices) == k
    assert jnp.all(indices < len(points))


def test_query_nearest_neighbors_on_boundary(setup_tree):
    approx_tree = setup_tree
    points = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    tree = approx_tree.build_tree(points)
    test_point = jnp.array([0.5, 0.5])
    k = 1

    distances, indices = approx_tree.query(tree, test_point, k)

    assert len(distances) == k
    assert len(indices) == k
    assert indices[0] == 1
    assert jnp.allclose(distances[0], 0.0)


def test_build_tree_handles_empty_points(setup_tree):
    approx_tree = setup_tree
    points = jnp.array([]).reshape(0, 2)
    with pytest.raises(ValueError, match="No points provided to build the tree."):
        _ = approx_tree.build_tree(points)


def brute_force_nearest_neighbors(points: jnp.ndarray, test_point: jnp.ndarray, k: int) -> Tuple[
    jnp.ndarray, jnp.ndarray]:
    """
    A brute-force approach to find the k nearest neighbors to a test point.

    Parameters:
        points (jax.numpy.ndarray): Array of points with shape (n_points, 2).
        test_point (jax.numpy.ndarray): A point in [0,1]^2 with shape (2,).
        k (int): The number of nearest neighbors to find.

    Returns:
        distances (jax.numpy.ndarray): Distances to the k nearest neighbors.
        indices (jax.numpy.ndarray): Indices of the k nearest neighbors.
    """
    distances = jnp.linalg.norm(points - test_point, axis=1)
    top_k_neg_distances, top_k_indices = jax.lax.top_k(-distances, k)
    return -top_k_neg_distances, top_k_indices


