import jax
import numpy as np
import pytest
from jax import random as random, numpy as jnp
from jax._src.tree_util import Partial

from dsa2000_cal.common.nearest_neighbours import ApproximateTreeNN2D, GridTree2D, ApproximateTreeNN3D, GridTree3D


@pytest.fixture
def setup_tree_2d():
    approx_tree = ApproximateTreeNN2D(average_points_per_cell=16, kappa=5.0)
    return approx_tree


def test_build_tree_2d(setup_tree_2d):
    approx_tree = setup_tree_2d
    n_points = 100
    points = random.uniform(random.PRNGKey(0), (n_points, 2))

    tree = approx_tree.build_tree(points)

    assert isinstance(tree, GridTree2D)
    assert tree.grid.shape == (4, 25 + 5 * np.sqrt(25))
    assert jnp.all(tree.points == points)

    tree = jax.jit(approx_tree.build_tree)(points)

    assert isinstance(tree, GridTree2D)
    assert tree.grid.shape == (4, 25 + 5 * np.sqrt(25))
    assert jnp.all(tree.points == points)


def test_query_within_single_cell_2d(setup_tree_2d):
    approx_tree = setup_tree_2d
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


def test_query_no_points_in_cell_2d(setup_tree_2d):
    approx_tree = setup_tree_2d
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


def test_query_with_exactly_k_points_2d(setup_tree_2d):
    approx_tree = setup_tree_2d
    points = jnp.array([[0.1, 0.1], [0.15, 0.15], [0.2, 0.2]])

    tree = approx_tree.build_tree(points)
    test_point = jnp.array([0.12, 0.12])
    k = 3

    distances, indices = approx_tree.query(tree, test_point, k)

    assert len(distances) == k
    assert len(indices) == k
    assert jnp.all(indices < len(points))


def test_query_nearest_neighbors_on_boundary_2d(setup_tree_2d):
    approx_tree = setup_tree_2d
    points = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    tree = approx_tree.build_tree(points)
    test_point = jnp.array([0.5, 0.5])
    k = 1

    distances, indices = approx_tree.query(tree, test_point, k)

    assert len(distances) == k
    assert len(indices) == k
    assert indices[0] == 1
    assert jnp.allclose(distances[0], 0.0)


def test_build_tree_handles_empty_points_2d(setup_tree_2d):
    approx_tree = setup_tree_2d
    points = jnp.array([]).reshape(0, 2)
    with pytest.raises(ValueError, match="No points provided to build the tree."):
        _ = approx_tree.build_tree(points)


@pytest.fixture
def setup_tree_3d():
    approx_tree = ApproximateTreeNN3D(average_points_per_cell=16, kappa=5.0)
    return approx_tree


def test_build_tree_3d(setup_tree_3d):
    approx_tree = setup_tree_3d
    n_points = 100
    points = random.uniform(random.PRNGKey(0), (n_points, 3))

    tree = approx_tree.build_tree(points)

    assert isinstance(tree, GridTree3D)
    assert tree.grid.shape == (4, 25 + 5 * np.cbrt(25))  # Adjusted for 3D
    assert jnp.all(tree.points == points)

    tree = jax.jit(approx_tree.build_tree)(points)

    assert isinstance(tree, GridTree3D)
    assert tree.grid.shape == (4, 25 + 5 * np.cbrt(25))  # Adjusted for 3D
    assert jnp.all(tree.points == points)


def test_query_within_single_cell_3d(setup_tree_3d):
    approx_tree = setup_tree_3d
    points = jnp.array([[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]])

    tree = approx_tree.build_tree(points)
    test_point = jnp.array([0.12, 0.12, 0.12])
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


def test_query_no_points_in_cell_3d(setup_tree_3d):
    approx_tree = setup_tree_3d
    points = jnp.array([[0.8, 0.8, 0.8], [0.9, 0.9, 0.9], [0.85, 0.85, 0.85]])

    tree = approx_tree.build_tree(points)
    tree = tree._replace(grid=tree.grid.at[0, :].set(-1))
    test_point = jnp.array([0.1, 0.1, 0.1])
    k = 2

    distances, indices = approx_tree.query(tree, test_point, k)

    assert distances.size == 2
    assert indices.size == 2

    np.testing.assert_allclose(distances, jnp.inf)
    np.testing.assert_allclose(indices, -1)


def test_query_with_exactly_k_points_3d(setup_tree_3d):
    approx_tree = setup_tree_3d
    points = jnp.array([[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]])

    tree = approx_tree.build_tree(points)
    test_point = jnp.array([0.12, 0.12, 0.12])
    k = 3

    distances, indices = approx_tree.query(tree, test_point, k)

    assert len(distances) == k
    assert len(indices) == k
    assert jnp.all(indices < len(points))


def test_query_nearest_neighbors_on_boundary_3d(setup_tree_3d):
    approx_tree = setup_tree_3d
    points = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])

    tree = approx_tree.build_tree(points)
    test_point = jnp.array([0.5, 0.5, 0.5])
    k = 1

    distances, indices = approx_tree.query(tree, test_point, k)

    assert len(distances) == k
    assert len(indices) == k
    assert indices[0] == 1
    assert jnp.allclose(distances[0], 0.0)


def test_build_tree_handles_empty_points_3d(setup_tree_3d):
    approx_tree = setup_tree_3d
    points = jnp.array([]).reshape(0, 3)  # Adjusted for 3D
    with pytest.raises(ValueError, match="No points provided to build the tree."):
        _ = approx_tree.build_tree(points)
