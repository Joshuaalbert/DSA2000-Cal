import dataclasses
import warnings
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import KDTree

from dsa2000_cal.common.types import int_type


class GridTree2D(NamedTuple):
    grid: jax.Array  # [num_grids, max_points_per_cell]
    points: jax.Array  # [n_points, 2]
    extent: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]  # [4] (min_x, max_x, min_y, max_y)


@dataclasses.dataclass(eq=False)
class ApproximateTreeNN2D:
    """
    Approximate tree for nearest neighbor search on 2D box.

    A tree structure is used to find the k nearest neighbors to a given point in 2D space, by constructing a grid of
    shape (n_grid, n_grid) where,

    n_grid = int(sqrt(n / average_points_per_cell)), where n is the number of points.

    The memory usage goes as O(n * ( 2 + kappa / sqrt(average_points_per_cell))).

    The tree build time goes as O(n).

    The tree query time goes as O((average_points_per_cell + kappa * sqrt(average_points_per_cell)) + k log k)

    Accuracy generally increases with more points in the tree. Accuracy decreasses with larger `k` queries due to cell
    edge effects.

    Args:
        average_points_per_cell: Average number of points per cell in the grid.
        kappa: how many sigmas above the expected number of points per cell to allow.
    """
    average_points_per_cell: int = 16
    kappa: float = 5.0

    def _point_to_cell(self, point: jax.Array, n_grid: int,
                       extent: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:
        x_min, x_max, y_min, y_max = extent
        # x = point[0]
        # y = point[1]
        # cell_x = jnp.floor(x * n_grid).astype(int)
        # cell_y = jnp.floor(y * n_grid).astype(int)
        # return cell_x, cell_y
        cell_x = jnp.clip(jnp.floor((point[0] - x_min) / (x_max - x_min) * n_grid), 0, n_grid - 1).astype(int)
        cell_y = jnp.clip(jnp.floor((point[1] - y_min) / (y_max - y_min) * n_grid), 0, n_grid - 1).astype(int)
        return cell_x, cell_y

    def _grid_to_idx(self, cell_x: jax.Array, cell_y: jax.Array, n_grid: int) -> jax.Array:
        """Maps (cell_x, cell_y) to grid_idx."""
        return cell_y * n_grid + cell_x

    def _idx_to_grid(self, grid_idx: jax.Array, n_grid: int) -> Tuple[jax.Array, jax.Array]:
        """Maps grid_idx back to (cell_x, cell_y)."""
        cell_x = grid_idx % n_grid
        cell_y = grid_idx // n_grid
        return cell_x, cell_y

    def build_tree(self, points: jax.Array) -> GridTree2D:
        """
        Builds the tree structure given the points in the space [a,b]x[c,d].

        Parameters:
            points (jax.numpy.ndarray): Array of points with shape (n_points, 2).

        Returns:
            GridTree2D: A named tuple containing the grid, grid size, max points per cell, and the original points.
        """
        n_points = points.shape[0]
        if n_points == 0:
            raise ValueError("No points provided to build the tree.")
        n_grid = int(np.sqrt(n_points / self.average_points_per_cell))
        if n_grid < 1:
            warnings.warn("Number of points is too small to meet desired average points per cell.")
            n_grid = 1
        num_cells = n_grid * n_grid

        max_points_per_cell = int(
            n_points / num_cells + self.kappa * np.sqrt(n_points / num_cells)
        )
        if max_points_per_cell < 1:
            raise ValueError("max_points_per_cell must be at least 1.")

        grid = -1 * jnp.ones((num_cells, max_points_per_cell), dtype=int)
        storage_indices = jnp.zeros(num_cells, dtype=int)  # To track where to store the next point in each grid
        points_min = jnp.min(points, axis=0)
        points_max = jnp.max(points, axis=0)
        extent = (points_min[0], points_max[0], points_min[1], points_max[1])

        def assign_point(i, state):
            grid, storage_indices = state
            point = points[i]
            cell_x, cell_y = self._point_to_cell(point, n_grid, extent)
            grid_idx = self._grid_to_idx(cell_x, cell_y, n_grid)

            storage_index = storage_indices[grid_idx]
            grid = grid.at[grid_idx, storage_index].set(i)
            storage_indices = storage_indices.at[grid_idx].set((storage_index + 1) % max_points_per_cell)

            return grid, storage_indices

        grid, storage_indices = jax.lax.fori_loop(0, n_points, assign_point, (grid, storage_indices))

        # Some cells may not have any points, thus test points that fall within that cell will have no neighbors.
        # We can solve this and improve edge effects by filling up all -1 with random points from neighboring cells.

        def body(state):
            i, grid, storage_indices = state
            # Get a random neighbour for each unfilled point in each cell
            G, P = jnp.meshgrid(jnp.arange(np.shape(grid)[0]), jnp.arange(np.shape(grid)[1]), indexing='ij')
            cell_x, cell_y = self._idx_to_grid(G, n_grid)  # [num_cells, max_points_per_cell]
            neighbour_inc = jax.random.randint(
                jax.random.PRNGKey(42), np.shape(G) + (2,),
                -1, 2
            )  # [num_cells, max_points_per_cell, 2]
            neighbour_x = jnp.clip(cell_x + neighbour_inc[:, :, 0], 0, n_grid - 1)
            neighbour_y = jnp.clip(cell_y + neighbour_inc[:, :, 1], 0, n_grid - 1)
            neighbour_grid_idx = self._grid_to_idx(neighbour_x, neighbour_y, n_grid)  # [num_cells, max_points_per_cell]
            random_select = jax.random.randint(jax.random.PRNGKey(42), np.shape(G), 0,
                                               storage_indices[:, None])  # [num_cells, max_points_per_cell]
            random_neighbour = grid[neighbour_grid_idx, random_select]  # [num_cells, max_points_per_cell]

            # Check that random neighbour is not already in the cell it would go to
            @partial(jax.vmap, in_axes=(0, 0))
            @partial(jax.vmap, in_axes=(0, 0))
            def check_cell(i, j):
                return jnp.logical_not(jnp.any(grid[i] == random_neighbour[i, j]))

            replace = (grid == -1) & check_cell(G, P)

            grid = jnp.where(replace, random_neighbour, grid)
            grid = jnp.sort(grid, axis=1, descending=True)
            storage_indices = jnp.sum(grid != -1, axis=1)
            return i + 1, grid, storage_indices

        def cond(state):
            # Until all -1 are replaced
            i, grid, storage_indices = state
            return jnp.any(grid == -1) & (i < 10)

        _, grid, _ = jax.lax.while_loop(cond, body, (0, grid, storage_indices))

        return GridTree2D(grid=grid, points=points, extent=extent)

    def query(self, tree: GridTree2D, test_point: jax.Array, k: int = 1) -> Tuple[jax.Array, jax.Array]:
        """
        Queries the tree structure to find the k nearest neighbors to the test point.

        Parameters:
            tree (GridTree2D): The tree structure built by the `build_tree` method.
            test_point (jax.numpy.ndarray): A point in [a,b]x[c,d] with shape (2,).
            k (int): The number of nearest neighbors to find.

        Returns:
            distances (jax.numpy.ndarray): Distances to the k nearest neighbors.
            indices (jax.numpy.ndarray): Indices of the k nearest neighbors.
        """
        n_grid = int(np.sqrt(np.shape(tree.grid)[0]))
        cell_x, cell_y = self._point_to_cell(test_point, n_grid, tree.extent)
        grid_idx = self._grid_to_idx(cell_x, cell_y, n_grid)
        point_indices = tree.grid[grid_idx]  # [max_points_per_cell]
        points_in_cell = tree.points[point_indices]  # [max_points_per_cell, 2]

        # Gather valid indices using jax.numpy.where and take them instead of boolean indexing
        valid_mask = point_indices >= 0

        distances = jnp.linalg.norm(points_in_cell - test_point, axis=1)  # [max_points_per_cell]
        neg_distances = jnp.where(valid_mask, -distances, -jnp.inf)
        top_k_neg_distances, top_k_indices_within_cell = jax.lax.top_k(neg_distances, k)
        top_k_distances = -top_k_neg_distances

        # Return the actual distances and the corresponding indices in the original points array
        return top_k_distances, point_indices[top_k_indices_within_cell]


class GridTree3D(NamedTuple):
    grid: jax.Array  # [num_grids, max_points_per_cell]
    points: jax.Array  # [n_points, 3]
    extent: Tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]  # [6] (min_x, max_x, min_y, max_y, min_z, max_z)


def kd_tree_nn(points: jax.Array, test_points: jax.Array, k: int = 1) -> Tuple[jax.Array, jax.Array]:
    """
    Uses a KD-tree to find the k nearest neighbors to a test point in 3D space.

    Parameters:
        points: [n, d] Array of points.
        test_points: [m, d] points to query
        k: The number of nearest neighbors to find.

    Returns:
        distances: [m, k] Distances to the k nearest neighbors.
        indices: [m, k] Indices of the k nearest neighbors.
    """
    m, d = np.shape(test_points)
    k = int(k)
    args = (
        points,
        test_points,
        k
    )

    distance_shape_dtype = jax.ShapeDtypeStruct(
        shape=(m, k),
        dtype=points.dtype
    )
    index_shape_dtype = jax.ShapeDtypeStruct(
        shape=(m, k),
        dtype=int_type
    )

    return jax.pure_callback(_kd_tree_nn_host, (distance_shape_dtype, index_shape_dtype), *args)


def _kd_tree_nn_host(points: jax.Array, test_points: jax.Array, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses a KD-tree to find the k nearest neighbors to a test point in 3D space.

    Parameters:
        points: [n, d] Array of points.
        test_points: [m, d] points to query
        k: The number of nearest neighbors to find.

    Returns:
        distances: [m, k] Distances to the k nearest neighbors.
        indices: [m, k] Indices of the k nearest neighbors.
    """
    points, test_points = jax.tree.map(np.asarray, (points, test_points))
    k = int(k)
    tree = KDTree(points, compact_nodes=False, balanced_tree=False)
    if k == 1:
        distances, indices = tree.query(test_points, k=[1])  # unsqueeze k
    else:
        distances, indices = tree.query(test_points, k=k)
    return distances, indices.astype(int_type)
