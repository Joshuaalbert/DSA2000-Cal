import dataclasses
import warnings
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class GridTree(NamedTuple):
    grid: jax.Array  # [num_grids, max_points_per_cell]
    points: jax.Array  # [n_points, 2]
    extent: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]  # [4] (min_x, max_x, min_y, max_y)


@dataclasses.dataclass(eq=False)
class ApproximateTreeNN:
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

    def build_tree(self, points: jax.Array) -> GridTree:
        """
        Builds the tree structure given the points in the space [a,b]x[c,d].

        Parameters:
            points (jax.numpy.ndarray): Array of points with shape (n_points, 2).

        Returns:
            GridTree: A named tuple containing the grid, grid size, max points per cell, and the original points.
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
        return GridTree(grid=grid, points=points, extent=extent)

    def query(self, tree: GridTree, test_point: jax.Array, k: int = 1) -> Tuple[jax.Array, jax.Array]:
        """
        Queries the tree structure to find the k nearest neighbors to the test point.

        Parameters:
            tree (GridTree): The tree structure built by the `build_tree` method.
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