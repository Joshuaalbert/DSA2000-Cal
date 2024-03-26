from dataclasses import dataclass

import jax

jax.config.update('jax_threefry_partitionable', True)

from jax import numpy as jnp

from jax.sharding import PartitionSpec

P = PartitionSpec

c = 2.99792458e8
two_pi_over_c = 2 * jnp.pi / c # Casa convention
minus_two_pi_over_c = -two_pi_over_c # Fourier convention



@dataclass(eq=False)
class Gridder:
    """
    Class to grid visibilities.
    """

    def get_support_indices(self, u: jnp.ndarray, v: jnp.ndarray, support: jnp.ndarray) -> jnp.ndarray:
        """
        Get the indices of the grid points within a certain distance from each uv coordinate.

        Args:
            u: [n_l] u coordinates of grid (in lambda units)
            v: [n_m] v coordinates of grid (in lambda units)
            support: [n_l, n_m] support in lambda units

        Returns:
            row_indices: [n_l, n_m, ...] row indices where distance in u,v is less than support
        """
        pass

    def grid(self, visibilities_degridded: jnp.ndarray,
             uvw: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray,
             baseline_idx: jnp.ndarray,
             kernel: jnp.ndarray,
             support: jnp.ndarray) -> jnp.ndarray:
        """
        Grid visibilities onto a grid specified.

        Args:
            visibilities_degridded: [row]
            uvw: [row, 3] uvw coordinates of `visibilities_degridded` (in lambda units)
            u: [n_l] u coordinates of grid (in lambda units)
            v: [n_m] v coordinates of grid (in lambda units)
            baseline_idx: [row] baseline index
            kernel: [n_ij, n_l', n_m']
            support:

        Returns:
            visibilities: [n_l, n_m]
        """

        # V_G[u, v, 0] = sum_row V_D[u[row], v[row], w[row]] * kernel(ij[row], u, v)
        U, V = jnp.meshgrid(u, v, indexing='ij')  # [n_l, n_m]

        # row_indices where distance in u,v is less than support
        row_indices = ...  # [n_l, n_m, ...]
        def grid_point(_u, _v):
            ...

        # kernel = jax.vmap(C, in_axes=(0, None, None, None))(baseline_idx, u, v, uvw[:, 2])  # [row, n_l, n_m]



        # We find the uvw within a certain distance from each grid point

