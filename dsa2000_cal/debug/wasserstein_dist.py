# Install necessary libraries
# You might need to install JAX and ott-jax if not already installed
# Uncomment and run the following lines in your environment:

# !pip install --upgrade pip
# !pip install jax jaxlib  # For CPU version
# !pip install ott-jax

import jax.numpy as jnp
import numpy as np
from ott.geometry import pointcloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear import sinkhorn

if __name__ == '__main__':
    # Generate two sets of N-dimensional points
    N = 1000  # Number of points in each set
    dim = 5  # Dimension of the space

    # Set random seed for reproducibility
    rng = np.random.default_rng(42)

    # First point cloud sampled from a standard normal distribution
    x = rng.normal(size=(N, dim))

    # Second point cloud sampled from a normal distribution with different mean and variance
    y = rng.normal(loc=0.0, scale=1., size=(N, dim))

    # Convert numpy arrays to JAX arrays
    x_jax = jnp.array(x)
    y_jax = jnp.array(y)

    # Create a point cloud geometry object
    geom = pointcloud.PointCloud(x_jax, x_jax, epsilon=None)

    # Run the Sinkhorn algorithm to compute the optimal transport plan
    problem = LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    ot_output = solver(problem)

    # Extract the regularized optimal transport cost (approximate Wasserstein distance)
    distance = ot_output.reg_ot_cost

    print(f"Approximate Wasserstein {ot_output.converged} distance between x and y: {distance}")
