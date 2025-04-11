import time

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from dsa2000_common.common.sum_utils import scan_sum
from dsa2000_fm.array_layout.fast_psf_evaluation import project_antennas, rotation_matrix_change_dec

tfpd = tfp.distributions


def wasserstein_point_clouds(x, y, p=1):
    """
    Compute the Wasserstein distance between two point clouds.

    Args:
        x: [N, D] points in first point cloud with uniform weight
        y: [M, D] points in second point cloud with uniform weight
        p: the weight norm

    Returns:
        a cost measure of matching
    """
    cost_matrix = cdist(x, y, metric='minkowski', p=p)

    # use linear assignment
    assignment = linear_sum_assignment(cost_matrix)

    min_flow = cost_matrix[assignment].sum() / assignment[0].shape[0]

    return min_flow


def test_wasserstein_point_clouds():
    x = y = np.array([[0, 0], [1, 1], [2, 2]])
    assert wasserstein_point_clouds(x, y) == 0.0
    x = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([[0, 0], [1, 1], [3, 3]])
    assert wasserstein_point_clouds(x, y) > 0.0

    x = np.random.normal(size=(2000, 2))
    y = np.random.normal(size=(1000, 2))
    t0 = time.time()
    dist = wasserstein_point_clouds(x, y)
    print(dist)
    print(time.time() - t0)


def wasserstein_1D(x, y, x_weights, y_weights):
    """
    Compute the Wasserstein distance between two 1D point clouds.

    Args:
        x: [N] points in first point cloud with uniform weight
        y: [N] points in second point cloud with uniform weight

    Returns:
        a cost measure of matching
    """
    return _cdf_distance_jax(1, x, y, x_weights, y_weights)


def weighted_wasserstein_1d_gaussian(empirical_data, weights, target_mean, target_std, p=1):
    """
    Computes the 1D p-Wasserstein distance between a weighted empirical distribution and a Gaussian.

    Parameters:
        empirical_data: array of projected empirical values.
        weights: array of weights associated with the empirical data (should sum to 1).
        target_mean: mean of the target Gaussian.
        target_std: standard deviation of the target Gaussian.
        p: order of the Wasserstein distance (typically 1 or 2).
    Returns:
        The p-Wasserstein distance.
    """
    # Sort the data and corresponding weights
    sort_idx = jnp.argsort(empirical_data)
    sorted_data = empirical_data[sort_idx]

    if weights is not None:
        sorted_weights = weights[sort_idx]

        # Compute cumulative weights (the weighted quantile levels)
        cumulative_weights_ = jnp.cumsum(sorted_weights)
        cumulative_weights_ /= cumulative_weights_[-1]  # Normalize to sum to 1

        # Prepend 0 for the integration measure differences
        _cumulative_weights = jnp.concatenate([jnp.array([0.], cumulative_weights_.dtype), cumulative_weights_[:-1]])
        quantile_levels = 0.5 * (cumulative_weights_ + _cumulative_weights)
        # Compute target quantiles at these levels (excluding the initial 0)
        mass_differences = cumulative_weights_ - _cumulative_weights
    else:
        quantile_levels = ((jnp.arange(sorted_data.shape[0]) + 0.5) / sorted_data.shape[0]).astype(sorted_data.dtype)
        mass_differences = jnp.asarray(1 / sorted_data.shape[0], sorted_data.dtype)

    target_dist = tfpd.Normal(loc=target_mean, scale=target_std)
    target_quantiles = target_dist.quantile(quantile_levels)

    # Approximate the Wasserstein distance using a Riemann sum
    # Differences in cumulative weights give the "mass" associated to each sample gap.

    print(sorted_data.shape, target_quantiles.shape, mass_differences.shape)
    if p == 1:
        return jnp.sum(jnp.multiply(jnp.abs(sorted_data - target_quantiles), mass_differences))
    if p == 2:
        return jnp.sqrt(jnp.sum(jnp.multiply(jnp.square(sorted_data - target_quantiles), mass_differences)))

    distance = np.sum(np.abs(sorted_data - target_quantiles) ** p * mass_differences)
    return distance ** (1 / p)


def sliced_wasserstein(key, x, y, x_weights, y_weights, num_samples: int):
    """
    Compute the sliced Wasserstein distance between two point clouds.
    This is a Monte Carlo approximation of the Wasserstein distance.

    Args:
        key: PRNGKey
        x: [N, D] points in first point cloud with uniform weight
        y: [M, D] points in second point cloud with uniform weight
        x_weights: [N] weights for the first point cloud
        y_weights: [M] weights for the second point cloud
        num_samples: number of samples
        p: the weight norm

    Returns:
        scalar
    """
    D = np.shape(x)[1]

    def accum_fn(key):
        proj = jax.random.normal(key, shape=(D,))
        proj /= jnp.linalg.norm(proj)
        x_proj = x @ proj
        y_proj = y @ proj
        return wasserstein_1D(x_proj, y_proj, x_weights, y_weights)

    sum = scan_sum(accum_fn, jnp.zeros(()), jax.random.split(key, num_samples), unroll=1)
    return sum / num_samples


def sliced_wasserstein_gaussian(key, x, x_weights, target_mean, target_Sigma, num_samples: int, p=1, unroll=1):
    """
    Compute the sliced Wasserstein distance between two point clouds.
    This is a Monte Carlo approximation of the Wasserstein distance.

    Args:
        key: PRNGKey
        x: [N, D] points in first point cloud with uniform weight
        x_weights: [N] weights for the first point cloud
        target_mean: [D] the target mean or none
        target_Sigma: [D, D] the target covariance
        num_samples: number of samples
        p: the weight norm

    Returns:
        scalar
    """
    D = np.shape(x)[1]

    def accum_fn(key):
        proj = jax.random.normal(key, shape=(D,))
        proj /= jnp.linalg.norm(proj)
        x_proj = x @ proj
        if target_mean is not None:
            mean_proj = proj @ target_mean
        else:
            mean_proj = 0
        Sigma_proj = proj @ target_Sigma @ proj.T
        return weighted_wasserstein_1d_gaussian(x_proj, x_weights, target_mean=mean_proj, target_std=Sigma_proj, p=p)

    sum = scan_sum(accum_fn, jnp.zeros(()), jax.random.split(key, num_samples), unroll=unroll)
    return sum / num_samples


def wasserstein_uvw_vs_gaussian(key, uvw, weights, sigma, latitude, transit_dec, num_samples, p=1, unroll=1):
    """
    Compute the sliced wasserstein distance between a 2D set of weighted saamples and a 2D gaussian with
    optional dec tilt.
    This is a Monte Carlo approximation of the Wasserstein distance.

    Args:
        key: PRNGKey
        uvw: [N, 3] points in first point cloud with uniform weight
        weights: [N] the weights or None
        sigma: the circular sigma at zenith
        latitude: the latitude
        transit_dec: the transit DEC of the array
        num_samples: num slices
        p: p norm

    Returns:
        the distance
    """

    uvw_rotated = project_antennas(uvw, latitude, transit_dec)
    R = rotation_matrix_change_dec(transit_dec - latitude)
    # mean -> R.mean = 0
    # sigma -> R.sigma.R^T
    target_mean = None
    target_Sigma = R @ jnp.diag(jnp.array([sigma, sigma, 0.])) @ R.T

    return sliced_wasserstein_gaussian(
        key, uvw_rotated, weights, target_mean=target_mean, target_Sigma=target_Sigma,
        p=p, num_samples=num_samples, unroll=unroll
    )


def test_sliced_wasserstein():
    # 2D gaussians same mean, sigma -> 0
    # assert sliced_wasserstein(jax.random.PRNGKey(0), np.random.normal(size=(2000, 2)), np.random.normal(size=(2000, 2)), 100) < sliced_wasserstein(jax.random.PRNGKey(0), np.random.normal(size=(100, 2)), np.random.normal(size=(100, 2)), 100)

    # Different means
    x = np.random.normal(size=(2000, 2))
    y = np.random.normal(size=(2000, 2)) + 2

    dist = sliced_wasserstein(jax.random.PRNGKey(0), x, y,None, None, 100)
    dist_2 = sliced_wasserstein_gaussian(jax.random.PRNGKey(0), y, None, target_mean=2*jnp.ones(2), target_Sigma=jnp.eye(2), num_samples=100, p=1)
    print(dist, dist_2)



def _cdf_distance_np(p, u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p), deltas)), 1 / p)


def _cdf_distance_jax(p, u_values, v_values, u_weights=None, v_weights=None):
    u_sorter = jnp.argsort(u_values)
    v_sorter = jnp.argsort(v_values)
    all_values = jnp.concatenate((u_values, v_values))
    all_values = jnp.sort(all_values, stable=True)
    deltas = jnp.diff(all_values)
    u_cdf_indices = jnp.searchsorted(u_values[u_sorter], all_values[:-1], 'right')
    v_cdf_indices = jnp.searchsorted(v_values[v_sorter], all_values[:-1], 'right')
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = jnp.concatenate((jnp.array([0.], u_weights.dtype), jnp.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = jnp.concatenate((jnp.array([0.], u_weights.dtype), jnp.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    if p == 1:
        return jnp.sum(jnp.multiply(jnp.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return jnp.sqrt(jnp.sum(jnp.multiply(jnp.square(u_cdf - v_cdf), deltas)))
    return jnp.power(jnp.sum(jnp.multiply(jnp.power(jnp.abs(u_cdf - v_cdf), p), deltas)), 1 / p)


def test_cdf_distance():
    p = 1
    u = np.random.normal(size=(2000,))
    v = np.random.normal(size=(1000,))
    u_weights = np.random.uniform(size=(2000,))
    v_weights = np.random.uniform(size=(1000,))
    np.testing.assert_allclose(_cdf_distance_jax(p, u, v, u_weights, v_weights),
                               _cdf_distance_np(p, u, v, u_weights, v_weights), atol=1e-6)
