from functools import partial

import astropy.constants as const
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax._src.scipy.signal import convolve2d
from scipy.optimize import brentq
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from dsa2000_common.common.interp_utils import InterpolatedArray
from dsa2000_common.common.quantity_utils import quantity_to_np, quantity_to_jnp
from dsa2000_common.common.sum_utils import scan_sum
from dsa2000_common.delay_models.uvw_utils import geometric_uvw_from_gcrs
from dsa2000_fm.array_layout.fast_psf_evaluation import project_antennas, rotation_matrix_change_dec
from dsa2000_fm.systematics.ionosphere import evolve_gcrs

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


def wasserstein_1D_p(p, x, y, x_weights, y_weights, resolution: int):
    # Sort
    x_sort = jnp.argsort(x)
    y_sort = jnp.argsort(y)
    x = x[x_sort]
    y = y[y_sort]
    q = (0.5 + jnp.arange(resolution + 1)) / (resolution + 1)
    dq = 1. / resolution
    if x_weights is not None:
        x_weights = x_weights[x_sort]
        x_cdf = jnp.cumsum(x_weights)
        x_cdf /= x_cdf[-1]  # Normalize to sum to 1
        x_interp = InterpolatedArray(x_cdf, x, regular_grid=False, check_spacing=False, clip_out_of_bounds=True)
    else:
        x_cdf = jnp.arange(1, 1 + len(x)) / len(x)
        x_interp = InterpolatedArray(x_cdf, x, regular_grid=True, check_spacing=False, clip_out_of_bounds=True)
    x_gridded = x_interp(q)
    if y_weights is not None:
        y_weights = y_weights[y_sort]
        y_cdf = jnp.cumsum(y_weights)
        y_cdf /= y_cdf[-1]  # Normalize to sum to 1
        y_interp = InterpolatedArray(y_cdf, y, regular_grid=False, check_spacing=False, clip_out_of_bounds=True)
    else:
        y_cdf = jnp.arange(1, 1 + len(y)) / len(y)
        y_interp = InterpolatedArray(y_cdf, y, regular_grid=True, check_spacing=False, clip_out_of_bounds=True)
    y_gridded = y_interp(q)
    if p == 1:
        d = jnp.sum(jnp.abs(x_gridded - y_gridded)) * dq
    elif p == 2:
        d = jnp.sqrt(jnp.sum(jnp.square(x_gridded - y_gridded)) * dq)
    else:
        d = (jnp.sum(jnp.abs(x_gridded - y_gridded) ** p) * dq) ** (1 / p)
    return d


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

    if p == 1:
        return jnp.sum(jnp.multiply(jnp.abs(sorted_data - target_quantiles), mass_differences))
    if p == 2:
        return jnp.sqrt(jnp.sum(jnp.multiply(jnp.square(sorted_data - target_quantiles), mass_differences)))

    distance = np.sum(np.abs(sorted_data - target_quantiles) ** p * mass_differences)
    return distance ** (1 / p)


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


def sliced_wasserstein(key, x, y, x_weights, y_weights, p=1, num_samples: int = 100, resolution: int = 100, unroll=1):
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
        return wasserstein_1D_p(p, x_proj, y_proj, x_weights, y_weights, resolution)

    sum = scan_sum(accum_fn, jnp.zeros(()), jax.random.split(key, num_samples), unroll=unroll)
    return sum / num_samples


def compute_uv_gaussian_width(target_psf_fwhm: au.Quantity, freq: au.Quantity) -> au.Quantity:
    wavelength = quantity_to_np(const.c / freq, 'm')
    gamma_lambda = np.sqrt(2 * np.log(2)) / (np.pi * quantity_to_np(target_psf_fwhm, 'rad'))
    gamma = (gamma_lambda * wavelength) * au.m
    return gamma


def compute_ideal_uv_distribution(du: au.Quantity, R: au.Quantity, target_fwhm: au.Quantity, freq: au.Quantity):
    """
    Compute the ideal UV distribution for a Gaussian beam.
    The UV distribution is a 2D Gaussian with a given FWHM and a circular aperture, constrained by the maximum
    baseline length.

    Args:
        du: the UV bin size in meters
        R: the maximum baseline length in meters
        target_fwhm: the target FWHM in radians
        freq: the frequency in Hz

    Returns:
        uv_bins: [M + 1] bins for the UV distribution
        uv_grid: [M, M, 2] UV coordinates in lambda
        target_dist: [M, M] target UV distribution
    """
    gamma = quantity_to_np(compute_uv_gaussian_width(target_fwhm, freq), 'm')
    R_jax = quantity_to_np(R, 'm')
    du_jax = quantity_to_np(du, 'm')
    uvec_bins = np.arange(-R_jax - du_jax, R_jax + du_jax, du_jax)
    uvec = 0.5 * (uvec_bins[1:] + uvec_bins[:-1])
    U, V = np.meshgrid(uvec, uvec, indexing='ij')
    norm = np.reciprocal(2 * np.pi * gamma ** 2)
    mask = U ** 2 + V ** 2 < R_jax ** 2

    def target_gaussian(u, v):
        return np.where(mask, norm * np.exp(-0.5 * (u ** 2 + v ** 2) / gamma ** 2), 0.)

    target_dist = target_gaussian(U, V)
    uv_grid = np.stack((U, V), axis=-1)
    return uvec_bins, uv_grid, target_dist


def find_optimal_ref_fwhm(freqs, ref_freq, target_fwhm):
    """
    Find the optimal gamma for a given frequency and target FWHM.

    Args:
        freqs: frequencies in Hz
        ref_freq: reference frequency in Hz
        target_fwhm: target FWHM in radians

    Returns:
        the optimal FWHM of ref frequency
    """

    # given arrays nu, and scalars gamma, nu0:
    w = ref_freq / freqs
    W = w.sum()

    def gaussian(x, gamma):
        return np.exp(-0.5 * x ** 2 / gamma ** 2)

    def f(ref_fwhm):
        ref_gamma = ref_fwhm / (2 * np.sqrt(2 * np.log(2)))
        return 0.5 - np.sum(gaussian(0.5 * target_fwhm, ref_gamma * w) * w) / W

    # bracket the root between Delta_min and Delta_max:
    ref_fwhm = brentq(f, 0.001 * target_fwhm, 10 * target_fwhm)  # adjust upper bound as needed
    return ref_fwhm


@partial(jax.jit, static_argnames=['unroll'])
def accumulate_uv_distribution_lambda(antennas_gcrs, times, freqs, ra0, dec0, uvec_bins, unroll=1):
    def get_idx(x):
        # bins [0, 1, 2] -> mid points [0, 1], i0=0, i1=1 -> i0
        dx = uvec_bins[1] - uvec_bins[0]
        one = jnp.ones((), jnp.int32)
        _i1 = jnp.ceil((x - uvec_bins[0]) / dx).astype(jnp.int32)
        i1 = jnp.clip(_i1, one, len(uvec_bins) - 1).astype(jnp.int32)
        i0 = i1 - one
        return i0

    zero_accum = jnp.zeros((uvec_bins.size - 1, uvec_bins.size - 1), dtype=jnp.float64)

    def accumm_time(time):
        antennas_uvw = geometric_uvw_from_gcrs(evolve_gcrs(antennas_gcrs, time), ra0, dec0)
        i_idxs, j_idxs = np.triu_indices(antennas_uvw.shape[0], k=1)
        # uvw = antennas_uvw[i_idxs] - antennas_uvw[j_idxs]  # [..., 3]
        uvw = jnp.concatenate(
            [antennas_uvw[i_idxs] - antennas_uvw[j_idxs],
             antennas_uvw[j_idxs] - antennas_uvw[i_idxs]],
            axis=0
        )  # [..., 3]

        u = uvw[..., 0]
        v = uvw[..., 1]

        def accum_freq(freq):
            wavelength = quantity_to_jnp(const.c) / freq
            u_idx = get_idx(u / wavelength)
            v_idx = get_idx(v / wavelength)
            accum = zero_accum.at[u_idx, v_idx].add(1.0)
            return accum

        accum = scan_sum(accum_freq, zero_accum, freqs, unroll=unroll)
        return accum

    accum = scan_sum(accumm_time, zero_accum, times, unroll=unroll)
    return accum


@partial(jax.jit, static_argnames=['conv_size', 'unroll'])
def accumulate_uv_distribution(antennas_gcrs, times, ra0, dec0, uvec_bins, conv_size=1, unroll=1):
    if conv_size % 2 == 0:
        raise ValueError("conv_size must be odd")

    def get_idx(x):
        # bins [0, 1, 2] -> mid points [0, 1], i0=0, i1=1 -> i0
        dx = uvec_bins[1] - uvec_bins[0]
        one = jnp.ones((), jnp.int32)
        _i1 = jnp.ceil((x - uvec_bins[0]) / dx).astype(jnp.int32)
        i1 = jnp.clip(_i1, one, len(uvec_bins) - 1).astype(jnp.int32)
        i0 = i1 - one
        return i0

    zero_accum = jnp.zeros((uvec_bins.size - 1, uvec_bins.size - 1), dtype=jnp.float64)

    def accumm_time(time):
        antennas_uvw = geometric_uvw_from_gcrs(evolve_gcrs(antennas_gcrs, time), ra0, dec0)
        i_idxs, j_idxs = np.triu_indices(antennas_uvw.shape[0], k=1)
        # uvw = antennas_uvw[i_idxs] - antennas_uvw[j_idxs]  # [..., 3]
        uvw = jnp.concatenate(
            [antennas_uvw[i_idxs] - antennas_uvw[j_idxs],
             antennas_uvw[j_idxs] - antennas_uvw[i_idxs]],
            axis=0
        )  # [..., 3]

        u = uvw[..., 0]
        v = uvw[..., 1]

        u_idx = get_idx(u)
        v_idx = get_idx(v)
        accum = zero_accum.at[u_idx, v_idx].add(1.0)

        return accum

    accum = scan_sum(accumm_time, zero_accum, times, unroll=unroll)

    if conv_size > 1:
        k_vec = jnp.linspace(-1, 1, conv_size)
        kx, ky = jnp.meshgrid(k_vec, k_vec, indexing='ij')
        mask = jnp.sqrt(kx ** 2 + ky ** 2) <= 1 # [conv_size, conv_size]
        kernel = mask.astype(accum.dtype)
        kernel /= jnp.sum(kernel)
        # do 2D convolution
        accum = convolve2d(accum, kernel, mode='same')

    return accum


@jax.jit
def evaluate_uv_distribution(antennas_gcrs, times, ra0, dec0, uv_bins, target_uv, target_dist):
    """
    Evaluate the UV distribution using the negative sliced Wasserstein distance.

    Args:
        antennas_gcrs: [N, 3] GCRS coordinates of antennas at the reference time
        times: [T] times in seconds since the reference time
        ra0: the right ascension of the tracking center in radians
        dec0: the declination of the tracking center in radians
        uv_bins: [M + 1] bins for the UV distribution
        target_uv: [M, M, 2] target UV coordinates
        target_dist: [M, M] target UV distribution

    Returns:
        the negative sliced Wasserstein distance, a scalar
    """
    dist = accumulate_uv_distribution(antennas_gcrs, times, ra0, dec0, uv_bins)
    target_dist *= jnp.sum(dist) / jnp.sum(target_dist)
    diff = dist - target_dist
    return -jnp.max(jnp.abs(diff))

    # uv_grid = jnp.reshape(target_uv, (-1, 2))
    # x_weights = jnp.reshape(dist, (-1,))
    # x_weights /= jnp.sum(x_weights)
    # y_weights = jnp.reshape(target_dist, (-1,))
    # y_weights /= jnp.sum(y_weights)
    # dist = sliced_wasserstein(
    #     key=key,
    #     x=uv_grid,
    #     y=uv_grid,
    #     x_weights=x_weights,
    #     y_weights=y_weights,
    #     p=1,
    #     num_samples=100,
    #     resolution=100
    # )
    # return -dist


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
