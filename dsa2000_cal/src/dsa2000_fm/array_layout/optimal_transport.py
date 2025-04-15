import time
from functools import partial

import astropy.constants as const
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.fourier_utils import ApertureTransform
from dsa2000_common.common.interp_utils import InterpolatedArray
from dsa2000_common.common.quantity_utils import quantity_to_np, quantity_to_jnp, time_to_jnp
from dsa2000_common.common.sum_utils import scan_sum
from dsa2000_common.delay_models.uvw_utils import geometric_uvw_from_gcrs
from dsa2000_fm.array_layout.fast_psf_evaluation import project_antennas, rotation_matrix_change_dec
from dsa2000_fm.imaging.base_imagor import fit_beam
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


def test_wasserstein_1D_p():
    x = np.random.normal(size=(2000,))
    y = np.random.normal(size=(1000,))
    x_weights = np.random.uniform(size=(2000,))
    y_weights = np.random.uniform(size=(1000,))
    resolution = 100
    p = 1
    dist1 = wasserstein_1D_p(p, x, y, x_weights, y_weights, resolution)
    dist2 = wasserstein_1D(x, y, x_weights, y_weights)
    print(dist1, dist2)
    np.testing.assert_allclose(dist1, dist2, atol=1e-2)


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


def test_weighted_wasserstein_1d_gaussian():
    x = np.random.normal(size=(20000,))
    y = 1 + np.random.normal(size=(20000,))
    x_weights = None  # np.random.uniform(size=(2000,))
    y_weights = None  # np.random.uniform(size=(2000,))
    target_mean = 1
    target_std = 1
    p = 1

    dist2 = wasserstein_1D(x, y, x_weights, y_weights)
    dist1 = weighted_wasserstein_1d_gaussian(x, x_weights, target_mean, target_std, p)
    np.testing.assert_allclose(dist1, dist2, atol=1e-2)


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


def test_sliced_wasserstein():
    # 2D gaussians same mean, sigma -> 0
    # assert sliced_wasserstein(jax.random.PRNGKey(0), np.random.normal(size=(2000, 2)), np.random.normal(size=(2000, 2)), 100) < sliced_wasserstein(jax.random.PRNGKey(0), np.random.normal(size=(100, 2)), np.random.normal(size=(100, 2)), 100)

    # Different means
    x = np.random.normal(size=(2000, 2))
    y = np.random.normal(size=(2000, 2)) + 2

    dist1 = sliced_wasserstein(jax.random.PRNGKey(0), x, y, None, None, p=1, num_samples=1000, resolution=100)
    dist2 = sliced_wasserstein_gaussian(jax.random.PRNGKey(0), x, None, target_mean=2 * jnp.ones(2),
                                        target_Sigma=jnp.eye(2), num_samples=1000, p=1)
    print(dist1, dist2)
    np.testing.assert_allclose(dist1, dist2, atol=1e-2)


def compute_ideal_uv_distribution(du: au.Quantity, R: au.Quantity, target_fwhm: au.Quantity, max_freq: au.Quantity):
    min_wavelength = quantity_to_np(const.c / max_freq, 'm')
    gamma = np.sqrt(2) * np.log(2) / (np.pi * quantity_to_np(target_fwhm, 'rad'))
    R_jax = quantity_to_np(R, 'm') / min_wavelength
    du_jax = quantity_to_np(du, 'm') / min_wavelength
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


@partial(jax.jit, static_argnames=['unroll'])
def accumulate_uv_distribution(antennas_gcrs, times, freqs, ra0, dec0, uvec_bins, unroll=1):
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
        uvw = antennas_uvw[i_idxs] - antennas_uvw[j_idxs]  # [..., 3]

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

@jax.jit
def evaluate_uv_distribution(key, antennas_gcrs, times, freqs, ra0, dec0, uv_bins, target_uv, target_dist):
    """
    Evaluate the UV distribution using the negative sliced Wasserstein distance.

    Args:
        antennas_gcrs: [N, 3] GCRS coordinates of antennas at the reference time
        times: [T] times in seconds since the reference time
        freqs: [C] frequencies in Hz
        ra0: the right ascension of the tracking center in radians
        dec0: the declination of the tracking center in radians
        uv_bins: [M + 1] bins for the UV distribution
        target_uv: [M, M, 2] target UV coordinates
        target_dist: [M, M] target UV distribution

    Returns:
        the negative sliced Wasserstein distance, a scalar
    """
    dist = accumulate_uv_distribution(antennas_gcrs, times, freqs, ra0, dec0, uv_bins)
    uv_grid = jnp.reshape(target_uv, (-1, 2))
    x_weights = jnp.reshape(dist, (-1, ))
    y_weights = jnp.reshape(target_dist, (-1, ))
    dist = sliced_wasserstein(
        key=key,
        x=uv_grid,
        y=uv_grid,
        x_weights=x_weights,
        y_weights=y_weights,
        p=1,
        num_samples=100,
        resolution=100
    )
    return -dist


def test_compute_ideal_uv_distribution():
    du = 10 * au.m
    R = 16000 * au.m
    target_fwhm = 3. * au.arcsec
    max_freq = 2 * au.GHz
    uv_bins, uv_grid, target_dist = compute_ideal_uv_distribution(du, R, target_fwhm, max_freq)

    a = ApertureTransform()
    dx = uv_bins[1] - uv_bins[0]

    dl = 1 / (target_dist.shape[0] * dx)
    psf = a.to_image(target_dist, axes=(-2, -1), dx=dx, dy=dx).real
    psf /= np.max(psf)
    import pylab as plt

    plt.imshow(psf.T, cmap='gray')
    plt.colorbar()
    plt.show()

    major, minor, pos_angle = fit_beam(psf, dl, dl)
    rad2arcsec = 180 / np.pi * 3600

    print(
        f"Beam major: {major * rad2arcsec:.2f}arcsec, "
        f"minor: {minor * rad2arcsec:.2f}arcsec, "
        f"posang: {pos_angle * 180 / np.pi:.2f}deg"
    )


def test_accumulate_uv_distribution():
    du = 100 * au.m
    R = 16000 * au.m
    target_fwhm = 3.3 * au.arcsec
    max_freq = 2 * au.GHz
    uv_bins, uv_grid, target_dist = compute_ideal_uv_distribution(du, R, target_fwhm, max_freq)
    import pylab as plt
    import astropy.time as at
    import astropy.coordinates as ac

    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa1650_9P'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    obstimes = ref_time + np.arange(2) * array.get_integration_time()
    phase_center = ENU(0, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())

    antennas_gcrs = quantity_to_jnp(antennas.get_gcrs(ref_time).cartesian.xyz.T, 'm')
    times = time_to_jnp(obstimes, ref_time)
    obsfreqs = array.get_channels()[:2]
    freqs = quantity_to_jnp(obsfreqs, 'Hz')
    ra0 = quantity_to_jnp(phase_center.ra, 'rad')
    dec0 = quantity_to_jnp(phase_center.dec, 'rad')

    dist = accumulate_uv_distribution(antennas_gcrs, times, freqs, ra0, dec0, uv_bins)

    target_dist /= jnp.sum(target_dist)
    dist /= jnp.sum(dist)

    plt.imshow(dist.T, origin='lower', extent=(-R.value, R.value, -R.value, R.value),
               interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.xlabel(r'U ($\lambda$)')
    plt.ylabel(r'V ($\lambda$)')
    plt.show()

    plt.imshow(target_dist.T, origin='lower', extent=(-R.value, R.value, -R.value, R.value),
               interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.xlabel(r'U ($\lambda$)')
    plt.ylabel(r'V ($\lambda$)')
    plt.show()

    dist = sliced_wasserstein(
        key=jax.random.PRNGKey(0),
        x=jnp.asarray(uv_grid.reshape((-1, 2))),
        y=jnp.asarray(uv_grid.reshape((-1, 2))),
        x_weights=jnp.asarray(dist.reshape((-1,))),
        y_weights=jnp.asarray(target_dist.reshape((-1,))),
        p=1,
        num_samples=100,
        resolution=100
    )
    print(dist)


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
