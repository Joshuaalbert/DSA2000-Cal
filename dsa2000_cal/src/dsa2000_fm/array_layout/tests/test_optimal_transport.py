import time

import astropy.coordinates as ac
import astropy.time as at
import jax
import numpy as np
import pylab as plt
from astropy import units as au
from jax import numpy as jnp

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.fourier_utils import ApertureTransform
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_fm.array_layout.optimal_transport import wasserstein_point_clouds, wasserstein_1D_p, wasserstein_1D, \
    weighted_wasserstein_1d_gaussian, sliced_wasserstein, sliced_wasserstein_gaussian, compute_ideal_uv_distribution, \
    accumulate_uv_distribution, _cdf_distance_jax, find_optimal_ref_fwhm
from dsa2000_fm.imaging.base_imagor import fit_beam


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


def test_compute_ideal_uv_distribution():
    du = 50 * au.m
    R = 16000 * au.m
    target_fwhm = 3 * au.arcsec
    freq = 1350 * au.MHz
    uv_bins, uv_grid, target_dist = compute_ideal_uv_distribution(du, R, target_fwhm, freq)

    plt.imshow(target_dist.T, extent=(-R.value, R.value, -R.value, R.value),
               interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.xlabel(r'U ($\lambda$)')
    plt.ylabel(r'V ($\lambda$)')
    plt.show()

    wavelength = 299792458 / freq.to('Hz').value
    a = ApertureTransform()
    dx = (uv_bins[1] - uv_bins[0]) / wavelength

    dl = 1 / (target_dist.shape[0] * dx)
    psf = a.to_image(target_dist, axes=(-2, -1), dx=dx, dy=dx).real
    psf /= np.max(psf)

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
    np.testing.assert_allclose(major * rad2arcsec, target_fwhm.value, atol=4e-2)
    np.testing.assert_allclose(minor * rad2arcsec, target_fwhm.value, atol=4e-2)


def test_accumulate_uv_distribution():
    du = 20 * au.m

    dconv = 200 * au.m

    conv_size = (int(dconv / du) // 2) * 2 + 1
    print(f"conv_size: {conv_size}")
    R = 16000 * au.m
    target_fwhm = 3.14 * au.arcsec
    freq = 1350 * au.MHz
    uv_bins, uv_grid, target_dist = compute_ideal_uv_distribution(du, R, target_fwhm, freq)

    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa1650_9P'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    obstimes = ref_time + np.arange(1) * array.get_integration_time()
    phase_center = ENU(0, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())

    antennas_gcrs = quantity_to_jnp(antennas.get_gcrs(ref_time).cartesian.xyz.T, 'm')
    times = time_to_jnp(obstimes, ref_time)
    ra0 = quantity_to_jnp(phase_center.ra, 'rad')
    dec0 = quantity_to_jnp(phase_center.dec, 'rad')

    dist = accumulate_uv_distribution(antennas_gcrs, times, ra0, dec0, uv_bins, conv_size=conv_size)

    # dist /= jnp.sum(dist)
    target_dist *= jnp.sum(dist) / jnp.sum(target_dist)

    diff = target_dist - dist

    plt.imshow(dist.T, origin='lower', extent=(-R.value, R.value, -R.value, R.value),
               interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.xlabel(r'U (m)')
    plt.ylabel(r'V (m)')
    plt.show()

    plt.imshow(target_dist.T, origin='lower', extent=(-R.value, R.value, -R.value, R.value),
               interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.xlabel(r'U (m)')
    plt.ylabel(r'V (m)')
    plt.show()

    plt.imshow(diff.T, origin='lower', extent=(-R.value, R.value, -R.value, R.value),
               interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.xlabel(r'U (m)')
    plt.ylabel(r'V (m)')
    plt.show()

    # dist = sliced_wasserstein(
    #     key=jax.random.PRNGKey(0),
    #     x=jnp.asarray(uv_grid.reshape((-1, 2))),
    #     y=jnp.asarray(uv_grid.reshape((-1, 2))),
    #     x_weights=jnp.asarray(dist.reshape((-1,))),
    #     y_weights=jnp.asarray(target_dist.reshape((-1,))),
    #     p=1,
    #     num_samples=100,
    #     resolution=100
    # )
    # print(dist)


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


def test_cdf_distance():
    p = 1
    u = np.random.normal(size=(2000,))
    v = np.random.normal(size=(1000,))
    u_weights = np.random.uniform(size=(2000,))
    v_weights = np.random.uniform(size=(1000,))
    np.testing.assert_allclose(_cdf_distance_jax(p, u, v, u_weights, v_weights),
                               _cdf_distance_np(p, u, v, u_weights, v_weights), atol=1e-6)


def test_find_optimal_gamma():
    freqs = np.linspace(700e6, 2000e6, 10000)
    ref_freq = 1350e6

    for target_fwhm_arcsec in [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6]:
        ref_fwhm = find_optimal_ref_fwhm(freqs, ref_freq, target_fwhm_arcsec * np.pi / 180 / 3600)
        ref_fwhm_arcsec = ref_fwhm * 3600 * 180 / np.pi

        print(
            f"Target Fullband FWHM: {target_fwhm_arcsec} arcsec ==> FWHM: {ref_fwhm_arcsec:.2f} arcsec at {ref_freq / 1e6:.0f} MHz")
