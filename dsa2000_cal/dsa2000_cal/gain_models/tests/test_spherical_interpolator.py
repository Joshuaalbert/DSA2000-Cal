import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as au, coordinates as ac, time as at

from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.geodesic_model import GeodesicModel
from dsa2000_cal.gain_models.spherical_interpolator import lmn_from_phi_theta, SphericalInterpolatorGainModel, \
    phi_theta_from_lmn


def test_lmn_from_phi_theta():
    # L = -Y, M = X, N = Z

    # Bore-sight
    phi = 0.
    theta = 0.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, 0, 1], atol=5e-8)

    phi = np.pi / 2.
    theta = 0.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, 0, 1], atol=5e-8)

    # Points to right on sky == -L
    phi = np.pi / 2.
    theta = np.pi / 2.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [-1, 0, 0], atol=5e-8)

    # Points to left on sky == L
    phi = - np.pi / 2.
    theta = np.pi / 2.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [1, 0, 0], atol=5e-8)

    # Points up on sky == M
    phi = 0.
    theta = np.pi / 2.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, 1, 0], atol=5e-8)

    # Points down on sky == -M
    phi = np.pi
    theta = np.pi / 2.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, -1, 0], atol=1e-7)


def test_phi_theta_from_lmn():
    # Test bore-sight
    l, m, n = 0., 0., 1.
    phi, theta = phi_theta_from_lmn(l, m, n)
    np.testing.assert_allclose(theta, 0., atol=5e-8)

    # Points to right on sky == -L
    l, m, n = -1., 0., 0.
    phi, theta = phi_theta_from_lmn(l, m, n)
    np.testing.assert_allclose([phi, theta], [np.pi / 2., np.pi / 2.], atol=5e-8)

    # Points to left on sky == L
    l, m, n = 1., 0., 0.
    phi, theta = phi_theta_from_lmn(l, m, n)
    np.testing.assert_allclose([phi, theta], [np.pi * 3 / 2., np.pi / 2.], atol=5e-8)

    # Points up on sky == M
    l, m, n = 0., 1., 0.
    phi, theta = phi_theta_from_lmn(l, m, n)
    np.testing.assert_allclose([phi, theta], [0., np.pi / 2.], atol=5e-8)

    # Points down on sky == -M
    l, m, n = 0., -1., 0.
    phi, theta = phi_theta_from_lmn(l, m, n)
    np.testing.assert_allclose([phi, theta], [np.pi, np.pi / 2.], atol=1e-7)


@pytest.fixture(scope='function')
def mock_setup():
    num_time = 2
    num_freq = 3
    num_dir = 4

    freqs = au.Quantity(np.linspace(1000, 2000, num_freq), unit=au.MHz)
    theta = au.Quantity(np.linspace(0., 180., num_dir), unit=au.deg)
    phi = au.Quantity(np.linspace(0., 360., num_dir), unit=au.deg)
    times = at.Time.now() + np.zeros((num_time,)) * au.s
    model_gains = au.Quantity(np.ones((num_time, num_dir, num_freq, 2, 2)), unit=au.dimensionless_unscaled)
    antennas = ac.EarthLocation.of_site('vla').reshape((1,))
    return freqs, theta, phi, times, model_gains, antennas


@pytest.fixture(scope='function')
def mock_spherical_interpolator_gain_model(mock_setup):
    freqs, theta, phi, times, model_gains, antennas = mock_setup
    antennas = ac.concatenate([antennas.get_itrs(), antennas.get_itrs()]).earth_location
    model_gains = np.tile(model_gains[:, :, None, :, :, :], (1, 1, 2, 1, 1, 1))

    gain_model = SphericalInterpolatorGainModel(
        antennas=antennas,
        model_freqs=freqs,
        model_theta=theta,
        model_phi=phi,
        model_times=times,
        model_gains=model_gains,
        tile_antennas=False
    )

    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)

    geodesic_model = GeodesicModel(
        phase_center=phase_tracking,
        antennas=antennas,
        array_location=antennas[0],
        obstimes=times,
        ref_time=times[0],
        pointings=None
    )

    return gain_model, geodesic_model


@pytest.fixture(scope='function')
def mock_spherical_interpolator_gain_model_tile(mock_setup):
    freqs, theta, phi, times, model_gains, antennas = mock_setup

    gain_model = SphericalInterpolatorGainModel(
        antennas=antennas,
        model_freqs=freqs,
        model_theta=theta,
        model_phi=phi,
        model_times=times,
        model_gains=model_gains,
        tile_antennas=True
    )

    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)

    geodesic_model = GeodesicModel(
        phase_center=phase_tracking,
        antennas=antennas,
        array_location=antennas[0],
        obstimes=times,
        ref_time=times[0],
        pointings=None
    )

    return gain_model, geodesic_model


def test_beam_gain_model_shape(mock_spherical_interpolator_gain_model,
                               mock_spherical_interpolator_gain_model_tile):
    for (mock_gain_model, geodesic_model) in [mock_spherical_interpolator_gain_model,
                                              mock_spherical_interpolator_gain_model_tile]:
        print(f"Tiled: {mock_gain_model.tile_antennas}")
        # Near field source
        array_location = mock_gain_model.antennas[0]
        time = mock_gain_model.model_times[0]
        freqs = mock_gain_model.model_freqs
        num_sources = 3
        obstimes = quantity_to_jnp((mock_gain_model.model_times - mock_gain_model.model_times[0]).sec * au.s)

        for near_sources in [True, False]:
            print(f'Near field sources: {near_sources}')
            if near_sources:
                geodesics = geodesic_model.compute_near_field_geodesics(
                    times=obstimes,
                    source_positions_enu=jnp.zeros((num_sources, 3))
                )
            else:
                geodesics = geodesic_model.compute_far_field_geodesic(
                    times=obstimes,
                    lmn_sources=jnp.zeros((num_sources, 3))
                )

            gains = mock_gain_model.compute_gain(
                quantity_to_jnp(mock_gain_model.model_freqs),
                obstimes,
                geodesics
            )
            if mock_gain_model.is_full_stokes():
                assert gains.shape == (num_sources, len(obstimes), len(mock_gain_model.antennas), len(freqs), 2, 2)
            else:
                assert gains.shape == (num_sources, len(obstimes), len(mock_gain_model.antennas), len(freqs))
