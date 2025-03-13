import numpy as np
import pytest
from astropy import units as au, coordinates as ac, time as at
from jax import numpy as jnp

from dsa2000_common.gain_models.base_spherical_interpolator import regrid_to_regular_grid, lmn_from_phi_theta, \
    phi_theta_from_lmn, build_spherical_interpolator
from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model


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

    phi = 0.
    theta = np.pi
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, 0, -1], atol=5e-8)

    phi = np.pi / 2.
    theta = np.pi
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, 0, -1], atol=5e-8)

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


def build_mock_spherical_interpolator_gain_model(tile_antennas, full_stokes, num_model_times, num_model_freqs):
    num_dir = 4
    num_ant = 5

    freqs = au.Quantity(np.linspace(1000, 2000, num_model_freqs), unit=au.MHz)
    theta = au.Quantity(np.linspace(0., 180., num_dir), unit=au.deg)
    phi = au.Quantity(np.linspace(0., 360., num_dir), unit=au.deg)
    times = at.Time.now() + np.zeros((num_model_times,)) * au.s
    antennas = ac.EarthLocation.from_geocentric([0] * num_ant * au.m, [0] * num_ant * au.m, [0] * num_ant * au.m)

    if full_stokes:
        if tile_antennas:
            model_gains = au.Quantity(np.ones((num_model_times, num_dir, num_model_freqs, 2, 2)),
                                      unit=au.dimensionless_unscaled)
        else:
            model_gains = au.Quantity(np.ones((num_model_times, num_dir, num_ant, num_model_freqs, 2, 2)),
                                      unit=au.dimensionless_unscaled)
    else:
        if tile_antennas:
            model_gains = au.Quantity(np.ones((num_model_times, num_dir, num_model_freqs)),
                                      unit=au.dimensionless_unscaled)
        else:
            model_gains = au.Quantity(np.ones((num_model_times, num_dir, num_ant, num_model_freqs)),
                                      unit=au.dimensionless_unscaled)

    gain_model = build_spherical_interpolator(
        antennas=antennas,
        model_freqs=freqs,
        model_theta=theta,
        model_phi=phi,
        model_times=times,
        ref_time=times[0],
        model_gains=model_gains,
        tile_antennas=tile_antennas
    )

    phase_center = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)

    geodesic_model = build_geodesic_model(
        phase_center=phase_center,
        antennas=antennas,
        array_location=antennas[0],
        obstimes=times,
        ref_time=times[0],
        pointings=None
    )

    return gain_model, geodesic_model


@pytest.mark.parametrize('full_stokes', [True, False])
@pytest.mark.parametrize('tile_antennas', [True, False])
@pytest.mark.parametrize('num_model_times', [1, 2])
@pytest.mark.parametrize('num_model_freqs', [1, 4])
@pytest.mark.parametrize('num_sources', [1, 3])
def test_spherical_interpolator(full_stokes, tile_antennas, num_model_times, num_model_freqs, num_sources):
    mock_gain_model, geodesic_model = build_mock_spherical_interpolator_gain_model(
        tile_antennas=tile_antennas,
        full_stokes=full_stokes,
        num_model_freqs=num_model_freqs,
        num_model_times=num_model_times
    )

    print(f"Tiled: {mock_gain_model.tile_antennas}")

    for near_sources in [True, False]:

        for freqs, obstimes in [
            (mock_gain_model.model_freqs, mock_gain_model.model_times),
            (mock_gain_model.model_freqs[:1], mock_gain_model.model_times[:1]),
        ]:

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
                freqs,
                obstimes,
                geodesics
            )
            if full_stokes:
                assert mock_gain_model.is_full_stokes()
                if tile_antennas:
                    assert gains.shape == (len(obstimes), 5, len(freqs), num_sources, 2, 2)
                else:
                    assert gains.shape == (len(obstimes), 5, len(freqs), num_sources, 2, 2)
            else:
                assert not mock_gain_model.is_full_stokes()
                if tile_antennas:
                    assert gains.shape == (len(obstimes), 5, len(freqs), num_sources,)
                else:
                    assert gains.shape == (len(obstimes), 5, len(freqs), num_sources,)


def test_regrid_to_regular_grid():
    num_model_times = 2
    num_model_dir = 4
    num_ant = 4
    num_model_freqs = 5
    resolution = 11

    model_theta = jnp.linspace(0, 180, num_model_dir)
    model_phi = jnp.linspace(0, 360, num_model_dir)
    model_lmn = jnp.stack(lmn_from_phi_theta(model_phi, model_theta), axis=-1)
    # Full shape: A, full stokes
    model_gains = jnp.ones((num_model_times, num_model_dir, num_ant, num_model_freqs, 2, 2))
    theta, phi, gains = regrid_to_regular_grid(model_lmn, model_gains, resolution)
    assert np.shape(gains) == (num_model_times, resolution, resolution, num_ant, num_model_freqs, 2, 2)

    # Partial: no A full stokes
    model_gains = jnp.ones((num_model_times, num_model_dir, num_model_freqs, 2, 2))
    theta, phi, gains = regrid_to_regular_grid(model_lmn, model_gains, resolution)
    assert np.shape(gains) == (num_model_times, resolution, resolution, num_model_freqs, 2, 2)

    # Partial: no A, no full stokes
    model_gains = jnp.ones((num_model_times, num_model_dir, num_model_freqs))
    theta, phi, gains = regrid_to_regular_grid(model_lmn, model_gains, resolution)
    assert np.shape(gains) == (num_model_times, resolution, resolution, num_model_freqs)

    # Partial: A, no full stokes
    model_gains = jnp.ones((num_model_times, num_model_dir, num_ant, num_model_freqs))
    theta, phi, gains = regrid_to_regular_grid(model_lmn, model_gains, resolution)
    assert np.shape(gains) == (num_model_times, resolution, resolution, num_ant, num_model_freqs)


@pytest.mark.parametrize('array_name', ['dsa2000W_small'])
def test_spherical_beams(array_name):
    beam_gain_model = build_beam_gain_model(array_name=array_name, full_stokes=False)

    beam_gain_model.plot_regridded_beam()
