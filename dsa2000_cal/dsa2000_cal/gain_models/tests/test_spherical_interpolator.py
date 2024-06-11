import numpy as np
import pytest
from astropy import units as au, coordinates as ac, time as at
from tomographic_kernel.frames import ENU

from dsa2000_cal.gain_models.spherical_interpolator import lmn_from_phi_theta, SphericalInterpolatorGainModel


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

    return gain_model


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

    return gain_model


def test_beam_gain_model_shape(mock_spherical_interpolator_gain_model,
                               mock_spherical_interpolator_gain_model_tile):
    for mock_gain_model in [mock_spherical_interpolator_gain_model,
                            mock_spherical_interpolator_gain_model_tile]:
        print(f"Tiled: {mock_gain_model.tile_antennas}")
        # Near field source
        array_location = ac.EarthLocation(lat=0, lon=0, height=0)
        time = mock_gain_model.model_times[0]
        freqs = mock_gain_model.model_freqs

        for use_scan in [True, False]:
            print(f"Use scan {use_scan}")
            mock_gain_model.use_scan = use_scan

            for near_sources in [True, False]:
                print(f'Near field sources: {near_sources}')
                if near_sources:
                    sources = ENU(east=[0, 1] * au.km, north=[1, 0] * au.km, up=[20, 20] * au.m,
                                  location=array_location, obstime=time)
                else:
                    sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg).reshape((2, 1))

                # scalar pointing
                print("Scalar pointing")
                pointing = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
                gains = mock_gain_model.compute_gain(
                    freqs=freqs,
                    sources=sources,
                    array_location=array_location,
                    time=time,
                    pointing=pointing
                )
                assert gains.shape == sources.shape + (len(mock_gain_model.antennas),
                                                       len(freqs), 2, 2)

                # 1D pointing
                print("1D pointing")
                pointing = ac.ICRS(ra=[0 for _ in mock_gain_model.antennas] * au.deg,
                                   dec=[0 for _ in mock_gain_model.antennas] * au.deg)

                gains = mock_gain_model.compute_gain(
                    freqs=freqs,
                    sources=sources,
                    array_location=array_location,
                    time=time,
                    pointing=pointing
                )
                assert gains.shape == sources.shape + (len(mock_gain_model.antennas),
                                                       len(freqs), 2, 2)

                # Zenith
                print('Zenith')
                gains = mock_gain_model.compute_gain(
                    freqs=freqs,
                    sources=sources,
                    array_location=array_location,
                    time=time,
                    pointing=None
                )
                assert gains.shape == sources.shape + (len(mock_gain_model.antennas),
                                                       len(freqs), 2, 2)
