import numpy as np
import pylab as plt
import pytest
from astropy import units as au, coordinates as ac, time as at
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.gain_models.beam_gain_model import lmn_from_phi_theta, BeamGainModel, beam_gain_model_factory


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


def test_beam_gain_model():
    num_freqs = 3
    num_dir = 30
    freqs = au.Quantity(np.linspace(1000, 2000, num_freqs), unit=au.MHz)
    theta = au.Quantity(np.linspace(0., 180., num_dir), unit=au.deg)
    phi = au.Quantity(np.linspace(0., 360., num_dir), unit=au.deg)
    model_gains = au.Quantity(np.ones((num_dir, num_freqs, 2, 2)), unit=au.dimensionless_unscaled)
    antennas = ac.EarthLocation.of_site('vla').reshape((1,))

    beam_gain_model = BeamGainModel(
        model_freqs=freqs,
        model_theta=theta,
        model_phi=phi,
        model_gains=model_gains,
        antennas=antennas
    )

    sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg).reshape((2, 1))
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    gains = beam_gain_model.compute_gain(freqs=freqs, sources=sources, array_location=array_location,
                                         time=time, pointing=phase_tracking)

    assert gains.shape == sources.shape + (1, len(freqs), 2, 2)


def test_zenith_pointing():
    num_freqs = 3
    num_dir = 30
    freqs = au.Quantity(np.linspace(1000, 2000, num_freqs), unit=au.MHz)
    theta = au.Quantity(np.linspace(0., 180., num_dir), unit=au.deg)
    phi = au.Quantity(np.linspace(0., 360., num_dir), unit=au.deg)
    model_gains = au.Quantity(np.ones((num_dir, num_freqs, 2, 2)), unit=au.dimensionless_unscaled)
    antennas = ac.EarthLocation.of_site('vla').reshape((1,))

    beam_gain_model = BeamGainModel(
        model_freqs=freqs,
        model_theta=theta,
        model_phi=phi,
        model_gains=model_gains,
        antennas=antennas
    )

    sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg).reshape((2, 1))
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    gains = beam_gain_model.compute_gain(freqs=freqs, sources=sources, array_location=array_location,
                                         time=time, pointing=None)

    assert gains.shape == sources.shape + (1, len(freqs), 2, 2)


@pytest.mark.parametrize('array_name, freq, zenith', [['lwa', 50e6, True], ['dsa2000W', 700e6, False]])
def test_beam_gain_model_real_data(array_name, freq, zenith):
    freqs = au.Quantity([freq], unit=au.Hz)
    beam_gain_model = beam_gain_model_factory(array_name=array_name)
    # print(beam_gain_model)

    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=45 * au.deg)
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # Simple shape test
    sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg)
    gains = beam_gain_model.compute_gain(freqs=freqs, sources=sources, array_location=array_location,
                                         time=time,
                                         pointing=None if zenith else phase_tracking)  # (source_shape) + [num_ant, num_freq, 2, 2]
    assert gains.shape == (len(sources), len(beam_gain_model.antennas), len(freqs), 2, 2)

    # Test meshgrid
    lvec = np.linspace(-1, 1, 100) * au.dimensionless_unscaled
    mvec = np.linspace(-1, 1, 100) * au.dimensionless_unscaled
    M, L = np.meshgrid(mvec, lvec, indexing='ij')
    lmn = np.stack([L, M, np.sqrt(1. - L ** 2 - M ** 2)], axis=-1)  # [100, 100, 3]
    sources = lmn_to_icrs(lmn=lmn, time=time, phase_tracking=phase_tracking)
    gains = beam_gain_model.compute_gain(freqs=freqs, sources=sources, array_location=array_location,
                                         time=time, pointing=None)  # [100, 100, num_ant, num_freq, 2, 2]
    gains = gains[..., 0, 0, 0, 0]  # [100, 100]

    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True, squeeze=False)
    im = axs[0, 0].imshow(
        np.abs(gains),
        extent=(lvec[0].value, lvec[-1].value, mvec[0].value, mvec[-1].value),
        origin='lower',
        cmap='PuOr',
        vmin=0.
    )
    fig.colorbar(im, ax=axs[0, 0])
    axs[0, 0].set_xlabel('l')
    axs[0, 0].set_ylabel('m')
    axs[0, 0].set_title('Amplitude')

    im = axs[1, 0].imshow(
        np.angle(gains),
        extent=(lvec[0].value, lvec[-1].value, mvec[0].value, mvec[-1].value),
        origin='lower',
        cmap='hsv',
        vmin=-np.pi,
        vmax=np.pi
    )
    fig.colorbar(im, ax=axs[1, 0])
    axs[1, 0].set_xlabel('l')
    axs[1, 0].set_ylabel('m')
    axs[1, 0].set_title('phase')

    fig.tight_layout()
    plt.show()


@pytest.mark.parametrize('array_name', ['lwa', 'dsa2000W'])
def test_beam_gain_model_factory(array_name: str):
    beam_gain_model = beam_gain_model_factory(array_name=array_name)
    select = beam_gain_model.lmn_data[:, 2] >= 0.  # Select only positive N
    sc = plt.scatter(beam_gain_model.lmn_data[select, 0], beam_gain_model.lmn_data[select, 1], s=1, alpha=0.5,
                     c=np.log10(np.abs(beam_gain_model.model_gains[select, 0, 0, 0])))
    plt.colorbar(sc)
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('log10(Amplitude)')
    plt.show()

    sc = plt.scatter(beam_gain_model.lmn_data[select, 0], beam_gain_model.lmn_data[select, 1], s=1, alpha=0.5,
                     c=np.angle(beam_gain_model.model_gains[select, 0, 0, 0]),
                     cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(sc)
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('Phase')
    plt.show()


def test_near_field_source():
    freqs = au.Quantity([50e6], unit=au.Hz)
    beam_gain_model = beam_gain_model_factory(array_name='lwa')
    # print(beam_gain_model)

    array_location = beam_gain_model.antennas[0]
    time = at.Time('2021-01-01T00:00:00', scale='utc')
    phase_tracking = ENU(east=0, north=0, up=1, obstime=time, location=array_location).transform_to(ac.ICRS())
    sources = ENU(east=[0, 1] * au.km, north=[1, 0] * au.km, up=[20, 20] * au.m, location=array_location, obstime=time)

    # Simple shape test
    gains = beam_gain_model.compute_gain(freqs=freqs, sources=sources, array_location=array_location,
                                         time=time,
                                         pointing=phase_tracking)  # (source_shape) + [num_ant, num_freq, 2, 2]
    assert gains.shape == (len(sources), len(beam_gain_model.antennas), len(freqs), 2, 2)