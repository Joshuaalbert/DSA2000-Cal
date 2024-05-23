import numpy as np
import pylab as plt
from astropy import units as au, coordinates as ac, time as at

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
    freqs = au.Quantity([1000, 2000], unit=au.Hz)
    theta = au.Quantity([0, 90], unit=au.deg)
    phi = au.Quantity([0, 90], unit=au.deg)
    amplitude = au.Quantity([[1, 2], [3, 4]], unit=au.dimensionless_unscaled)
    num_antenna = 5

    beam_gain_model = BeamGainModel(
        model_freqs=freqs,
        model_theta=theta,
        model_phi=phi,
        model_amplitude=amplitude,
        num_antenna=num_antenna
    )

    sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg).reshape((2, 1))
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    gains = beam_gain_model.compute_gain(
        freqs=freqs,
        sources=sources,
        phase_tracking=phase_tracking,
        array_location=array_location,
        time=time
    )

    assert gains.shape == sources.shape + (num_antenna, len(freqs), 2, 2)


def test_beam_gain_model_real_data():
    freqs = au.Quantity([700e6, 2000e6], unit=au.Hz)
    beam_gain_model = beam_gain_model_factory(array_name='dsa2000W')
    # print(beam_gain_model)

    sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg)
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    gains = beam_gain_model.compute_gain(
        freqs=freqs,
        sources=sources, phase_tracking=phase_tracking, array_location=array_location, time=time
    )  # (source_shape) + [num_ant, num_freq, 2, 2]

    # print(gains)
    assert gains.shape == (len(sources), beam_gain_model.num_antenna, len(freqs), 2, 2)

    # Plot amplitude
    lvec = np.linspace(-1, 1, 100) * au.dimensionless_unscaled
    mvec = np.linspace(-1, 1, 100) * au.dimensionless_unscaled
    M, L = np.meshgrid(mvec, lvec, indexing='ij')
    lmn = np.stack([L, M, np.sqrt(1. - L ** 2 - M ** 2)], axis=-1)  # [100, 100, 3]
    sources = lmn_to_icrs(lmn=lmn, time=time, phase_tracking=phase_tracking)
    gains = beam_gain_model.compute_gain(
        freqs=freqs[:1],
        sources=sources, phase_tracking=phase_tracking, array_location=array_location, time=time
    )  # [100, 100, num_ant, num_freq, 2, 2]
    gains = gains[..., 0, 0, 0, 0]  # [100, 100]
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True, squeeze=False)
    im = axs[0, 0].imshow(
        np.abs(gains),
        extent=(lvec[0].value, lvec[-1].value, mvec[0].value, mvec[-1].value),
        origin='lower',
        cmap='PuOr'
    )
    fig.colorbar(im, ax=axs[0, 0])
    axs[0, 0].set_xlabel('l')
    axs[0, 0].set_ylabel('m')
    axs[0, 0].set_title('Amplitude')
    plt.show()


def test_beam_gain_model_factory():
    beam_gain_model = beam_gain_model_factory(array_name='lwa')
    sc = plt.scatter(beam_gain_model.lmn_data[:, 0], beam_gain_model.lmn_data[:, 1], s=1, alpha=0.5,
                     c=np.log10(np.abs(beam_gain_model.model_gains[:, 0, 0, 0])))
    plt.colorbar(sc)
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('log10(Amplitude)')
    plt.show()

    sc = plt.scatter(beam_gain_model.lmn_data[:, 0], beam_gain_model.lmn_data[:, 1], s=1, alpha=0.5,
                     c=np.angle(beam_gain_model.model_gains[:, 0, 0, 0]),
                     cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(sc)
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('Phase')
    plt.show()
