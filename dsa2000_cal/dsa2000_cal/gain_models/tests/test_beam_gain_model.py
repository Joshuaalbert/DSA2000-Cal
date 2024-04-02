import numpy as np
from astropy import units as au, coordinates as ac, time as at

from dsa2000_cal.gain_models.beam_gain_model import lmn_from_phi_theta, BeamGainModel, beam_gain_model_factory


def test_phi_theta_to_cartesian():
    # L = -Y, M = X, N = Z

    phi = 0.
    theta = 0.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, 0, 1], atol=5e-8)

    phi = np.pi / 2.
    theta = np.pi / 2.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [-1, 0, 0], atol=5e-8)

    phi = 0.
    theta = np.pi / 2.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, 1, 0], atol=5e-8)


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
    )

    # print(gains)
    assert gains.shape == (len(sources), beam_gain_model.num_antenna, len(freqs), 2, 2)
