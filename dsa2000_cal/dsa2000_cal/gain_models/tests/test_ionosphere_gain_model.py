import jax
import numpy as np
from astropy import units as au, coordinates as ac, time as at

from dsa2000_cal.gain_models.ionosphere_gain_model import IonosphereGainModel, msqrt


def test_ionosphere_gain_model():
    freqs = au.Quantity([1, 2], unit=au.Hz)
    directions = ac.ICRS([0, 1, 2] * au.deg, [3, 4, 5] * au.deg)
    times = at.Time([1, 2, 3, 4], format='jd')

    num_ant = 6

    # mTECU dtec

    dtec = au.Quantity(np.ones((len(times), len(directions), num_ant)))

    ionosphere_gain_model = IonosphereGainModel(freqs=freqs, model_directions=directions, model_times=times, dtec=dtec)

    array_location = ac.EarthLocation.from_geodetic(0, 0, 0)
    time = at.Time(1, format='jd')
    phase_tracking = ac.ICRS(0 * au.deg, 0 * au.deg)

    sources = ac.ICRS([0, 1, 2] * au.deg, [3, 4, 5] * au.deg)
    gains = ionosphere_gain_model.compute_beam(sources, phase_tracking, array_location, time)
    print(gains.shape)
    assert gains.shape == sources.shape + (num_ant, len(freqs), 2, 2)

    sources = ac.ICRS([0, 1, 2, 3] * au.deg, [3, 4, 5, 6] * au.deg).reshape((2, 2))
    gains = ionosphere_gain_model.compute_beam(sources, phase_tracking, array_location, time)
    print(gains.shape)
    assert gains.shape == sources.shape + (num_ant, len(freqs), 2, 2)


def test_msqrt():
    M = jax.random.normal(jax.random.PRNGKey(42), (100, 100))
    A = M @ M.T
    max_eig, min_eig, L = msqrt(A)
    np.testing.assert_allclose(A, L @ L.T, atol=2e-4)
