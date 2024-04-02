import jax
import numpy as np
from astropy import coordinates as ac, time as at, units as au

from dsa2000_cal.gain_models.gain_model import GainModel, ProductGainModel


class MockGainModel(GainModel):
    def __init__(self, gain):
        self.gain = gain

    def compute_gain(self, sources: ac.ICRS, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time,
                     **kwargs) -> jax.Array:
        return self.gain


def test_product_gain_model():
    num_sources = 2
    num_ant = 3
    num_freq = 4
    gain = np.arange(num_sources * num_ant * num_freq * 2 * 2).reshape((num_sources, num_ant, num_freq, 2, 2))
    gain_model = MockGainModel(gain)
    product_gain_model = ProductGainModel([gain_model, gain_model])
    sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg)
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    result = product_gain_model.compute_gain(sources, phase_tracking, array_location, time)
    expected = gain @ gain
    np.testing.assert_array_equal(result, expected)
