import jax
import jax.numpy as jnp
import numpy as np
from astropy import coordinates as ac, time as at, units as au

from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from src.dsa2000_cal.gain_models.gain_model import GainModel, ProductGainModel
from dsa2000_cal.geodesics.base_geodesic_model import build_geodesic_model


class MockGainModel(GainModel):
    def __init__(self, gain):
        self.gain = gain

    def compute_gain(self, freqs: jax.Array, times: jax.Array, lmn_geodesic: jax.Array) -> jax.Array:
        return self.gain

    def is_full_stokes(self) -> bool:
        return True


def test_product_gain_model():
    num_sources = 2
    num_ant = 3
    num_freq = 4
    gain = np.arange(num_sources * num_ant * num_freq * 2 * 2).reshape((num_sources, num_ant, num_freq, 2, 2))
    gain_model = MockGainModel(gain)
    product_gain_model = ProductGainModel([gain_model, gain_model])
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    obstimes = at.Time(['2021-01-01T00:00:00', '2021-01-01T00:00:30'], scale='utc')
    antennas = ac.EarthLocation.from_geocentric([0] * num_ant * au.m, [0] * num_ant * au.m, [0] * num_ant * au.m)

    lmn_sources = jnp.zeros((num_sources, 3))

    geodesic_model = build_geodesic_model(
        phase_center=phase_tracking,
        antennas=antennas,
        array_location=array_location,
        obstimes=obstimes,
        ref_time=obstimes[0],
        pointings=None
    )

    times = quantity_to_jnp((obstimes.tt - obstimes[0].tt).sec * au.s)
    freqs = jnp.linspace(0.1, 0.5, num_freq)

    result = product_gain_model.compute_gain(
        freqs=freqs, times=times, lmn_geodesic=geodesic_model.compute_far_field_geodesic(times, lmn_sources)
    )

    expected = gain @ gain
    np.testing.assert_array_equal(result, expected)
