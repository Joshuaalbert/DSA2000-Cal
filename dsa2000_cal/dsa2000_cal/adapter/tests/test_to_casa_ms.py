import os

import astropy.time as at
import astropy.units as au
import numpy as np
from astropy import coordinates as ac

from dsa2000_cal.adapter.to_casa_ms import create_makems_config
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0


def test_create_makems_config(tmp_path):
    meta = MeasurementSetMetaV0(
        array_name="test_array",
        array_location=ac.EarthLocation.from_geodetic(0 * au.deg, 0 * au.deg, 0 * au.m),
        phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
        channel_width=au.Quantity(1, au.Hz),
        integration_time=au.Quantity(1, au.s),
        coherencies=['XX','XY','YX','YY'],
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=at.Time.now() + np.arange(10) * au.s,
        freqs=au.Quantity([1, 2, 3], au.Hz),
        antennas=ac.EarthLocation.from_geodetic(np.arange(5) * au.deg, np.arange(5) * au.deg, np.arange(5) * au.m),
        antenna_names=[f"antenna_{i}" for i in range(5)],
        antenna_diameters=au.Quantity(np.ones(5), au.m),
        with_autocorr=False,
        mount_types='ALT-AZ'
    )
    config_file = create_makems_config(
        casa_ms=str(tmp_path / 'test.ms'),
        meta=meta
    )
    assert os.path.exists(config_file)
    os.remove(config_file)
