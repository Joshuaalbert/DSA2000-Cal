from datetime import datetime

from astropy import coordinates as ac

from dsa2000_cal.create_ms_cfg import create_makems_config


def test_create_makems_config():
    create_makems_config(
        array_name='dsa2000W',
        start_freq=800e6,
        step_freq=2e6,
        start_time=datetime(2019, 3, 19, 19, 58, 15),
        step_time=1.5,
        pointing_direction=ac.ICRS(ra=ac.Angle('00h00m0.0s'), dec=ac.Angle('+37d07m47.400s')),
        num_freqs=32,
        num_times=30
    )
