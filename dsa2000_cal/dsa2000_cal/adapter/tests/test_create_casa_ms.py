import os
from datetime import datetime

from astropy import coordinates as ac

from dsa2000_cal.adapter.create_casa_ms import create_makems_config


def test_create_makems_config(tmp_path):
    config_file = create_makems_config(
        array_name='dsa2000W',
        ms_name=tmp_path / 'test.ms',
        start_freq=700e6,
        step_freq=2e6,
        start_time=datetime(2019, 3, 19, 19, 58, 15),
        step_time=1.5,
        phase_tracking=ac.ICRS(ra=ac.Angle('00h00m0.0s'), dec=ac.Angle('+37d07m47.400s')),
        num_freqs=32,
        num_times=30
    )
    assert os.path.exists(config_file)
    os.remove(config_file)

