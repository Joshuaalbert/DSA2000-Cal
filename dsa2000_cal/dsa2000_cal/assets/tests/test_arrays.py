import os

import astropy.coordinates as ac
import astropy.time as at
import numpy as np
from astropy import units as au
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries

fill_registries()
from dsa2000_cal.assets.arrays.array import extract_itrs_coords
from dsa2000_cal.assets.registries import array_registry


def test_extract_itrs_coords():
    with open('test_array.txt', 'w') as f:
        f.write('#X Y Z dish_diam station mount\n'
                '-1601614.0612 -5042001.67655 3554652.4556 25 vla-00 ALT-AZ\n'
                '-1602592.82353 -5042055.01342 3554140.65277 25 vla-01 ALT-AZ\n'
                '-1604008.70191 -5042135.83581 3553403.66677 25 vla-02 ALT-AZ\n'
                '-1605808.59818 -5042230.07046 3552459.16736 25 vla-03 ALT-AZ')
    stations, antenna_coords = extract_itrs_coords('test_array.txt')
    assert len(stations) == 4
    assert len(antenna_coords) == 4
    assert antenna_coords[0].x == -1601614.0612 * au.m
    assert antenna_coords[0].y == -5042001.67655 * au.m
    assert antenna_coords[0].z == 3554652.4556 * au.m
    assert stations == ['vla-00', 'vla-01', 'vla-02', 'vla-03']
    os.remove('test_array.txt')


def test_array_beam():
    array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    antenna_beam = array.get_antenna_beam()
    antenna_beam.get_model().plot_polar_amplitude()

    amplitude = antenna_beam.compute_beam_amplitude(
        pointing=ac.ICRS(ra=0 * au.deg, dec=0 * au.deg),
        source=ac.ICRS(ra=0 * au.deg, dec=0 * au.deg),
        freq_hz=800e6,
        enu_frame=ENU(
            location=ac.EarthLocation.from_geodetic(lon=0 * au.deg, lat=0 * au.deg, height=0 * au.m),
            obstime=at.Time('2000-01-01T00:00:00', format='isot', scale='utc')
        ),
        pol='X'
    )
    assert not np.any(np.isnan(amplitude))
    assert not np.isnan(antenna_beam.get_model().get_voltage_gain())
    assert amplitude.shape == (1,)
    # assert amplitude[0] <= 1.
