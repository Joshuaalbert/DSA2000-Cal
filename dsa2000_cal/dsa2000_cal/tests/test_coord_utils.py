import numpy as np
from astropy import coordinates as ac, time as at, units as au
from tomographic_kernel.frames import ENU

from dsa2000_cal.coord_utils import earth_location_to_uvw


def test_enu_to_uvw():
    antennas = ac.EarthLocation.of_site('vla')
    array_location = ac.EarthLocation.of_site('vla')
    time = at.Time('2000-01-01T00:00:00', format='isot')
    pointing = ac.ICRS(0 * au.deg, 90 * au.deg)
    uvw = earth_location_to_uvw(antennas, array_location, time, pointing)
    assert np.linalg.norm(uvw) < 6400 * au.km


    enu_frame = ENU(location=ac.EarthLocation.of_site('vla'), obstime=time)
    antennas = ac.SkyCoord(
        east=np.random.uniform(size=(10,), low=-5, high=5) * au.km,
        north=np.random.uniform(size=(10,), low=-5, high=5) * au.km,
        up=np.random.uniform(size=(10,), low=-5, high=5) * au.km,
        frame=enu_frame
    ).transform_to(ac.ITRS).earth_location
    uvw = earth_location_to_uvw(antennas, array_location, time, pointing)
    assert np.all(np.linalg.norm(uvw, axis=-1) < 6400 * au.km)
