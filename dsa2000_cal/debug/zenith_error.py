import numpy as np
from astropy import time as at, coordinates as ac, units as au


def test_altaz_origin_maps_to_location():
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    location = ac.EarthLocation.of_site('vla')
    coord = ac.AltAz(alt=0. * au.deg, az=0. * au.deg, distance=0.0 * au.m, obstime=time, location=location)
    coord_itrs = coord.transform_to(ac.ITRS(obstime=time))
    location_itrs = location.get_itrs(obstime=time)
    print("Separation:", coord.separation_3d(location_itrs))
    np.testing.assert_allclose(coord_itrs.cartesian.xyz, location_itrs.cartesian.xyz, atol=1e-6 * au.m)
