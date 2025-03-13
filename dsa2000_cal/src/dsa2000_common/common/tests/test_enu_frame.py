import numpy as np
import pytest
from astropy import time as at, coordinates as ac, units as au

from dsa2000_common.common.enu_frame import ENU


@pytest.mark.parametrize("array_centre", [
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=0. * au.deg, height=0. * au.m),  # Equator
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=90. * au.deg, height=0. * au.m),  # North Pole
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=-90. * au.deg, height=0. * au.m),  # South Pole
])
@pytest.mark.parametrize("unit", [au.dimensionless_unscaled, au.km])
def test_altaz_to_enu(array_centre: ac.EarthLocation, unit: au.Unit):
    # Define the time and location of the array
    obstime = at.Time("2019-03-19T19:58:14.9", format='isot')
    enu_frame = ENU(obstime=obstime, location=array_centre)

    # Test Zenith direction
    zenith = ac.AltAz(az=0. * au.deg, alt=90. * au.deg, distance=1 * unit, location=array_centre, obstime=obstime)
    coords_enu = zenith.transform_to(enu_frame)
    np.testing.assert_allclose(coords_enu.north, 0. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.east, 0. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.up, 1. * unit, atol=1e-6)

    # Test East direction
    east = ac.AltAz(az=90. * au.deg, alt=0. * au.deg, distance=1 * unit, location=array_centre, obstime=obstime)
    coords_enu = east.transform_to(enu_frame)
    np.testing.assert_allclose(coords_enu.north, 0. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.east, 1. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.up, 0. * unit, atol=1e-6)

    # Test North direction
    north = ac.AltAz(az=0. * au.deg, alt=0. * au.deg, distance=1 * unit, location=array_centre, obstime=obstime)
    coords_enu = north.transform_to(enu_frame)
    np.testing.assert_allclose(coords_enu.north, 1. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.east, 0. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.up, 0. * unit, atol=1e-6)

    # Test South direction
    south = ac.AltAz(az=180. * au.deg, alt=0. * au.deg, distance=1 * unit, location=array_centre, obstime=obstime)
    coords_enu = south.transform_to(enu_frame)
    np.testing.assert_allclose(coords_enu.north, -1. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.east, 0. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.up, 0. * unit, atol=1e-6)

    # Test West direction
    west = ac.AltAz(az=270. * au.deg, alt=0. * au.deg, distance=1 * unit, location=array_centre, obstime=obstime)
    coords_enu = west.transform_to(enu_frame)
    np.testing.assert_allclose(coords_enu.north, 0. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.east, -1. * unit, atol=1e-6)
    np.testing.assert_allclose(coords_enu.up, 0. * unit, atol=1e-6)


@pytest.mark.parametrize("array_centre", [
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=0. * au.deg, height=0. * au.m),  # Equator
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=90. * au.deg, height=0. * au.m),  # North Pole
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=-90. * au.deg, height=0. * au.m),  # South Pole
])
@pytest.mark.parametrize("unit", [au.dimensionless_unscaled, au.km])
def test_enu_to_altaz(array_centre: ac.EarthLocation, unit: au.Unit):
    # Define the time and location of the array
    obstime = at.Time("2019-03-19T19:58:14.9", format='isot')
    altaz_frame = ac.AltAz(obstime=obstime, location=array_centre)

    # Test Zenith direction
    zenith = ENU(north=0. * unit, east=0. * unit, up=1 * unit, location=array_centre, obstime=obstime)
    coords_altaz = zenith.transform_to(altaz_frame)
    np.testing.assert_allclose(coords_altaz.az, 0. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.alt, 90. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.distance, 1 * unit, atol=1e-6)

    # Test East direction
    east = ENU(north=0. * unit, east=1. * unit, up=0 * unit, location=array_centre, obstime=obstime)
    coords_altaz = east.transform_to(altaz_frame)
    np.testing.assert_allclose(coords_altaz.az, 90. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.alt, 0. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.distance, 1 * unit, atol=1e-6)

    # Test North direction
    north = ENU(north=1. * unit, east=0. * unit, up=0 * unit, location=array_centre, obstime=obstime)
    coords_altaz = north.transform_to(altaz_frame)
    np.testing.assert_allclose(coords_altaz.az, 0. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.alt, 0. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.distance, 1 * unit, atol=1e-6)

    # Test South direction
    south = ENU(north=-1. * unit, east=0. * unit, up=0 * unit, location=array_centre, obstime=obstime)
    coords_altaz = south.transform_to(altaz_frame)
    np.testing.assert_allclose(coords_altaz.az, 180. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.alt, 0. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.distance, 1 * unit, atol=1e-6)

    # Test West direction
    west = ENU(north=0. * unit, east=-1. * unit, up=0 * unit, location=array_centre, obstime=obstime)
    coords_altaz = west.transform_to(altaz_frame)
    np.testing.assert_allclose(coords_altaz.az, 270. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.alt, 0. * au.deg, atol=1e-6)
    np.testing.assert_allclose(coords_altaz.distance, 1 * unit, atol=1e-6)


@pytest.mark.parametrize("to_array_centre", [
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=0. * au.deg, height=0. * au.m),  # Equator
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=90. * au.deg, height=0. * au.m),  # North Pole
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=-90. * au.deg, height=0. * au.m),  # South Pole
])
@pytest.mark.parametrize("from_array_centre", [
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=0. * au.deg, height=0. * au.m),  # Equator
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=90. * au.deg, height=0. * au.m),  # North Pole
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=-90. * au.deg, height=0. * au.m),  # South Pole
])
@pytest.mark.parametrize("from_time", [
    at.Time("2019-03-19T19:58:14.9", format='isot'),
    at.Time("2020-03-19T19:58:14.9", format='isot'),
])
@pytest.mark.parametrize("to_time", [
    at.Time("2019-03-19T19:58:14.9", format='isot'),
    at.Time("2020-03-19T19:58:14.9", format='isot'),
])
@pytest.mark.parametrize("unit", [au.dimensionless_unscaled, au.km])
def test_enu_to_enu(from_array_centre: ac.EarthLocation,
                    to_array_centre: ac.EarthLocation,
                    from_time: at.Time,
                    to_time: at.Time,
                    unit: au.Unit):
    # Transform from one to the other, and back, and get the same result

    # Test Zenith direction
    zenith = ENU(north=0. * unit, east=0. * unit, up=1 * unit, location=from_array_centre, obstime=from_time)
    zenith_to = zenith.transform_to(ENU(location=to_array_centre, obstime=to_time))
    zenith_return = zenith_to.transform_to(ENU(location=from_array_centre, obstime=from_time))
    np.testing.assert_allclose(zenith.north, zenith_return.north, atol=1e-6)
    np.testing.assert_allclose(zenith.east, zenith_return.east, atol=1e-6)
    np.testing.assert_allclose(zenith.up, zenith_return.up, atol=1e-6)

    # Test East direction
    east = ENU(north=0. * unit, east=1. * unit, up=0 * unit, location=from_array_centre, obstime=from_time)
    east_to = east.transform_to(ENU(location=to_array_centre, obstime=to_time))
    east_return = east_to.transform_to(ENU(location=from_array_centre, obstime=from_time))
    np.testing.assert_allclose(east.north, east_return.north, atol=1e-6)
    np.testing.assert_allclose(east.east, east_return.east, atol=1e-6)
    np.testing.assert_allclose(east.up, east_return.up, atol=1e-6)

    # Test North direction
    north = ENU(north=1. * unit, east=0. * unit, up=0 * unit, location=from_array_centre, obstime=from_time)
    north_to = north.transform_to(ENU(location=to_array_centre, obstime=to_time))
    north_return = north_to.transform_to(ENU(location=from_array_centre, obstime=from_time))
    np.testing.assert_allclose(north.north, north_return.north, atol=1e-6)
    np.testing.assert_allclose(north.east, north_return.east, atol=1e-6)
    np.testing.assert_allclose(north.up, north_return.up, atol=1e-6)

    # Test South direction
    south = ENU(north=-1. * unit, east=0. * unit, up=0 * unit, location=from_array_centre, obstime=from_time)
    south_to = south.transform_to(ENU(location=to_array_centre, obstime=to_time))
    south_return = south_to.transform_to(ENU(location=from_array_centre, obstime=from_time))
    np.testing.assert_allclose(south.north, south_return.north, atol=1e-6)
    np.testing.assert_allclose(south.east, south_return.east, atol=1e-6)
    np.testing.assert_allclose(south.up, south_return.up, atol=1e-6)

    # Test West direction
    west = ENU(north=0. * unit, east=-1. * unit, up=0 * unit, location=from_array_centre, obstime=from_time)
    west_to = west.transform_to(ENU(location=to_array_centre, obstime=to_time))
    west_return = west_to.transform_to(ENU(location=from_array_centre, obstime=from_time))
    np.testing.assert_allclose(west.north, west_return.north, atol=1e-6)
    np.testing.assert_allclose(west.east, west_return.east, atol=1e-6)
    np.testing.assert_allclose(west.up, west_return.up, atol=1e-6)


@pytest.mark.parametrize("array_centre", [
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=0. * au.deg, height=0. * au.m),  # Equator
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=90. * au.deg, height=0. * au.m),  # North Pole
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=-90. * au.deg, height=0. * au.m),  # South Pole
])
def test_earth_centre(array_centre: ac.EarthLocation):
    obstime = at.Time("2019-03-19T19:58:14.9", format='isot')
    # earth location
    earth_centre = ac.EarthLocation.from_geocentric(x=0 * au.m, y=0 * au.m, z=0 * au.m).get_itrs(obstime=obstime)
    coords_enu = earth_centre.transform_to(ENU(location=array_centre, obstime=obstime))
    print(coords_enu)
    np.testing.assert_allclose(coords_enu.up, -6360 * au.km, atol=20 * au.km)


@pytest.mark.parametrize("array_centre", [
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=0. * au.deg, height=0. * au.m),  # Equator
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=90. * au.deg, height=0. * au.m),  # North Pole
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=-90. * au.deg, height=0. * au.m),  # South Pole
])
@pytest.mark.parametrize("unit", [au.dimensionless_unscaled, au.km])
def test_enu_to_itrs(array_centre: ac.EarthLocation, unit: au.Unit):
    obstime = at.Time("2019-03-19T19:58:14.9", format='isot')

    # Test Zenith direction
    zenith_altaz = ac.AltAz(az=0. * au.deg, alt=90. * au.deg, distance=1 * unit, location=array_centre, obstime=obstime)
    zenith_enus = zenith_altaz.transform_to(ENU(location=array_centre, obstime=obstime))
    zenith_itrs = zenith_enus.transform_to(ac.ITRS(obstime=obstime))
    zenith_via_altaz = zenith_altaz.transform_to(ac.ITRS(obstime=obstime))

    print(zenith_itrs)
    print(zenith_via_altaz)

    np.testing.assert_allclose(zenith_itrs.cartesian.xyz, zenith_via_altaz.cartesian.xyz, atol=1e-6)


@pytest.mark.parametrize("array_centre", [
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=0. * au.deg, height=0. * au.m),  # Equator
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=90. * au.deg, height=0. * au.m),  # North Pole
    ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=-90. * au.deg, height=0. * au.m),  # South Pole
])
def test_enu_of_array_centre_should_be_zero(array_centre: ac.EarthLocation):
    obstime = at.Time("2019-03-19T19:58:14.9", format='isot')
    enu_frame = ENU(obstime=obstime, location=array_centre)
    array_centre_enu = array_centre.get_itrs(obstime=obstime, location=array_centre).transform_to(enu_frame)
    np.testing.assert_allclose(array_centre_enu.east, 0. * au.m, atol=1e-6)
    np.testing.assert_allclose(array_centre_enu.north, 0. * au.m, atol=1e-6)
    np.testing.assert_allclose(array_centre_enu.up, 0. * au.m, atol=1e-6)


def test_vector_location():
    locations = ac.EarthLocation.from_geodetic(lon=[0., 0., 0.] * au.deg,
                                               lat=[0., 90., -90.] * au.deg,
                                               height=[0., 0., 0.] * au.m)
    time = at.Time("2019-03-19T19:58:14.9", format='isot')
    zenith = ENU(up=1, east=0, north=0, obstime=time, location=locations)
    assert zenith.shape == (3,)


def test_sep_3d():
    location = ac.EarthLocation.from_geodetic(lon=0. * au.deg, lat=0. * au.deg, height=0. * au.m)
    time = at.Time("2019-03-19T19:58:14.9", format='isot')
    loc1 = ENU(up=0, east=0, north=0, obstime=time, location=location)
    loc2 = ENU(up=0, east=0, north=1, obstime=time, location=location)
    sep = loc1.cartesian.xyz - loc2.cartesian.xyz
    print(sep)
