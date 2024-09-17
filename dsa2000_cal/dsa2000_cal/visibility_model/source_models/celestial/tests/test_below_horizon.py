import numpy as np

from dsa2000_cal.visibility_model.source_models.celestial.below_horizon import compute_elevation


def test_elevation():
    # Test with astropy
    from astropy.coordinates import EarthLocation, AltAz, ICRS
    from astropy.time import Time
    from astropy import units as au

    # Set up the observer
    array_location = EarthLocation.from_geodetic(0, 0, 0)
    obstime = Time('2021-01-01T00:00:00', scale='utc')

    # Set up the source
    source = ICRS(ra=0 * au.deg, dec=0 * au.deg)

    lat = array_location.lat
    dec = source.dec
    lst = obstime.sidereal_time(
        kind='apparent',
        longitude=array_location.lon
    )
    hour_angle = lst - source.ra

    print(compute_elevation(hour_angle.rad, lat.rad, dec.rad))

    altaz = source.transform_to(AltAz(obstime=obstime, location=array_location))
    print(altaz.alt.rad)
    np.testing.assert_allclose(compute_elevation(hour_angle.rad, lat.rad, dec.rad), altaz.alt.rad, atol=0.01)
