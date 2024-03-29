import numpy as np
from astropy import coordinates as ac, time as at, units as au
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import earth_location_to_uvw, icrs_to_lmn, lmn_to_icrs


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


def test_lmn_to_icrs():
    array_location = ac.EarthLocation.from_geodetic(0, 0, 0)
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    pointing = ac.ICRS(0 * au.deg, 0 * au.deg)
    sources = ac.ICRS(4 * au.deg, 2 * au.deg)
    lmn = icrs_to_lmn(sources, array_location, time, pointing)
    reconstructed_sources = lmn_to_icrs(lmn, array_location, time, pointing)
    print(sources)
    print(lmn)
    print(reconstructed_sources)
    assert sources.separation(reconstructed_sources).max() < 1e-10 * au.deg

    sources = ac.ICRS([1,2,3,4]*au.deg, [1,2,3,4]*au.deg).reshape((2,2))
    lmn = icrs_to_lmn(sources, array_location, time, pointing)
    assert lmn.shape == (2, 2, 3)
    reconstructed_sources = lmn_to_icrs(lmn, array_location, time, pointing)
    assert reconstructed_sources.shape == (2, 2)
    assert sources.separation(reconstructed_sources).max() < 1e-10 * au.deg


def test_lmn_to_icrs_near_poles():
    array_location = ac.EarthLocation.from_geodetic(0, 0, 0)
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    lmn = au.Quantity(
        [
            [0.05, 0.0, np.sqrt(1 - 0.05 ** 2)],
            [0.0, 0.05, np.sqrt(1 - 0.05 ** 2)],
            [0.0, 0.0, 1],
        ]
    )

    pointing_north_pole = ac.ICRS(0 * au.deg, 90 * au.deg)
    sources = lmn_to_icrs(lmn, array_location, time, pointing_north_pole)
    print(sources)

    lmn_reconstructed = icrs_to_lmn(sources, array_location, time, pointing_north_pole)
    print(lmn_reconstructed)

    np.testing.assert_allclose(lmn, lmn_reconstructed, atol=1e-10)

    # Near south pole
    pointing_south_pole = ac.ICRS(0 * au.deg, -90 * au.deg)
    sources = lmn_to_icrs(lmn, array_location, time, pointing_south_pole)
    print(sources)

    lmn_reconstructed = icrs_to_lmn(sources, array_location, time, pointing_south_pole)
    print(lmn_reconstructed)

    np.testing.assert_allclose(lmn, lmn_reconstructed, atol=1e-10)
