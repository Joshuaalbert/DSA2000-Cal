import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.units import Quantity


def create_uvw_frame(array_location: EarthLocation, time: at.Time, pointing: ac.ICRS) -> ac.SkyOffsetFrame:
    """
    Create a UVW frame for a given array location, time, and pointing.

    Args:
        array_location: the location of array reference location
        time: the time of the observation
        pointing:

    Returns:

    """
    # Convert antenna pos terrestrial to celestial.  For astropy use
    # get_gcrs_posvel(t)[0] rather than get_gcrs(t) because if a velocity
    # is attached to the coordinate astropy will not allow us to do additional
    # transformations with it (https://github.com/astropy/astropy/issues/6280)
    array_position, array_velocity = array_location.get_gcrs_posvel(time)

    array_gcrs = ac.GCRS(array_position,
                         obstime=time,
                         obsgeoloc=array_position,
                         obsgeovel=array_velocity)

    # Define the UVW frame relative to a certain point on the sky.  There are
    # two versions, depending on whether the sky offset is done in ICRS
    # or GCRS:
    # frame_uvw = ac.SkyOffsetFrame(origin=pointing)  # ICRS
    frame_uvw = ac.SkyOffsetFrame(origin=pointing.transform_to(array_gcrs))  # GCRS

    return frame_uvw


def earth_location_to_uvw(antennas: EarthLocation, array_location: EarthLocation, time: at.Time,
                          pointing: ac.ICRS) -> Quantity:
    # Convert antenna pos terrestrial to celestial.  For astropy use
    # get_gcrs_posvel(t)[0] rather than get_gcrs(t) because if a velocity
    # is attached to the coordinate astropy will not allow us to do additional
    # transformations with it (https://github.com/astropy/astropy/issues/6280)
    array_position, array_velocity = array_location.get_gcrs_posvel(time)
    antennas_position = antennas.get_gcrs_posvel(time)[0]
    antennas_gcrs = ac.GCRS(antennas_position,
                            obstime=time, obsgeoloc=array_position, obsgeovel=array_velocity)

    frame_uvw = create_uvw_frame(array_location=array_location, time=time, pointing=pointing)

    # Rotate antenna positions into UVW frame.
    antennas_uvw = antennas_gcrs.transform_to(frame_uvw)

    w, u, v = antennas_uvw.cartesian.xyz
    return ac.CartesianRepresentation(u, v, w).xyz.T


def icrs_to_lmn(sources: ac.ICRS, array_location: ac.EarthLocation, time: at.Time, pointing: ac.ICRS) -> Quantity:
    """
    Convert ICRS coordinates to LMN coordinates.

    Args:
        sources: [num_sources] ICRS coordinates
        array_location: the location of array reference location
        time: the time of the observation
        pointing: the pointing direction

    Returns:
        [num_sources, 3] LMN coordinates
    """
    frame = create_uvw_frame(array_location=array_location, time=time, pointing=pointing)
    # Unsure why, but order is not l,m,n but rather n,l,m (verified)
    n, l, m = sources.transform_to(frame).cartesian.xyz.value
    lmn = ac.CartesianRepresentation(l, m, n).xyz.T
    return lmn


def lmn_to_icrs(lmn: Quantity, array_location: ac.EarthLocation, time: at.Time, pointing: ac.ICRS) -> ac.ICRS:
    """
    Convert LMN coordinates to ICRS coordinates.

    Args:
        lmn: [num_sources, 3] LMN coordinates
        array_location: the location of array reference location
        time: the time of the observation
        pointing: the pointing direction

    Returns:
        [num_sources] ICRS coordinates
    """
    frame = create_uvw_frame(array_location=array_location, time=time, pointing=pointing)
    lmn = lmn.reshape((-1, 3))
    # Swap back order to n, l, m
    cartesian_rep = ac.CartesianRepresentation(lmn[:, 2], lmn[:, 0], lmn[:, 1])
    sources = ac.SkyCoord(cartesian_rep, frame=frame).transform_to(ac.ICRS)
    return sources


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