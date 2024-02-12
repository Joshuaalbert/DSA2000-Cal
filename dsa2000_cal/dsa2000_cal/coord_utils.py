import astropy.coordinates as ac
import astropy.time as at
from astropy.coordinates import EarthLocation
from astropy.units import Quantity
import numpy as np

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

