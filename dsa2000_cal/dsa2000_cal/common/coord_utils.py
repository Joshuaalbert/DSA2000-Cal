import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.units import Quantity
from tomographic_kernel.frames import ENU


def create_uvw_frame(obs_time: at.Time, phase_tracking: ac.ICRS, barycentre: str = 'earth') -> ac.SkyOffsetFrame:
    """
    Create a UVW frame for a given array location, time, and pointing.

    Note: comparison with CASA shows accuracy of UVW coordinates with error standard deviation of 1.6cm over DSA2000W.

    Args:
        obs_time: the time of the observation
        phase_tracking: the phase tracking direction
        barycentre: the barycentre of the observation, either 'earth' or 'sun'.
            Sun should be more accurate, but in checks with CASA seems to use Earth.

    Returns:
        UVW frame
    """

    if barycentre == 'sun':
        earth_location, earth_velocity = ac.get_body_barycentric_posvel('earth', obs_time, ephemeris='builtin')
        obs_position = sun_location = -earth_location
        obs_velocity = sun_velocity = -earth_velocity
    elif barycentre == 'earth':
        obs_location = ac.EarthLocation.from_geocentric(0 * au.m, 0 * au.m, 0 * au.m)
        obs_position, obs_velocity = obs_location.get_gcrs_posvel(obs_time)
    else:
        raise ValueError("barycentre must be 'earth' or 'sun'.")

    # print(f"Obs position: {obs_position.xyz.decompose()}")
    # print(f"Obs velocity: {obs_velocity.xyz.decompose()}")

    gcrs_frame = ac.GCRS(
        obstime=obs_time,
        obsgeoloc=obs_position,
        obsgeovel=obs_velocity
    )

    frame_uvw = ac.SkyOffsetFrame(origin=phase_tracking.transform_to(gcrs_frame),
                                  obstime=obs_time,
                                  obsgeoloc=obs_position,
                                  obsgeovel=obs_velocity
                                  )  # GCRS
    return frame_uvw


def earth_location_to_uvw(antennas: EarthLocation, obs_time: at.Time, phase_tracking: ac.ICRS) -> Quantity:
    """
    Convert EarthLocation coordinates to UVW coordinates.

    Args:
        antennas: (shape) EarthLocation coordinates
        obs_time: observation time
        phase_tracking: the phase tracking direction

    Returns:
        (shape) + [3] UVW coordinates
    """

    shape = antennas.shape
    antennas: ac.EarthLocation = antennas.reshape((-1,))

    antennas_gcrs = antennas.get_gcrs(obstime=obs_time)
    frame_uvw = create_uvw_frame(obs_time=obs_time, phase_tracking=phase_tracking)

    # Rotate antenna positions into UVW frame.
    antennas_uvw = antennas_gcrs.transform_to(frame_uvw)

    w, u, v = antennas_uvw.cartesian.xyz
    uvw = ac.CartesianRepresentation(u, v, w).xyz.T
    uvw = uvw.reshape(shape + (3,))
    return uvw


def icrs_to_lmn(sources: ac.ICRS, time: at.Time, phase_tracking: ac.ICRS) -> Quantity:
    """
    Convert ICRS coordinates to LMN coordinates.

    Args:
        sources: [num_sources] ICRS coordinates
        time: the time of the observation
        phase_tracking: the pointing direction

    Returns:
        [num_sources, 3] LMN coordinates
    """
    source_shape = sources.shape
    time_shape = time.shape
    phase_tracking_shape = phase_tracking.shape
    # Assert that they broadcast
    try:
        np.broadcast_shapes(source_shape, time_shape, phase_tracking_shape)
    except ValueError:
        raise ValueError(
            f"Shapes of sources {source_shape}, time {time_shape}, "
            f"and phase_tracking {phase_tracking_shape} do not broadcast."
        )
    frame = create_uvw_frame(obs_time=time, phase_tracking=phase_tracking)
    # Unsure why, but order is not l,m,n but rather n,l,m (verified)
    n, l, m = sources.transform_to(frame).cartesian.xyz
    lmn = au.Quantity(np.stack([l, m, n], axis=-1))
    return lmn


def lmn_to_icrs(lmn: Quantity, time: at.Time, phase_tracking: ac.ICRS) -> ac.ICRS:
    """
    Convert LMN coordinates to ICRS coordinates.

    Args:
        lmn: [num_sources, 3] LMN coordinates
        time: the time of the observation
        phase_tracking: the pointing direction

    Returns:
        [num_sources] ICRS coordinates
    """
    frame = create_uvw_frame(obs_time=time, phase_tracking=phase_tracking)
    # Swap back order to n, l, m
    cartesian_rep = ac.CartesianRepresentation(lmn[..., 2], lmn[..., 0], lmn[..., 1])
    sources = ac.SkyCoord(cartesian_rep, frame=frame).transform_to(ac.ICRS)
    # Enforce instance type
    sources = ac.ICRS(sources.ra, sources.dec)
    return sources


def icrs_to_enu(sources: ac.ICRS, array_location: ac.EarthLocation, time: at.Time) -> Quantity:
    """
    Convert ICRS coordinates to ENU coordinates.

    Args:
        sources: [num_sources] ICRS coordinates
        array_location: the location of array reference location
        time: the time of the observation

    Returns:
        [num_sources, 3] ENU coordinates
    """
    shape = sources.shape
    sources = sources.reshape((-1,))

    enu_frame = ENU(location=array_location, obstime=time)
    enu = sources.transform_to(enu_frame).cartesian.xyz.T
    enu = enu.reshape(shape + (3,))
    return enu


def enu_to_icrs(enu: Quantity, array_location: ac.EarthLocation, time: at.Time) -> ac.ICRS:
    """
    Convert ENU coordinates to ICRS coordinates.

    Args:
        enu: [num_sources, 3] ENU coordinates
        array_location: the location of array reference location
        time: the time of the observation

    Returns:
        [num_sources] ICRS coordinates
    """
    shape = enu.shape
    enu = enu.reshape((-1, 3))

    enu_frame = ENU(location=array_location, obstime=time)
    sources = ac.SkyCoord(enu, frame=enu_frame).transform_to(ac.ICRS)
    sources = sources.reshape(shape[:-1])
    return sources


def earth_location_to_enu(antennas: EarthLocation, array_location: EarthLocation, time: at.Time) -> Quantity:
    """
    Convert EarthLocation coordinates to ENU coordinates.

    Args:
        antennas: [num_ant] EarthLocation coordinates
        array_location: the location of array reference location
        time: the time of the observation

    Returns:
        [num_ant, 3] ENU coordinates
    """
    shape = antennas.shape
    antennas = antennas.reshape((-1,))

    antennas_itrs = antennas.get_itrs(obstime=time)

    enu_frame = ENU(location=array_location, obstime=time)
    enu = antennas_itrs.transform_to(enu_frame).cartesian.xyz.T
    enu = enu.reshape(shape + (3,))
    return enu
