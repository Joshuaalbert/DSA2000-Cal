from astropy import coordinates as ac, units as au, time as at


def is_below_horizon(direction: ac.ICRS, obstime: at.Time, location: ac.EarthLocation) -> bool:
    """
    Check if a source is below the horizon at a given time and location.

    Args:
        direction: the direction of the source
        obstime: the time of the observation
        location: the location of the observer

    Returns:
        True if the source is below the horizon, False otherwise
    """
    altaz = direction.transform_to(ac.AltAz(obstime=obstime, location=location))
    return altaz.alt < 0 * au.deg
