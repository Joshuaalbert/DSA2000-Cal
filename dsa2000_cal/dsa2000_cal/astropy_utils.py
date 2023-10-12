import numpy as np
from astropy import coordinates as ac
from astropy import units as au


def mean_itrs(coords: ac.ITRS) -> ac.ITRS:
    """
    Returns the mean ITRS coordinate from a list of ITRS coordinates.

    Args:
        coords: list of ITRS coordinates

    Returns:
        the mean ITRS coordinate
    """
    return ac.ITRS(x=np.mean(coords.cartesian.x),
                   y=np.mean(coords.cartesian.y),
                   z=np.mean(coords.cartesian.z)
                   )


def rotate_icrs_direction(direction: ac.ICRS, ra_rotation: ac.Angle, dec_rotation: ac.Angle) -> ac.ICRS:
    """
    Rotates an ICRS direction by the specified RA and DEC angles.

    Args:
        direction (ac.ICRS): Initial direction in ICRS coordinates.
        ra_rotation (Angle): Rotation angle in RA.
        dec_rotation (Angle): Rotation angle in DEC.

    Returns:
        ac.ICRS: New rotated direction in ICRRS coordinates.
    """
    # Apply RA rotation
    new_ra = direction.ra + ra_rotation
    # Apply DEC rotation, and handle poles
    new_dec = direction.dec + dec_rotation
    if new_dec > 90 * au.deg:
        new_dec = 180 * au.deg - new_dec
        new_ra += 180 * au.deg
    elif new_dec < -90 * au.deg:
        new_dec = -180 * au.deg - new_dec
        new_ra += 180 * au.deg

    # Ensure RA is within [0, 360) range
    new_ra = new_ra.wrap_at(360 * au.deg)

    return ac.ICRS(ra=new_ra, dec=new_dec)
