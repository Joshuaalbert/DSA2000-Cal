import jax
from astropy import coordinates as ac, units as au, time as at
from jax import numpy as jnp

from dsa2000_cal.delay_models.uvw_utils import perley_icrs_from_lmn


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


def get_horizon_mask(lst: jax.Array, lat: jax.Array, ra0: jax.Array, dec0: jax.Array, num_l: int, num_m: int, dl: float,
                     dm: float, l0: float,
                     m0: float) -> jax.Array:
    """
    Get a mask for the image, where the horizon is masked out.

    Args:
        lst: local sidereal time in radians
        lat: latitude of the observer in radians
        ra0: right ascension of the pointing in radians
        dec0: declination of the pointing in radians
        num_l: the number of l points
        num_m: the number of m points
        dl: the l spacing
        dm: the m spacing
        l0: the center l
        m0: the center m


    Returns:
        mask: [num_l, num_m] True if the pixel is above the horizon
    """
    lvec = (-0.5 * num_l + jnp.arange(num_l)) * dl + l0  # [num_l]
    mvec = (-0.5 * num_m + jnp.arange(num_m)) * dm + m0  # [num_m]
    l, m = jnp.meshgrid(lvec, mvec, indexing='ij')  # [num_l, num_m]
    n = jnp.sqrt(1. - (jnp.square(l) + jnp.square(m)))  # [num_l, num_m]
    ra, dec = perley_icrs_from_lmn(l, m, n, ra0, dec0)
    # For each lmn get the ra,dec
    # given the pointing, determine the elevation of each lmn, and set to True if above the horizon.
    hour_angle = lst - ra
    return elevation(hour_angle=hour_angle, lat=lat, dec=dec) > 0.


def elevation(hour_angle: jax.Array, lat: jax.Array, dec: jax.Array) -> jax.Array:
    """
    Calculate the elevation of a source.

    Args:
        hour_angle: hour angle of the source in radians
        lat: latitude of the observer in radians
        dec: declination of the source in radians

    Returns:

    """
    # sin(alt) = sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(HA)
    sin_alt = jnp.sin(dec) * jnp.sin(lat) + jnp.cos(dec) * jnp.cos(lat) * jnp.cos(hour_angle)
    return jnp.arcsin(sin_alt)
