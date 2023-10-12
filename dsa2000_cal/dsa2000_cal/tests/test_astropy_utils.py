import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, Angle

from dsa2000_cal.astropy_utils import rotate_icrs_direction


# The rotate_icrs_direction function from the previous response should be here or imported

def test_rotate_near_north_pole():
    direction = SkyCoord(ra=10 * u.deg, dec=85 * u.deg, frame='icrs')
    new_direction = rotate_icrs_direction(direction, Angle(20 * u.deg), Angle(10 * u.deg))
    assert new_direction.dec.deg < 90
    assert np.isclose(new_direction.ra.deg, 30 + 180)  # 10 + 20 + 180 (because it flipped over the pole)


def test_rotate_near_south_pole():
    direction = SkyCoord(ra=10 * u.deg, dec=-85 * u.deg, frame='icrs')
    new_direction = rotate_icrs_direction(direction, Angle(20 * u.deg), Angle(-10 * u.deg))
    assert new_direction.dec.deg > -90
    assert np.isclose(new_direction.ra.deg, 30 + 180)  # 10 + 20 + 180 (because it flipped over the pole)


def test_ra_wrap_around():
    direction = SkyCoord(ra=350 * u.deg, dec=0 * u.deg, frame='icrs')
    new_direction = rotate_icrs_direction(direction, Angle(20 * u.deg), Angle(0 * u.deg))
    assert np.isclose(new_direction.ra.deg, 10)  # 350 + 20 - 360


def test_no_rotation():
    direction = SkyCoord(ra=10 * u.deg, dec=10 * u.deg, frame='icrs')
    new_direction = rotate_icrs_direction(direction, Angle(0 * u.deg), Angle(0 * u.deg))
    assert np.isclose(new_direction.ra.deg, 10)
    assert np.isclose(new_direction.dec.deg, 10)


# def test_large_rotation():
#     direction = SkyCoord(ra=10*u.deg, dec=10*u.deg, frame='icrs')
#     new_direction = rotate_icrs_direction(direction, Angle(400*u.deg), Angle(400*u.deg))
#     # The results here depend on how the function handles rotations larger than 360 degrees.
#     # Adjust assertions accordingly. This example assumes such rotations get wrapped.

def test_negative_rotation():
    direction = SkyCoord(ra=10 * u.deg, dec=10 * u.deg, frame='icrs')
    new_direction = rotate_icrs_direction(direction, Angle(-20 * u.deg), Angle(-20 * u.deg))
    assert np.isclose(new_direction.ra.deg, 350)
    assert np.isclose(new_direction.dec.deg, -10)
