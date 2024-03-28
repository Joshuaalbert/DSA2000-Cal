import astropy.units as u
import numpy as np
from astropy import coordinates as ac, units as au
from astropy.coordinates import SkyCoord, Angle
from matplotlib import pyplot as plt

from dsa2000_cal.common.astropy_utils import rotate_icrs_direction, random_discrete_skymodel, mean_icrs, \
    create_spherical_grid, create_spherical_earth_grid


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


def test_random_discrete_skymodel():
    pointing = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    angular_width = 1 * au.deg
    n = 100
    coords = random_discrete_skymodel(pointing, angular_width, n)
    assert np.all(coords.separation(pointing).max() <= angular_width)

    # Near poles
    pointing = ac.ICRS(ra=0 * au.deg, dec=90 * au.deg)
    coords = random_discrete_skymodel(pointing, angular_width, n)
    assert np.all(coords.separation(pointing).max() <= angular_width)

    pointing = ac.ICRS(ra=0 * au.deg, dec=-90 * au.deg)
    coords = random_discrete_skymodel(pointing, angular_width, n)
    assert np.all(coords.separation(pointing).max() <= angular_width)


def test_mean_icrs():
    coords = ac.ICRS([0, 1] * au.deg, [2, 3] * au.deg)
    mean_coord = mean_icrs(coords)
    print(mean_coord)
    np.testing.assert_allclose(mean_coord.ra.deg, 0.5, atol=5e-4)
    np.testing.assert_allclose(mean_coord.dec.deg, 2.5, atol=5e-4)


def test_create_spherical_grid():
    pointing = ac.ICRS(ra=10 * au.deg, dec=0 * au.deg)
    angular_width = 1 * au.deg
    dr = 0.1 * au.deg
    grid = create_spherical_grid(pointing, angular_width, dr)
    plt.scatter(grid.ra.rad, grid.dec.rad, marker='o')
    plt.show()

    assert len(grid) > 0
    np.testing.assert_allclose(grid.separation(pointing).max(), angular_width, atol=1e-10)
    np.testing.assert_allclose(np.diff(grid.separation(pointing)).max(), dr, atol=1e-10)


def test_create_spherical_earth_grid():
    center = ac.EarthLocation.from_geodetic(10 * au.deg, 0 * au.deg, 0 * au.m)
    radius = 10. * au.km
    dr = 0.8 * au.km
    grid_earth = create_spherical_earth_grid(center, radius, dr)
    import pylab as plt
    plt.scatter(grid_earth.geodetic[0], grid_earth.geodetic[1], marker='o')
    # plt.scatter(center.geodetic[0], center.geodetic[1], marker='x')
    plt.show()
    assert np.linalg.norm(grid_earth.get_itrs().cartesian.xyz.T - center.get_itrs().cartesian.xyz, axis=-1).max() <= radius

    print(len(grid_earth))
