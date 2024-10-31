import numpy as np
import pytest
from astropy import coordinates as ac, units as au
from matplotlib import pyplot as plt

from dsa2000_cal.common.astropy_utils import random_discrete_skymodel, mean_icrs, \
    create_spherical_grid_old, create_spherical_earth_grid, create_random_spherical_layout, create_mosaic_tiling, \
    fibonacci_celestial_sphere


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
    grid = create_spherical_grid_old(pointing, angular_width, dr)
    plt.scatter(grid.ra.rad, grid.dec.rad, marker='o')
    plt.show()

    assert len(grid) > 0
    np.testing.assert_allclose(grid.separation(pointing).max(), angular_width, atol=1e-10)
    np.testing.assert_allclose(np.diff(grid.separation(pointing)).max(), dr, atol=1e-10)

    pointing = ac.ICRS(ra=10 * au.deg, dec=0 * au.deg)
    angular_width = 0 * au.deg
    dr = 0.1 * au.deg
    grid = create_spherical_grid_old(pointing, angular_width, dr)
    assert not grid.isscalar
    assert len(grid) == 1


def test_create_spherical_earth_grid():
    center = ac.EarthLocation.from_geodetic(10 * au.deg, 0 * au.deg, 0 * au.m)
    radius = 10. * au.km
    dr = 0.8 * au.km
    grid_earth = create_spherical_earth_grid(center, radius, dr)
    import pylab as plt
    plt.scatter(grid_earth.geodetic[0], grid_earth.geodetic[1], marker='o')
    # plt.scatter(center.geodetic[0], center.geodetic[1], marker='x')
    plt.show()
    assert np.linalg.norm(grid_earth.get_itrs().cartesian.xyz.T - center.get_itrs().cartesian.xyz,
                          axis=-1).max() <= radius

    print(len(grid_earth))


def test_create_spherical_grid_all_sky():
    grid = create_random_spherical_layout(10000)
    plt.scatter(grid.ra.rad, grid.dec.rad, marker='o', alpha=0.1)
    plt.show()

    assert len(grid) == 10000


def test_create_mosaic_tiling():
    tiling_coords = create_mosaic_tiling()
    print(tiling_coords.size)
    import pylab as plt
    plt.scatter(tiling_coords.ra, tiling_coords.dec, s=1)
    plt.show()


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_fibonacci_celestial_sphere(n: int):
    pointings = fibonacci_celestial_sphere(n=n)
    import pylab as plt
    plt.scatter(pointings.ra, pointings.dec, s=1)
    plt.show()

    mean_area = (4 * np.pi / n) * au.rad ** 2
    print(n, mean_area.to('deg^2'))
