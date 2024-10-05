import astropy.coordinates as ac
import astropy.units as au
import numpy as np
import pylab as plt


def fibonacci_celestial_sphere(n: int) -> ac.ICRS:
    """
    Generates 'n' points on the surface of a sphere using the Fibonacci sphere algorithm.

    Args:
        n (int): Number of points to generate.

    Returns:
        lon (jnp.ndarray): Array of longitudes in radians.
        lat (jnp.ndarray): Array of latitudes in radians.
    """
    # Golden angle in radians
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # Approximately 2.39996 radians

    # Indices from 0 to n-1
    indices = np.arange(n)

    # Compute y coordinates (latitude component)
    y = 1 - (2 * indices) / (n - 1)  # y ranges from 1 to -1

    # Compute latitude in radians
    lat = np.arcsin(y)

    # Compute longitude in radians
    theta = golden_angle * indices
    lon = theta % (2 * np.pi)  # Ensure longitude is within [0, 2π)

    return ac.ICRS(lon * au.rad, lat * au.rad)


if __name__ == '__main__':
    for n in [100, 1000, 10000]:
        pointings = fibonacci_celestial_sphere(n=n)
        plt.scatter(pointings.ra, pointings.dec, s=1)
        plt.show()

        mean_area = (4 * np.pi / n) * au.rad ** 2
        print(n, mean_area.to('deg^2'))
