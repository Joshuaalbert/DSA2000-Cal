import logging

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
from h5parm import DataPack
from scipy.spatial import KDTree
from tomographic_kernel.tomographic_kernel import TEC_CONV
from tomographic_kernel.utils import make_coord_array, wrap

logger = logging.getLogger(__name__)


def haversine_distance(lon1: Union[float, np.ndarray], lat1: Union[float, np.ndarray], lon2: Union[float, np.ndarray],
                       lat2: Union[float, np.ndarray]) -> np.ndarray:
    """
    Calculate the great circle distance between two points

    Args:
        lon1: the longitude of the first point
        lat1: the latitude of the first point
        lon2: the longitude of the second point
        lat2: the latitude of the second point

    Returns:
        the distance between the two points in radians
    """
    return np.abs(np.arctan2(np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1),
                             np.sin(lon2 - lon1) * np.cos(lat2)))


def calculate_midpoint(lon1: Union[float, np.ndarray], lat1: Union[float, np.ndarray], lon2: Union[float, np.ndarray],
                       lat2: Union[float, np.ndarray]) -> np.ndarray:
    """
    Calculate the midpoint between two coordinates on a sphere.

    Args:
        lon1: longitude of first coordinate
        lat1: latitude of first coordinate
        lon2: longitude of second coordinate
        lat2: latitude of second coordinate

    Returns:
        [2] array of longitude and latitude in radians
    """
    dLon = lon2 - lon1
    Bx = np.cos(lat2) * np.cos(dLon)
    By = np.cos(lat2) * np.sin(dLon)
    lon_mid = np.arctan2(np.sin(lon1) + np.sin(lon2), np.sqrt((np.cos(lat1) + Bx) ** 2 + By ** 2))

    # Find the midpoint latitude
    lat_mid = np.arctan2(np.sin(lat1) + np.sin(lat2), np.sqrt((np.cos(lat1) + Bx) ** 2 + By ** 2))
    return np.asarray([lon_mid, lat_mid])


def compute_mean_coordinates(coords_rad: np.ndarray) -> np.ndarray:
    """
    Compute the mean coordinates of a set of coordinates on a sphere.

    Args:
        coords_rad: [N, 2] array of longitude and latitude in radians

    Returns:
        [2] array of mean longitude and latitude in radians
    """
    # Convert coordinates to radians
    # Convert to unit vectors
    x = np.cos(coords_rad[:, 0]) * np.cos(coords_rad[:, 1])
    y = np.sin(coords_rad[:, 0]) * np.cos(coords_rad[:, 1])
    z = np.sin(coords_rad[:, 1])

    # Compute the mean of unit vectors
    mean_vector = np.asarray([np.mean(x), np.mean(y), np.mean(z)])

    # Convert back to longitude and latitude in radians
    mean_lon = np.arctan2(mean_vector[1], mean_vector[0])
    mean_lat = np.arcsin(mean_vector[2])

    # Convert mean coordinates back to degrees
    mean_coord = np.asarray([mean_lon, mean_lat])

    dist = list(map(lambda coord: haversine_distance(*mean_coord, *coord), coords_rad))
    idx1 = np.argmax(dist)

    dist = list(map(lambda coord: haversine_distance(*coords_rad[idx1], *coord), coords_rad))
    idx2 = np.argmax(dist)

    return calculate_midpoint(*coords_rad[idx1], *coords_rad[idx2])


def map_spherical_to_plane(directions: np.ndarray) -> np.ndarray:
    """
    Map spherical coordinates to a plane using the mean coordinate as the origin.

    Args:
        directions: [Nd, 2] array of spherical coordinates

    Returns:
        [Nd, 2] array of plane coordinates
    """
    lon_mean, lat_mean = compute_mean_coordinates(directions)
    dx = lon_mean - directions[:, 0]
    lon_prime = np.cos(directions[:, 1]) * np.sin(dx)
    lat_prime = lat_mean - directions[:, 1]
    return np.stack([lon_prime, lat_prime], axis=1)


def make_coords(antennas: ac.ITRS, directions: ac.ICRS, times: at.Time):
    logger.info("Interpolating...")
    antennas = antennas.cartesian.xyz.to(au.km).value.T
    directions = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    directions = map_spherical_to_plane(directions=directions)
    times = times.mjd * 86400.
    times -= times[0]
    X = make_coord_array(directions, antennas, times[:, None])
    return X


def interpolate_h5parm(input_h5parm: str, output_h5parm: str, k: int = 3):
    """
    Interpolates a given h5parm onto another one, using simple interpolation which will fail to give good results
    when spacing is too sparse. Works by interpolating dtec.

    Args:
        input_h5parm: input h5parm with tec
        output_h5parm: output h5parm with space for tec, phase
    """
    with DataPack(input_h5parm, readonly=True) as dp:
        assert dp.axes_order == ['pol', 'dir', 'ant', 'freq', 'time']
        dp.current_solset = 'sol000'
        dp.select(pol=slice(0, 1, 1))
        tec_grid, axes = dp.tec
        tec_grid = tec_grid[0]  # remove pol
        tec_grid_flat = tec_grid.flatten()
        _, directions_grid = dp.get_directions(axes['dir'])
        _, antennas_grid = dp.get_antennas(axes['ant'])
        _, times_grid = dp.get_times(axes['time'])

    with DataPack(output_h5parm, readonly=True) as dp:
        assert dp.axes_order == ['pol', 'dir', 'ant', 'freq', 'time']
        dp.current_solset = 'sol000'
        dp.select(pol=slice(0, 1, 1))
        axes = dp.axes_phase
        _, freqs = dp.get_freqs(axes['freq'])
        _, directions = dp.get_directions(axes['dir'])
        _, antennas = dp.get_antennas(axes['ant'])
        _, times = dp.get_times(axes['time'])

    logger.info("Normalising coordinates")
    X_grid = make_coords(directions=directions_grid, antennas=antennas_grid, times=times_grid)
    X_grid -= np.mean(X_grid, axis=0, keepdims=True)
    X_grid /= np.std(X_grid, axis=0, keepdims=True) + 1e-3

    X = make_coords(directions=directions, antennas=antennas, times=times)
    X -= np.mean(X, axis=0, keepdims=True)
    X /= np.std(X, axis=0, keepdims=True) + 1e-3

    logger.info("Interpolating")
    # Step 2: Build the tree and find closest points
    tree = KDTree(X_grid)
    dist, ind = tree.query(X, k=k)  # Change k to find more than 1 nearest point
    if k == 1:  # nearest
        tec_flat = tec_grid_flat[ind]
    else:  # inverse distance weighted
        weights = 1.0 / dist  # [N, K]
        tec_flat = np.sum(weights * tec_grid_flat[ind], axis=1) / np.sum(weights, axis=1)
    tec = tec_flat.reshape((1, len(directions), len(antennas), len(times)))
    logger.info("Storing results.")
    with DataPack(output_h5parm, readonly=False) as dp:
        dp.current_solset = 'sol000'
        dp.select(pol=slice(0, 1, 1))
        dp.tec = tec
        dp.phase = np.asarray(wrap(tec[..., None, :] * (TEC_CONV / freqs.to('Hz').value[:, None])))
