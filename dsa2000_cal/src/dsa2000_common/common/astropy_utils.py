from typing import Tuple, List

import astropy.time as at
import astropy.units as au
import jax.random
import numpy as np
from astropy import constants as const
from astropy import coordinates as ac
from astropy.coordinates import Angle, ITRS, CartesianRepresentation
from astropy.coordinates.angles import offset_by

from dsa2000_common.common.coord_utils import lmn_to_icrs


def create_random_spherical_layout(num_sources: int, key=None) -> ac.ICRS:
    if key is None:
        key = jax.random.PRNGKey(0)
    lmn = jax.random.normal(key, (num_sources, 3))
    lmn /= np.linalg.norm(lmn, axis=1, keepdims=True)
    return lmn_to_icrs(lmn=lmn * au.dimensionless_unscaled, phase_center=ac.ICRS(ra=0 * au.deg, dec=90 * au.deg))


def choose_dr(field_of_view: au.Quantity, total_n: int) -> au.Quantity:
    """
    Choose the dr for a given field of view and total number of sources.
    Approximate number of sources result.

    Args:
        field_of_view:
        total_n:

    Returns:
        the sky spacing
    """
    possible_n = [1, 7, 19, 37, 62, 93, 130, 173, 223, 279, 341, 410, 485,
                  566, 653, 747, 847, 953, 1066, 1185]
    if total_n == 1:
        return field_of_view
    dr = 0.5 * field_of_view / np.arange(1, total_n + 1)
    N = np.floor(0.5 * field_of_view / dr)
    num_n = [1 + np.sum(np.floor(2 * np.pi * np.arange(1, n + 1))) for n in N]
    if total_n not in possible_n:
        print(f"total_n {total_n} not exactly achievable. Will be rounded to nearest achievable value "
              f"This is based on a uniform dithering of the sky. Possible values are {possible_n}.")
    return au.Quantity(np.interp(total_n, num_n, dr))


def create_spherical_spiral_grid(pointing: ac.ICRS, num_points: int, angular_radius: au.Quantity,
                                 inner_radius: au.Quantity | None = None) -> ac.ICRS:
    if num_points == 1:
        return pointing[None]
    if inner_radius is None:
        inner_radius = 0 * au.deg
    ra0 = pointing.ra
    dec0 = pointing.dec
    dtheta = 2 * np.pi / num_points ** 0.5
    ra_grid = []
    dec_grid = []
    for i in range(num_points):
        # r = angular_radius * (i / (num_points - 1)) ** 0.5
        r = inner_radius + (angular_radius - inner_radius) * (i / (num_points - 1)) ** 0.5
        theta = i * dtheta
        ra_i, dec_i = offset_by(ra0, dec0, theta, r)
        ra_grid.append(ra_i)
        dec_grid.append(dec_i)
    return ac.ICRS(ra=au.Quantity(ra_grid), dec=au.Quantity(dec_grid))


def create_spherical_grid(pointing: ac.ICRS, angular_radius: au.Quantity, num_shells: int,
                          sky_rotation: au.Quantity | None = None) -> ac.ICRS:
    """
    Create a spherical grid around a given coordinate with evenly spaced shells.

    Parameters:
    center_coord (SkyCoord): The central ICRS coordinate.
    angular_width (float): The radius, in degrees.
    dr (float): The angular distance between shells, in degrees.
    sky_rotation (float): The rotation of the grid in the plane of the sky, in degrees.

    Returns:
        the coordinates forming the spherical grid.
    """

    if not angular_radius.unit.is_equivalent(au.deg):
        raise ValueError("Angular width and dr must be in angluar units.")

    if num_shells == 1:
        return pointing[None]

    dr = angular_radius / (num_shells - 1)

    ra = pointing.ra
    dec = pointing.dec

    grid_ra = [ra]
    grid_dec = [dec]

    for i in range(1, 1 + num_shells):
        ri = i * dr  # Radius of the shell
        ni = int(2 * np.pi * i)  # Number of points in the shell, ensuring even spacing
        for j in range(ni):
            angle_offset = j * 2 * np.pi / ni
            if sky_rotation is not None:
                angle_offset += sky_rotation

            shell_ra, shell_dec = offset_by(ra, dec, angle_offset, ri)

            grid_ra.append(shell_ra)
            grid_dec.append(shell_dec)

    return ac.ICRS(ra=au.Quantity(grid_ra), dec=au.Quantity(grid_dec))


def create_spherical_grid_old(pointing: ac.ICRS, angular_radius: au.Quantity, dr: au.Quantity,
                              sky_rotation: au.Quantity | None = None) -> ac.ICRS:
    """
    Create a spherical grid around a given coordinate with evenly spaced shells.

    Parameters:
    center_coord (SkyCoord): The central ICRS coordinate.
    angular_width (float): The radius, in degrees.
    dr (float): The angular distance between shells, in degrees.
    sky_rotation (float): The rotation of the grid in the plane of the sky, in degrees.

    Returns:
    list of SkyCoord: The coordinates forming the spherical grid.
    """
    if not angular_radius.unit.is_equivalent(au.deg) or not dr.unit.is_equivalent(au.deg):
        raise ValueError("Angular width and dr must be in angluar units.")

    pointing = ac.SkyCoord(ra=pointing.ra, dec=pointing.dec, frame='icrs')

    grid_points = [pointing]
    num_shells = int(angular_radius / dr)

    for i in range(1, num_shells + 1):
        ri = i * dr  # Radius of the shell
        ni = int(2 * np.pi * i)  # Number of points in the shell, ensuring even spacing

        # Create even distribution of points on the shell

        angle_offsets = np.linspace(0, 2 * np.pi, ni, endpoint=False) * au.rad
        if sky_rotation is not None:
            angle_offsets += sky_rotation

        distances = np.ones(ni) * ri

        # Calculate the point's position in 3D space
        points = pointing.directional_offset_by(angle_offsets, distances)
        grid_points.append(points)

    if len(grid_points) > 1:
        coords = ac.concatenate(grid_points).transform_to(ac.ICRS)
    else:
        coords = grid_points[0].reshape((1,))
    return ac.ICRS(ra=coords.ra, dec=coords.dec)


def create_mosaic_tiling(theta_row: au.Quantity = 1.8 * au.deg, theta_hex: au.Quantity = 2.0 * au.deg) -> ac.ICRS:
    dec = np.arange(-90, 90, theta_row.to(au.deg).value) * au.deg
    ra = np.arange(0, 360, theta_hex.to(au.deg).value) * au.deg
    ra, dec = np.meshgrid(ra, dec, indexing='ij')
    return ac.ICRS(ra.flatten(), dec.flatten())


def create_spherical_earth_grid(center: ac.EarthLocation, radius: au.Quantity, dr: au.Quantity) -> ac.EarthLocation:
    if not radius.unit.is_equivalent(au.m) or not dr.unit.is_equivalent(au.m):
        raise ValueError("Radius and dr must be in meters.")
    earth_radius = np.sqrt(center.geocentric[0] ** 2 + center.geocentric[1] ** 2 + center.geocentric[2] ** 2)
    angular_width = (radius / earth_radius).decompose() * au.rad
    dr = (dr / earth_radius).decompose() * au.rad

    slon, slat, height = center.geodetic

    if not angular_width.unit.is_equivalent(au.deg) or not dr.unit.is_equivalent(au.deg):
        raise ValueError("Angular width and dr must be in angluar units.")

    grid_points_lon = [slon]
    grid_points_lat = [slat]
    grid_points_height = [height]
    num_shells = int(angular_width / dr)

    for i in range(1, num_shells + 1):
        ri = i * dr  # Radius of the shell
        ni = int(2 * np.pi * ri / dr)  # Number of points in the shell, ensuring even spacing

        # Create even distribution of points on the shell

        angle_offsets = np.linspace(0, 2 * np.pi, ni, endpoint=False) * au.rad
        distances = np.ones(ni) * ri

        # Calculate the point's position in 3D space

        newlon, newlat = offset_by(
            lon=slon, lat=slat, posang=angle_offsets, distance=distances
        )

        for lon, lat in zip(newlon, newlat):
            grid_points_lon.append(lon)
            grid_points_lat.append(lat)
            grid_points_height.append(height)

    return ac.EarthLocation.from_geodetic(grid_points_lon, grid_points_lat, grid_points_height)


def random_discrete_skymodel(pointing: ac.ICRS, angular_width: au.Quantity, n: int, seed: int = None) -> ac.ICRS:
    """
    Randomly sample n points around a given ICRS coordinate within a specified angular width.

    Parameters:
    coord (SkyCoord): The ICRS coordinate to sample around.
    angular_width (float): The angular width within which to sample points.
    n (int): The number of points to sample.

    Returns:
    list of SkyCoord: The randomly sampled ICRS coordinates.
    """
    if seed is not None:
        np.random.seed(seed)
    # Generate n random angular distances within the specified angular width
    random_distances = angular_width * np.sqrt(np.random.rand(n))

    # Generate n random position angles from 0 to 360 degrees
    random_angles = Angle(360 * np.random.rand(n), unit=au.deg)

    # Compute the new coordinates using the random distances and angles
    pointing = ac.SkyCoord(ra=pointing.ra, dec=pointing.dec, frame='icrs')
    sampled_coords = pointing.directional_offset_by(random_angles, random_distances)

    return ac.ICRS(ra=sampled_coords.ra, dec=sampled_coords.dec)


def mean_icrs(coords: ac.ICRS) -> ac.ICRS:
    """
    Returns the mean ITRS coordinate from a list of ITRS coordinates.

    Args:
        coords: list of ITRS coordinates

    Returns:
        the mean ITRS coordinate
    """
    mean_coord = coords.cartesian.xyz.T.mean(axis=0)
    mean_coord /= np.linalg.norm(mean_coord)
    spherical = ac.ICRS(mean_coord, representation_type='cartesian').spherical
    return ac.ICRS(ra=spherical.lon, dec=spherical.lat)


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


def dimensionless(q: au.Quantity) -> au.Quantity:
    """
    Convert a quantity to dimensionless units.

    Args:
        q: Quantity to convert.

    Returns:
        Quantity in dimensionless units.
    """
    return q.to(au.dimensionless_unscaled)


def get_angular_diameter(coords_icrs: ac.ICRS) -> au.Quantity:
    """
    Get the maximum angular diameter of a set of ICRS coordinates.

    Args:
        coords_icrs: The ICRS coordinates.

    Returns:
        The maximum angular diameter.
    """
    if coords_icrs.isscalar or coords_icrs.shape[0] == 1:
        return au.Quantity(0, 'rad')

    # Get maximal separation
    return max(coord.separation(coords_icrs).max() for coord in coords_icrs).to('deg')


def fraunhofer_far_field_limit(diameter: au.Quantity, freq: au.Quantity) -> au.Quantity:
    """
    Calculate the Fraunhofer far field limit for a given diameter and wavelength.

    Args:
        diameter: The diameter of the aperture.
        freq: The wavelength of the light.

    Returns:
        The Fraunhofer far field limit.
    """
    if not diameter.unit.is_equivalent(au.m):
        raise ValueError("Diameter must be in meters.")
    if not freq.unit.is_equivalent(au.Hz):
        raise ValueError("Frequency must be in Hz.")
    wavelength = const.c / freq
    return (diameter ** 2 / wavelength).to('m')


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


def extract_itrs_coords(filename: str, delim=' ') -> Tuple[List[str], ac.ITRS]:
    """
    Extract stations and antenna ITRS coordinates from a file.

    Args:
        filename: file to read
        delim: delimiter to use for splitting the file

    Returns:
        a tuple of lists of stations and antenna ITRS coordinates
    """
    header = []
    # Initialize lists to store stations and coordinates
    stations = []
    coordinates = []
    station_idx = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if header:
                    raise ValueError(f"Multiple header lines found in {filename}")
                header = list(filter(lambda s: len(s) > 0, map(str.strip, line[1:].lower().split(delim))))
                continue
            if not header:
                raise ValueError(f"No header line found in {filename}")

            # Process each line in the file

            # Split the line into its components
            parsed_line = list(filter(lambda s: len(s) > 0, map(str.strip, line.split(delim))))
            line_dict = dict(zip(header, parsed_line))

            # Convert x, y, z to float and append to the coordinates list
            coordinates.append(
                ITRS(CartesianRepresentation(float(line_dict['x']) * au.m,
                                             float(line_dict['y']) * au.m,
                                             float(line_dict['z']) * au.m))
            )

            # Append the station name to the stations list
            stations.append(line_dict.get('station', f"station_{station_idx}"))
            station_idx += 1
    if len(stations) != len(coordinates):
        raise ValueError(
            f"Number of stations ({len(stations)}) does not match number of coordinates ({len(coordinates)})")
    if len(coordinates) == 0:
        raise ValueError(f"No coordinates found in {filename}")
    if len(coordinates) == 1:
        return stations, coordinates[0].reshape((1,))
    return stations, ac.concatenate(coordinates).transform_to(ITRS())


def get_time_of_local_meridean(coord: ac.ICRS, location: ac.EarthLocation, ref_time: at.Time) -> at.Time:
    """
    Get the closest time from ref_time when a coordinate would be in local transit.

    Args:
        coord:
        ref_time:

    Returns:

    """
    lst = ref_time.sidereal_time('apparent', longitude=location.lon)

    # Compute the next transit time when LST = RA
    delta_lst = (coord.ra - lst).wrap_at(180 * au.deg)  # Ensure proper wraparound
    delta_t = (delta_lst / (15 * au.deg)) * au.hour  # Convert LST difference to hours

    # Calculate exact transit time
    t_transit = ref_time + delta_t

    return t_transit
