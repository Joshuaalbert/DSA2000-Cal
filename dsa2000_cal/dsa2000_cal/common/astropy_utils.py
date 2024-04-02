import numpy as np
from astropy import coordinates as ac
from astropy import units as au
from astropy.coordinates import Angle
from astropy.coordinates.angles import offset_by


def create_spherical_grid(pointing: ac.ICRS, angular_width: au.Quantity, dr: au.Quantity) -> ac.ICRS:
    """
    Create a spherical grid around a given coordinate with evenly spaced shells.

    Parameters:
    center_coord (SkyCoord): The central ICRS coordinate.
    angular_width (float): The radius, in degrees.
    dr (float): The angular distance between shells, in degrees.

    Returns:
    list of SkyCoord: The coordinates forming the spherical grid.
    """
    if not angular_width.unit.is_equivalent(au.deg) or not dr.unit.is_equivalent(au.deg):
        raise ValueError("Angular width and dr must be in angluar units.")

    pointing = ac.SkyCoord(ra=pointing.ra, dec=pointing.dec, frame='icrs')

    grid_points = [pointing]
    num_shells = int(angular_width / dr)

    for i in range(1, num_shells + 1):
        ri = i * dr  # Radius of the shell
        ni = int(2 * np.pi * ri / dr)  # Number of points in the shell, ensuring even spacing

        # Create even distribution of points on the shell

        angle_offsets = np.linspace(0, 2 * np.pi, ni, endpoint=False) * au.rad
        distances = np.ones(ni) * ri

        # Calculate the point's position in 3D space
        points = pointing.directional_offset_by(angle_offsets, distances)
        grid_points.append(points)

    if len(grid_points) > 1:
        coords = ac.concatenate(grid_points).transform_to(ac.ICRS)
    else:
        coords = grid_points[0].reshape((1,))
    return ac.ICRS(ra=coords.ra, dec=coords.dec)


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
