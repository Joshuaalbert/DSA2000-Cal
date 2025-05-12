import astropy.coordinates as ac
import astropy.units as au
import numpy as np


def main():
    lon, lat, height = [], [], []

    with open('brad_verify.txt', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.strip() == '':
                continue
            _, _lon, _lat, _ele = line.split()
            lon.append(float(_lon))
            lat.append(float(_lat))
            height.append(float(_ele))

    antennas = ac.EarthLocation.from_geodetic(lon * au.deg, lat * au.deg, height * au.m)
    print(antennas)

    solution_file = '/home/albert/Downloads/dsa1650_a_P305_v2.4.6.txt'
    x, y, z = [], [], []
    with open(solution_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.strip() == '':
                continue
            _x, _y, _z = line.split(',')
            x.append(float(_x))
            y.append(float(_y))
            z.append(float(_z))

    antennas_sol = ac.EarthLocation.from_geocentric(x * au.m, y * au.m, z * au.m)
    print(antennas_sol)

    # For each antenna in `antennas`, find the closest antenna in `antennas_sol`
    antennas_itrs = antennas.get_itrs().cartesian.xyz.T
    antennas_sol_itrs = antennas_sol.get_itrs().cartesian.xyz.T
    dist = []
    for i, antenna in enumerate(antennas_itrs):
        # Calculate the distance to each antenna in antennas_sol
        distances = np.linalg.norm(antennas_sol_itrs - antenna, axis=1)
        # Find the index of the closest antenna
        closest_index = np.argmin(distances)
        # Print the result
        print(
            f"Antenna {i}: Closest antenna in solution is {closest_index} with distance {distances[closest_index]:.2f}")
        dist.append(distances[closest_index].value)

    # Print the average distance
    print(f"Average distance: {np.mean(dist):.2f}")
    # Print the maximum distance
    print(f"Maximum distance: {np.max(dist):.2f}")
    # Print the minimum distance
    print(f"Minimum distance: {np.min(dist):.2f}")


if __name__ == '__main__':
    main()
