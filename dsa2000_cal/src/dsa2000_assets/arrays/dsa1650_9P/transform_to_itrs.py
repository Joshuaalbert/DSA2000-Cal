import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au

from dsa2000_assets.arrays.dsa2000_optimal_v1.array import DSA2000OptimalV1
from dsa2000_common.common.enu_frame import ENU


def transfer_and_add_station_names():
    old_array = DSA2000OptimalV1(seed='abc')
    array_location = old_array.get_array_location()
    enu = []
    with open('antenna_enu.txt', 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.strip() == "":
                continue
            e, n = map(float, line.split())
            u = 0
            enu.append([e, n, u])

    antennas_enu = enu * au.m
    obstime = at.Time.now()
    antennas = ENU(
        antennas_enu[:, 0], antennas_enu[:, 1], antennas_enu[:, 2], obstime=obstime, location=array_location
    ).transform_to(ac.ITRS(obstime=obstime, location=array_location)).earth_location
    # add station names
    idx = 0
    with open('antenna_config.txt', 'w') as f:
        f.write("#station,X,Y,Z\n")
        for antenna in antennas:
            antenna_label = f"dsa-{idx:04d}"
            f.write(f"{antenna_label},{antenna.x.to('m').value},{antenna.y.to('m').value},{antenna.z.to('m').value}\n")
            idx += 1

if __name__ == "__main__":
    transfer_and_add_station_names()