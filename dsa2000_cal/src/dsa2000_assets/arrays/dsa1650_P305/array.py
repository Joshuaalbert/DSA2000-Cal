import os

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au

from dsa2000_assets.arrays.dsa2000W.array import DSA2000WArray
from dsa2000_assets.arrays.dsa2000_optimal_v1.array import DSA2000OptimalV1
from dsa2000_assets.registries import array_registry, beam_model_registry
from dsa2000_common.common.enu_frame import ENU
from dsa2000_fm.antenna_model.abc import AbstractAntennaModel


@array_registry(template='dsa1650_P305')
class DSA1650_P305(DSA2000WArray):
    """
    DSA2000W array class, optimised array layout.
    """

    def get_array_file(self) -> str:
        return os.path.join(*self.content_path, 'antenna_config.txt')

    def get_antenna_model(self) -> AbstractAntennaModel:
        beam_model = beam_model_registry.get_instance(beam_model_registry.get_match('dsa_nominal'))
        return beam_model.get_antenna_model()

    def get_system_equivalent_flux_density(self) -> au.Quantity:
        return 3360. * au.Jy  # Jy

    def get_antenna_diameter(self) -> au.Quantity:
        return 6.1 * au.m


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

