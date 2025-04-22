import os

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au

from dsa2000_assets.arrays.dsa2000W.array import DSA2000WArray
from dsa2000_assets.arrays.dsa2000_optimal_v1.array import DSA2000OptimalV1
from dsa2000_assets.registries import array_registry, beam_model_registry
from dsa2000_common.common.enu_frame import ENU
from dsa2000_fm.antenna_model.abc import AbstractAntennaModel


@array_registry(template='dsa1650_a35')
class DSA1650_A35(DSA2000WArray):
    """
    DSA2000W array class, optimised array layout.
    """

    def get_array_file(self) -> str:
        return os.path.join(*self.content_path, 'dsa1650_a35_antenna_config.txt')

    def get_antenna_model(self) -> AbstractAntennaModel:
        beam_model = beam_model_registry.get_instance(beam_model_registry.get_match('dsa_nominal'))
        return beam_model.get_antenna_model()

    def get_system_equivalent_flux_density(self) -> au.Quantity:
        return 3360. * au.Jy  # Jy

    def get_antenna_diameter(self) -> au.Quantity:
        return 6.1 * au.m



@array_registry(template='dsa1650_a35')
class DSA1650_A29_Prelim(DSA2000WArray):
    """
    DSA2000W array class, optimised array layout.
    """

    def get_array_file(self) -> str:
        return os.path.join(*self.content_path, 'dsa1650_a29_prelim_antenna_config.txt')

    def get_antenna_model(self) -> AbstractAntennaModel:
        beam_model = beam_model_registry.get_instance(beam_model_registry.get_match('dsa_nominal'))
        return beam_model.get_antenna_model()

    def get_system_equivalent_flux_density(self) -> au.Quantity:
        return 3360. * au.Jy  # Jy

    def get_antenna_diameter(self) -> au.Quantity:
        return 6.1 * au.m


def transfer_and_add_station_names():
    x, y, z = [], [], []
    with open('dsa1650_a_2.9_v2-prelim.txt', 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            _x, _y, _z = line.strip().split(',')
            x.append(float(_x))
            y.append(float(_y))
            z.append(float(_z))
        antennas = ac.EarthLocation.from_geocentric(
            x * au.m,
            y * au.m,
            z * au.m
        )

    # add station names
    idx = 0
    with open('dsa1650_a29_prelim_antenna_config.txt', 'w') as f:
        f.write("#station,X,Y,Z\n")
        for antenna in antennas:
            antenna_label = f"dsa-{idx:04d}"
            f.write(f"{antenna_label},{antenna.x.to('m').value},{antenna.y.to('m').value},{antenna.z.to('m').value}\n")
            idx += 1
