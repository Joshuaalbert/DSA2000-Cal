import os
from typing import List

from astropy import coordinates as ac
from astropy import units as au

from dsa2000_cal.abc import AbstractAntennaModel
from dsa2000_cal.antenna_model.h5_efield_model import H5AntennaModelV1
from dsa2000_cal.assets.arrays.array import AbstractArray, extract_itrs_coords
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.astropy_utils import mean_itrs


@array_registry(template='lwa')
class LWAArray(AbstractArray):
    """
    DSA2000W array class.
    """

    def get_channel_width(self) -> au.Quantity:
        return (86874511.71875 - 40960937.5) / 1920 * au.Hz

    def get_array_location(self) -> ac.EarthLocation:
        return mean_itrs(self.get_antennas().get_itrs()).earth_location

    def get_antenna_diameter(self) -> au.Quantity:
        return 2. * au.m

    def get_focal_length(self) -> au.Quantity:
        raise NotImplementedError("Focal length not implemented for LWA")

    def get_mount_type(self) -> str:
        return 'ALT-AZ'

    def get_antennas(self) -> ac.EarthLocation:
        _, coords = extract_itrs_coords(self.get_array_file(), delim=',')
        return coords.earth_location

    def get_antenna_names(self) -> List[str]:
        stations, _ = extract_itrs_coords(self.get_array_file(), delim=',')
        return stations

    def get_array_file(self) -> str:
        return os.path.join(*self.content_path, 'antenna_config.txt')

    def get_station_name(self) -> str:
        return 'OVRO'

    def get_system_equivalent_flux_density(self) -> au.Quantity:
        # At 55 MHz
        return 5070. * au.Jy  # Jy

    def get_system_efficiency(self) -> au.Quantity:
        return 1. * au.dimensionless_unscaled

    def get_antenna_model(self) -> AbstractAntennaModel:
        return H5AntennaModelV1(
            beam_file=os.path.join(*self.content_path, 'OVRO-LWA_soil_pt.h5')
        )
