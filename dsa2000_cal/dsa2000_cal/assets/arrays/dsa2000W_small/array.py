import os

from astropy import coordinates as ac

from dsa2000_cal.assets.arrays.array import AbstractArray, extract_itrs_coords
from dsa2000_cal.assets.registries import array_registry


@array_registry(template='dsa2000W_small')
class DSA2000WArray(AbstractArray):
    """
    DSA2000W array class, smaller for testing
    """

    def get_antenna_diameter(self) -> float:
        return 5.

    def get_mount_type(self) -> str:
        return 'ALT-AZ'

    def get_antennas(self) -> ac.ITRS:
        _, coords = extract_itrs_coords(self.get_array_file(), delim=',')
        return coords

    def get_antenna_names(self) -> list[str]:
        stations, _ = extract_itrs_coords(self.get_array_file(), delim=',')
        return stations

    def get_array_file(self) -> str:
        return os.path.join(*self.content_path, 'antenna_config.txt')

    def get_station_name(self) -> str:
        return 'OVRO'

    def system_equivalent_flux_density(self) -> float:
        return 2.5

    def system_efficency(self) -> float:
        return 0.7