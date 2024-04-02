import os
from typing import List

from astropy import coordinates as ac
from astropy import units as au

from dsa2000_cal.abc import AbstractAntennaBeam
from dsa2000_cal.antenna_beam import AltAzAntennaBeam, MatlabAntennaModelV1
from dsa2000_cal.assets.arrays.array import AbstractArray, extract_itrs_coords
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.astropy_utils import mean_itrs


@array_registry(template='dsa2000W')
class DSA2000WArray(AbstractArray):
    """
    DSA2000W array class.
    """

    def get_channel_width(self) -> au.Quantity:
        return (2000e6 * au.Hz - 700e6 * au.Hz) / 8000

    def get_array_location(self) -> ac.EarthLocation:
        return mean_itrs(self.get_antennas().get_itrs()).earth_location

    def get_antenna_diameter(self) -> au.Quantity:
        return 5. * au.m

    def get_focal_length(self) -> au.Quantity:
        return 2. * au.m

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
        return 5022. * au.Jy  # Jy

    def get_system_efficency(self) -> au.Quantity:
        return 0.7 * au.dimensionless_unscaled

    def get_antenna_beam(self) -> AbstractAntennaBeam:
        return AltAzAntennaBeam(
            antenna_model=MatlabAntennaModelV1(
                antenna_model_file=os.path.join(*self.content_path, 'dsa2000_antenna_model.mat'),
                model_name='coPolPattern_dBi_Freqs_15DegConicalShield'
            )
        )


def test_dsa2000_antenna_beam():
    antenna_beam = array_registry.get_instance(array_registry.get_match('dsa2000W')).get_antenna_beam()
    import pylab as plt
    import numpy as np
    amplitude = antenna_beam.get_model().get_amplitude()
    circular_mean = np.mean(amplitude, axis=1)
    phi = antenna_beam.get_model().get_phi()
    theta = antenna_beam.get_model().get_theta()
    print(amplitude.shape)
    for i, freq in enumerate(antenna_beam.get_model().get_freqs()):
        plt.plot(theta[:60], amplitude[:60, i], label=freq)
    plt.legend()
    plt.show()
    print(antenna_beam.get_model().get_freqs())
    for i, freq in enumerate(antenna_beam.get_model().get_freqs()):
        for k, th in enumerate(theta):
            if circular_mean[k, i] < 0.01:
                break
        print(f"Freq: {i}, {freq}, theta: {k}, {th}")
