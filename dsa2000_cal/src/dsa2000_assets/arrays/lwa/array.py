import os
from typing import List

import numpy as np
from astropy import coordinates as ac
from astropy import units as au

from dsa2000_fm.antenna_model.abc import AbstractAntennaModel
from dsa2000_common.abc import AbstractArray
from dsa2000_assets.registries import array_registry, beam_model_registry
from dsa2000_cal.common.astropy_utils import mean_itrs, extract_itrs_coords
from dsa2000_common.common.types import DishEffectsParams


@array_registry(template='lwa')
class LWAArray(AbstractArray):
    """
    LWA array class.
    """

    def get_integration_time(self) -> au.Quantity:
        return 10. * au.s

    def get_channel_width(self) -> au.Quantity:
        return 23913.3199056 * au.Hz

    def get_channels(self) -> au.Quantity:
        return au.Quantity(np.linspace(42, 88, 1920) * au.MHz)

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
        return 5570. * au.Jy  # Jy

    def get_system_efficiency(self) -> au.Quantity:
        return 1. * au.dimensionless_unscaled

    def get_antenna_model(self) -> AbstractAntennaModel:
        beam_model = beam_model_registry.get_instance(beam_model_registry.get_match('lwa_highres'))
        return beam_model.get_antenna_model()

    def get_dish_effect_params(self) -> DishEffectsParams:
        raise NotImplementedError("Dish effects not implemented for LWA")
