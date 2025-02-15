import os
from typing import List

import numpy as np
from astropy import coordinates as ac
from astropy import units as au

from dsa2000_cal.antenna_model.abc import AbstractAntennaModel
from dsa2000_assets.arrays.array import AbstractArray, extract_itrs_coords
from dsa2000_assets.registries import array_registry, beam_model_registry
from dsa2000_cal.common.astropy_utils import mean_itrs
from dsa2000_common.common.types import DishEffectsParams


@array_registry(template='dsa2000W')
class DSA2000WArray(AbstractArray):
    """
    DSA2000W array class.
    """

    def get_integration_time(self) -> au.Quantity:
        return 1.5 * au.s

    def get_channel_width(self) -> au.Quantity:
        return 1300 * au.MHz / 10000

    def get_channels(self) -> au.Quantity:
        return au.Quantity(np.linspace(700, 2000, 10000) * au.MHz)

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

    def get_system_efficiency(self) -> au.Quantity:
        return 0.7 * au.dimensionless_unscaled

    def get_antenna_model(self) -> AbstractAntennaModel:
        beam_model = beam_model_registry.get_instance(beam_model_registry.get_match('dsa_original'))
        return beam_model.get_antenna_model()

    def get_dish_effect_params(self) -> DishEffectsParams:
        # Could provide in terms of zenike polynomial coefficients
        return DishEffectsParams(
            # dish parameters
            dish_diameter=self.get_antenna_diameter(),
            focal_length=self.get_focal_length(),
            elevation_pointing_error_stddev=2. * au.arcmin,
            cross_elevation_pointing_error_stddev=2. * au.arcmin,
            axial_focus_error_stddev=3. * au.mm,
            elevation_feed_offset_stddev=3. * au.mm,
            cross_elevation_feed_offset_stddev=3. * au.mm,
            horizon_peak_astigmatism_stddev=5. * au.mm,
            surface_error_mean=3. * au.mm,
            surface_error_stddev=1. * au.mm,
        )
