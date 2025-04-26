import os

import astropy.units as au

from dsa2000_assets.arrays.dsa2000W.array import DSA2000WArray
from dsa2000_assets.registries import array_registry, beam_model_registry
from dsa2000_fm.antenna_model.abc import AbstractAntennaModel


@array_registry(template='dsa1650_9P')
class DSA1650_9P(DSA2000WArray):
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
