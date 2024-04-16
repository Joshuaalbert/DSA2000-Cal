from functools import cached_property
from typing import List

import numpy as np
from astropy import coordinates as ac

from dsa2000_cal.abc import AbstractAntennaBeam
from dsa2000_cal.assets.arrays.dsa2000W.array import DSA2000WArray
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.astropy_utils import create_spherical_earth_grid


@array_registry(template='dsa2000W_small')
class DSA2000WSmallArray(DSA2000WArray):
    """
    DSA2000W array class, smaller for testing
    """

    @cached_property
    def _get_antenna_beam(self) -> AbstractAntennaBeam:
        array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
        return array.get_antenna_beam()

    def get_antenna_beam(self) -> AbstractAntennaBeam:
        return self._get_antenna_beam

    @cached_property
    def _get_antennas(self) -> ac.EarthLocation:
        array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
        all_antennas = array.get_antennas()
        array_centre = array.get_array_location()
        all_antennas_itrs = all_antennas.get_itrs()
        all_antennas_itrs_xyz = all_antennas_itrs.cartesian.xyz.T
        max_baseline = np.max(
            np.linalg.norm(
                all_antennas_itrs_xyz[:, None, :] - all_antennas_itrs_xyz[None, :, :],
                axis=-1
            )
        )
        radius = 0.5 * max_baseline

        spatial_separation = radius * 0.25

        model_antennas = create_spherical_earth_grid(
            center=array_centre,
            radius=radius,
            dr=spatial_separation
        )

        # filter out model antennas that are too far from an actual antenna
        def keep(model_antenna: ac.EarthLocation):
            dist = np.linalg.norm(
                model_antenna.get_itrs().cartesian.xyz - all_antennas_itrs_xyz,
                axis=-1
            )
            return np.any(dist < spatial_separation)

        # List of EarthLocation
        model_antennas = list(filter(keep, model_antennas))
        # Via ITRS then back to EarthLocation
        model_antennas = ac.concatenate(list(map(lambda x: x.get_itrs(), model_antennas))).earth_location
        return model_antennas

    def get_antennas(self) -> ac.EarthLocation:
        return self._get_antennas

    def get_antenna_names(self) -> List[str]:
        return [f'ANT{i}' for i in range(len(self.get_antennas()))]
