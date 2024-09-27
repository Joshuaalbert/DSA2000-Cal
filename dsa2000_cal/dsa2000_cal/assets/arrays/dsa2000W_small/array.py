from functools import cached_property
from typing import List

import astropy.units as au
import numpy as np
from astropy import coordinates as ac

import dsa2000_cal.common.mixed_precision_utils
from dsa2000_cal.abc import AbstractAntennaModel
from dsa2000_cal.antenna_model.antenna_beam import AltAzAntennaModel
from dsa2000_cal.assets.arrays.dsa2000W.array import DSA2000WArray
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.astropy_utils import create_spherical_earth_grid


class MockAntennaModel(AltAzAntennaModel):

    def __init__(self):
        self.model_name = 'mock_antenna_model'
        self._num_theta = 61
        self._num_phi = 41
        self._num_freqs = 20

    @cached_property
    def _get_amplitude(self) -> au.Quantity:
        Theta, Phi, Freq = np.meshgrid(self.get_theta(), self.get_phi(), self.get_freqs(), indexing='ij')
        scalar_amplitude = np.exp(-0.5 * Theta ** 2 / (20 * au.deg) ** 2 * np.cos(Phi) ** 2) * np.cos(Theta) ** 2 * (
                Freq / (1 * au.GHz)) * au.dimensionless_unscaled  # [num_theta, num_phi, num_freqs]
        amplitude = np.zeros(scalar_amplitude.shape + (2, 2))
        amplitude[..., 0, 0] = scalar_amplitude
        amplitude[..., 1, 1] = scalar_amplitude
        return amplitude * au.dimensionless_unscaled

    @cached_property
    def _get_phase(self) -> au.Quantity:
        Theta, Phi, Freq = np.meshgrid(self.get_theta(), self.get_phi(), self.get_freqs(), indexing='ij')
        scalar_phase = np.cos(Phi) * np.sin(Theta)  # [num_theta, num_phi, num_freqs]
        phase = np.zeros(scalar_phase.shape + (2, 2))
        phase[..., 0, 0] = scalar_phase
        phase[..., 1, 1] = scalar_phase
        return phase * au.rad

    def get_phase(self) -> au.Quantity:
        return self._get_phase

    def get_amplitude(self) -> au.Quantity:
        return self._get_amplitude

    @cached_property
    def _get_voltage_gain(self) -> au.Quantity:
        return np.max(np.max(self.get_amplitude()[..., 0, 0], axis=0),
                      axis=0) * au.dimensionless_unscaled  # [num_freqs]

    def get_voltage_gain(self) -> au.Quantity:
        """
        Get the voltage gain of the antenna model. This is used in the correlator to account for amplification of
        the signal.

        Returns:
            voltage gain [Nf]
        """
        return self._get_voltage_gain  # [Nf]

    @cached_property
    def _get_freqs(self) -> au.Quantity:
        return np.linspace(0.7, 2, self._num_freqs) * au.GHz

    def get_freqs(self) -> au.Quantity:
        return self._get_freqs

    @cached_property
    def _get_theta(self) -> au.Quantity:
        return np.linspace(0, 180, self._num_theta) * au.deg  # [num_theta]

    def get_theta(self) -> au.Quantity:
        return self._get_theta

    @cached_property
    def _get_phi(self) -> au.Quantity:
        return np.linspace(0, 360, self._num_phi) * au.deg  # [num_phi]

    def get_phi(self) -> au.Quantity:
        return self._get_phi


@array_registry(template='dsa2000W_small')
class DSA2000WSmallArray(DSA2000WArray):
    """
    DSA2000W array class, smaller for testing
    """

    @cached_property
    def _get_antenna_model(self) -> AbstractAntennaModel:
        return MockAntennaModel()

    def get_antenna_model(self) -> AbstractAntennaModel:
        return self._get_antenna_model

    @cached_property
    def _get_antennas(self) -> ac.EarthLocation:
        array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
        all_antennas = array.get_antennas()
        array_centre = array.get_array_location()
        all_antennas_itrs = all_antennas.get_itrs()
        all_antennas_itrs_xyz = dsa2000_cal.common.mixed_precision_utils.T
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
