import os
from functools import cached_property

import numpy as np
from astropy import units as au
from scipy.io import loadmat

from src.dsa2000_cal.antenna_model.antenna_beam import AltAzAntennaModel


class MatlabAntennaModelV1(AltAzAntennaModel):
    """
    Antenna beam model, assumes an Alt-Az mount, with isotropic dish so that rotating by 90 deg gives other
    polarisation, based on what jonas provided ghellbourg provided.
    """

    def __init__(self, antenna_model_file: str, model_name: str):
        if not os.path.exists(antenna_model_file):
            raise ValueError(f"Antenna model file {antenna_model_file} does not exist")
        self.antenna_model_file = antenna_model_file
        self.ant_model = loadmat(antenna_model_file)
        self.model_name = model_name

    @cached_property
    def _get_amplitude(self) -> au.Quantity:
        scalar_amplitude = 10 ** (self.ant_model[self.model_name] / 20.)  # [num_theta, num_phi, num_freqs]
        # Note: this is amplification factor, not peak of 1.
        scalar_amplitude = scalar_amplitude * au.dimensionless_unscaled
        amplitude = np.zeros(scalar_amplitude.shape + (2, 2))
        amplitude[..., 0, 0] = scalar_amplitude
        amplitude[..., 1, 1] = scalar_amplitude
        return amplitude * au.dimensionless_unscaled

    @cached_property
    def _get_phase(self) -> au.Quantity:
        return np.zeros_like(self.get_amplitude()) * au.rad

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
        return np.reshape(self.ant_model['freqListGHz'], (-1,)) * au.GHz

    def get_freqs(self) -> au.Quantity:
        return self._get_freqs

    @cached_property
    def _get_theta(self) -> au.Quantity:
        return self.ant_model['ThetaDeg'][:, 0] * au.deg  # [num_theta, 1] -> [num_theta]

    def get_theta(self) -> au.Quantity:
        return self._get_theta

    @cached_property
    def _get_phi(self) -> au.Quantity:
        return self.ant_model['PhiDeg'][0, :] * au.deg  # [1, num_phi] -> [num_phi]

    def get_phi(self) -> au.Quantity:
        return self._get_phi
