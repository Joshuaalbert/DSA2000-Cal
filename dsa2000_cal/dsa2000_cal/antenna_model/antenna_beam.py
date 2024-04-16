import os
from functools import cached_property
from typing import Literal
import astropy.units as au
import astropy.coordinates as ac
import numpy as np
import pylab as plt
from scipy.io import loadmat
from tomographic_kernel.frames import ENU

from dsa2000_cal.abc import AbstractAntennaBeam, AbstractAntennaModel


class AltAzAntennaModel(AbstractAntennaModel):
    """
    Antenna beam model, assumes an Alt-Az mount.
    """
    ...


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

    def plot_polar_amplitude(self):
        # Generate theta and phi values for the meshgrid
        print(f"Min, Max: {np.min(self.get_amplitude())}, {np.max(self.get_amplitude())}")

        # Create a 2D meshgrid of theta and phi values
        Theta, Phi = np.meshgrid(self.get_theta(), self.get_phi())
        # Plot the data
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        c = ax.pcolormesh(Phi, Theta, np.log10(self.get_amplitude())[:-1, :-1, 0].T,
                          shading='auto')
        fig.colorbar(c, ax=ax, label='log10(Amplitude)')

        ax.set_title(f"'{self.model_name}' Beam")
        plt.show()

    @cached_property
    def _get_amplitude(self) -> au.Quantity:
        amplitude = 10 ** (self.ant_model[self.model_name] / 20.)  # [num_theta, num_phi, num_freqs]
        # Note: this is amplification factor, not peak of 1.
        return amplitude * au.dimensionless_unscaled

    def get_amplitude(self) -> au.Quantity:
        return self._get_amplitude

    @cached_property
    def _get_voltage_gain(self) -> au.Quantity:
        return np.max(np.max(self.get_amplitude(), axis=0), axis=0) * au.dimensionless_unscaled # [num_freqs]

    def get_voltage_gain(self) -> au.Quantity:
        """
        Get the voltage gain of the antenna model. This is used in the correlator to account for amplification of
        the signal.

        Returns:
            voltage gain [Nf]
        """
        return self._get_voltage_gain # [Nf]

    @cached_property
    def _get_freqs(self) -> au.Quantity:
        return np.reshape(self.ant_model['freqListGHz'], (-1,)) * au.GHz

    def get_freqs(self) -> au.Quantity:
        return self._get_freqs

    @cached_property
    def _get_theta(self) -> au.Quantity:
        return self.ant_model['ThetaDeg'][:, 0] * au.deg # [num_theta, 1] -> [num_theta]

    def get_theta(self) -> au.Quantity:
        return self._get_theta

    @cached_property
    def _get_phi(self) -> au.Quantity:
        return self.ant_model['PhiDeg'][0, :] * au.deg # [1, num_phi] -> [num_phi]

    def get_phi(self) -> au.Quantity:
        return self._get_phi


class AltAzAntennaBeam(AbstractAntennaBeam):
    """
    Antenna beam class.
    """

    def __init__(self, antenna_model: AltAzAntennaModel):
        """
        Initialise the antenna beam class.

        Args:
            antenna_model: an antenna model.
        """
        self.antenna_model = antenna_model

    def get_model(self) -> AbstractAntennaModel:
        return self.antenna_model

