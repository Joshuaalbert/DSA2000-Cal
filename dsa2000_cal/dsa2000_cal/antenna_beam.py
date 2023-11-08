import os
from functools import cached_property
from typing import Literal

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
    def _get_amplitude(self) -> np.ndarray:
        amplitude = 10 ** (self.ant_model[self.model_name] / 20.)  # [num_theta, num_phi, num_freqs]
        amplitude /= np.max(amplitude)  # Normalise to 1
        return amplitude

    def get_amplitude(self) -> np.ndarray:
        return self._get_amplitude

    @cached_property
    def _get_freqs(self) -> np.ndarray:
        return self.ant_model['freqListGHz'] * 1e9

    def get_freqs(self) -> np.ndarray:
        return self._get_freqs

    @cached_property
    def _get_theta(self) -> np.ndarray:
        return self.ant_model['ThetaDeg'][:, 0]  # [num_theta, 1] -> [num_theta]

    def get_theta(self) -> np.ndarray:
        return self._get_theta

    @cached_property
    def _get_phi(self) -> np.ndarray:
        return self.ant_model['PhiDeg'][0, :]  # [1, num_phi] -> [num_phi]

    def get_phi(self) -> np.ndarray:
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

    def get_amplitude(self, pointing: ac.ICRS, source: ac.ICRS, freq_hz: float, enu_frame: ENU,
                      pol: Literal['X', 'Y']) -> np.ndarray:
        if pointing.shape == ():
            pointing = pointing.reshape((1,))
        if source.shape == ():
            source = source.reshape((1,))
        if pointing.shape != source.shape:
            raise ValueError(f"pointing and source must have the same shape, got {pointing.shape} and {source.shape}")

        freq_idx = np.argmin(np.abs(self.antenna_model.get_freqs() - freq_hz))

        pointing_enu_xyz = pointing.transform_to(enu_frame).cartesian.xyz.T  # [num_ant, 3] (normed)
        line_of_sight_enu_xyz = source.transform_to(enu_frame).cartesian.xyz.T  # [num_ant, 3] (normed)

        # Theta is in [0, 180] measured from pointing direction, aka bore sight
        theta = np.arccos(np.sum(line_of_sight_enu_xyz.value * pointing_enu_xyz.value, axis=-1))  # [num_ant]
        theta_idx = np.asarray(
            [
                np.argmin(np.abs(self.antenna_model.get_theta() - np.rad2deg(theta_i)))
                for theta_i in theta
            ]
        )  # [num_ant]

        # Phi is in [0, 360] measured from x-axis (see below)
        line_of_sight_proj = line_of_sight_enu_xyz - np.sum(line_of_sight_enu_xyz.value * pointing_enu_xyz.value,
                                                            axis=-1,
                                                            keepdims=True) * pointing_enu_xyz  # [num_ant, 3]
        line_of_sight_proj /= np.linalg.norm(line_of_sight_proj, axis=-1, keepdims=True)  # [num_ant, 3] (normed)

        # Assume Alt-Az mount, so x-axis always stays parallel to ground tangent
        # x' = a x + b y where x, y are east and north. To solve for a, b use x'.z'=0 and x'.x'=1.
        # a = (y.z' / x.z') / sqrt((y.z' / x.z') ^ 2 + 1)
        # b = 1 / sqrt((y.z' / x.z') ^ 2 + 1)

        east_proj = pointing_enu_xyz[:, 0]  # [num_ant]
        north_proj = pointing_enu_xyz[:, 1]  # [num_ant]

        a = (north_proj / east_proj) / np.sqrt((north_proj / east_proj) ** 2 + 1)  # [num_ant]
        b = 1 / np.sqrt((north_proj / east_proj) ** 2 + 1)  # [num_ant]
        x = a[:, None] * np.asarray([1, 0, 0]) + b[:, None] * np.asarray([0, 1, 0])  # [num_ant, 3] (normed)

        phi = np.arccos(np.sum((line_of_sight_proj * x).value, axis=-1))  # [num_ant]
        if pol == 'Y':
            phi += np.pi / 2

        phi_idx = np.asarray(
            [
                np.argmin(np.abs(self.antenna_model.get_phi() - np.rad2deg(phi_i)))
                for phi_i in phi
            ]
        )  # [num_ant]

        return self.antenna_model.get_amplitude()[theta_idx, phi_idx, freq_idx]
