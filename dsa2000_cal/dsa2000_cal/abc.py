from abc import ABC, abstractmethod
from typing import Literal

import astropy.units as au
import astropy.time as at
import jax
import numpy as np
from astropy import coordinates as ac
from tomographic_kernel.frames import ENU


class AbstractSourceModel(ABC):
    @abstractmethod
    def predict(self, uvw: au.Quantity) -> jax.Array:
        """
        Predict visibilities for the source model.

        Args:
            uvw: UVW coordinates from the measurement set.

        Returns:
            [num_row, num_freqs] Predicted visibilities.
        """
        ...

    @abstractmethod
    def compute_calibration_lmn(self,  phase_tracking: ac.ICRS, time: at.Time) -> np.ndarray:
        """
        Compute the direction cosines this calibration source model.
        """
        ...


class AbstractAntennaModel(ABC):
    """
    Antenna beam model.
    """

    @abstractmethod
    def plot_polar_amplitude(self):
        """
        Plot the antenna beam model in polar coordinates.
        """
        ...

    @abstractmethod
    def get_amplitude(self) -> np.ndarray:
        """
        Get the antenna beam model amplitude. This has peak value of 1 typically along bore sight.

        Returns:
            A 3D array of shape [num_theta, num_phi, num_freqs] where num_theta is the number of theta values,
            num_phi is the number of phi values, and num_freqs is the number of frequency values.
        """
        ...

    @abstractmethod
    def get_voltage_gain(self) -> float:
        """
        Get the voltage gain of the antenna beam model. This is basically the maximum value of the amplitude.

        Returns:
            A float.
        """
        ...

    @abstractmethod
    def get_freqs(self) -> np.ndarray:
        """
        Get the frequency axis, in Hz.

        Returns:
            A 1D array of shape [num_freqs] where num_freqs is the number of frequency values.
        """
        ...

    @abstractmethod
    def get_theta(self) -> np.ndarray:
        """
        Get the theta axis, in degrees, measured from the pointing direction. Domain is [0, 180].

        Returns:
            A 1D array of shape [num_theta] where num_theta is the number of theta values.
        """
        ...

    @abstractmethod
    def get_phi(self) -> np.ndarray:
        """
        Get the phi axis, in degrees, measured from the x-axis. Domain is [0, 360].

        Returns:
            A 1D array of shape [num_phi] where num_phi is the number of phi values.
        """
        ...

    @abstractmethod
    def compute_amplitude(self, pointing: ac.ICRS, source: ac.ICRS, freq_hz: float, enu_frame: ENU,
                          pol: Literal['X', 'Y']) -> np.ndarray:
        """
        Compute the amplitude of the antenna at a given pointing and source.

        Args:
            pointing: The pointing direction in ICRS frame.
            source: The source direction in ICRS frame.
            freq_hz: The frequency in Hz.
            enu_frame: The ENU frame.
            pol: The polarisation, one of ['X', 'Y'].

        Returns:
            The amplitude of the antenna beam.
        """
        ...


class AbstractAntennaBeam(ABC):

    @abstractmethod
    def get_model(self) -> AbstractAntennaModel:
        """
        Get the antenna model.

        Returns:
            antenna_model: AbstractAntennaModel
        """
        ...

    @abstractmethod
    def compute_beam_amplitude(self, pointing: ac.ICRS, source: ac.ICRS, freq_hz: float, enu_frame: ENU,
                               pol: Literal['X', 'Y']) -> np.ndarray:
        """
        Get the amplitude of the antenna beam at a given pointing and source.

        Args:
            pointing: The pointing direction in ICRS frame.
            source: The source direction in ICRS frame.
            freq_hz: The frequency in Hz.
            enu_frame: The ENU frame.
            pol: The polarisation, one of ['X', 'Y'].

        Returns:
            The amplitude of the antenna beam.
        """
        ...
