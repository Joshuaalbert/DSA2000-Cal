from abc import ABC, abstractmethod

import astropy.units as au


class AbstractSourceModel(ABC):
    ...


class AbstractAntennaModel(ABC):
    """
    Antenna beam model.
    """

    @abstractmethod
    def plot_polar_amplitude(self):
        ...

    @abstractmethod
    def plot_polar_phase(self):
        ...

    @abstractmethod
    def get_amplitude(self) -> au.Quantity:
        """
        Get the antenna beam model amplitude. This has peak value of 1 typically along bore sight.

        Returns:
            A 3D array of shape [num_theta, num_phi, num_freqs, 2, 2] where num_theta is the number of theta values,
            num_phi is the number of phi values, and num_freqs is the number of frequency values.
        """
        ...

    @abstractmethod
    def get_phase(self) -> au.Quantity:
        """
        Get the antenna beam model phase.

        Returns:
            A 3D array of shape [num_theta, num_phi, num_freqs, 2, 2] where num_theta is the number of theta values,
            num_phi is the number of phi values, and num_freqs is the number of frequency values.
        """
        ...

    @abstractmethod
    def get_voltage_gain(self) -> au.Quantity:
        """
        Get the voltage gain of the antenna beam model. This is basically the maximum value of the amplitude.

        Returns:
            A float.
        """
        ...

    @abstractmethod
    def get_freqs(self) -> au.Quantity:
        """
        Get the frequency axis, in Hz.

        Returns:
            A 1D array of shape [num_freqs] where num_freqs is the number of frequency values.
        """
        ...

    @abstractmethod
    def get_theta(self) -> au.Quantity:
        """
        Get the theta axis, in degrees, measured from the pointing direction. Domain is [0, 180].

        Returns:
            A 1D array of shape [num_theta] where num_theta is the number of theta values.
        """
        ...

    @abstractmethod
    def get_phi(self) -> au.Quantity:
        """
        Get the phi axis, in degrees, measured from the x-axis. Domain is [0, 360].

        Returns:
            A 1D array of shape [num_phi] where num_phi is the number of phi values.
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
