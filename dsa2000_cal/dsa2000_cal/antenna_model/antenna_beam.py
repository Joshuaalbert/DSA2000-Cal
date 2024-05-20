import numpy as np
import pylab as plt

from dsa2000_cal.abc import AbstractAntennaBeam, AbstractAntennaModel


class AltAzAntennaModel(AbstractAntennaModel):
    """
    Antenna beam model, assumes an Alt-Az mount.
    """

    def plot_polar_amplitude(self):
        # Generate theta and phi values for the meshgrid
        print(f"Min, Max: {np.min(self.get_amplitude())}, {np.max(self.get_amplitude())}")

        # Create a 2D meshgrid of theta and phi values
        Theta, Phi = np.meshgrid(self.get_theta().to('deg').value, self.get_phi().to('deg').value)
        # Plot the data
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        print(self.get_amplitude().shape)
        c = ax.pcolormesh(Phi, Theta, np.log10(self.get_amplitude()[..., 0, 0].value)[:-1, :-1, 0].T,
                          shading='auto')
        fig.colorbar(c, ax=ax, label='log10(Amplitude)')

        ax.set_title(f"'{self.__class__.__name__}' Beam")
        plt.show()

    def plot_polar_phase(self):
        # Generate theta and phi values for the meshgrid
        print(f"Min, Max: {np.min(self.get_phase())}, {np.max(self.get_phase())}")

        # Create a 2D meshgrid of theta and phi values
        Theta, Phi = np.meshgrid(self.get_theta().to('deg').value, self.get_phi().to('deg').value)
        # Plot the data
        print(self.get_phase()[..., 0, 0])
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        c = ax.pcolormesh(Phi, Theta, self.get_phase()[..., 0, 0][:-1, :-1, 0].T,
                          shading='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(c, ax=ax, label='Phase')

        ax.set_title(f"'{self.__class__.__name__}' Beam")
        plt.show()


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
