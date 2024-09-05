import numpy as np
import pylab as plt

from dsa2000_cal.abc import AbstractAntennaModel


class AltAzAntennaModel(AbstractAntennaModel):
    """
    Antenna beam model, assumes an Alt-Az mount.
    """

    def plot_polar_amplitude(self, nu: int = 0, p: int = 0, q: int = 0):
        amplitude = self.get_amplitude()[..., nu, p, q]
        # Generate theta and phi values for the meshgrid
        print(f"Min, Max: {np.min(amplitude)}, {np.max(amplitude)}")

        # Create a 2D meshgrid of theta and phi values
        Theta, Phi = np.meshgrid(self.get_theta(), self.get_phi(),
                                 indexing='ij')

        # Use scatter plot

        radius = Theta
        x = radius * np.sin(Phi)
        y = radius * np.cos(Phi)

        # Plot the data
        fig, ax = plt.subplots(1, 1)
        c = ax.scatter(x.flatten(), y.flatten(), c=amplitude.flatten(),
                       cmap='jet', s=1)
        fig.colorbar(c, ax=ax, label='Amplitude')

        ax.set_title(f"Amplitude '{self.__class__.__name__}' Beam")
        plt.show()

    def plot_polar_phase(self, nu: int = 0, p: int = 0, q: int = 0):
        phase = self.get_phase()[..., nu, p, q]  # [num_theta, num_phi]
        # Generate theta and phi values for the meshgrid
        print(f"Min, Max: {np.min(phase)}, {np.max(phase)}")

        # Create a 2D meshgrid of theta and phi values
        Theta, Phi = np.meshgrid(self.get_theta(), self.get_phi(),
                                 indexing='ij')

        # Use scatter plot

        radius = Theta
        x = radius * np.cos(Phi)
        y = radius * np.sin(Phi)

        # Plot the data
        fig, ax = plt.subplots(1, 1)
        c = ax.scatter(x.flatten(), y.flatten(), c=phase.flatten(),
                       cmap='hsv', s=1,
                       vmin=-np.pi, vmax=np.pi)
        fig.colorbar(c, ax=ax, label='Phase')

        ax.set_title(f"Phase '{self.__class__.__name__}' Beam")
        plt.show()
