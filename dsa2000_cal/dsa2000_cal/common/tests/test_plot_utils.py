import numpy as np
from astropy import time as at, units as au, coordinates as ac
from matplotlib import pyplot as plt

from dsa2000_cal.common.plot_utils import figs_to_gif, plot_antenna_gains
from dsa2000_cal.types import SystemGains


def test_figs_to_gif():
    # Example usage:
    def example_figure_generator():
        x = np.linspace(0, 2 * np.pi, 100)
        for i in range(50):
            fig, ax = plt.subplots()
            ax.plot(x, np.sin(x + i * 0.1))
            ax.set_title(f'Step {i}')
            yield fig
            plt.close(fig)

    # Convert figures to GIF
    fig_generator = example_figure_generator()
    figs_to_gif(fig_generator, 'example_animation.gif', loop=0)


def test_plot_antenna_gains():
    # Create a dummy SystemGains object
    times = at.Time('2022-01-01T00:00:00', format='isot', scale='utc') + np.linspace(0, 600, 30) * au.s
    freqs = np.linspace(700, 2000, 100) * au.MHz
    antennas = ac.EarthLocation.from_geocentric(np.arange(10), np.arange(10), np.arange(10), unit='m')
    directions = ac.ICRS(ra=np.arange(10) * au.deg, dec=np.arange(10) * au.deg)
    num_times = len(times)
    num_freqs = len(freqs)
    num_ants = len(antennas)
    num_dirs = len(directions)
    gains = np.random.normal(size=(num_dirs, num_times, num_ants, num_freqs, 2, 2)) + 1j * np.random.normal(
        size=(num_dirs, num_times, num_ants, num_freqs, 2, 2))
    antenna_labels = [f"ANT{i}" for i in range(num_ants)]

    gain_obj = SystemGains(
        directions=directions,
        times=times,
        antennas=antennas,
        antenna_labels=antenna_labels,
        freqs=freqs,
        gains=gains
    )
    fig = plot_antenna_gains(gain_obj, antenna_idx=0, direction_idx=0)
    fig.show()
