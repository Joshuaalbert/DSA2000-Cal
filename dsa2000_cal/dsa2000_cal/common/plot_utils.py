import tempfile

import astropy.units as au
import imageio
import matplotlib.pyplot as plt
import numpy as np

from dsa2000_cal.types import SystemGains
from dsa2000_cal.calibration.probabilistic_models.gains_per_facet_model import CalibrationSolutions


def figs_to_gif(fig_generator, gif_path, duration=0.5, loop=0, dpi=80):
    """
    Convert a generator of matplotlib figures to a GIF using a temporary directory, with options for loop and resolution.

    Parameters:
        fig_generator (generator): A generator that yields matplotlib figures. Generator should close figs.
        gif_path (str): Path where the GIF should be saved.
        duration (float): Duration of each frame in the GIF in seconds.
        loop (int): Number of times the GIF should loop (0 for infinite).
        dpi (int): Dots per inch (resolution) of images in the GIF.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        filenames = []
        for i, fig in enumerate(fig_generator):
            # Save each figure to a temporary file
            filename = f'{tmp_dir}/frame_{i}.png'
            fig.savefig(filename, dpi=dpi)  # Specify DPI for image quality
            filenames.append(filename)

        # Create a GIF using the saved frames
        with imageio.get_writer(gif_path, mode='I', duration=duration, loop=loop) as writer:
            for filename in filenames:
                image = imageio.v2.imread(filename)
                writer.append_data(image)

        # Temporary files are automatically cleaned up when exiting the block

    print(f"GIF saved as {gif_path}")


def plot_antenna_gains(gain_obj: SystemGains | CalibrationSolutions, antenna_idx: int,
                       direction_idx: int, ref_ant: int = 0) -> plt.Figure:
    """
    Plots the gains as a function of time to a file.

    Args:
        gain_obj: gains along with times, antennas, directions, freqs.
        antenna_idx: index of the antenna to plot.
        direction_idx: index of the direction to plot.
        ref_ant: index of the reference antenna for normalisation.

    Returns:
        a matplotlib figure.
    """
    # directions: ac.ICRS  # [source]
    # times: at.Time  # [time]
    # antennas: ac.EarthLocation  # [ant]
    # antenna_labels: List[str]  # [ant]
    # freqs: au.Quantity  # [chan]
    # gains: np.ndarray  # [source, time, ant, chan, 2, 2]

    # We'll use a waterfall (time on x-axis, freq on y-axis) plot to show the phase and amplitude for each antenna,
    # for each p,q polarisation pair.

    # Rows are [XX, YX, XY, YY], cols are {amplitude, phase}
    fig, axs = plt.subplots(4, 2, figsize=(12, 12), sharex=True, squeeze=False)

    # Plot amplitude
    ref_phase = np.angle(gain_obj.gains[direction_idx, :, ref_ant, :, :, :])  # [time, chan, 2, 2]
    ref_gain = np.exp(1j * ref_phase)
    amplitude = np.abs(gain_obj.gains[direction_idx, :, antenna_idx, :, :, :] / ref_gain)  # [time, chan, 2, 2]
    avmin = np.min(amplitude)
    avmax = np.max(amplitude)
    phase = np.angle(gain_obj.gains[direction_idx, :, antenna_idx, :, :, :] / ref_gain)  # [time, chan, 2, 2]
    extent = (gain_obj.times[0].mjd, gain_obj.times[-1].mjd, gain_obj.freqs[0].value, gain_obj.freqs[-1].value)
    direction = gain_obj.directions[direction_idx]
    # pretty string
    direction_str = f"J{direction.ra.to_string(unit=au.hourangle, sep='', precision=2, pad=True)}{direction.dec.to_string(sep='', precision=2, alwayssign=True, pad=True)}"
    antenna_label = gain_obj.antenna_labels[antenna_idx]
    row_pols = ['XX', 'YX', 'XY', 'YY']
    for p in range(2):
        for q in range(2):
            # X,X -> 0, Y,X -> 1, X,Y -> 2, Y,Y -> 3
            # 0,0 -> 0, 1,0 -> 1, 0,1 -> 2, 1,1 -> 3
            row = 2 * q + p
            axs[row][0].imshow(
                amplitude[:, :, p, q].T,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                extent=extent,
                interpolation='none',
                vmin=avmin,
                vmax=avmax
            )
            axs[row, 0].set_title(f"Amplitude {row_pols[row]} {antenna_label} {direction_str}")
            axs[row, 0].set_ylabel("Frequency (Hz)")

            axs[row][1].imshow(
                phase[:, :, p, q].T,
                aspect='auto',
                origin='lower',
                cmap='hsv',
                extent=extent,
                vmin=-np.pi,
                vmax=np.pi,
                interpolation='none'
            )
            axs[row, 1].set_title(f"Phase {row_pols[row]} {antenna_label} {direction_str}")
    axs[-1, 0].set_xlabel("Time (MJD)")
    axs[-1, 1].set_xlabel("Time (MJD)")

    # Put colorbar horizontal, below the last row. One for amp, one or phase.
    # Shift down a little so that the colorbar doesn't overlap with the xlabels.
    # And make sure all the axes shrink the same, not just the last one to make space
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.1, 0.06, 0.8, 0.02])
    fig.colorbar(axs[0, 0].images[0], cax=cbar_ax, orientation='horizontal')
    cbar_ax = fig.add_axes([0.1, 0.02, 0.8, 0.02])
    fig.colorbar(axs[0, 1].images[0], cax=cbar_ax, orientation='horizontal')
    return fig
