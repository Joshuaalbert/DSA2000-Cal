# Matplotlib backend for server
import matplotlib

matplotlib.use('Agg')

import astropy.coordinates as ac
import numpy as np
import pylab as plt

from dsa2000_fm.measurement_sets.measurement_set import MeasurementSet


def main(ms_folder: str):
    ms = MeasurementSet(ms_folder)
    antennas = ms.meta.antennas  # [num_ant] EarthLocations
    times = ms.meta.times  # [num_times] Time
    pointings = ms.meta.pointings  # [num_ant] ICRS (or None ==> Zenith)

    if pointings is None:
        print("Zenith pointing")
    else:

        pointings_altaz = pointings.reshape((-1, 1)).transform_to(
            ac.AltAz(
                obstime=times.reshape((1, -1)),
                location=antennas.reshape((-1, 1))
            )
        )  # [num_ant, num_times]

        elevation = pointings_altaz.alt
        print(f"Elevation: {elevation}")
        # Plot histograms of both
        plt.hist(elevation.flatten(), bins='auto')
        plt.xlabel("Elevation")
        plt.ylabel("Count")
        plt.savefig("elevation.png")
        # plt.show()

    # Get local sidereal time and phase tracking RA
    lst = ms.meta.times.reshape((1, -1)).sidereal_time(
        kind='apparent',
        longitude=antennas.reshape((-1, 1)).lon
    )  # [num_ant, num_times]
    ra = ms.meta.phase_center.ra
    ha = lst - ra
    print(f"Hour angle: {ha}")

    plt.hist(ha.flatten(), bins='auto')
    plt.xlabel("Hour angle")
    plt.ylabel("Count")
    plt.savefig("hour_angle.png")
    # plt.show()

    # Plot a histogram of uv weighted by magnitude of vis
    gen = ms.create_block_generator(vis=True, weights=True, flags=True)
    gen_response = None
    hist = None
    unweighted_hist = None

    x_edges = np.linspace(-20e3, 20e3, 100)
    y_edges = np.linspace(-20e3, 20e3, 100)

    while True:
        try:
            time, visibility_coords, data = gen.send(gen_response)
        except StopIteration:
            break
        mag = np.mean(np.mean(np.abs(data.vis), axis=-1), axis=-1)
        uv = visibility_coords.uvw[:, :2]
        # acculumate hist
        _hist, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=(x_edges, y_edges), weights=mag, density=False)
        if hist is None:
            hist = _hist
        else:
            hist += _hist
        _hist, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=(x_edges, y_edges), density=False)
        if unweighted_hist is None:
            unweighted_hist = _hist
        else:
            unweighted_hist += _hist

    plt.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower')
    plt.xlabel("u [m]")
    plt.ylabel("v [m]")
    plt.title("uv weighted by magnitude of vis")
    plt.colorbar(label="Weighted count")
    plt.savefig("uv_weighted_by_magnitude_of_vis.png")
    # plt.show()

    plt.imshow(unweighted_hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower')
    plt.xlabel("u [m]")
    plt.ylabel("v [m]")
    plt.title("uv")
    plt.colorbar(label="Weighted count")
    plt.savefig("uv.png")
    # plt.show()


if __name__ == '__main__':
    main('path/to/ms')
