import astropy.coordinates as ac
import pylab as plt

from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


def main(ms_folder: str):
    ms = MeasurementSet(ms_folder)
    antennas = ms.meta.antennas  # [num_ant] EarthLocations
    times = ms.meta.times  # [num_times] Time
    pointings = ms.meta.pointings  # [num_ant] ICRS (or None ==> Zenith)

    if pointings is None:
        print("Zenith pointing")
        return

    pointings_altaz = pointings.reshape((-1, 1)).transform_to(
        ac.AltAz(
            obstime=times.reshape((1, -1)),
            location=antennas.reshape((-1, 1))
        )
    )  # [num_ant, num_times]

    elevation = pointings_altaz.alt
    print(f"Elevation: {elevation}")

    # Get local sidereal time and phase tracking RA
    lst = ms.meta.times.reshape((1, -1)).sidereal_time(
        kind='apparent',
        longitude=antennas.reshape((-1, 1)).lon
    )  # [num_ant, num_times]
    ra = ms.meta.phase_tracking.ra
    ha = lst - ra
    print(f"Hour angle: {ha}")

    # Plot histograms of both
    plt.hist(elevation.flatten(), bins='auto')
    plt.xlabel("Elevation")
    plt.ylabel("Count")
    plt.show()

    plt.hist(ha.flatten(), bins='auto')
    plt.xlabel("Hour angle")
    plt.ylabel("Count")
    plt.show()


if __name__ == '__main__':
    main('path/to/ms')
