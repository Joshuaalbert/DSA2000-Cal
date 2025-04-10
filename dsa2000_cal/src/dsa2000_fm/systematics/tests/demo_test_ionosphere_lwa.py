import jax
import numpy as np
from astropy import time as at, units as au, coordinates as ac
from matplotlib import pyplot as plt

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.astropy_utils import create_spherical_spiral_grid, get_time_of_local_meridean
from dsa2000_common.common.enu_frame import ENU
from dsa2000_fm.systematics.ionosphere import compute_x0_radius, simulate_ionosphere, construct_ionosphere_model, \
    construct_eval_interp_struct


def test_ionosphere_tec_simulation():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('lwa'))
    antennas = array.get_antennas()
    ref_location = array.get_array_location()
    phase_center = ENU(
        0, 0, 1, obstime=at.Time('2025-06-10T09:00:00', scale='utc'),
        location=ref_location
    ).transform_to(ac.ICRS())
    # Or if you know the ICRS coord you can find the time when it is at the local transit
    ref_time = get_time_of_local_meridean(phase_center, array.get_array_location(),
                                          at.Time('2025-06-10T09:00:00', scale='utc'))
    times = ref_time + 10 * np.arange(10) * au.s

    directions = create_spherical_spiral_grid(
        pointing=phase_center,
        num_points=20,
        angular_radius=1 * au.deg
    )
    print(f"Number of directions: {len(directions)}")

    x0_radius = compute_x0_radius(ref_location, ref_time)
    ionosphere = construct_ionosphere_model(
        x0_radius=x0_radius,
        f0E=3.51,
        f0F1=5.08,
        f0F2=9.475,
        hmE=101.6,
        hmF1=171.0,
        hmF2=288.7,
        yE=11.3,
        yF1=55.4,
        yF2=95.7,
        vtec=29.0,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        turbulent=True
    )
    # ionosphere = construct_canonical_ionosphere(
    #     x0_radius=x0_radius,
    #     turbulent=True,
    #     dawn=True,
    #     high_sun_spot=True
    # )

    simulate_ionosphere(
        key=jax.random.PRNGKey(0),
        ionosphere=ionosphere,
        antennas=antennas,
        ref_location=ref_location,
        times=times,
        ref_time=ref_time,
        directions=directions,
        spatial_resolution=0.5 * au.km,
        predict_batch_size=512,
        do_tec=True,
        save_file='simulated_lwa_tec.json'
    )


def test_ionosphere_frozen_flow_tec_longtime_single_antenna():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('lwa'))
    antennas = array.get_antennas()[:1]
    ref_location = array.get_array_location()
    phase_center = ENU(
        0, 0, 1, obstime=at.Time('2025-06-10T09:00:00', scale='utc'),
        location=ref_location
    ).transform_to(ac.ICRS())
    # Or if you know the ICRS coord you can find the time when it is at the local transit
    ref_time = get_time_of_local_meridean(phase_center, array.get_array_location(),
                                          at.Time('2025-06-10T09:00:00', scale='utc'))
    times = ref_time + (10 * au.s) * np.arange(360)

    directions = phase_center[None]

    x0_radius = compute_x0_radius(ref_location, ref_time)
    ionosphere = construct_ionosphere_model(
        x0_radius=x0_radius,
        f0E=3.51,
        f0F1=5.08,
        f0F2=9.475,
        hmE=101.6,
        hmF1=171.0,
        hmF2=288.7,
        yE=11.3,
        yF1=55.4,
        yF2=95.7,
        vtec=29.0,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        turbulent=True
    )

    T = int((times.max() - times.min()) / (1 * au.min)) + 1
    model_times = times.min() + np.arange(0., T) * au.min

    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions, model_times
    )

    key = jax.random.PRNGKey(0)

    K, mean = ionosphere.compute_tec_process_params(
        antennas_gcrs, times_jax, directions_gcrs,
        resolution=27
    )
    D, T, A = np.shape(mean)
    K = jax.lax.reshape(K, (D * T * A, D * T * A))
    mean = jax.lax.reshape(mean, [D * T * A])

    plt.plot(mean)
    plt.title("Mean TEC")
    plt.ylabel("TEC [mTECU]")
    plt.show()

    plt.imshow(K)
    plt.title("K matrix")
    plt.colorbar()
    plt.show()

    sample = ionosphere.sample_tec(
        key=key,
        antennas_gcrs=antennas_gcrs,
        directions_gcrs=directions_gcrs,
        times=times_jax
    )

    for d in range(len(directions)):
        plt.scatter(times.datetime, sample[d, :, 0])
        plt.xlabel("Time")
        plt.ylabel("TEC [mTECU]")
        plt.title(f"Direction: {directions[d]}")
        plt.show()
