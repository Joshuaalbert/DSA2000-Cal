import jax
import numpy as np
import pylab as plt
from astropy import time as at, units as au, coordinates as ac
from tomographic_kernel.frames import ENU

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_fm.systematics.ionosphere import construct_eval_interp_struct, IonosphereLayer


def test_ionosphere_frozen_flow_dtec():
    ref_time = at.Time.now()
    times = ref_time + 20 * np.arange(3) * au.s
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()[::3]
    ref_location = array.get_array_location()
    phase_center = ENU(0, 0.9, 1, obstime=ref_time, location=ref_location).transform_to(ac.ICRS())

    directions = phase_center[None]

    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions
    )

    reference_antenna_gcrs = antennas_gcrs[0:1]

    key = jax.random.PRNGKey(0)

    ionosphere = IonosphereLayer(
        length_scale=2.,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        bottom_velocity=0.120,
        radial_velocity=0.,
        x0_radius=x0_radius,
        bottom=200,
        width=200,
        # fed_mu=50.,  # 5 * 10^11 e-/m^3 (low sun spot noon)
        # fed_sigma=25.  # 2.5 * 10^11 e-/m^3 (low sun spot noon)
        fed_mu=200.,  # 2 * 10^12 e-/m^3 (high sun spot noon)
        fed_sigma=50.  # 5 * 10^11 e-/m^3 (high sun spot noon)
    )

    K, mean = ionosphere.compute_dtec_process_params(reference_antenna_gcrs, antennas_gcrs, times_jax, directions_gcrs,
                                                     resolution=27)
    D, T, A = np.shape(mean)
    K = jax.lax.reshape(K, (D * T * A, D * T * A))
    mean = jax.lax.reshape(mean, [D * T * A])

    plt.plot(mean)
    plt.show()

    plt.imshow(K)
    plt.colorbar()
    plt.show()

    sample = ionosphere.sample_dtec(
        key=key,
        reference_antenna_gcrs=reference_antenna_gcrs,
        antennas_gcrs=antennas_gcrs,
        directions_gcrs=directions_gcrs,
        times=times_jax
    )

    for t in range(T):
        sc = plt.scatter(antennas.lon, antennas.lat, c=sample[0, t, :])
        plt.title(f"Time: {t}")
        plt.colorbar(sc)
        plt.show()


def test_ionosphere_frozen_flow_tec():
    ref_time = at.Time.now()
    times = ref_time + 20 * np.arange(3) * au.s
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()[::3]
    ref_location = array.get_array_location()
    phase_center = ENU(0, 0.9, 1, obstime=ref_time, location=ref_location).transform_to(ac.ICRS())

    directions = phase_center[None]

    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions
    )

    key = jax.random.PRNGKey(0)

    ionosphere = IonosphereLayer(
        length_scale=2.,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        bottom_velocity=0.120,
        radial_velocity=0.,
        x0_radius=x0_radius,
        bottom=200,
        width=200,
        # fed_mu=50.,  # 5 * 10^11 e-/m^3 (low sun spot noon)
        # fed_sigma=25.  # 2.5 * 10^11 e-/m^3 (low sun spot noon)
        fed_mu=200.,  # 2 * 10^12 e-/m^3 (high sun spot noon)
        fed_sigma=50.  # 5 * 10^11 e-/m^3 (high sun spot noon)
    )

    K, mean = ionosphere.compute_tec_process_params(antennas_gcrs, times_jax, directions_gcrs, resolution=27)
    D, T, A = np.shape(mean)
    K = jax.lax.reshape(K, (D * T * A, D * T * A))
    mean = jax.lax.reshape(mean, [D * T * A])

    plt.plot(mean)
    plt.show()

    plt.imshow(K)
    plt.colorbar()
    plt.show()

    sample = ionosphere.sample_tec(
        key=key,
        antennas_gcrs=antennas_gcrs,
        directions_gcrs=directions_gcrs,
        times=times_jax
    )

    for t in range(T):
        sc = plt.scatter(antennas.lon, antennas.lat, c=sample[0, t, :])
        plt.title(f"Time: {t}")
        plt.colorbar(sc)
        plt.show()
