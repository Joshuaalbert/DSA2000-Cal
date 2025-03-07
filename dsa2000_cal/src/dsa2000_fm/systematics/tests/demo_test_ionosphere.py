from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
from astropy import time as at, units as au, coordinates as ac
from tomographic_kernel.frames import ENU

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.astropy_utils import create_spherical_grid_old
from dsa2000_common.common.plot_utils import figs_to_gif
from dsa2000_fm.systematics.ionosphere import construct_eval_interp_struct, IonosphereLayer, IonosphereMultiLayer, \
    build_ionosphere_gain_model


def test_ionosphere_dtec_gain_model():
    ref_time = at.Time.now()
    times = ref_time + 2 * np.arange(10) * au.s
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()[:2]
    ref_location = array.get_array_location()
    phase_center = ENU(0, 0, 1, obstime=ref_time, location=ref_location).transform_to(ac.ICRS())

    # directions = phase_center[None]

    angular_resolution = 0.5 * au.deg

    model_directions = create_spherical_grid_old(
        pointing=phase_center,
        angular_radius=4 * au.deg,
        dr=angular_resolution
    )
    print(f"Number of model directions: {len(model_directions)}")

    # T = int((times.max() - times.min()) / (1 * au.min)) + 1
    # model_times = times.min() + np.arange(0., T) * au.min
    model_freqs = [700, 1350, 2000] * au.MHz
    gain_model = build_ionosphere_gain_model(
        model_freqs=model_freqs,
        antennas=antennas,
        ref_location=ref_location,
        times=times,
        ref_time=ref_time,
        directions=model_directions,
        phase_centre=phase_center,
        dt=1 * au.min
    )
    gain_model.plot_regridded_beam(ant_idx=-1)
    gain_model.to_aperture().plot_regridded_beam(ant_idx=-1, is_aperture=True)


def test_ionosphere_tec_multi_layer_conditional_flow():
    ref_time = at.Time.now()
    times = ref_time + 6 * np.arange(100) * au.s
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()[::3]
    ref_location = array.get_array_location()
    phase_center = ENU(0, 0, 1, obstime=ref_time, location=ref_location).transform_to(ac.ICRS())

    directions = phase_center[None]

    T = int((times.max() - times.min()) / (1 * au.min)) + 1
    model_times = times.min() + np.arange(0., T) * au.min
    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions, model_times
    )

    layer1 = IonosphereLayer(
        length_scale=1.,
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

    layer2 = IonosphereLayer(
        length_scale=2.,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        bottom_velocity=0.120,
        radial_velocity=0.,
        x0_radius=x0_radius,
        bottom=100,
        width=100,
        # fed_mu=50.,  # 5 * 10^11 e-/m^3 (low sun spot noon)
        # fed_sigma=25.  # 2.5 * 10^11 e-/m^3 (low sun spot noon)
        fed_mu=200.,  # 2 * 10^12 e-/m^3 (high sun spot noon)
        fed_sigma=50.  # 5 * 10^11 e-/m^3 (high sun spot noon)
    )

    ionosphere = IonosphereMultiLayer([layer1, layer2])

    def gen():

        key = jax.random.PRNGKey(0)

        past_sample = deque(maxlen=1)

        sample_tec_jit = jax.jit(ionosphere.sample_tec)
        conditional_sample_tec_jit = jax.jit(ionosphere.sample_conditional_tec)

        for t in range(len(times_jax)):
            sample_key, key = jax.random.split(key)

            if len(past_sample) == 0:
                sample = sample_tec_jit(
                    key=sample_key,
                    antennas_gcrs=antennas_gcrs,
                    directions_gcrs=directions_gcrs,
                    times=times_jax[t:t + 1]
                )
                past_sample.append(sample)
            else:
                n_past = len(past_sample)
                sample, _ = conditional_sample_tec_jit(
                    key=sample_key,
                    antennas_gcrs=antennas_gcrs,
                    times=times_jax[t:t + 1],
                    directions_gcrs=directions_gcrs,
                    antennas_gcrs_other=antennas_gcrs,
                    times_other=times_jax[t - n_past:t],
                    directions_gcrs_other=directions_gcrs,
                    tec_other=jnp.concatenate(past_sample, axis=1)
                )
                past_sample.append(sample)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sc = ax.scatter(antennas.lon, antennas.lat, c=sample[0, 0, :], cmap='jet')
            ax.set_title(f"Time: {t}")
            plt.colorbar(sc, ax=ax, label=r'$\mathrm{TEC}$')
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('Latitude (deg)')
            fig.tight_layout()
            yield fig
            plt.close(fig)

    figs_to_gif(gen(), 'multi_layer_conditional_tec_flow_3.gif', duration=5, loop=0, dpi=100)


def test_ionosphere_dtec_multi_layer_conditional_flow():
    ref_time = at.Time.now()
    times = ref_time + 6 * np.arange(100) * au.s
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()[::3]
    ref_location = array.get_array_location()
    phase_center = ENU(0, 0, 1, obstime=ref_time, location=ref_location).transform_to(ac.ICRS())

    directions = phase_center[None]

    T = int((times.max() - times.min()) / (1 * au.min)) + 1
    model_times = times.min() + np.arange(0., T) * au.min
    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions, model_times
    )

    reference_antenna_gcrs = antennas_gcrs[0:1]

    layer1 = IonosphereLayer(
        length_scale=1.,
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

    layer2 = IonosphereLayer(
        length_scale=2.,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        bottom_velocity=0.120,
        radial_velocity=0.,
        x0_radius=x0_radius,
        bottom=100,
        width=100,
        # fed_mu=50.,  # 5 * 10^11 e-/m^3 (low sun spot noon)
        # fed_sigma=25.  # 2.5 * 10^11 e-/m^3 (low sun spot noon)
        fed_mu=200.,  # 2 * 10^12 e-/m^3 (high sun spot noon)
        fed_sigma=50.  # 5 * 10^11 e-/m^3 (high sun spot noon)
    )

    ionosphere = IonosphereMultiLayer([layer1, layer2])

    def gen():

        key = jax.random.PRNGKey(0)

        past_sample = deque(maxlen=1)
        cache = None

        sample_dtec_jit = jax.jit(ionosphere.sample_dtec)
        conditional_sample_dtec_jit = jax.jit(ionosphere.sample_conditional_dtec)

        for t in range(len(times_jax)):
            t0 = time.time()
            sample_key, key = jax.random.split(key)

            if len(past_sample) == 0:
                sample = sample_dtec_jit(
                    key=sample_key,
                    reference_antenna_gcrs=reference_antenna_gcrs,
                    antennas_gcrs=antennas_gcrs,
                    directions_gcrs=directions_gcrs,
                    times=times_jax[t:t + 1]
                )
                past_sample.append(sample)
            else:
                n_past = len(past_sample)
                sample, cache = jax.block_until_ready(
                    conditional_sample_dtec_jit(
                        key=sample_key,
                        reference_antenna_gcrs=reference_antenna_gcrs,
                        antennas_gcrs=antennas_gcrs,
                        times=times_jax[t:t + 1],
                        directions_gcrs=directions_gcrs,
                        antennas_gcrs_other=antennas_gcrs,
                        times_other=times_jax[t - n_past:t],
                        directions_gcrs_other=directions_gcrs,
                        dtec_other=jnp.concatenate(past_sample, axis=1),
                        cache=cache
                    )
                )
                past_sample.append(sample)
            t1 = time.time()
            print(f"Sample iteration took {t1 - t0:.2f}s")
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sc = ax.scatter(antennas.lon, antennas.lat, c=sample[0, 0, :], vmin=-50, vmax=50, cmap='jet')
            ax.set_title(f"Time: {t}")
            plt.colorbar(sc, ax=ax, label=r'$\Delta\mathrm{TEC}$')
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('Latitude (deg)')
            fig.tight_layout()
            yield fig
            plt.close(fig)

    figs_to_gif(gen(), 'multi_layer_conditional_flow_4.gif', duration=5, loop=0, dpi=100)


def test_ionosphere_frozen_flow_dtec_multi_layer():
    ref_time = at.Time.now()
    times = ref_time + 20 * np.arange(3) * au.s
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()[::3]
    ref_location = array.get_array_location()
    phase_center = ENU(0, 0, 1, obstime=ref_time, location=ref_location).transform_to(ac.ICRS())

    directions = phase_center[None]

    T = int((times.max() - times.min()) / (1 * au.min)) + 1
    model_times = times.min() + np.arange(0., T) * au.min
    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions, model_times
    )

    reference_antenna_gcrs = antennas_gcrs[0:1]

    key = jax.random.PRNGKey(0)

    layer1 = IonosphereLayer(
        length_scale=1.,
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

    layer2 = IonosphereLayer(
        length_scale=2.,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        bottom_velocity=0.120,
        radial_velocity=0.,
        x0_radius=x0_radius,
        bottom=100,
        width=100,
        # fed_mu=50.,  # 5 * 10^11 e-/m^3 (low sun spot noon)
        # fed_sigma=25.  # 2.5 * 10^11 e-/m^3 (low sun spot noon)
        fed_mu=200.,  # 2 * 10^12 e-/m^3 (high sun spot noon)
        fed_sigma=50.  # 5 * 10^11 e-/m^3 (high sun spot noon)
    )

    ionosphere = IonosphereMultiLayer([layer1, layer2])

    K, mean = jax.jit(ionosphere.compute_dtec_process_params,
                      static_argnames=['resolution'])(reference_antenna_gcrs, antennas_gcrs, times_jax, directions_gcrs,
                                                      resolution=27)
    D, T, A = np.shape(mean)
    K = jax.lax.reshape(K, (D * T * A, D * T * A))
    mean = jax.lax.reshape(mean, [D * T * A])

    plt.plot(mean)
    plt.show()

    plt.imshow(K)
    plt.colorbar()
    plt.show()

    sample = jax.jit(ionosphere.sample_dtec)(
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


def test_ionosphere_frozen_flow_dtec():
    ref_time = at.Time.now()
    times = ref_time + 20 * np.arange(3) * au.s
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()[::3]
    ref_location = array.get_array_location()
    phase_center = ENU(0, 0.9, 1, obstime=ref_time, location=ref_location).transform_to(ac.ICRS())

    directions = phase_center[None]

    T = int((times.max() - times.min()) / (1 * au.min)) + 1
    model_times = times.min() + np.arange(0., T) * au.min

    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions, model_times
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

    T = int((times.max() - times.min()) / (1 * au.min)) + 1
    model_times = times.min() + np.arange(0., T) * au.min
    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions, model_times
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
