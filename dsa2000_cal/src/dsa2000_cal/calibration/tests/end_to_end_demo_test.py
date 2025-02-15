import os

os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import dataclasses
import os
import time
from functools import partial
from typing import Generator, Tuple, List

import astropy.time as at
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from astropy import units as au, coordinates as ac
from matplotlib import pyplot as plt
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import GainPriorModel
from dsa2000_cal.calibration.solvers.multi_step_lm import MultiStepLevenbergMarquardtDiagnostic
from dsa2000_cal.common.array_types import ComplexArray, FloatArray, BoolArray
from dsa2000_cal.common.astropy_utils import create_spherical_spiral_grid
from dsa2000_cal.common.corr_utils import broadcast_translate_corrs
from dsa2000_cal.common.fits_utils import ImageModel, save_image_to_fits
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.pure_callback_utils import construct_threaded_callback
from dsa2000_cal.common.quantity_utils import time_to_jnp, quantity_to_jnp, quantity_to_np
from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.common.wgridder import vis_to_image_np
from dsa2000_common.delay_models.base_far_field_delay_engine import build_far_field_delay_engine

, BaseFarFieldDelayEngine
from dsa2000_common.delay_models.base_near_field_delay_engine import build_near_field_delay_engine

, \
    BaseNearFieldDelayEngine
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model, BaseGeodesicModel
from dsa2000_cal.imaging.utils import get_array_image_parameters
from dsa2000_common.visibility_model.source_models.celestial.base_gaussian_source_model import build_gaussian_source_model, \
    BaseGaussianSourceModel
from dsa2000_common.visibility_model.source_models.celestial.base_point_source_model import build_point_source_model, \
    BasePointSourceModel
from dsa2000_fm.forward_models.streaming.distributed.average_utils import average_rule

tfpd = tfp.distributions


@dataclasses.dataclass
class TimerLog:
    msg: str

    def __post_init__(self):
        self.t0 = time.time()

    def __enter__(self):
        print(f"{self.msg}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"... took {time.time() - self.t0:.3f} seconds")
        return False


def build_mock_obs_setup(array_name: str, num_sol_ints_time: int, frac_aperture: float = 1.):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))

    array_location = array.get_array_location()

    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    num_times_per_sol_int = 1
    num_times = num_times_per_sol_int * num_sol_ints_time
    obstimes = ref_time + np.arange(num_times) * array.get_integration_time()

    phase_center = ENU(0, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())
    freqs = array.get_channels()[:1]

    # Point dishes exactly at phase center
    pointing = phase_center

    antennas = array.get_antennas()
    if frac_aperture < 1.:
        keep_ant_idxs = np.random.choice(len(antennas), max(2, int(frac_aperture * len(antennas))), replace=False)
        antennas = antennas[keep_ant_idxs]

    geodesic_model = build_geodesic_model(
        antennas=antennas,
        array_location=array_location,
        phase_center=phase_center,
        obstimes=obstimes,
        ref_time=ref_time,
        pointings=pointing
    )

    far_field_delay_engine = build_far_field_delay_engine(
        antennas=antennas,
        phase_center=phase_center,
        start_time=obstimes.min(),
        end_time=obstimes.max(),
        ref_time=ref_time
    )

    near_field_delay_engine = build_near_field_delay_engine(
        antennas=antennas,
        start_time=obstimes.min(),
        end_time=obstimes.max(),
        ref_time=ref_time
    )

    system_equivalent_flux_density, chan_width, integration_time = array.get_system_equivalent_flux_density(), array.get_channel_width(), array.get_integration_time()

    chan_width *= 40  # simulate wider band to lower nosie
    integration_time *= 4  # simulate longer integration time

    return ref_time, obstimes, freqs, phase_center, antennas, geodesic_model, far_field_delay_engine, near_field_delay_engine, system_equivalent_flux_density, chan_width, integration_time


def create_sky_model(phase_center: ac.ICRS, num_sources: int, model_freqs: au.Quantity, full_stokes: bool,
                     fov: au.Quantity, psf_size: au.Quantity):
    num_model_freqs = len(model_freqs)
    if full_stokes:
        A = 0.5 * np.tile(np.eye(2)[None, None, :, :], (num_sources, num_model_freqs, 1, 1)) * au.Jy
    else:
        A = np.ones((num_sources, num_model_freqs)) * au.Jy

    cal_source_pointings = create_spherical_spiral_grid(
        pointing=phase_center,
        num_points=num_sources,
        angular_radius=fov * 0.5
    )

    point_model_data = build_point_source_model(
        model_freqs=model_freqs,
        ra=cal_source_pointings.ra,
        dec=cal_source_pointings.dec,
        A=A
    )

    # Calibrator models
    # Turn each facet into a 1 facet sky model
    cal_sky_models = []
    for facet_idx in range(num_sources):
        cal_sky_models.append(
            BasePointSourceModel(
                model_freqs=point_model_data.model_freqs,
                A=point_model_data.A[facet_idx:facet_idx + 1],  # [facet=1,num_model_freqs, [2,2]]
                ra=point_model_data.ra[facet_idx:facet_idx + 1],  # [facet=1]
                dec=point_model_data.dec[facet_idx:facet_idx + 1],  # [facet=1]
                convention=point_model_data.convention
            )
        )

    background_source_pointings = create_spherical_spiral_grid(
        pointing=phase_center,
        num_points=2 * num_sources,
        angular_radius=fov * 0.5
    )
    # remove the first source which overlaps with central one
    background_source_pointings = background_source_pointings[1:]
    plt.scatter(cal_source_pointings.ra.deg, cal_source_pointings.dec.deg, label='cal sources', marker='*', color='r')
    plt.scatter(background_source_pointings.ra.deg, background_source_pointings.dec.deg, label='background sources',
                marker='o', color='b')
    plt.legend()
    plt.show()
    if full_stokes:
        total_flux = 0.5 * np.tile(np.eye(2)[None, None, :, :],
                                   (len(background_source_pointings), num_model_freqs, 1, 1)) * au.Jy
    else:
        total_flux = np.ones((len(background_source_pointings), num_model_freqs)) * au.Jy
    major_axis = 100 * np.ones((len(background_source_pointings))) * psf_size
    minor_axis = 50 * np.ones((len(background_source_pointings))) * psf_size
    position_angle = np.random.uniform(-np.pi, np.pi, size=(len(background_source_pointings))) * au.rad

    gaussian_model_data = build_gaussian_source_model(
        model_freqs=model_freqs,
        ra=background_source_pointings.ra,
        dec=background_source_pointings.dec,
        A=total_flux,
        major_axis=major_axis,
        minor_axis=minor_axis,
        pos_angle=position_angle,
        order_approx=1
    )

    return point_model_data, gaussian_model_data, cal_sky_models


def grid_residuls(visibilities: ComplexArray, weights: FloatArray,
                  visibility_coords: VisibilityCoords, flags: BoolArray, full_stokes: bool, num_l, num_m, dl, dm, l0,
                  m0):
    """
    Grids the visibilities for a single solution interval.

    Args:
        visibilities: [Ts, B, F[,2,2]] the visibilities
        weights: [Ts, B, F[,2,2]] the weights
        flags: [Ts, B, F[,2,2]] the flags, True means flagged, don't grid.

    Returns:
        the gridded image and psf
    """

    freqs = np.asarray(visibility_coords.freqs)  # [C]
    uvw = np.array(visibility_coords.uvw)  # [T, B, 3]
    num_rows = visibilities.shape[0] * visibilities.shape[1]
    num_chan = np.shape(freqs)[0]
    if full_stokes:
        image_buffer = np.zeros((num_l, num_m, 2, 2), dtype=np.float32, order='F')
        psf_buffer = np.zeros((num_l, num_m, 2, 2), dtype=np.float32, order='F')
        pol_array = np.arange(2)
    else:
        # Add extra axes
        visibilities = visibilities[..., None, None]
        weights = weights[..., None, None]
        flags = flags[..., None, None]
        image_buffer = np.zeros((num_l, num_m, 1, 1), dtype=np.float32, order='F')
        psf_buffer = np.zeros((num_l, num_m, 1, 1), dtype=np.float32, order='F')
        pol_array = np.arange(1)

    if full_stokes:
        num_threads_outer = 4
        num_threads_inner = 8
    else:
        num_threads_outer = 1
        num_threads_inner = 32

    def single_run(p_idx, q_idx):
        _visibilities = visibilities[..., p_idx, q_idx].reshape((num_rows, num_chan))
        _weights = weights[..., p_idx, q_idx].reshape((num_rows, num_chan))
        _mask = np.logical_not(flags[..., p_idx, q_idx].reshape((num_rows, num_chan)))

        vis_to_image_np(
            uvw=uvw.reshape((num_rows, 3)),
            freqs=freqs,
            vis=_visibilities,
            pixsize_m=quantity_to_np(dm, 'rad'),
            pixsize_l=quantity_to_np(dl, 'rad'),
            center_l=quantity_to_np(l0, 'rad'),
            center_m=quantity_to_np(m0, 'rad'),
            npix_l=num_l,
            npix_m=num_m,
            wgt=_weights,
            mask=_mask,
            epsilon=1e-6,
            double_precision_accumulation=False,
            scale_by_n=True,
            normalise=True,
            output_buffer=image_buffer[:, :, p_idx, q_idx],
            num_threads=num_threads_inner
        )
        # todo: PB correction
        vis_to_image_np(
            uvw=uvw.reshape((num_rows, 3)),
            freqs=freqs,
            vis=np.ones_like(_visibilities),
            pixsize_m=quantity_to_np(dm, 'rad'),
            pixsize_l=quantity_to_np(dl, 'rad'),
            center_l=quantity_to_np(l0, 'rad'),
            center_m=quantity_to_np(m0, 'rad'),
            npix_l=num_l,
            npix_m=num_m,
            wgt=_weights,
            mask=_mask,
            epsilon=1e-6,
            double_precision_accumulation=False,
            scale_by_n=True,
            normalise=True,
            output_buffer=psf_buffer[:, :, p_idx, q_idx],
            num_threads=num_threads_inner
        )

    cb = construct_threaded_callback(
        single_run, 0, 0,
        num_threads=num_threads_outer
    )
    _ = cb(pol_array[:, None], pol_array[None, :])

    if np.all(image_buffer == 0) or not np.all(np.isfinite(image_buffer)):
        print(f"Image buffer is all zeros or contains NaNs/Infs")
    if np.all(psf_buffer == 0) or not np.all(np.isfinite(psf_buffer)):
        print(f"PSF buffer is all zeros or contains NaNs/Infs")

    if full_stokes:
        image_buffer = np.asarray(
            broadcast_translate_corrs(
                jnp.asarray(image_buffer),
                (('XX', 'XY'), ('YX', 'YY')), ('I', 'Q', 'U', 'V')
            )
        )
        psf_buffer = np.asarray(
            broadcast_translate_corrs(
                jnp.asarray(psf_buffer),
                (('XX', 'XY'), ('YX', 'YY')), ('I', 'Q', 'U', 'V')
            )
        )
        return image_buffer, psf_buffer
    else:
        # remove the last dimensions, already I, so remove 1 axis
        return image_buffer[..., 0], psf_buffer[..., 0]


@partial(jax.jit, static_argnames=['full_stokes'])
def predict_and_sample(key, freqs, times, point_model_data: BasePointSourceModel,
                       gaussian_model_data: BaseGaussianSourceModel,
                       geodesic_model: BaseGeodesicModel, far_field_delay_engine: BaseFarFieldDelayEngine,
                       near_field_delay_engine: BaseNearFieldDelayEngine,
                       system_equivalent_flux_density_Jy,
                       chan_width_hz,
                       t_int_s,
                       full_stokes: bool
                       ):
    visibility_coords = far_field_delay_engine.compute_visibility_coords(
        freqs=freqs,
        times=times,
        with_autocorr=False
    )
    vis_points = point_model_data.predict(
        visibility_coords=visibility_coords,
        gain_model=None,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        geodesic_model=geodesic_model
    )  # [T, B, C[, 2, 2]]

    vis_gaussians = gaussian_model_data.predict(
        visibility_coords=visibility_coords,
        gain_model=None,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        geodesic_model=geodesic_model
    )  # [T, B, C[, 2, 2]]

    vis = vis_points + vis_gaussians

    # Add noise
    num_pol = 2 if full_stokes else 1
    noise_scale = calc_baseline_noise(
        system_equivalent_flux_density=system_equivalent_flux_density_Jy,
        chan_width_hz=chan_width_hz,
        t_int_s=t_int_s
    )
    key1, key2 = jax.random.split(key)
    noise = mp_policy.cast_to_vis(
        (noise_scale / np.sqrt(num_pol)) * jax.lax.complex(
            jax.random.normal(key1, np.shape(vis)),
            jax.random.normal(key2, np.shape(vis))
        )
    )

    vis += noise
    weights = jnp.full(np.shape(vis), 1 / noise_scale ** 2, mp_policy.weight_dtype)
    flags = jnp.full(np.shape(vis), False, mp_policy.flag_dtype)
    return vis, weights, flags, visibility_coords


@partial(jax.jit, static_argnames=['full_stokes'])
def predict_model(model_freqs, model_times, cal_sky_models: List[BasePointSourceModel],
                  background_sky_models: List[BaseGaussianSourceModel],
                  geodesic_model: BaseGeodesicModel, far_field_delay_engine: BaseFarFieldDelayEngine,
                  near_field_delay_engine: BaseNearFieldDelayEngine,
                  full_stokes: bool
                  ):
    visibility_coords = far_field_delay_engine.compute_visibility_coords(
        freqs=model_freqs,
        times=model_times,
        with_autocorr=False
    )
    vis_cal = []
    for source_model in cal_sky_models:
        _vis = source_model.predict(
            visibility_coords=visibility_coords,
            gain_model=None,
            near_field_delay_engine=near_field_delay_engine,
            far_field_delay_engine=far_field_delay_engine,
            geodesic_model=geodesic_model
        )  # [T, B, C[, 2, 2]]
        vis_cal.append(_vis)
    vis_cal = jnp.stack(vis_cal, axis=0)  # [S, T, B, C[, 2, 2]]
    vis_background = []
    for source_model in background_sky_models:
        _vis = source_model.predict(
            visibility_coords=visibility_coords,
            gain_model=None,
            near_field_delay_engine=near_field_delay_engine,
            far_field_delay_engine=far_field_delay_engine,
            geodesic_model=geodesic_model
        )  # [T, B, C[, 2, 2]]
        vis_background.append(_vis)
    vis_background = jnp.stack(vis_background, axis=0)  # [E, T, B, C[, 2, 2]]

    return vis_cal, vis_background, visibility_coords


@partial(jax.jit, static_argnames=['full_stokes', 'num_antennas'])
def calibrate(vis_model, vis_data_avg, weights_avg, flags_avg, cal_visibility_coords, solve_state, full_stokes: bool,
              num_antennas: int):
    cal = Calibration(
        gain_probabilistic_model=GainPriorModel(
            gain_stddev=1.,
            dd_dof=1,
            di_dof=1,
            double_differential=True,
            dd_type='unconstrained',
            di_type='unconstrained',
            full_stokes=full_stokes
        ),
        full_stokes=full_stokes,
        num_ant=num_antennas,
        verbose=True
    )
    return cal.step(
        vis_model=vis_model,
        vis_data=vis_data_avg,
        weights=weights_avg,
        flags=flags_avg,
        freqs=cal_visibility_coords.freqs,
        times=cal_visibility_coords.times,
        antenna1=cal_visibility_coords.antenna1,
        antenna2=cal_visibility_coords.antenna2,
        state=solve_state
    )


@jax.jit
def calc_residual(vis_model, vis_data, gains, antenna1, antenna2):
    return compute_residual(vis_model, vis_data, gains, antenna1, antenna2)


def main(plot_folder: str, image_name: str, array_name: str, num_sources: int, num_sol_ints_time: int,
         full_stokes: bool,
         fov: au.Quantity,
         oversample_factor: float = 3.8, skip_calibration: bool = False, frac_aperture: float = 1.):
    os.makedirs(plot_folder, exist_ok=True)

    num_model_times_per_solution_interval = 1  # Tm = num_model_times_per_solution_interval
    num_model_freqs_per_solution_interval = 1  # Cm = num_model_freqs_per_solution_interval

    # Create array setup
    (ref_time, obstimes, obsfreqs, phase_center, antennas, geodesic_model, far_field_delay_engine,
     near_field_delay_engine,
     system_equivalent_flux_density, chan_width, integration_time) = build_mock_obs_setup(
        array_name, num_sol_ints_time, frac_aperture)

    system_equivalent_flux_density_Jy = quantity_to_jnp(system_equivalent_flux_density, 'Jy')
    chan_width_hz = quantity_to_jnp(chan_width, 'Hz')
    t_int_s = quantity_to_jnp(integration_time, 's')

    # Create sky model of grid of point sources
    point_model_data, gaussian_model_data, cal_sky_models = create_sky_model(
        phase_center=phase_center, num_sources=num_sources, model_freqs=obsfreqs,
        full_stokes=full_stokes, fov=fov, psf_size=3.3 * au.arcsec
    )

    num_pixel, dl, dm, l0, m0 = get_array_image_parameters(array_name, fov, oversample_factor)

    def generate_data(key) -> Generator[Tuple[ComplexArray, FloatArray, BoolArray, VisibilityCoords], None, None]:
        freqs = quantity_to_jnp(obsfreqs, 'Hz')
        for sol_int_time_idx in range(num_sol_ints_time):
            key, sample_key = jax.random.split(key)
            times = time_to_jnp(obstimes[sol_int_time_idx * 4:(sol_int_time_idx + 1) * 4], ref_time)
            model_times = average_rule(times, num_model_times_per_solution_interval, axis=0)
            model_freqs = average_rule(freqs, num_model_freqs_per_solution_interval, axis=0)
            with TimerLog(f"Predicting data for solution interval {sol_int_time_idx}"):
                vis_data, weights, flags, visibility_coords = jax.block_until_ready(
                    predict_and_sample(
                        key=key,
                        freqs=freqs,
                        times=times,
                        point_model_data=point_model_data,
                        gaussian_model_data=gaussian_model_data,
                        geodesic_model=geodesic_model,
                        far_field_delay_engine=far_field_delay_engine,
                        near_field_delay_engine=near_field_delay_engine,
                        system_equivalent_flux_density_Jy=system_equivalent_flux_density_Jy,
                        chan_width_hz=chan_width_hz,
                        t_int_s=t_int_s,
                        full_stokes=full_stokes
                    )
                )
            with TimerLog(f"Predicting model or solution interval {sol_int_time_idx}"):
                vis_cal, vis_background, cal_visibility_coords = jax.block_until_ready(
                    predict_model(
                        model_freqs=model_freqs,
                        model_times=model_times,
                        cal_sky_models=cal_sky_models,
                        background_sky_models=[gaussian_model_data],
                        geodesic_model=geodesic_model,
                        far_field_delay_engine=far_field_delay_engine,
                        near_field_delay_engine=near_field_delay_engine,
                        full_stokes=full_stokes
                    )
                )

            yield sol_int_time_idx, (vis_data, weights, flags, visibility_coords), (
                vis_cal, vis_background, cal_visibility_coords)

    # average data to match model: [Ts, B, Cs[, 2, 2]] -> [Tm, B, Cm[, 2, 2]]
    time_average_rule = partial(
        average_rule,
        num_model_size=num_model_times_per_solution_interval,
        axis=0
    )
    freq_average_rule = partial(
        average_rule,
        num_model_size=num_model_freqs_per_solution_interval,
        axis=2
    )
    # Predict data and model
    solver_state = None
    for sol_int_time_idx, (vis_data, weights, flags, visibility_coords), (
            vis_cal, vis_background, cal_visibility_coords) in generate_data(jax.random.PRNGKey(0)):
        # print(vis_data, weights, flags, visibility_coords)

        if skip_calibration:
            vis_residuals = vis_data
        else:
            vis_model = jnp.concatenate([vis_cal, vis_background], axis=0)  # [S + E, T, B, C[, 2, 2]]

            # Average using average rule
            with TimerLog("Averaging data"):
                vis_data_avg = time_average_rule(freq_average_rule(vis_data))
                weights_avg = jnp.reciprocal(time_average_rule(freq_average_rule(jnp.reciprocal(weights))))
                flags_avg = freq_average_rule(time_average_rule(flags.astype(jnp.float16))).astype(jnp.bool_)

            # Construct calibration

            with TimerLog("Calibrating"):
                gains, solver_state, diagnostics = jax.block_until_ready(
                    calibrate(vis_model, vis_data_avg, weights_avg, flags_avg,
                              cal_visibility_coords, None,
                              full_stokes, len(antennas))
                )

            with TimerLog("Plotting calibration diagnostics"):

                # plot phase, amp over aperature
                for i in range(np.shape(gains)[0]):
                    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
                    _gain = gains[i]  # [Tm, A, Cm, 2, 2]
                    if full_stokes:
                        _gain = _gain[0, :, 0, 0, 0]
                    else:
                        _gain = _gain[0, :, 0]
                    _gain = _gain / _gain[0]
                    _phase = np.angle(_gain)
                    _amplitude = np.abs(_gain)
                    lon = antennas.geodetic.lon
                    lat = antennas.geodetic.lat
                    sc = axs[0].scatter(lon, lat, c=_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
                    plt.colorbar(sc, ax=axs[0], label='Phase (rad)')
                    axs[0].set_title('Phase')
                    sc = axs[1].scatter(lon, lat, c=_amplitude, cmap='jet')
                    plt.colorbar(sc, ax=axs[1], label='Amplitude')
                    axs[1].set_title('Amplitude')
                    plt.savefig(
                        os.path.join(plot_folder, f'{image_name}_calibration_{sol_int_time_idx}_dir{i:03d}.png')
                    )
                    plt.close(fig)

                    _gain = gains[i]  # [Tm, A, Cm, 2, 2]
                    if full_stokes:
                        _gain = _gain[0, :, 0, 0, 0]
                    else:
                        _gain = _gain[0, :, 0]

                    G = _gain[:, None] * _gain.conj()[None, :]  # [A, A]
                    _phase = np.angle(G)
                    _amplitude = np.abs(G)
                    with open(os.path.join(plot_folder, f'{image_name}_aperture_phase_stats.txt'), 'a') as f:
                        mean_phase = np.mean(_phase)
                        std_phase = np.mean(np.square(_phase))
                        mean_amp = np.mean(_amplitude)
                        std_amp = np.std(_amplitude)
                        iteration = np.max(diagnostics.iteration)
                        f.write(f'{sol_int_time_idx},{i},{mean_phase},{std_phase},{mean_amp},{std_amp},{iteration}\n')

                    with open(os.path.join(plot_folder, f'aperture_stats.txt'), 'a') as f:
                        f.write(
                            f'{frac_aperture},{sol_int_time_idx},{i},{mean_phase},{std_phase},{mean_amp},{std_amp},{iteration}\n')

                    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
                    sc = axs[0].imshow(_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi, interpolation='nearest',
                                       origin='lower')
                    plt.colorbar(sc, ax=axs[0], label='Phase (rad)')
                    axs[0].set_title('baseline-based G phase')
                    sc = axs[1].imshow(_amplitude, cmap='jet', interpolation='nearest', origin='lower')
                    plt.colorbar(sc, ax=axs[1], label='Amplitude')
                    axs[1].set_title('baseline-based G amplitude')
                    plt.savefig(
                        os.path.join(plot_folder,
                                     f'{image_name}_calibration_baseline_{sol_int_time_idx}_dir{i:03d}.png')
                    )
                    plt.close(fig)

                # row 1: Plot error
                # row 2: Plot r
                # row 3: plot chi-2 (F_norm)
                # row 4: plot damping

                fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
                diagnostics: MultiStepLevenbergMarquardtDiagnostic
                axs[0].plot(diagnostics.iteration, diagnostics.error)
                axs[0].set_title('Error')
                axs[1].plot(diagnostics.iteration, diagnostics.r)
                axs[1].set_title('r')
                axs[2].plot(diagnostics.iteration, diagnostics.F_norm)
                axs[2].set_title('|F|')
                axs[3].plot(diagnostics.iteration, diagnostics.damping)
                axs[3].set_title('Damping')
                axs[3].set_xlabel('Iteration')
                plt.savefig(
                    os.path.join(plot_folder,
                                 f'{image_name}_diagnostics_{sol_int_time_idx}.png')
                )
                plt.close(fig)
            # Compute residuals
            with TimerLog("Computing residuals"):
                num_cals = np.shape(vis_cal)[0]
                vis_residuals = jax.block_until_ready(
                    calc_residual(vis_cal, vis_data, gains[:num_cals], visibility_coords.antenna1,
                                  visibility_coords.antenna2)
                )
        # Grid result
        with TimerLog("Gridding residuals"):
            image, psf = grid_residuls(
                visibilities=vis_residuals, weights=weights, visibility_coords=visibility_coords,
                flags=flags, full_stokes=full_stokes,
                num_l=num_pixel, num_m=num_pixel, dl=dl, dm=dm, l0=l0, m0=m0
            )

        with TimerLog("Plotting image"):
            img = image[..., 0]
            fig, ax = plt.subplots(1, 1)

            vmin = np.min(img)
            vmax = -vmin * 10
            lmax = (dl.to('rad') * num_pixel / 2).value
            mmax = (dm.to('rad') * num_pixel / 2).value
            im = ax.imshow(
                img.T, interpolation='nearest', origin='lower',
                extent=(-lmax, lmax, -mmax, mmax),
                vmin=vmin, vmax=vmax,
                cmap='jet'
            )
            ax.set_xlabel('l [rad]')
            ax.set_ylabel('m [rad]')
            ax.set_title('I')
            # colorbar to right of ax
            cbar = fig.colorbar(im, ax=ax)
            fig.savefig(os.path.join(plot_folder, f"{image_name}_residuals_{sol_int_time_idx:03d}.png"))
            plt.close(fig)

            image_model = ImageModel(
                phase_center=phase_center,
                obs_time=ref_time,
                dl=dl,
                dm=dm,
                freqs=np.mean(obsfreqs)[None],
                bandwidth=len(obsfreqs) * chan_width,
                coherencies=('I', 'Q', 'U', 'V') if full_stokes else ('I',),
                beam_major=np.asarray(3) * au.arcsec,
                beam_minor=np.asarray(3) * au.arcsec,
                beam_pa=np.asarray(0) * au.rad,
                unit='JY/PIXEL',
                object_name='demo',
                image=image[:, :, None, :] * au.Jy  # [num_l, num_m, 1, 4/1]
            )
            save_image_to_fits(os.path.join(plot_folder, f"{image_name}_image_{sol_int_time_idx:03d}.fits"),
                               image_model=image_model,
                               overwrite=True)
            image_model.image = psf[:, :, None, :] * au.Jy  # [num_l, num_m, 1, 4/1]
            save_image_to_fits(os.path.join(plot_folder, f"{image_name}_psf_{sol_int_time_idx:03d}.fits"),
                               image_model=image_model,
                               overwrite=True)


if __name__ == '__main__':
    # main(
    #     plot_folder='plots',
    #     image_name='dsa2000W',
    #     array_name='dsa2000W',
    #     num_sources=3,
    #     num_sol_ints_time=1,
    #     full_stokes=False,
    #     fov=1 * au.deg,
    #     oversample_factor=4.5,
    #     skip_calibration=False,
    #     frac_aperture=1
    # )
    for frac_aperture in np.linspace(50 / 2048, 1., 20):
        main(
            plot_folder='plots',
            image_name=f'dsa2000W_{int(frac_aperture * 2048)}',
            array_name='dsa2000W',
            num_sources=3,
            num_sol_ints_time=1,
            full_stokes=False,
            fov=1 * au.deg,
            oversample_factor=3.8,
            skip_calibration=False,
            frac_aperture=frac_aperture
        )
