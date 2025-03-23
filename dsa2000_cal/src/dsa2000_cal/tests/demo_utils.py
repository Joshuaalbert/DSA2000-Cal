import os

os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

from functools import partial
from typing import List

import astropy.time as at
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from astropy import units as au, coordinates as ac
from matplotlib import pyplot as plt
from dsa2000_common.common.enu_frame import ENU

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.array_types import ComplexArray, FloatArray, BoolArray
from dsa2000_common.common.astropy_utils import create_spherical_spiral_grid
from dsa2000_common.common.corr_utils import broadcast_translate_corrs
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.noise import calc_baseline_noise
from dsa2000_common.common.pure_callback_utils import construct_threaded_callback
from dsa2000_common.common.quantity_utils import quantity_to_np
from dsa2000_common.common.types import VisibilityCoords
from dsa2000_common.common.wgridder import vis_to_image_np
from dsa2000_common.delay_models.base_far_field_delay_engine import build_far_field_delay_engine, \
    BaseFarFieldDelayEngine
from dsa2000_common.delay_models.base_near_field_delay_engine import build_near_field_delay_engine, \
    BaseNearFieldDelayEngine
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model, BaseGeodesicModel
from dsa2000_common.visibility_model.source_models.celestial.base_gaussian_source_model import \
    build_gaussian_source_model, \
    BaseGaussianSourceModel
from dsa2000_common.visibility_model.source_models.celestial.base_point_source_model import build_point_source_model, \
    BasePointSourceModel

tfpd = tfp.distributions


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
