import json
import os
import time
from functools import partial
from typing import NamedTuple, Dict

import astropy.constants as const
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import jax.random
import numpy as np
import pylab as plt

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import source_model_registry, array_registry
from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.astropy_utils import create_spherical_spiral_grid, get_time_of_local_meridean
from dsa2000_common.common.jax_utils import get_pytree_size
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.noise import calc_image_noise, calc_baseline_noise
from dsa2000_common.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_common.common.ray_utils import TimerLog
from dsa2000_common.common.serialise_utils import SerialisableBaseModel
from dsa2000_common.delay_models.base_far_field_delay_engine import build_far_field_delay_engine, \
    BaseFarFieldDelayEngine
from dsa2000_common.delay_models.uvw_utils import perley_lmn_from_icrs
from dsa2000_common.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_common.gain_models.gain_model import GainModel
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model, BaseGeodesicModel
from dsa2000_common.visibility_model.source_models.celestial.base_point_source_model import \
    build_point_source_model_from_wsclean_components, BasePointSourceModel
from dsa2000_fm.systematics.dish_aperture_effects import build_dish_aperture_effects
from dsa2000_fm.systematics.ionosphere import compute_x0_radius, construct_canonical_ionosphere, \
    build_ionosphere_gain_model


class Result(SerialisableBaseModel):
    phase_center: ac.ICRS
    ref_time: at.Time
    array_name: str
    bright_source_id: str
    num_measure_points: int
    angular_radius: au.Quantity
    prior_psf_sidelobe_peak: float
    pointing_offset_stddev: au.Quantity
    axial_focus_error_stddev: au.Quantity
    horizon_peak_astigmatism_stddev: au.Quantity
    turbulent: bool
    dawn: bool
    high_sun_spot: bool
    with_ionosphere: bool
    with_dish_effects: bool
    with_smearing: bool
    run_time: float
    thermal_noise: au.Quantity
    baseline_noise: au.Quantity
    result_values: Dict[str, float]


class ValuesAndRMS(NamedTuple):
    rms_no_noise: FloatArray
    max_no_noise: FloatArray
    min_no_noise: FloatArray
    mean_no_noise: FloatArray
    std_no_noise: FloatArray
    rms_noise: FloatArray
    max_noise: FloatArray
    min_noise: FloatArray
    mean_noise: FloatArray
    std_noise: FloatArray

    # mean_abs_R: FloatArray
    # std_abs_R: FloatArray
    mean_time_smear: FloatArray
    std_time_smear: FloatArray
    mean_freq_smear: FloatArray
    std_freq_smear: FloatArray
    mean_smear: FloatArray
    std_smear: FloatArray

    lmn: FloatArray  # [M]
    image: FloatArray  # [M]
    image_noise: FloatArray  # [M]


def sinc(x):
    return jax.lax.div(jnp.sin(x), x)


@partial(jax.jit, static_argnames=['smearing'])
def compute_image_and_smear_values_single_freq(
        key,
        freq,
        phase, R, A,
        phase_eval,
        n: FloatArray,
        g12,
        integration_time: FloatArray,
        channel_width: FloatArray,
        baseline_noise: FloatArray,
        smearing: bool
):
    c = quantity_to_jnp(const.c)
    # scalar -> [B] -> [B, M] -> scalar
    wavelength = c / freq
    coeff = 2 * jnp.pi / wavelength
    # compute vis with optional smearing
    fringe = jax.lax.complex(jnp.cos(coeff * phase), jnp.sin(coeff * phase)).astype(jnp.complex64)  # [B, N]
    if smearing:
        time_smear_modulation = sinc(R * (jnp.pi * integration_time / wavelength))  # [B, N]
        freq_smear_moduluation = sinc(phase * (jnp.pi * channel_width / c))  # [B, N]
        smear_modulation = time_smear_modulation * freq_smear_moduluation  # [B, N]
        mean_time_smear = jnp.mean(time_smear_modulation)
        var_time_smear = jnp.var(time_smear_modulation)
        mean_freq_smear = jnp.mean(freq_smear_moduluation)
        var_freq_smear = jnp.var(freq_smear_moduluation)
        mean_smear = jnp.mean(smear_modulation)
        var_smear = jnp.var(smear_modulation)
        vis = jnp.sum((g12) * (A * (fringe * smear_modulation)), axis=1).astype(jnp.complex64)  # [B]
    else:
        mean_time_smear = var_time_smear = mean_freq_smear = var_freq_smear = mean_smear = var_smear = jnp.asarray(
            1., jnp.float32)
        vis = jnp.sum((g12) * (A * fringe), axis=1).astype(jnp.complex64)  # [B]
    key1, key2 = jax.random.split(key)
    # divide by sqrt(2) for real and imag part
    noise = (baseline_noise / np.sqrt(2)) * jax.lax.complex(
        jax.random.normal(key1, shape=vis.shape, dtype=vis.real.dtype),
        jax.random.normal(key2, shape=vis.shape, dtype=vis.imag.dtype)).astype(
        vis.dtype)
    vis_noise = vis + noise
    # compute image, normalising
    fringe = jax.lax.complex(jnp.cos(coeff * phase_eval), jnp.sin(coeff * phase_eval)).astype(jnp.complex64)
    image_noise = n.astype(jnp.float32) * jnp.sum((vis_noise * fringe).real, axis=1)  # [M]
    image_noise /= vis.size
    delta = (
        image_noise,
        mean_time_smear, var_time_smear, mean_freq_smear, var_freq_smear, mean_smear, var_smear
    )
    return delta


@partial(jax.jit)
def pre_compute_image_and_smear_values(
        freq,
        l: FloatArray, m: FloatArray, n: FloatArray,
        bright_sky_model: BasePointSourceModel,
        total_gain_model: GainModel,
        times: FloatArray,
        far_field_delay_engine: BaseFarFieldDelayEngine,
        geodesic_model: BaseGeodesicModel,
        freqs: FloatArray,
        integration_time: FloatArray,
        ra0: FloatArray, dec0: FloatArray
):
    """
    Compute the RMS in the field due to uncalibrated extra-field sources.

    Args:
        l: [M]
        m: [M]
        n: [M]
        bright_sky_model: sky model of [N] sources
        total_gain_model: gain model
        times: [T] the times
        far_field_delay_engine: far field delay engine
        geodesic_model: geodesic model
        freqs: [F] the frequencies to compute over

    Returns:
        [M], [M], scalar, scalar, scalar, scalar, scalar, scalar
    """
    # For each frequency
    visibilty_coords = far_field_delay_engine.compute_visibility_coords(
        freqs=freqs,
        times=times,
        with_autocorr=False
    )  # [B]
    visibilty_coords_dt = far_field_delay_engine.compute_visibility_coords(
        freqs=freqs,
        times=times + 0.5 * integration_time,
        with_autocorr=False
    )  # [B]
    uvw = jax.lax.reshape(visibilty_coords.uvw, (visibilty_coords.antenna1.shape[0], 3))  # [B, 3]
    uvw_dt = jax.lax.reshape(visibilty_coords_dt.uvw, (visibilty_coords_dt.antenna1.shape[0], 3))  # [B, 3]
    # DFT
    w = uvw[..., 2]  # [B]
    w_dt = uvw_dt[..., 2]  # [B]

    lmn_sources = jnp.stack(perley_lmn_from_icrs(bright_sky_model.ra, bright_sky_model.dec, ra0, dec0),
                            axis=-1)  # [N, 3]
    lmn_geodesic, elevation = geodesic_model.compute_far_field_geodesic(times, lmn_sources,
                                                                        return_elevation=True)  # [T, A, N, 3], [T, A, N]

    A = jnp.where(elevation[0, 0, :] > 0, jnp.mean(bright_sky_model.A, axis=1), 0.)  # [N]

    phase = -(jnp.sum(lmn_sources[None, ...] * uvw[:, None, :], axis=-1) - w[:, None])  # [B, N]
    phase_dt = -(jnp.sum(lmn_sources[None, ...] * uvw_dt[:, None, :], axis=-1) - w_dt[:, None])  # [B, N]
    R = (phase_dt - phase) / (0.5 * integration_time)  # [B, N]

    lmn_eval = jnp.stack([l, m, n], axis=-1)  # [M, 3]
    phase_eval = (jnp.sum(lmn_eval[:, None, ...] * uvw[None, ...], axis=-1) - w[None, :])  # [M, B]

    gains = total_gain_model.compute_gain(freq[None], times, lmn_geodesic)  # [T, A, 1, N]
    g1 = gains[0, visibilty_coords.antenna1, 0, :]  # [B, N]
    g2 = gains[0, visibilty_coords.antenna2, 0, :]  # [B, N]
    g12 = g1 * g2.conj()
    return phase, R, A, phase_eval, g12


def compute_image_and_smear_values(
        key,
        l: FloatArray, m: FloatArray, n: FloatArray,
        bright_sky_model: BasePointSourceModel,
        total_gain_model: GainModel,
        times: FloatArray,
        far_field_delay_engine: BaseFarFieldDelayEngine,
        geodesic_model: BaseGeodesicModel,
        freqs: FloatArray,
        integration_time: FloatArray,
        channel_width: FloatArray,
        ra0: FloatArray, dec0: FloatArray,
        baseline_noise: FloatArray,
        smearing: bool
):
    """
    Compute the RMS in the field due to uncalibrated extra-field sources.

    Args:
        l: [M]
        m: [M]
        n: [M]
        bright_sky_model: sky model of [N] sources
        total_gain_model: gain model
        times: [T] the times
        far_field_delay_engine: far field delay engine
        geodesic_model: geodesic model
        freqs: [F] the frequencies to compute over

    Returns:
        [M], [M], scalar, scalar, scalar, scalar, scalar, scalar
    """

    def accumulate_over_freq(accumulate, x):
        # scalar -> [B] -> [B, M] -> scalar
        (freq, key) = x  # scalar

        phase, R, A, phase_eval, g12 = jax.tree.map(
            np.array, jax.block_until_ready(
                pre_compute_image_and_smear_values(
                    freq,
                    l, m, n,
                    bright_sky_model,
                    total_gain_model,
                    times,
                    far_field_delay_engine,
                    geodesic_model,
                    freqs,
                    integration_time,
                    ra0, dec0
                )
            )
        )
        delta = jax.tree.map(
            np.array, jax.block_until_ready(
                compute_image_and_smear_values_single_freq(
                    key,
                    freq,
                    phase, R, A,
                    phase_eval,
                    n,
                    g12,
                    integration_time, channel_width, baseline_noise,
                    smearing
                )
            )
        )
        accumulate = jax.tree.map(lambda x, y: x + y, accumulate, delta)
        return accumulate

    accumulate = (
        np.zeros(l.shape, np.float32),
        np.zeros((), np.float32), np.zeros((), np.float32),
        np.zeros((), np.float32), np.zeros((), np.float32),
        np.zeros((), np.float32), np.zeros((), np.float32)
    )

    for freq, key in zip(freqs, jax.random.split(key, len(freqs))):
        accumulate = accumulate_over_freq(accumulate, (freq, key))

    # Normalize by number of freqs
    accumulate = jax.tree.map(lambda x: x / len(freqs), accumulate)
    (
        image_noise,
        mean_time_smear, var_time_smear, mean_freq_smear, var_freq_smear, mean_smear, var_smear
    ) = accumulate

    return (
        image_noise,
        mean_time_smear, var_time_smear, mean_freq_smear, var_freq_smear, mean_smear, var_smear
    )


def compute_rms_and_values(
        key,
        l: FloatArray, m: FloatArray, n: FloatArray,
        bright_sky_model: BasePointSourceModel,
        total_gain_model: GainModel,
        times: FloatArray,
        far_field_delay_engine: BaseFarFieldDelayEngine,
        geodesic_model: BaseGeodesicModel,
        freqs: FloatArray,
        zero_point: FloatArray,
        integration_time: FloatArray,
        channel_width: FloatArray,
        ra0: FloatArray, dec0: FloatArray,
        baseline_noise: FloatArray,
        smearing: bool,
        image_batch_size: int | None = None,
        source_batch_size: int | None = None
):
    """
    Compute the RMS in the field due to uncalibrated extra-field sources.

    Args:
        key: random key for noise sampling
        l: [M]
        m: [M]
        n: [M]
        bright_sky_model: sky model of [N] sources
        total_gain_model: gain model
        times: [T] the times
        far_field_delay_engine: far field delay engine
        geodesic_model: geodesic model
        freqs: [F] the frequencies to compute over
        zero_point: the zero-point adjustment due to excluding auto-correlations

    Returns:
        scalar
    """

    if image_batch_size is None:
        image_batch_size = np.shape(l)[0]

    if source_batch_size is None:
        source_batch_size = np.shape(bright_sky_model.ra)[0]

    (
        image, image_noise,
        mean_time_smear, var_time_smear, mean_freq_smear, var_freq_smear, mean_smear, var_smear
    ) = [], [], [], [], [], [], [], []
    for image_start_idx in range(0, np.shape(l)[0], image_batch_size):
        image_stop_idx = min(image_start_idx + image_batch_size, np.shape(l)[0])
        l_batch = l[image_start_idx: image_stop_idx]
        m_batch = m[image_start_idx: image_stop_idx]
        n_batch = n[image_start_idx: image_stop_idx]
        _image, _mean_freq_smear, _mean_smear, _mean_time_smear, _var_freq_smear, _var_smear, _var_time_smear = single_compute_image_and_values(
            key, 0., bright_sky_model, channel_width, integration_time, ra0, dec0, freqs, times,
            far_field_delay_engine, geodesic_model, l_batch, m_batch, n_batch, source_batch_size, total_gain_model,
            smearing)
        _image_noise, _, _, _, _, _, _ = single_compute_image_and_values(
            key, baseline_noise, bright_sky_model, channel_width, integration_time, ra0, dec0, freqs, times,
            far_field_delay_engine, geodesic_model, l_batch, m_batch, n_batch, source_batch_size, total_gain_model,
            smearing)
        image.append(np.array(sum(_image[1:], _image[0])))
        image_noise.append(np.array(sum(_image_noise[1:], _image_noise[0])))
        mean_time_smear.append(np.mean(_mean_time_smear))
        var_time_smear.append(np.mean(_var_time_smear))
        mean_freq_smear.append(np.mean(_mean_freq_smear))
        var_freq_smear.append(np.mean(_var_freq_smear))
        mean_smear.append(np.mean(_mean_smear))
        var_smear.append(np.mean(_var_smear))

    image = np.concatenate(image, axis=0)
    image_noise = np.concatenate(image_noise, axis=0)
    mean_time_smear = np.mean(np.stack(mean_time_smear), axis=0)
    var_time_smear = np.mean(np.stack(var_time_smear), axis=0)
    mean_freq_smear = np.mean(np.stack(mean_freq_smear), axis=0)
    var_freq_smear = np.mean(np.stack(var_freq_smear), axis=0)
    mean_smear = np.mean(np.stack(mean_smear), axis=0)
    var_smear = np.mean(np.stack(var_smear), axis=0)

    # Compute RMS and image normal stats
    rms_no_noise = np.sqrt(np.mean((image - zero_point) ** 2))
    max_no_noise = np.max(image)
    min_no_noise = np.min(image)
    mean_no_noise = np.mean(image)
    std_no_noise = np.std(image)

    rms_noise = np.sqrt(np.mean((image_noise - zero_point) ** 2))
    max_noise = np.max(image_noise)
    min_noise = np.min(image_noise)
    mean_noise = np.mean(image_noise)
    std_noise = np.std(image_noise)

    lmn_eval = np.stack([l, m, n], axis=-1)  # [M, 3]

    return ValuesAndRMS(
        rms_no_noise=rms_no_noise,
        max_no_noise=max_no_noise,
        min_no_noise=min_no_noise,
        mean_no_noise=mean_no_noise,
        std_no_noise=std_no_noise,
        rms_noise=rms_noise,
        max_noise=max_noise,
        min_noise=min_noise,
        mean_noise=mean_noise,
        std_noise=std_noise,
        # mean_abs_R=mean_abs_R,
        # std_abs_R=np.sqrt(var_abs_R),
        mean_time_smear=mean_time_smear,
        std_time_smear=np.sqrt(var_time_smear),
        mean_freq_smear=mean_freq_smear,
        std_freq_smear=np.sqrt(var_freq_smear),
        mean_smear=mean_smear,
        std_smear=np.sqrt(var_smear),
        lmn=lmn_eval,
        image=image,
        image_noise=image_noise
    )


def single_compute_image_and_values(key, baseline_noise, bright_sky_model, channel_width, integration_time, ra0, dec0,
                                    freqs, times,
                                    far_field_delay_engine, geodesic_model, l_batch, m_batch, n_batch,
                                    source_batch_size, total_gain_model,
                                    smearing):
    (
        _image_noise, _mean_time_smear, _var_time_smear,
        _mean_freq_smear, _var_freq_smear,
        _mean_smear, _var_smear
    ) = [], [], [], [], [], [], []
    _key = key  # Use the same sequence of keys for accumulation over directions (important)
    for source_start_idx in range(0, np.shape(bright_sky_model.ra)[0], source_batch_size):
        source_stop_idx = min(source_start_idx + source_batch_size, np.shape(bright_sky_model.ra)[0])
        bright_sky_model_batch = BasePointSourceModel(
            model_freqs=bright_sky_model.model_freqs,
            ra=bright_sky_model.ra[source_start_idx: source_stop_idx],
            dec=bright_sky_model.dec[source_start_idx: source_stop_idx],
            A=bright_sky_model.A[source_start_idx: source_stop_idx]
        )
        _key, sample_key = jax.random.split(_key)
        kwargs = dict(
            key=sample_key,
            l=l_batch, m=m_batch, n=n_batch,
            bright_sky_model=bright_sky_model_batch,
            total_gain_model=total_gain_model, times=times, far_field_delay_engine=far_field_delay_engine,
            geodesic_model=geodesic_model, freqs=freqs, integration_time=integration_time,
            channel_width=channel_width, ra0=ra0, dec0=dec0, baseline_noise=baseline_noise,
            smearing=smearing
        )
        for k, v in kwargs.items():
            dsa_logger.info(f"Size of {k} is {get_pytree_size(v) / 2 ** 30} GB")
        (
            __image_noise, __mean_time_smear, __var_time_smear,
            __mean_freq_smear, __var_freq_smear,
            __mean_smear, __var_smear
        ) = jax.block_until_ready(
            compute_image_and_smear_values(
                **kwargs
            )
        )
        _image_noise.append(__image_noise)
        _mean_time_smear.append(__mean_time_smear)
        _var_time_smear.append(__var_time_smear)
        _mean_freq_smear.append(__mean_freq_smear)
        _var_freq_smear.append(__var_freq_smear)
        _mean_smear.append(__mean_smear)
        _var_smear.append(__var_smear)
    return _image_noise, _mean_freq_smear, _mean_smear, _mean_time_smear, _var_freq_smear, _var_smear, _var_time_smear


@jax.jit
def apply_dish_effects(sample_key, dish_aperture_effects, beam_model, geodesic_model):
    return dish_aperture_effects.apply_dish_aperture_effects(
        sample_key,
        beam_model,
        geodesic_model=geodesic_model
    )


@jax.jit
def get_beam_amp_per_bright_source(freqs, times, lmn_bright_sources, geodesic_model,
                                   beam_model: BaseSphericalInterpolatorGainModel):
    lmn_geodesic = geodesic_model.compute_far_field_geodesic(
        times=times,
        lmn_sources=lmn_bright_sources,
        antenna_indices=jnp.asarray([0]),
        return_elevation=False
    )  # [num_time, num_ant=1, num_sources, 3]
    if beam_model.tile_antennas:
        beam_gain = beam_model.compute_gain(
            freqs=freqs,
            times=times,
            lmn_geodesic=lmn_geodesic
        )  # [num_time, 1, num_freq, num_sources]
    else:
        beam_gain = beam_model.compute_gain(
            freqs=freqs,
            times=times,
            lmn_geodesic=lmn_geodesic,
            antenna_indices=jnp.asarray([0])
        )  # [num_time, 1, num_freq, num_sources]
    beam_amp = jnp.mean(jnp.abs(beam_gain), axis=(0, 1)).T  # [num_sources, num_freq]
    return beam_amp


def simulate_rms(
        cpu,
        gpu,
        result_num: int,
        seed: int,
        save_folder: str,
        array_name: str,
        pointing: ac.ICRS,
        num_measure_points: int,
        image_batch_size: int,
        source_batch_size: int,
        angular_radius: au.Quantity,
        prior_psf_sidelobe_peak: float,
        bright_source_id: str,
        pointing_offset_stddev: au.Quantity,
        axial_focus_error_stddev: au.Quantity,
        horizon_peak_astigmatism_stddev: au.Quantity,
        turbulent: bool,
        dawn: bool,
        high_sun_spot: bool,
        with_ionosphere: bool = False,
        with_dish_effects: bool = False,
        with_smearing: bool = True
):
    with jax.default_device(cpu):
        plt.close('all')
        t0 = time.time()

        key = jax.random.PRNGKey(seed)
        fill_registries()
        os.makedirs(save_folder, exist_ok=True)
        array = array_registry.get_instance(array_registry.get_match(array_name))
        array_location = array.get_array_location()
        ref_time = get_time_of_local_meridean(pointing, array_location, at.Time('2022-01-01T00:00:00', scale='utc'))
        times = ref_time[None]
        antennas = array.get_antennas()

        phase_center = pointing

        freqs = array.get_channels()
        freqs_jax = quantity_to_jnp(freqs)
        times_jax = time_to_jnp(times, ref_time)

        thermal_noise = float(calc_image_noise(
            system_equivalent_flux_density=quantity_to_jnp(array.get_system_equivalent_flux_density(), 'Jy'),
            bandwidth_hz=quantity_to_jnp(array.get_channel_width()) * len(freqs),
            t_int_s=quantity_to_jnp(array.get_integration_time(), 's'),
            num_antennas=len(antennas),
            flag_frac=0.33,
            num_pol=2
        )) * au.Jy

        baseline_noise = float(calc_baseline_noise(
            system_equivalent_flux_density=quantity_to_jnp(array.get_system_equivalent_flux_density(), 'Jy'),
            chan_width_hz=quantity_to_jnp(array.get_channel_width(), 'Hz'),
            t_int_s=quantity_to_jnp(array.get_integration_time(), 's')
        ) / np.sqrt(2)) * au.Jy  # assume stokes I so 2 cross pols combined reduces noise by sqrt(2)

        far_field_delay_engine = build_far_field_delay_engine(
            antennas=antennas,
            phase_center=phase_center,
            start_time=times.min(),
            end_time=times.max(),
            ref_time=ref_time
        )

        # near_field_delay_engine = build_near_field_delay_engine(
        #     antennas=antennas,
        #     start_time=times.min(),
        #     end_time=times.max(),
        #     ref_time=ref_time
        # )

        geodesic_model = build_geodesic_model(
            antennas=antennas,
            array_location=array_location,
            phase_center=phase_center,
            obstimes=times,
            ref_time=ref_time,
            pointings=phase_center
        )

    with TimerLog("Constructing the beam model"):
        with jax.default_device(cpu):
            beam_model = build_beam_gain_model(
                array_name=array_name,
                full_stokes=False,
                times=times,
                ref_time=ref_time,
                freqs=freqs,
                resolution=127
            )

            beam_model.plot_regridded_beam(
                save_fig=os.path.join(save_folder, f'beam_model_{result_num:03d}.png'),
                show=False
            )

    if with_dish_effects:
        with TimerLog('Simulating dish aperture effects'):
            with jax.default_device(cpu):
                dish_aperture_effects = build_dish_aperture_effects(
                    dish_diameter=array.get_antenna_diameter(),
                    focal_length=array.get_focal_length(),
                    elevation_pointing_error_stddev=pointing_offset_stddev,
                    cross_elevation_pointing_error_stddev=pointing_offset_stddev,
                    axial_focus_error_stddev=axial_focus_error_stddev,
                    elevation_feed_offset_stddev=1 * au.mm,  # Assume perpendicular accuracy is quite good.
                    cross_elevation_feed_offset_stddev=1 * au.mm,
                    horizon_peak_astigmatism_stddev=horizon_peak_astigmatism_stddev,
                    # surface_error_mean=0 * au.mm, # TODO: update to use a GP model for RMS surface error
                    # surface_error_stddev=1 * au.mm
                )
                key, sample_key = jax.random.split(key)
                beam_model = apply_dish_effects(
                    sample_key,
                    dish_aperture_effects,
                    beam_model,
                    geodesic_model
                )
                beam_model.plot_regridded_beam(
                    save_fig=os.path.join(save_folder, f'beam_model_with_aperture_effects_{result_num:03d}.png'),
                    ant_idx=-1,
                    show=False
                )

    with TimerLog('Constructing the bright source model'):
        with jax.default_device(cpu):
            # Get the crest peak outside the angular radius
            L, M = jnp.meshgrid(beam_model.lvec, beam_model.mvec)
            mask = jnp.sqrt(L ** 2 + M ** 2) > quantity_to_jnp(angular_radius, 'rad')
            global_crest_peak = 0.
            for freq_idx in range(len(beam_model.model_freqs)):
                # model_gains [num_model_times, lres, mres, [num_ant,] num_model_freqs[, 2, 2]]
                global_crest_peak = max(global_crest_peak,
                                        float(jnp.max(jnp.abs(beam_model.model_gains[0, :, :, freq_idx][mask]))))

            wsclean_clean_component_file = source_model_registry.get_instance(
                source_model_registry.get_match(bright_source_id)
            ).get_wsclean_clean_component_file()

            unfiltered_bright_sky_model = build_point_source_model_from_wsclean_components(
                wsclean_clean_component_file=wsclean_clean_component_file,
                model_freqs=freqs[[0, -1]],
                full_stokes=False
            )

            dist_from_pointing = ac.ICRS(ra=unfiltered_bright_sky_model.ra * au.rad,
                                         dec=unfiltered_bright_sky_model.dec * au.rad).separation(
                phase_center)

            _, _, n = perley_lmn_from_icrs(
                unfiltered_bright_sky_model.ra,
                unfiltered_bright_sky_model.dec,
                phase_center.ra.rad,
                phase_center.dec.rad
            )

            select_mask = jnp.logical_and(
                n > 0,
                quantity_to_jnp(dist_from_pointing, 'rad') > quantity_to_jnp(angular_radius, 'rad')
            )

            bright_sky_model = BasePointSourceModel(
                model_freqs=unfiltered_bright_sky_model.model_freqs,
                ra=unfiltered_bright_sky_model.ra[select_mask],
                dec=unfiltered_bright_sky_model.dec[select_mask],
                A=unfiltered_bright_sky_model.A[select_mask]
            )

            lmn_bright_sources = jnp.stack(
                perley_lmn_from_icrs(bright_sky_model.ra, bright_sky_model.dec, phase_center.ra.rad,
                                     phase_center.dec.rad), axis=-1)  # [N, 3]

            batch_size = 1000
            beam_amp = []
            for start_idx in range(0, len(bright_sky_model.A), batch_size):
                stop_idx = min(start_idx + batch_size, len(bright_sky_model.A))
                _beam_amp = jax.block_until_ready(
                    get_beam_amp_per_bright_source(
                        bright_sky_model.model_freqs, times_jax,
                        lmn_bright_sources[start_idx: stop_idx],
                        geodesic_model, beam_model
                    )
                )  # [N, F]
                beam_amp.append(_beam_amp)
            beam_amp = jnp.concatenate(beam_amp, axis=0)

            with open(os.path.join(save_folder, f'beam_amps_{result_num:03d}.json'), 'w') as f:
                f.write(json.dumps(beam_amp.tolist()))

            # A * beam^2 * psf > sigma => A > 1muJy / beam^2 / psf
            global_flux_cut = thermal_noise / (global_crest_peak ** 2 * prior_psf_sidelobe_peak)
            dsa_logger.info(f"Thermal floor: {thermal_noise}")
            dsa_logger.info(f"Global crest peak outside {angular_radius}: {global_crest_peak}")
            dsa_logger.info(f"PSF sidelobe peak: {prior_psf_sidelobe_peak}")
            dsa_logger.info(f"==> Flux cut: {global_flux_cut}")
            select_mask = jnp.any(bright_sky_model.A > quantity_to_jnp(global_flux_cut, 'Jy'), axis=1)  # [N]
            dsa_logger.info(
                f"Global: {np.sum(select_mask)} selected brightest sources out of {len(bright_sky_model.A)}")

            flux_cut = thermal_noise / (beam_amp ** 2 * prior_psf_sidelobe_peak)  # [N, F]
            select_mask = jnp.any(bright_sky_model.A > flux_cut, axis=1)  # [N]
            dsa_logger.info(f"Mean beam amp: {jnp.mean(beam_amp)}")
            dsa_logger.info(f"Mean flux cut: {jnp.mean(flux_cut[select_mask])}")
            dsa_logger.info(
                f"Per-source: {np.sum(select_mask)} selected brightest sources out of {len(bright_sky_model.A)}")

            bright_sky_model = BasePointSourceModel(
                model_freqs=bright_sky_model.model_freqs,
                ra=bright_sky_model.ra[select_mask],
                dec=bright_sky_model.dec[select_mask],
                A=bright_sky_model.A[select_mask]
            )

            # directions = ac.ICRS(ra=np.asarray(bright_sky_model.ra) * au.rad,
            #                      dec=np.asarray(bright_sky_model.dec) * au.rad)
            l, m, n = perley_lmn_from_icrs(unfiltered_bright_sky_model.ra, unfiltered_bright_sky_model.dec,
                                           phase_center.ra.rad,
                                           phase_center.dec.rad)
            fig, axs = plt.subplots(2, 1, figsize=(10, 16))
            sc = axs[0].scatter(l, m, c=jnp.log10(unfiltered_bright_sky_model.A.mean(axis=1)), s=1, marker='*')
            plt.colorbar(sc, ax=axs[0], label='log10(A)')
            axs[0].set_xlabel('l [proj.rad]')
            axs[0].set_ylabel('m [proj.rad]')
            axs[0].set_title(f'Unfiltered Bright Sources: Direction {pointing}')

            sc = axs[1].scatter(unfiltered_bright_sky_model.ra * 180 / np.pi,
                                unfiltered_bright_sky_model.dec * 180 / np.pi,
                                c=jnp.log10(unfiltered_bright_sky_model.A.mean(axis=1)), s=1, marker='*')
            plt.colorbar(sc, ax=axs[1], label='log10(A)')
            axs[1].set_xlabel('RA (deg)')
            axs[1].set_ylabel('DEC (deg)')

            fig.tight_layout()
            fig.savefig(os.path.join(save_folder, f'unfiltered_bright_sources_{result_num:03d}.png'))
            plt.close(fig)

            l, m, n = perley_lmn_from_icrs(bright_sky_model.ra, bright_sky_model.dec, phase_center.ra.rad,
                                           phase_center.dec.rad)

            fig, axs = plt.subplots(2, 1, figsize=(10, 16))
            sc = axs[0].scatter(l, m, c=jnp.log10(bright_sky_model.A.mean(axis=1)), s=10, marker='*')
            plt.colorbar(sc, ax=axs[0], label='log10(A)')
            axs[0].set_xlabel('l [proj.rad]')
            axs[0].set_ylabel('m [proj.rad]')
            axs[0].set_title(f'Filtered Bright Sources: Direction {pointing}')

            sc = axs[1].scatter(bright_sky_model.ra * 180 / np.pi, bright_sky_model.dec * 180 / np.pi,
                                c=jnp.log10(bright_sky_model.A.mean(axis=1)), s=10, marker='*')
            plt.colorbar(sc, ax=axs[1], label='log10(A)')
            axs[1].set_xlabel('RA (deg)')
            axs[1].set_ylabel('DEC (deg)')

            fig.tight_layout()
            fig.savefig(os.path.join(save_folder, f'filtered_bright_sources_{result_num:03d}.png'))
            plt.close(fig)

    if with_ionosphere:
        with TimerLog("Simulating ionosphere..."):
            with jax.default_device(cpu):
                x0_radius = compute_x0_radius(array_location, ref_time)
                ionosphere = construct_canonical_ionosphere(
                    x0_radius=x0_radius,
                    turbulent=turbulent,
                    dawn=dawn,
                    high_sun_spot=high_sun_spot
                )

                ionosphere_model_directions = create_spherical_spiral_grid(
                    pointing=phase_center,
                    num_points=20,
                    angular_radius=90 * au.deg
                )
                dsa_logger.info(f"Number of ionosphere sample directions: {len(ionosphere_model_directions)}")

                key, sim_key = jax.random.split(key)
                ionosphere_gain_model = build_ionosphere_gain_model(
                    key=sim_key,
                    ionosphere=ionosphere,
                    model_freqs=freqs[[0, -1]],
                    antennas=antennas,
                    ref_location=array_location,
                    times=times,
                    ref_time=ref_time,
                    directions=ionosphere_model_directions,
                    phase_centre=phase_center,
                    full_stokes=False,
                    predict_batch_size=512,
                    resolution=127,
                    save_file=os.path.join(save_folder, f'simulated_dtec_{result_num:03d}.json')
                )
                ionosphere_gain_model.plot_regridded_beam(
                    save_fig=os.path.join(save_folder, f'ionosphere_model_{result_num:03d}.png'),
                    ant_idx=-1,
                    show=False
                )
                total_gain_model = beam_model @ ionosphere_gain_model
    else:
        with jax.default_device(cpu):
            total_gain_model = beam_model

    with TimerLog("Computing RMS for pointing..."):
        with jax.default_device(gpu):
            measure_directions = create_spherical_spiral_grid(
                pointing=phase_center,
                num_points=num_measure_points,
                angular_radius=angular_radius
            )  # [M]

            l, m, n = perley_lmn_from_icrs(measure_directions.ra.rad, measure_directions.dec.rad, phase_center.ra.rad,
                                           phase_center.dec.rad)
            # For each frequency:
            #   For each bright source:
            #     compute model vis
            #     accumulate visibilities
            #   DFT vis only M directions
            #   Compute RMS, with zero-point adjustment +1/(N-1) or not
            zero_point = 0.  # - 1 / (len(antennas) - 1)
            key, sample_key = jax.random.split(key)
            values = compute_rms_and_values(
                key=sample_key,
                l=l, m=m, n=n,
                bright_sky_model=bright_sky_model,
                total_gain_model=total_gain_model,
                times=times_jax,
                far_field_delay_engine=far_field_delay_engine,
                geodesic_model=geodesic_model,
                freqs=freqs_jax,
                zero_point=zero_point,
                integration_time=quantity_to_jnp(array.get_integration_time(), 's'),
                channel_width=quantity_to_jnp(array.get_channel_width(), 'Hz'),
                ra0=phase_center.ra.rad,
                dec0=phase_center.dec.rad,
                baseline_noise=quantity_to_jnp(baseline_noise, 'Jy'),
                smearing=with_smearing,
                image_batch_size=image_batch_size,
                source_batch_size=source_batch_size
            )

            result_values = jax.tree.map(np.array, values)
            t1 = time.time()
            result = Result(
                phase_center=phase_center,
                ref_time=ref_time,
                array_name=array_name,
                bright_source_id=bright_source_id,
                num_measure_points=num_measure_points,
                angular_radius=angular_radius,
                prior_psf_sidelobe_peak=prior_psf_sidelobe_peak,
                pointing_offset_stddev=pointing_offset_stddev,
                axial_focus_error_stddev=axial_focus_error_stddev,
                horizon_peak_astigmatism_stddev=horizon_peak_astigmatism_stddev,
                turbulent=turbulent,
                dawn=dawn,
                high_sun_spot=high_sun_spot,
                with_ionosphere=with_ionosphere,
                with_dish_effects=with_dish_effects,
                with_smearing=with_smearing,
                run_time=float(t1 - t0),
                thermal_noise=thermal_noise,
                baseline_noise=baseline_noise,
                result_values=result_values._asdict()
            )

            with open(os.path.join(save_folder, f'result_{result_num:03d}.json'), 'w') as f:
                f.write(result.json(indent=2))
