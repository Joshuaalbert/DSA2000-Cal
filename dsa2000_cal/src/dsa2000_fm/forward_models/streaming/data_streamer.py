import asyncio
import dataclasses
import logging
import os
from datetime import timedelta
from typing import NamedTuple, List, AsyncGenerator

import astropy.coordinates as ac
import astropy.units as au
import jax
import numpy as np
import ray
from jax import numpy as jnp

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import source_model_registry
from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.ray_utils import TimerLog, resource_logger
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_common.common.array_types import FloatArray, ComplexArray
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_common.common.types import VisibilityCoords
from dsa2000_common.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_common.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_common.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_common.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_common.visibility_model.source_models.celestial.base_fits_source_model import BaseFITSSourceModel, \
    build_fits_source_model_from_wsclean_components
from dsa2000_common.visibility_model.source_models.celestial.base_point_source_model import BasePointSourceModel, \
    build_point_source_model_from_wsclean_components
from dsa2000_fm.forward_models.streaming.common import ForwardModellingRunParams
from dsa2000_fm.forward_models.streaming.degridding_predictor import DegriddingPredictor
from dsa2000_fm.forward_models.streaming.dft_predictor import DFTPredictorResponse
from dsa2000_fm.forward_models.streaming.supervisor import Supervisor
from dsa2000_fm.forward_models.streaming.system_gain_simulator import SystemGainSimulatorResponse

logger = logging.getLogger('ray')


class DataStreamerParams(SerialisableBaseModel):
    sky_model_id: str
    bright_sky_model_id: str
    num_facets_per_side: int
    crop_box_size: au.Quantity | None
    near_field_delay_engine: BaseNearFieldDelayEngine
    far_field_delay_engine: BaseFarFieldDelayEngine
    geodesic_model: BaseGeodesicModel


class DataStreamerResponse(NamedTuple):
    vis: np.ndarray  # [B,[, 2, 2]]
    weights: np.ndarray  # [B, [, 2, 2]]
    flags: np.ndarray  # [B, [, 2, 2]]
    visibility_coords: VisibilityCoords


def compute_data_streamer_options(run_params: ForwardModellingRunParams):
    # memory is 2 * B * num_coh * (itemsize(vis) + itemsize(weights) + itemsize(flags))
    # num_coh = 4 if run_params.full_stokes else 1
    # B = run_params.chunk_params.num_baselines
    # itemsize_vis = np.dtype(np.complex64).itemsize
    # itemsize_weights = np.dtype(np.float16).itemsize
    # itemsize_flags = np.dtype(np.bool_).itemsize
    # memory = 2 * B * num_coh * (itemsize_vis + itemsize_weights + itemsize_flags)
    # memory is 17GB
    memory = 17 * 1024 ** 3
    return {
        "num_cpus": 0,
        "num_gpus": 0,
        'memory': 1.1 * memory
    }


@ray.remote
class DataStreamer:
    def __init__(self, params: ForwardModellingRunParams, predict_params: DataStreamerParams,
                 system_gain_simulator: Supervisor[SystemGainSimulatorResponse],
                 dft_predictor: Supervisor[DFTPredictorResponse],
                 degridding_predictor: Supervisor[DegriddingPredictor]
                 ):
        self.params = params
        self.predict_params = predict_params
        self._system_gain_simulator = system_gain_simulator
        self._dft_predictor = dft_predictor
        self._degridding_predictor = degridding_predictor

        self.params.plot_folder = os.path.join(self.params.plot_folder, 'data_streamer')
        os.makedirs(self.params.plot_folder, exist_ok=True)
        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

    async def init(self):
        if self._initialised:
            return
        self._initialised = True
        self._memory_logger_task = asyncio.create_task(
            resource_logger(task='data_streamer', cadence=timedelta(seconds=5)))

        predict_and_sample = PredictAndSample(
            faint_sky_model_id=self.predict_params.sky_model_id,
            bright_sky_model_id=self.predict_params.bright_sky_model_id,
            freqs=self.params.ms_meta.freqs,
            full_stokes=self.params.full_stokes,
            crop_box_size=self.predict_params.crop_box_size,
            num_facets_per_side=self.predict_params.num_facets_per_side,
            system_equivalent_flux_density=self.params.ms_meta.system_equivalent_flux_density,
            channel_width=self.params.ms_meta.channel_width,
            integration_time=self.params.ms_meta.integration_time,
            with_autocorr=self.params.ms_meta.with_autocorr,
            convention=self.params.ms_meta.convention,
            pointing=self.params.ms_meta.phase_center
        )
        self.state = predict_and_sample.get_state()
        # self._step_jit = jax.jit(predict_and_sample.step)

    async def __call__(self, key, sol_int_time_idxs: List[int], sol_int_freq_idxs: List[int]) -> AsyncGenerator[
        DataStreamerResponse, None]:

        await self.init()

        async def get_response(key, sol_int_time_idx: int, sol_int_freq_idx: int):
            # Get the time and freq indices for the solution interval (a regular grid)
            time_idxs = sol_int_time_idx * self.params.chunk_params.num_times_per_sol_int + np.arange(
                self.params.chunk_params.num_times_per_sol_int)
            freq_idxs = sol_int_freq_idx * self.params.chunk_params.num_freqs_per_sol_int + np.arange(
                self.params.chunk_params.num_freqs_per_sol_int)
            logger.info(f"Sampling visibilities for time_idxs={time_idxs} and freq_idxs={freq_idxs}")

            logger.info(f"Time indices: {time_idxs}")
            logger.info(f"Freq indices: {freq_idxs}")

            noise_key, sim_gain_key = jax.random.split(key)
            with TimerLog("Getting system gains"):
                system_gain_simulator_response: SystemGainSimulatorResponse = await self._system_gain_simulator(
                    sim_gain_key,
                    time_idxs,
                    freq_idxs
                )
            times = time_to_jnp(self.params.ms_meta.times[time_idxs], self.params.ms_meta.ref_time)
            freqs = quantity_to_jnp(self.params.ms_meta.freqs[freq_idxs], 'Hz')
            with TimerLog("Predicting and sampling visibilities"):
                tasks = []
                tasks.append(
                    self._degridding_predictor(
                        source_model=self.state.sky_model,
                        freqs=freqs,
                        times=times,
                        gain_model=system_gain_simulator_response.gain_model,
                        near_field_delay_engine=self.predict_params.near_field_delay_engine,
                        far_field_delay_engine=self.predict_params.far_field_delay_engine,
                        geodesic_model=self.predict_params.geodesic_model
                    )
                )
                tasks.append(
                    self._dft_predictor(
                        source_model=self.state.bright_sky_model,
                        freqs=freqs,
                        times=times,
                        gain_model=system_gain_simulator_response.gain_model,
                        near_field_delay_engine=self.predict_params.near_field_delay_engine,
                        far_field_delay_engine=self.predict_params.far_field_delay_engine,
                        geodesic_model=self.predict_params.geodesic_model
                    )
                )
                degridder_response, dft_response = await asyncio.gather(*tasks)
                vis = dft_response.vis + degridder_response.vis
                visibility_coords = dft_response.visibility_coords  # [T, B, ...]

                with TimerLog("...adding noise"):
                    # Add noise
                    vis, weights, flags = add_noise(
                        noise_key,
                        vis,
                        self.params.full_stokes,
                        self.params.ms_meta.system_equivalent_flux_density,
                        self.params.ms_meta.channel_width,
                        self.params.ms_meta.integration_time
                    )

            # Predict then send
            return DataStreamerResponse(
                vis=vis,
                weights=weights,
                flags=flags,
                visibility_coords=visibility_coords
            )

        for sol_int_time_idx, sol_int_freq_idx in zip(sol_int_time_idxs, sol_int_freq_idxs):
            key, run_key = jax.random.split(key, 2)
            response = await get_response(run_key, sol_int_time_idx, sol_int_freq_idx)
            yield response


def add_noise(key, vis: ComplexArray, full_stokes: bool, system_equivalent_flux_density: au.Quantity,
              channel_width: au.Quantity, integration_time: au.Quantity):
    # Add noise
    num_pol = 2 if full_stokes else 1
    noise_scale = calc_baseline_noise(
        system_equivalent_flux_density=quantity_to_jnp(system_equivalent_flux_density,
                                                       'Jy'),
        chan_width_hz=quantity_to_jnp(channel_width, 'Hz'),
        t_int_s=quantity_to_jnp(integration_time, 's')
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

    return jax.tree.map(np.asarray, (vis, weights, flags))


class PredictAndSampleState(NamedTuple):
    sky_model: BaseFITSSourceModel
    bright_sky_model: BasePointSourceModel


@dataclasses.dataclass(eq=False)
class PredictAndSample:
    faint_sky_model_id: str
    bright_sky_model_id: str
    freqs: au.Quantity
    full_stokes: bool
    crop_box_size: au.Quantity | None
    num_facets_per_side: int
    system_equivalent_flux_density: au.Quantity
    channel_width: au.Quantity
    integration_time: au.Quantity
    with_autocorr: bool
    pointing: ac.ICRS

    convention: str = 'physical'

    def get_state(self) -> PredictAndSampleState:
        fill_registries()

        model_freqs = au.Quantity([self.freqs[0], self.freqs[-1]])
        wsclean_fits_files = source_model_registry.get_instance(
            source_model_registry.get_match(self.faint_sky_model_id)).get_wsclean_fits_files()
        # -04:00:28.608,40.43.33.595

        faint_sky_model = build_fits_source_model_from_wsclean_components(
            wsclean_fits_files=wsclean_fits_files,
            model_freqs=model_freqs,
            full_stokes=self.full_stokes,
            crop_box_size=self.crop_box_size,
            num_facets_per_side=self.num_facets_per_side,
            repoint_centre=self.pointing
        )

        wsclean_clean_component_file = source_model_registry.get_instance(
            source_model_registry.get_match(self.bright_sky_model_id)).get_wsclean_clean_component_file()

        bright_sky_model_points = build_point_source_model_from_wsclean_components(
            wsclean_clean_component_file=wsclean_clean_component_file,
            model_freqs=model_freqs,
            full_stokes=self.full_stokes
        )

        # bright_sky_model_gaussians = build_gaussian_source_model_from_wsclean_components(
        #     wsclean_clean_component_file=wsclean_clean_component_file,
        #     model_freqs=model_freqs,
        #     full_stokes=self.full_stokes
        # )
        return PredictAndSampleState(
            sky_model=faint_sky_model,
            bright_sky_model=bright_sky_model_points
        )

    def step(self, key,
             freq: FloatArray,
             time: FloatArray,
             gain_model: BaseSphericalInterpolatorGainModel | None,
             near_field_delay_engine: BaseNearFieldDelayEngine,
             far_field_delay_engine: BaseFarFieldDelayEngine,
             geodesic_model: BaseGeodesicModel,
             state: PredictAndSampleState
             ):
        # Compute visibility coordinates
        visibility_coords = far_field_delay_engine.compute_visibility_coords(
            freqs=freq[None],
            times=time[None],
            with_autocorr=self.with_autocorr,
            convention=self.convention
        )
        # Predict visibilities
        sky_vis = state.sky_model.predict(
            visibility_coords=visibility_coords,
            gain_model=gain_model,
            near_field_delay_engine=near_field_delay_engine,
            far_field_delay_engine=far_field_delay_engine,
            geodesic_model=geodesic_model
        )
        bright_vis = state.bright_sky_model.predict(
            visibility_coords=visibility_coords,
            gain_model=gain_model,
            near_field_delay_engine=near_field_delay_engine,
            far_field_delay_engine=far_field_delay_engine,
            geodesic_model=geodesic_model
        )
        vis = sky_vis + bright_vis  # [T=1, B, C=1, 2, 2]
        vis = vis[0, :, 0]  # [B, 2, 2]
        # Add noise
        num_pol = 2 if self.full_stokes else 1
        noise_scale = calc_baseline_noise(
            system_equivalent_flux_density=quantity_to_jnp(self.system_equivalent_flux_density,
                                                           'Jy'),
            chan_width_hz=quantity_to_jnp(self.channel_width, 'Hz'),
            t_int_s=quantity_to_jnp(self.integration_time, 's')
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
