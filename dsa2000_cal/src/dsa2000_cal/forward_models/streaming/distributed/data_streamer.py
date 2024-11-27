import dataclasses
import logging
import os
from typing import NamedTuple

import astropy.units as au
import jax
import numpy as np
from jax import numpy as jnp
from ray import serve
from ray.serve.handle import DeploymentHandle

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.common.array_types import FloatArray
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_cal.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.visibility_model.source_models.celestial.base_fits_source_model import BaseFITSSourceModel, \
    build_fits_source_model_from_wsclean_components
from dsa2000_cal.visibility_model.source_models.celestial.base_point_source_model import BasePointSourceModel, \
    build_point_source_model_from_wsclean_components

logger = logging.getLogger('ray')


def test_bug():
    class A:
        def f(self, x):
            return x

    a = A()
    f_jit = jax.jit(a.f)
    print(f_jit(0))


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


@serve.deployment
class DataStreamer:
    def __init__(self, params: ForwardModellingRunParams, predict_params: DataStreamerParams,
                 system_gain_simulator: DeploymentHandle):
        self.params = params
        self.predict_params = predict_params
        self._system_gain_simulator = system_gain_simulator

        self.params.plot_folder = os.path.join(self.params.plot_folder, 'data_streamer')
        os.makedirs(self.params.plot_folder, exist_ok=True)

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
            convention=self.params.ms_meta.convention
        )
        self.state = predict_and_sample.get_state()
        self._step_jit = jax.jit(predict_and_sample.step)

    async def __call__(self, key, time_idx: int, freq_idx: int) -> DataStreamerResponse:
        logger.info(f"Sampling visibilities for time {time_idx} and freq {freq_idx}")
        gain_key, noise_key = jax.random.split(key)
        system_gain_main = await self._system_gain_simulator.remote(key, time_idx, freq_idx)
        time = time_to_jnp(self.params.ms_meta.times[time_idx], self.params.ms_meta.ref_time)
        freq = quantity_to_jnp(self.params.ms_meta.freqs[freq_idx], 'Hz')
        vis, weights, flags, visibility_coords = self._step_jit(
            key=noise_key,
            freq=freq,
            time=time,
            gain_model=system_gain_main,
            near_field_delay_engine=self.predict_params.near_field_delay_engine,
            far_field_delay_engine=self.predict_params.far_field_delay_engine,
            geodesic_model=self.predict_params.geodesic_model,
            state=self.state
        )
        vis = np.asarray(vis)
        weights = np.asarray(weights)
        flags = np.asarray(flags)
        visibility_coords = jax.tree.map(np.asarray, visibility_coords)
        # Predict then send
        return DataStreamerResponse(
            vis=vis,
            weights=weights,
            flags=flags,
            visibility_coords=visibility_coords
        )


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
            num_facets_per_side=self.num_facets_per_side
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
