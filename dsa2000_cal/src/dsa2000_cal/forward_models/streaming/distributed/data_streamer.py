import logging
import os
from typing import NamedTuple, Tuple

import jax
import numpy as np
from jax import numpy as jnp
from ray import serve
from ray.serve.handle import DeploymentHandle

from dsa2000_cal.common.array_types import FloatArray, ComplexArray, BoolArray
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.visibility_model.source_models.celestial.base_fits_source_model import BaseFITSSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.base_point_source_model import BasePointSourceModel

logger = logging.getLogger('ray')


class DataStreamerParams(SerialisableBaseModel):
    sky_model: BaseFITSSourceModel
    bright_sky_model: BasePointSourceModel
    near_field_delay_engine: BaseNearFieldDelayEngine
    far_field_delay_engine: BaseFarFieldDelayEngine
    geodesic_model: BaseGeodesicModel


class DataStreamerResponse(NamedTuple):
    vis: np.ndarray  # [B,[, 2, 2]]
    weights: np.ndarray  # [B, [, 2, 2]]
    flags: np.ndarray  # [B, [, 2, 2]]
    freq: np.ndarray  # [C]
    time: np.ndarray  # [T]


@serve.deployment
class DataStreamer:
    def __init__(self, params: ForwardModellingRunParams, predict_params: DataStreamerParams,
                 system_gain_simulator: DeploymentHandle):
        self.params = params
        self.predict_params = predict_params
        self._system_gain_simulator = system_gain_simulator

        self.params.plot_folder = os.path.join(self.params.plot_folder, 'data_streamer')
        os.makedirs(self.params.plot_folder, exist_ok=True)

        def predict(
                key,
                freq: FloatArray,
                time: FloatArray,
                sky_model: BaseFITSSourceModel,
                bright_sky_model: BasePointSourceModel,
                gain_model: GainModel | None,
                near_field_delay_engine: BaseNearFieldDelayEngine,
                far_field_delay_engine: BaseFarFieldDelayEngine,
                geodesic_model: BaseGeodesicModel
        ) -> Tuple[ComplexArray, FloatArray, BoolArray]:
            # Compute visibility coordinates
            visibility_coords = far_field_delay_engine.compute_visibility_coords(
                freqs=freq[None],
                times=time[None],
                with_autocorr=self.params.ms_meta.with_autocorr,
                convention='physical'
            )
            # Predict visibilities
            sky_vis = sky_model.predict(
                visibility_coords=visibility_coords,
                gain_model=gain_model,
                near_field_delay_engine=near_field_delay_engine,
                far_field_delay_engine=far_field_delay_engine,
                geodesic_model=geodesic_model
            )
            bright_vis = bright_sky_model.predict(
                visibility_coords=visibility_coords,
                gain_model=gain_model,
                near_field_delay_engine=near_field_delay_engine,
                far_field_delay_engine=far_field_delay_engine,
                geodesic_model=geodesic_model
            )
            vis = sky_vis + bright_vis  # [T=1, B, C=1, 2, 2]
            vis = vis[0, :, 0]
            # Add noise
            num_pol = 2 if self.predict_params.bright_sky_model.is_full_stokes() else 1
            noise_scale = calc_baseline_noise(
                system_equivalent_flux_density=quantity_to_jnp(self.params.ms_meta.system_equivalent_flux_density,
                                                               'Jy'),
                chan_width_hz=quantity_to_jnp(self.params.ms_meta.channel_width, 'Hz'),
                t_int_s=quantity_to_jnp(self.params.ms_meta.integration_time, 's')
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
            return vis, weights, flags

        self._predict_jit = jax.jit(predict)

    async def __call__(self, time_idx: int, freq_idx: int) -> DataStreamerResponse:
        system_gain_main = await self._system_gain_simulator.remote(time_idx, freq_idx)
        time = time_to_jnp(self.params.ms_meta.times[time_idx], self.params.ms_meta.ref_time)
        freq = quantity_to_jnp(self.params.ms_meta.freqs[freq_idx], 'Hz')
        key = jax.random.key(0)
        vis, weights, flags = self._predict_jit(
            key=key,
            freq=freq,
            time=time,
            sky_model=self.predict_params.sky_model,
            bright_sky_model=self.predict_params.bright_sky_model,
            gain_model=system_gain_main,
            near_field_delay_engine=self.predict_params.near_field_delay_engine,
            far_field_delay_engine=self.predict_params.far_field_delay_engine,
            geodesic_model=self.predict_params.geodesic_model
        )
        # Predict then send
        return DataStreamerResponse(
            vis=vis,
            weights=weights,
            flags=flags,
            freq=np.asarray(freq),
            time=np.asarray(time)
        )
