import dataclasses
import logging
import os
from typing import List, NamedTuple

import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import ray

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.common.array_types import FloatArray
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.ray_utils import TimerLog
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_fm.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.visibility_model.source_models.celestial.base_fits_source_model import BaseFITSSourceModel, \
    build_fits_calibration_source_model_from_wsclean_components

logger = logging.getLogger('ray')


class ModelPredictorParams(SerialisableBaseModel):
    sky_model_id: str
    num_facets_per_side: int
    crop_box_size: au.Quantity | None
    near_field_delay_engine: BaseNearFieldDelayEngine
    far_field_delay_engine: BaseFarFieldDelayEngine
    geodesic_model: BaseGeodesicModel


class ModelPredictorResponse(NamedTuple):
    vis: np.ndarray  # [D,  B, [, 2, 2]]


def compute_model_predictor_options(run_params: ForwardModellingRunParams):
    # memory is 2 * D * B * num_coh * itemsize(vis)
    # 2 is for buffer in predict
    num_coh = 4 if run_params.full_stokes else 1
    D = run_params.num_cal_facets
    B = run_params.chunk_params.num_baselines
    itemsize_vis = np.dtype(np.complex64).itemsize
    memory = 2 * D * B * num_coh * itemsize_vis
    return {
        "num_cpus": 1,
        "num_gpus": 0,
        'memory': 1.1 * memory
    }


@ray.remote
class ModelPredictor:
    def __init__(self, params: ForwardModellingRunParams, predict_params: ModelPredictorParams):
        self.params = params
        self.predict_params = predict_params
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'model_predictor')
        os.makedirs(self.params.plot_folder, exist_ok=True)
        self._initialised = False

    def init(self):
        if self._initialised:
            return
        self._initialised = True
        if self.predict_params.num_facets_per_side == 0:
            raise ValueError("At least one sky model is required.")

        self.beam_model = build_beam_gain_model(
            array_name=self.params.ms_meta.array_name,
            times=self.params.ms_meta.times,
            ref_time=self.params.ms_meta.ref_time,
            freqs=self.params.ms_meta.freqs,
            full_stokes=self.params.full_stokes
        )
        # plot_beam_model(beam_model)
        self.beam_model.plot_regridded_beam(save_fig=os.path.join(self.params.plot_folder, 'regridded_beam.png'))

        model_predict = ModelPredict(
            sky_model_id=self.predict_params.sky_model_id,
            freqs=self.params.ms_meta.freqs,
            full_stokes=self.params.full_stokes,
            crop_box_size=self.predict_params.crop_box_size,
            num_facets=self.predict_params.num_facets_per_side,
            system_equivalent_flux_density=self.params.ms_meta.system_equivalent_flux_density,
            channel_width=self.params.ms_meta.channel_width,
            integration_time=self.params.ms_meta.integration_time,
            with_autocorr=self.params.ms_meta.with_autocorr,
            convention=self.params.ms_meta.convention
        )
        self.state = model_predict.get_state()

        self._step_jit = jax.jit(model_predict.step)

    async def __call__(self, time: FloatArray, freq: FloatArray) -> ModelPredictorResponse:
        logger.info(f"Predicting visibilities for time {time} and freq {freq}")
        self.init()

        with TimerLog("Predicting..."):
            vis = block_until_ready(self._step_jit(
                freq=freq,
                time=time,
                gain_model=self.beam_model,
                near_field_delay_engine=self.predict_params.near_field_delay_engine,
                far_field_delay_engine=self.predict_params.far_field_delay_engine,
                geodesic_model=self.predict_params.geodesic_model,
                state=self.state
            ))
        vis = np.asarray(vis)
        return ModelPredictorResponse(
            vis=vis
        )


class ModelPredictState(NamedTuple):
    sky_models: List[BaseFITSSourceModel]


@dataclasses.dataclass(eq=False)
class ModelPredict:
    sky_model_id: str
    freqs: au.Quantity
    full_stokes: bool
    crop_box_size: au.Quantity | None
    num_facets: int
    system_equivalent_flux_density: au.Quantity
    channel_width: au.Quantity
    integration_time: au.Quantity
    with_autocorr: bool

    convention: str = 'physical'

    def get_state(self) -> ModelPredictState:
        fill_registries()

        model_freqs = au.Quantity([self.freqs[0], self.freqs[-1]])

        wsclean_fits_files = source_model_registry.get_instance(
            source_model_registry.get_match(self.sky_model_id)).get_wsclean_fits_files()
        # -04:00:28.608,40.43.33.595

        sky_models = build_fits_calibration_source_model_from_wsclean_components(
            wsclean_fits_files=wsclean_fits_files,
            model_freqs=model_freqs,
            full_stokes=self.full_stokes,
            crop_box_size=self.crop_box_size,
            num_facets=self.num_facets
        )

        return ModelPredictState(
            sky_models=sky_models,
        )

    def step(self,
             freq: FloatArray,
             time: FloatArray,
             gain_model: GainModel | None,
             near_field_delay_engine: BaseNearFieldDelayEngine,
             far_field_delay_engine: BaseFarFieldDelayEngine,
             geodesic_model: BaseGeodesicModel,
             state: ModelPredictState
             ):
        # Compute visibility coordinates
        visibility_coords = far_field_delay_engine.compute_visibility_coords(
            freqs=freq[None],
            times=time[None],
            with_autocorr=self.with_autocorr,
            convention=self.convention
        )
        vis_list = []
        for sky_model in state.sky_models:
            # Predict visibilities
            sky_vis = sky_model.predict(
                visibility_coords=visibility_coords,
                gain_model=gain_model,
                near_field_delay_engine=near_field_delay_engine,
                far_field_delay_engine=far_field_delay_engine,
                geodesic_model=geodesic_model
            )
            vis = sky_vis  # [T=1, B, C=1[, 2, 2]]
            vis = vis[0, :, 0]  # [B[, 2, 2]]
            vis_list.append(vis)
        vis = jnp.stack(vis_list, axis=0)  # [D, B[, 2, 2]]

        return vis
