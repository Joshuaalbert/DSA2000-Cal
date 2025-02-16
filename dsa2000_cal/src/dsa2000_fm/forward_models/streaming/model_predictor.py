import asyncio
import dataclasses
import logging
import os
from datetime import timedelta
from typing import List, NamedTuple, AsyncGenerator

import astropy.coordinates as ac
import astropy.units as au
import jax.numpy as jnp
import numpy as np
import ray

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import source_model_registry, array_registry
from dsa2000_cal.common.ray_utils import TimerLog, resource_logger
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_common.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_common.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_common.gain_models.gain_model import GainModel
from dsa2000_common.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_common.visibility_model.source_models.celestial.base_fits_source_model import BaseFITSSourceModel, \
    build_calibration_fits_source_models_from_wsclean
from dsa2000_common.visibility_model.source_models.celestial.base_point_source_model import \
    build_calibration_point_source_models_from_wsclean, BasePointSourceModel
from dsa2000_fm.antenna_model.antenna_model_utils import get_dish_model_beam_widths
from dsa2000_fm.forward_models.streaming.average_utils import average_rule
from dsa2000_fm.forward_models.streaming.common import ForwardModellingRunParams
from dsa2000_fm.forward_models.streaming.degridding_predictor import DegriddingPredictor, \
    DegriddingPredictorResponse
from dsa2000_fm.forward_models.streaming.dft_predictor import DFTPredictorResponse
from dsa2000_fm.forward_models.streaming.supervisor import Supervisor

logger = logging.getLogger('ray')


class ModelPredictorParams(SerialisableBaseModel):
    sky_model_id: str
    background_sky_model_id: str
    num_facets_per_side: int
    crop_box_size: au.Quantity | None
    near_field_delay_engine: BaseNearFieldDelayEngine
    far_field_delay_engine: BaseFarFieldDelayEngine
    geodesic_model: BaseGeodesicModel


class ModelPredictorResponse(NamedTuple):
    vis: np.ndarray  # [D,  T, B, C[, 2, 2]]
    vis_background: np.ndarray  # [E, T, B, C[, 2, 2]] the background visibilities, not subtracted
    model_times: np.ndarray  # [T]
    model_freqs: np.ndarray  # [C]


def compute_model_predictor_options(run_params: ForwardModellingRunParams):
    # memory is 10.2 GB
    memory = 10.2 * 1024 ** 3
    return {
        "num_cpus": 0,  # no comps, just memory
        "num_gpus": 0,
        'memory': 1.1 * memory
    }


@ray.remote
class ModelPredictor:
    def __init__(self,
                 params: ForwardModellingRunParams, predict_params: ModelPredictorParams,
                 dft_predictor: Supervisor[DFTPredictorResponse],
                 degridding_predictor: Supervisor[DegriddingPredictor]
                 ):
        self.params = params
        self.predict_params = predict_params
        self._dft_predictor = dft_predictor
        self._degridding_predictor = degridding_predictor
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'model_predictor')
        os.makedirs(self.params.plot_folder, exist_ok=True)
        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

    async def init(self):
        if self._initialised:
            return
        self._initialised = True
        self._memory_logger_task = asyncio.create_task(
            resource_logger(task='model_predictor', cadence=timedelta(seconds=5)))

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

        array = array_registry.get_instance(array_registry.get_match(self.params.ms_meta.array_name))

        _, beam_widths = get_dish_model_beam_widths(array.get_antenna_model())
        fov_fwhm = au.Quantity(np.mean(beam_widths))
        model_predict = ModelPredict(
            sky_model_id=self.predict_params.sky_model_id,
            background_sky_model_id=self.predict_params.background_sky_model_id,
            pointing=self.params.ms_meta.phase_center,
            freqs=self.params.ms_meta.freqs,
            full_stokes=self.params.full_stokes,
            crop_box_size=self.predict_params.crop_box_size,
            fov_fwhm=fov_fwhm,
            num_facets=self.predict_params.num_facets_per_side,
            system_equivalent_flux_density=self.params.ms_meta.system_equivalent_flux_density,
            channel_width=self.params.ms_meta.channel_width,
            integration_time=self.params.ms_meta.integration_time,
            with_autocorr=self.params.ms_meta.with_autocorr,
            convention=self.params.ms_meta.convention
        )
        self.state = model_predict.get_state()

        # self._step_jit = jax.jit(model_predict.step)

    async def __call__(self, sol_int_time_idxs: List[int], sol_int_freq_idxs: List[int]) -> AsyncGenerator[
        ModelPredictorResponse, None]:
        await self.init()

        async def get_response(sol_int_time_idx: int, sol_int_freq_idx: int):
            # Get the time and freq indices for the solution interval (a regular grid)
            time_idxs = sol_int_time_idx * self.params.chunk_params.num_times_per_sol_int + np.arange(
                self.params.chunk_params.num_times_per_sol_int)
            freq_idxs = sol_int_freq_idx * self.params.chunk_params.num_freqs_per_sol_int + np.arange(
                self.params.chunk_params.num_freqs_per_sol_int)

            # Get the model data at the averaged times and freqs
            model_times = time_to_jnp(self.params.ms_meta.times[time_idxs], self.params.ms_meta.ref_time)
            model_freqs = quantity_to_jnp(self.params.ms_meta.freqs[freq_idxs], 'Hz')
            model_times = average_rule(model_times, self.params.chunk_params.num_model_times_per_solution_interval,
                                       axis=0)
            model_freqs = average_rule(model_freqs, self.params.chunk_params.num_model_freqs_per_solution_interval,
                                       axis=0)

            logger.info(f"Predicting visibilities for model_times {model_times} and model_freqs {model_freqs}")
            with TimerLog("Predicting..."):
                # Calibrator component models
                tasks = []
                for source_model in self.state.sky_models:
                    tasks.append(
                        self._dft_predictor(
                            source_model=source_model,
                            freqs=model_freqs,
                            times=model_times,
                            gain_model=self.beam_model,
                            near_field_delay_engine=self.predict_params.near_field_delay_engine,
                            far_field_delay_engine=self.predict_params.far_field_delay_engine,
                            geodesic_model=self.predict_params.geodesic_model
                        )
                    )
                # Background models
                for source_model in self.state.background_sky_models:
                    tasks.append(
                        self._degridding_predictor(
                            source_model=source_model,
                            freqs=model_freqs,
                            times=model_times,
                            gain_model=self.beam_model,
                            near_field_delay_engine=self.predict_params.near_field_delay_engine,
                            far_field_delay_engine=self.predict_params.far_field_delay_engine,
                            geodesic_model=self.predict_params.geodesic_model
                        )
                    )
                results: List[DegriddingPredictorResponse | DFTPredictorResponse] = await asyncio.gather(*tasks)
                vis_list = []
                background_vis_list = []
                for _ in self.state.sky_models:
                    result: DFTPredictorResponse = results.pop(0)
                    vis_list.append(result.vis)  # each is [T,B,C[, 2, 2]]
                for _ in self.state.background_sky_models:
                    result: DegriddingPredictorResponse = results.pop(0)
                    background_vis_list.append(result.vis)  # each is [T,B,C[, 2, 2]]
                vis = np.stack(vis_list, axis=0)  # [D, T,B,C[, 2, 2]]
                background_vis = np.stack(background_vis_list, axis=0)  # [E, T,B,C[, 2, 2]]
            return ModelPredictorResponse(
                vis=vis,
                vis_background=background_vis,
                model_times=model_times,
                model_freqs=model_freqs
            )

        for sol_int_time_idx, sol_int_freq_idx in zip(sol_int_time_idxs, sol_int_freq_idxs):
            response = await get_response(sol_int_time_idx, sol_int_freq_idx)
            yield response


class ModelPredictState(NamedTuple):
    sky_models: List[BasePointSourceModel]
    background_sky_models: List[BaseFITSSourceModel]


@dataclasses.dataclass(eq=False)
class ModelPredict:
    sky_model_id: str
    background_sky_model_id: str
    pointing: ac.ICRS
    freqs: au.Quantity
    full_stokes: bool
    crop_box_size: au.Quantity | None
    num_facets: int
    fov_fwhm: au.Quantity
    system_equivalent_flux_density: au.Quantity
    channel_width: au.Quantity
    integration_time: au.Quantity
    with_autocorr: bool

    convention: str = 'physical'

    def get_state(self) -> ModelPredictState:
        fill_registries()

        model_freqs = au.Quantity([self.freqs[0], self.freqs[-1]])

        wsclean_component_file = source_model_registry.get_instance(
            source_model_registry.get_match(self.sky_model_id)).get_wsclean_clean_component_file()
        # -04:00:28.608,40.43.33.595

        # Treats each point source as a calibrator
        # TODO: for general combinations of components per calibrator, we'd need to generalise this.
        sky_models = build_calibration_point_source_models_from_wsclean(
            wsclean_component_file=wsclean_component_file,
            model_freqs=model_freqs,
            full_stokes=self.full_stokes,
            pointing=self.pointing,
            fov_fwhm=self.fov_fwhm
        )
        # Treats the backgrounds as FITS images
        background_sky_models = build_calibration_fits_source_models_from_wsclean(
            wsclean_fits_files=source_model_registry.get_instance(
                source_model_registry.get_match(self.background_sky_model_id)).get_wsclean_fits_files(),
            model_freqs=model_freqs,
            full_stokes=self.full_stokes,
            crop_box_size=self.crop_box_size,
            num_facets=self.num_facets,
            repoint_centre=self.pointing
        )

        return ModelPredictState(
            sky_models=sky_models,
            background_sky_models=background_sky_models
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
