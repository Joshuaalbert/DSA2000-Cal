import logging
import os
from typing import List, NamedTuple

import jax
import numpy as np
from ray import serve

from dsa2000_cal.common.array_types import FloatArray, ComplexArray
from dsa2000_cal.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMeta
from dsa2000_cal.visibility_model.source_models.celestial.base_fits_source_model import BaseFITSSourceModel

logger = logging.getLogger('ray')


class ModelPredictorParams(SerialisableBaseModel):
    ms_meta: MeasurementSetMeta
    sky_models: List[BaseFITSSourceModel]
    near_field_delay_engine: BaseNearFieldDelayEngine
    far_field_delay_engine: BaseFarFieldDelayEngine
    geodesic_model: BaseGeodesicModel
    plot_folder: str


class ModelPredictorResponse(NamedTuple):
    vis: np.ndarray  # [D,  B, [, 2, 2]]


@serve.deployment
class ModelPredictor:
    def __init__(self, params: ForwardModellingRunParams):
        self.params = params
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'model_predictor')
        os.makedirs(self.params.plot_folder, exist_ok=True)


        if len(self.params.sky_models) == 0:
            raise ValueError("At least one sky model is required.")

        self.beam_model = build_beam_gain_model(
            array_name=self.params.ms_meta.array_name, times=self.params.ms_meta.times,
            ref_time=self.params.ms_meta.ref_time,
            freqs=self.params.ms_meta.freqs, full_stokes=self.params.sky_models[0].is_full_stokes()
        )
        # plot_beam_model(beam_model)
        self.beam_model.plot_regridded_beam(save_fig=os.path.join(self.params.plot_folder, 'regridded_beam.png'))

        def predict(
                freq: FloatArray,
                time: FloatArray,
                sky_models: List[BaseFITSSourceModel],
                gain_model: GainModel | None,
                near_field_delay_engine: BaseNearFieldDelayEngine,
                far_field_delay_engine: BaseFarFieldDelayEngine,
                geodesic_model: BaseGeodesicModel
        ) -> ComplexArray:
            # Compute visibility coordinates
            visibility_coords = far_field_delay_engine.compute_visibility_coords(
                freqs=freq[None],
                times=time[None],
                with_autocorr=self.params.ms_meta.with_autocorr,
                convention='physical'
            )
            # Predict visibilities
            vis_list = []
            for sky_model in sky_models:
                sky_vis = sky_model.predict(
                    visibility_coords=visibility_coords,
                    gain_model=gain_model,
                    near_field_delay_engine=near_field_delay_engine,
                    far_field_delay_engine=far_field_delay_engine,
                    geodesic_model=geodesic_model
                )  # [T=1, B, C=1, 2, 2]
                vis_list.append(sky_vis[0, :, 0])

            return sum(vis_list[1:], vis_list[0])

        self._predict_jit = jax.jit(predict)

    async def __call__(self, time_idx: int, freq_idx: int) -> ModelPredictorResponse:
        time = time_to_jnp(self.params.ms_meta.times[time_idx], self.params.ms_meta.ref_time)
        freq = quantity_to_jnp(self.params.ms_meta.freqs[freq_idx], 'Hz')
        vis = self._predict_jit(
            freq=freq,
            time=time,
            sky_models=self.params.sky_models,
            gain_model=self.beam_model,
            near_field_delay_engine=self.params.near_field_delay_engine,
            far_field_delay_engine=self.params.far_field_delay_engine,
            geodesic_model=self.params.geodesic_model
        )
        return ModelPredictorResponse(
            vis=vis
        )
