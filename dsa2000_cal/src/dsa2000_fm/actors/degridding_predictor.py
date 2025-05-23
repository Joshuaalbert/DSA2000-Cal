import asyncio
import logging
import os
from datetime import timedelta
from typing import NamedTuple

import jax
import numpy as np
import ray
from ray.runtime_env import RuntimeEnv

from dsa2000_common.common.ray_utils import TimerLog, resource_logger
from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.types import VisibilityCoords
from dsa2000_common.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_common.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_common.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_common.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_common.visibility_model.source_models.celestial.base_fits_source_model import BaseFITSSourceModel
from dsa2000_fm.actors.common import ForwardModellingRunParams

from dsa2000_common.common.logging import dsa_logger as logger


class DegriddingPredictorResponse(NamedTuple):
    vis: np.ndarray  # [B,[, 2, 2]]
    visibility_coords: VisibilityCoords


def compute_degridding_predictor_options(run_params: ForwardModellingRunParams):
    # Distributed over max 32 CPUs
    T = run_params.chunk_params.num_times_per_sol_int
    C = run_params.chunk_params.num_freqs_per_sol_int
    total_num_execs = T * C * (4 if run_params.full_stokes else 1)
    num_threads = min(32, total_num_execs)
    num_threads_inner = (4 if run_params.full_stokes else 1)
    num_threads_outer = max(1, num_threads // num_threads_inner)
    num_cpus = num_threads_inner * num_threads_outer
    # memory is 42GB
    memory = 42 * 1024 ** 3
    return {
        "num_cpus": num_cpus,
        "num_gpus": 0,
        'memory': memory,
        "runtime_env": RuntimeEnv(
            env_vars={
                #         "XLA_PYTHON_CLIENT_MEM_FRACTION": ".1",
                #         "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                #         "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",  # Slow but more memory efficient
                "JAX_PLATFORMS": "cpu"
            }
        )
    }


@ray.remote
class DegriddingPredictor:
    def __init__(self, params: ForwardModellingRunParams):
        self.params = params

        self.params.plot_folder = os.path.join(self.params.plot_folder, 'dft_predictor')
        os.makedirs(self.params.plot_folder, exist_ok=True)
        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

    async def init(self):
        if self._initialised:
            return
        self._initialised = True
        self._memory_logger_task = asyncio.create_task(
            resource_logger(task='degridding_predictor', cadence=timedelta(seconds=5)))

        def predict(
                source_model: BaseFITSSourceModel,
                freqs: FloatArray,
                times: FloatArray,
                gain_model: BaseSphericalInterpolatorGainModel | None,
                near_field_delay_engine: BaseNearFieldDelayEngine,
                far_field_delay_engine: BaseFarFieldDelayEngine,
                geodesic_model: BaseGeodesicModel
        ):
            visibility_coords = far_field_delay_engine.compute_visibility_coords(
                freqs=np.asarray(freqs),
                times=np.asarray(times),
                with_autocorr=self.params.ms_meta.with_autocorr,
                convention=self.params.ms_meta.convention
            )
            vis = source_model.predict_np(
                visibility_coords=visibility_coords,
                gain_model=gain_model,
                near_field_delay_engine=near_field_delay_engine,
                far_field_delay_engine=far_field_delay_engine,
                geodesic_model=geodesic_model
            )  # [T, B, C[, 2, 2]]
            return DegriddingPredictorResponse(
                vis=vis,
                visibility_coords=visibility_coords
            )

        self._predict = predict

    async def __call__(self,
                       source_model: BaseFITSSourceModel,
                       freqs: FloatArray,
                       times: FloatArray,
                       gain_model: BaseSphericalInterpolatorGainModel | None,
                       near_field_delay_engine: BaseNearFieldDelayEngine,
                       far_field_delay_engine: BaseFarFieldDelayEngine,
                       geodesic_model: BaseGeodesicModel
                       ) -> DegriddingPredictorResponse:
        await self.init()

        with TimerLog(f"Predicting and sampling visibilities for times {times} and freqs {freqs}"):
            response = self._predict(
                source_model=source_model,
                freqs=freqs,
                times=times,
                gain_model=gain_model,
                near_field_delay_engine=near_field_delay_engine,
                far_field_delay_engine=far_field_delay_engine,
                geodesic_model=geodesic_model
            )
            # Check for nans
            if np.any(np.isnan(response.vis)):
                raise ValueError(f"NaNs in the response visibilities.")

        return jax.tree.map(np.asarray, response)
