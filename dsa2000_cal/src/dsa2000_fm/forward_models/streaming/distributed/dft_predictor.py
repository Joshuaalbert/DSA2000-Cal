import asyncio
import logging
import os
from datetime import timedelta
from typing import NamedTuple

import jax
import numpy as np
import ray
from ray.runtime_env import RuntimeEnv

from dsa2000_cal.common.array_types import FloatArray
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.ray_utils import TimerLog, set_all_gpus_visible, get_gpu_with_most_memory, resource_logger
from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_cal.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.visibility_model.source_models.celestial.base_gaussian_source_model import \
    BaseGaussianSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.base_point_source_model import BasePointSourceModel
from dsa2000_fm.forward_models.streaming.distributed.common import ForwardModellingRunParams

logger = logging.getLogger('ray')


class DFTPredictorResponse(NamedTuple):
    vis: np.ndarray  # [B,[, 2, 2]]
    visibility_coords: VisibilityCoords


def compute_dft_predictor_options(run_params: ForwardModellingRunParams):
    # memory is 2 * B * num_coh * (itemsize(vis))
    num_coh = 4 if run_params.full_stokes else 1
    B = run_params.chunk_params.num_baselines
    itemsize_vis = np.dtype(np.complex64).itemsize
    memory = 2 * B * num_coh * (itemsize_vis)
    return {
        "num_cpus": 0,
        "num_gpus": 0.1,
        'memory': 1.1 * memory,
        "runtime_env": RuntimeEnv(
            env_vars={
                # "XLA_PYTHON_CLIENT_MEM_FRACTION": ".1",  # 10% of GPU memory
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",  # if false allocate on demand as much as needed
                # "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",  # Slow, but releases memory
                "JAX_PLATFORMS": "cuda"
            }
        )
    }


@ray.remote
class DFTPredictor:
    def __init__(self, params: ForwardModellingRunParams):
        self.params = params
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'dft_predictor')
        os.makedirs(self.params.plot_folder, exist_ok=True)
        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

        set_all_gpus_visible()

    async def init(self):
        if self._initialised:
            return
        self._initialised = True

        self._memory_logger_task = asyncio.create_task(
            resource_logger(task='dft_predictor', cadence=timedelta(seconds=5)))

        def predict(
                source_model: BasePointSourceModel | BaseGaussianSourceModel,
                freq: FloatArray,
                time: FloatArray,
                gain_model: BaseSphericalInterpolatorGainModel | None,
                near_field_delay_engine: BaseNearFieldDelayEngine,
                far_field_delay_engine: BaseFarFieldDelayEngine,
                geodesic_model: BaseGeodesicModel
        ):
            visibility_coords = far_field_delay_engine.compute_visibility_coords(
                freqs=freq[None],
                times=time[None],
                with_autocorr=self.params.ms_meta.with_autocorr,
                convention=self.params.ms_meta.convention
            )
            vis = source_model.predict(
                visibility_coords=visibility_coords,
                gain_model=gain_model,
                near_field_delay_engine=near_field_delay_engine,
                far_field_delay_engine=far_field_delay_engine,
                geodesic_model=geodesic_model
            )  # [T=1, B, C=1[, 2, 2]]
            vis = vis[0, :, 0, ...]  # [B, [, 2, 2]]
            return DFTPredictorResponse(
                vis=vis,
                visibility_coords=visibility_coords
            )

        self._predict_jit = jax.jit(predict)

    async def __call__(self,
                       source_model,
                       freq: FloatArray,
                       time: FloatArray,
                       gain_model: BaseSphericalInterpolatorGainModel | None,
                       near_field_delay_engine: BaseNearFieldDelayEngine,
                       far_field_delay_engine: BaseFarFieldDelayEngine,
                       geodesic_model: BaseGeodesicModel
                       ) -> DFTPredictorResponse:
        await self.init()

        with TimerLog(f"Predicting and sampling visibilities for time {time} and freq {freq}"):
            data_dict = dict(
                source_model=source_model,
                freq=freq,
                time=time,
                gain_model=gain_model,
                near_field_delay_engine=near_field_delay_engine,
                far_field_delay_engine=far_field_delay_engine,
                geodesic_model=geodesic_model
            )

            gpu_idx, memory = get_gpu_with_most_memory()
            logger.info(f"Using GPU {gpu_idx} with {memory / 2 ** 30} GB free memory")

            data_dict = jax.device_put(data_dict, jax.local_devices(gpu_idx, backend='gpu')[0], donate=True)

            response = block_until_ready(
                self._predict_jit(
                    **data_dict
                )
            )
        return jax.tree.map(np.asarray, response)
