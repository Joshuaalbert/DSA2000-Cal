import asyncio
import os
from datetime import timedelta
from typing import NamedTuple, List

import astropy.units as au
import numpy as np
import ray
from ray.runtime_env import RuntimeEnv

from dsa2000_cal.iterative_calibrator import IterativeCalibrator, Data
from dsa2000_common.common.logging import dsa_logger as logger
from dsa2000_common.common.quantity_utils import jnp_to_time
from dsa2000_common.common.ray_utils import TimerLog, resource_logger
from dsa2000_common.common.serialise_utils import SerialisableBaseModel
from dsa2000_fm.actors.calibration_solution_cache import CalibrationSolution, \
    CalibrationSolutionCache
from dsa2000_fm.actors.common import ForwardModellingRunParams
from dsa2000_fm.actors.data_streamer import DataStreamerResponse
from dsa2000_fm.actors.model_predictor import ModelPredictorResponse
from dsa2000_fm.actors.supervisor import Supervisor


class CalibratorResponse(NamedTuple):
    visibilities: np.ndarray  # [Ts, B, Cs[, 2, 2]]
    weights: np.ndarray  # [Ts, B, Cs[, 2, 2]]
    flags: np.ndarray  # [Ts, B, Cs[, 2, 2]]
    uvw: np.ndarray  # [Ts, B, 3]


def compute_calibrator_options(run_params: ForwardModellingRunParams):
    # # memory for inputs from stream:
    # # Ts * B * Cs * num_coh * (itemsize(vis) + itemsize(weights) + itemsize(flags))
    #
    # # memory for averaged data:
    # # Tm * B * Cm * num_coh * (itemsize(vis) + itemsize(weights) + itemsize(flags))
    #
    # # memory for model:
    # # D * Tm * B * Cm * num_coh * itemsize(vis)
    #
    # # memory for solution:
    # # D * Tm * A * Cm * num_coh * itemsize(gains)
    #
    # num_coh = 4 if run_params.full_stokes else 1
    # Ts = run_params.chunk_params.num_times_per_sol_int
    # B = run_params.chunk_params.num_baselines
    # Cs = run_params.chunk_params.num_freqs_per_sol_int
    # D = run_params.num_cal_facets
    # Tm = run_params.chunk_params.num_model_times_per_solution_interval
    # Cm = run_params.chunk_params.num_model_freqs_per_solution_interval
    # A = len(run_params.ms_meta.antennas)
    # itemsize_vis = np.dtype(np.complex64).itemsize
    # itemsize_weights = np.dtype(np.float16).itemsize
    # itemsize_flags = np.dtype(np.bool_).itemsize
    # itemsize_gains = np.dtype(np.complex64).itemsize
    # memory = Ts * B * Cs * num_coh * (itemsize_vis + itemsize_weights + itemsize_flags) + \
    #          Tm * B * Cm * num_coh * (itemsize_vis + itemsize_weights + itemsize_flags) + \
    #          D * Tm * B * Cm * num_coh * itemsize_vis + \
    #          D * Tm * A * Cm * num_coh * itemsize_gains
    # memory used is 11.5GB
    memory = 11.5 * 1024 ** 3
    # {
    #     "num_cpus": 1,
    #     "num_gpus": 0,
    #     'memory': 1.1 * memory
    # }
    # TODO: if we change execution pattern, e.g. to GPU or sharded over CPUs then should change.
    return {
        "num_cpus": 1,
        "num_gpus": 0.1,
        'memory': 1.1 * memory,
        "runtime_env": RuntimeEnv(
            env_vars={
                # "XLA_PYTHON_CLIENT_MEM_FRACTION": ".1",  # 10% of GPU memory
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",  # if false allocate on demand as much as needed
                # "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",  # Slow, but releases memory
                "JAX_PLATFORMS": "cuda,cpu"
            }
        )
    }


class CalibratorParams(SerialisableBaseModel):
    do_calibration: bool


@ray.remote
class Calibrator:
    """
    Calibrates and subtracts model visibilities from data vis and streams results, per sol_idx.
    A sol_int corresponds to a chunk of time and frequency data of size (num_times_per_sol_int, num_freqs_per_sol_int).
    The total dataset is (num_times_per_obs, num_freqs_per_obs) or in terms of sol_ints (num_times_per_obs//num_times_per_sol_int, num_freqs_per_obs//num_freqs_per_sol_int).
    """

    def __init__(self, params: ForwardModellingRunParams, calibrator_params: CalibratorParams,
                 data_streamer: Supervisor[DataStreamerResponse],
                 model_predictor: Supervisor[ModelPredictorResponse],
                 calibration_solution_cache: CalibrationSolutionCache):
        self.params = params
        self.calibrator_params = calibrator_params
        self._calibration_solution_cache = calibration_solution_cache
        self._data_streamer = data_streamer
        self._model_predictor = model_predictor
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'calibrator')
        os.makedirs(self.params.plot_folder, exist_ok=True)

        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

    async def init(self):
        if self._initialised:
            return
        self._initialised = True
        self._memory_logger_task = asyncio.create_task(resource_logger(task='calibrator', cadence=timedelta(seconds=5)))

        iterative_calibrator = IterativeCalibrator(
            plot_folder=self.params.plot_folder,
            run_name=self.params.run_name,
            gain_stddev=1.,
            dd_dof=1,
            di_dof=1,
            double_differential=True,
            dd_type='unconstrained',
            di_type='unconstrained',
            full_stokes=self.params.full_stokes,
            antennas=self.params.ms_meta.antennas,
            verbose=False
        )
        self._main_step = iterative_calibrator.build_main_step(
            Ts=self.params.chunk_params.num_times_per_sol_int,
            Cs=self.params.chunk_params.num_freqs_per_sol_int,
        )

    async def __call__(self, key, sol_int_time_idxs: List[int], sol_int_freq_idxs: List[int]) -> CalibratorResponse:
        logger.info(f"Calibrating and subtracting model visibilities for sol_int_time_idxs={sol_int_time_idxs} and "
                    f"sol_int_freq_idxs={sol_int_freq_idxs}")
        await self.init()

        data_response_gen = self._data_streamer.stream(key, sol_int_time_idxs, sol_int_freq_idxs)
        if self.calibrator_params.do_calibration:
            model_response_gen = self._model_predictor.stream(sol_int_time_idxs, sol_int_freq_idxs)

        idx = 0
        while True:
            try:
                if self.calibrator_params.do_calibration:
                    with TimerLog("Gathering data and model visibilities"):
                        data_ref, model_ref = await asyncio.gather(
                            data_response_gen.__anext__(),
                            model_response_gen.__anext__()  # noqa
                        )
                        data, model = await asyncio.gather(data_ref, model_ref)
                else:
                    with TimerLog("Gathering data visibilities and skipping calibration..."):
                        data_ref = await data_response_gen.__anext__()
                        data = await data_ref
            except StopAsyncIteration:
                break
            # Ts = len(time_idxs)
            # Cs = len(freq_idxs)
            # Tm = len(model_times)
            # Cm = len(model_freqs)
            sol_int_time_idx, sol_int_freq_idx = sol_int_time_idxs[idx], sol_int_freq_idxs[idx]
            idx += 1

            vis_data = data.vis  # [Ts, B, Cs[, 2, 2]]
            weights = data.weights  # [Ts, B, Cs[, 2, 2]]
            flags = data.flags  # [Ts, B, Cs[, 2, 2]]
            uvw = data.visibility_coords.uvw  # [Ts, B, 3]
            freqs = data.visibility_coords.freqs  # [Cs]
            times = data.visibility_coords.times  # [Ts]
            antenna1 = data.visibility_coords.antenna1  # [B]
            antenna2 = data.visibility_coords.antenna2  # [B]

            if not self.calibrator_params.do_calibration:
                yield CalibratorResponse(
                    visibilities=np.asarray(vis_data),
                    weights=np.asarray(weights),
                    flags=np.asarray(flags),
                    uvw=np.asarray(uvw)
                )
            else:

                with TimerLog("Preparing data for calibration"):

                    vis_model = model.vis  # [D, Tm, B, Cm[, 2, 2]]
                    background_vis_model = model.vis_background  # [E, Tm, B, Cm[, 2, 2]]
                    model_freqs = model.model_freqs
                    model_times = model.model_times

                    if self.params.full_stokes:
                        coherencies = (('XX', 'XY'), ('YX', 'YY'))
                        _vis_data = vis_data  # [Ts, B, Cs, 4]
                        _weights = weights  # [Ts, B, Cs, 4]
                        _flags = flags  # [Ts, B, Cs, 4]
                        _vis_model = vis_model  # [D, Tm, B, Cm, 4]
                        _background_vis_model = background_vis_model  # [E, Tm, B, Cm, 4]
                    else:
                        coherencies = ('I',)
                        # add coh dim
                        _vis_data = vis_data[..., None]  # [Ts, B, Cs, 1]
                        _weights = weights[..., None]  # [Ts, B, Cs, 1]
                        _flags = flags[..., None]  # [Ts, B, Cs, 1]
                        _vis_model = vis_model[..., None]  # [D, Tm, B, Cm, 1]
                        _background_vis_model = background_vis_model[..., None]  # [E, Tm, B, Cm, 1]

                    main_data = Data(
                        sol_int_time_idx=sol_int_time_idx,
                        coherencies=coherencies,
                        vis_data=_vis_data,
                        weights=_weights,
                        flags=_flags,
                        vis_bright_sources=_vis_model,
                        vis_background=_background_vis_model,
                        model_times=jnp_to_time(model_times, self.params.ms_meta.ref_time),
                        model_freqs=au.Quantity(np.asarray(model_freqs), 'Hz'),
                        ref_time=self.params.ms_meta.ref_time,
                        antenna1=antenna1,
                        antenna2=antenna2
                    )

                with TimerLog("Getting last params in cache"):
                    params = await self._calibration_solution_cache.get_calibration_solution_snapshot(
                        sol_int_time_idx=sol_int_time_idx,
                        sol_int_freq_idx=sol_int_freq_idx
                    )

                with TimerLog("Running calibration"):
                    return_data = self._main_step(main_data, params=params)
                    if self.params.full_stokes:
                        vis_residual = return_data.vis_residuals
                    else:
                        vis_residual = return_data.vis_residuals[..., 0]

                with TimerLog("Storing params in cache"):
                    await self._calibration_solution_cache.store_calibration_solution(
                        sol_int_time_idx=sol_int_time_idx,
                        sol_int_freq_idx=sol_int_freq_idx,
                        solution=CalibrationSolution(
                            params=return_data.params,
                            gains=np.asarray(return_data.gains),
                            model_freqs=np.asarray(model_freqs),
                            model_times=np.asarray(model_times)
                        )
                    )

                yield CalibratorResponse(
                    visibilities=np.asarray(vis_residual),
                    weights=np.asarray(weights),
                    flags=np.asarray(flags),
                    uvw=np.asarray(uvw)
                )
