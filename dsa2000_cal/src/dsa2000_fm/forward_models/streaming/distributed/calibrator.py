import asyncio
import dataclasses
import logging
import os
from datetime import timedelta
from functools import partial
from typing import NamedTuple, Tuple, List, AsyncGenerator

import jax
import jaxns.framework.context as ctx
import numpy as np
import pylab as plt
import ray
from jax import numpy as jnp
from jaxns.framework.ops import simulate_prior_model

from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import AbstractGainPriorModel, GainPriorModel
from dsa2000_cal.calibration.solvers.multi_step_lm import MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardt, \
    MultiStepLevenbergMarquardtDiagnostic
from dsa2000_cal.common.array_types import ComplexArray, FloatArray, BoolArray, IntArray
from dsa2000_cal.common.jax_utils import block_until_ready, simple_broadcast
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.ray_utils import TimerLog, resource_logger
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_fm.forward_models.streaming.distributed.calibration_solution_cache import CalibrationSolution, \
    CalibrationSolutionCache
from dsa2000_fm.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_fm.forward_models.streaming.distributed.data_streamer import DataStreamerResponse
from dsa2000_fm.forward_models.streaming.distributed.model_predictor import ModelPredictorResponse
from dsa2000_fm.forward_models.streaming.distributed.supervisor import Supervisor

logger = logging.getLogger('ray')


class CalibratorResponse(NamedTuple):
    visibilities: np.ndarray  # [Ts, B, Cs[, 2, 2]]
    weights: np.ndarray  # [Ts, B, Cs[, 2, 2]]
    flags: np.ndarray  # [Ts, B, Cs[, 2, 2]]
    uvw: np.ndarray  # [Ts, B, 3]


def average_rule(array, num_model_size: int, axis: int):
    """
    Block average array along axis.

    Args:
        array: [..., N, ...] on axis `axis`
        num_model_size: how many blocks to average
        axis: the axis

    Returns:
        [..., num_model_size, ...] on axis `axis`
    """
    axis_size = np.shape(array)[axis]
    if axis_size % num_model_size != 0:
        raise ValueError(f"Axis {axis} must be divisible by {num_model_size}.")
    block_size = axis_size // num_model_size
    return array.reshape(np.shape(array)[:axis] + (num_model_size, block_size) + np.shape(array)[axis + 1:]).mean(
        axis=axis + 1)


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
    # TODO: if we change execution pattern, e.g. to GPU or sharded over CPUs then should change.
    return {
        "num_cpus": 1,
        "num_gpus": 0,
        'memory': 1.1 * memory
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
                 data_streamer: Supervisor[AsyncGenerator[DataStreamerResponse]],
                 model_predictor: Supervisor[AsyncGenerator[ModelPredictorResponse]],
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

        calibration = Calibration(
            full_stokes=self.params.full_stokes,
            num_ant=len(self.params.ms_meta.antennas),
            gain_probabilistic_model=GainPriorModel(
                full_stokes=self.params.full_stokes,
                gain_stddev=2.,
                dd_dof=1,
                double_differential=True,
                di_dof=1,
                di_type='unconstrained',
                dd_type='unconstrained'
            )
        )

        self.calibrate_jit = jax.jit(calibration.step)

        self._compute_residual_jit = jax.jit(compute_residual)

    async def __call__(self, key, sol_int_time_idxs: List[int], sol_int_freq_idxs: List[int]) -> CalibratorResponse:
        logger.info(f"Calibrating and subtracting model visibilities for sol_int_time_idxs={sol_int_time_idxs} and "
                    f"sol_int_freq_idxs={sol_int_freq_idxs}")
        await self.init()

        data_response_gen = await self._data_streamer(key, sol_int_time_idxs, sol_int_freq_idxs)
        if self.calibrator_params.do_calibration:
            model_response_gen = await self._model_predictor(sol_int_time_idxs, sol_int_freq_idxs)

        idx = 0
        while True:
            try:
                if self.calibrator_params.do_calibration:
                    with TimerLog("Gathering data and model visibilities"):
                        data, model = await asyncio.gather(
                            data_response_gen.__anext__(),
                            model_response_gen.__anext__()  # noqa
                        )
                else:
                    with TimerLog("Gathering data visibilities and skipping calibration..."):
                        data = await data_response_gen.__anext__()
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

            vis_model = model.vis  # [D, Tm, B, Cm[, 2, 2]]
            background_vis_model = model.vis_background  # [E, Tm, B, Cm[, 2, 2]]
            model_freqs = model.model_freqs
            model_times = model.model_times

            # Response generator can be used in an `async for` block.
            with TimerLog("Getting previous state..."):
                previous_state = await self._calibration_solution_cache.get_calibration_solution_snapshot(
                    sol_int_time_idx, sol_int_freq_idx)

            with TimerLog("Averaging data and model visibilities"):
                # vis_model, vis_data, weights, flags, freqs, times, antenna1, antenna2, state
                # average data to match model: [Ts, B, Cs[, 2, 2]] -> [Tm, B, Cm[, 2, 2]]
                time_average_rule = partial(
                    average_rule,
                    num_model_size=self.params.chunk_params.num_model_times_per_solution_interval,
                    axis=0
                )
                freq_average_rule = partial(
                    average_rule,
                    num_model_size=self.params.chunk_params.num_model_freqs_per_solution_interval,
                    axis=2
                )

                vis_data_avg = time_average_rule(freq_average_rule(vis_data))
                weights_avg = np.reciprocal(time_average_rule(freq_average_rule(np.reciprocal(weights))))
                flags_avg = freq_average_rule(time_average_rule(flags.astype(np.float16))).astype(np.bool_)

            with TimerLog("Calibrating..."):
                # combine model and background model
                full_vis_model = np.concatenate([vis_model, background_vis_model], axis=0)
                gains, solver_state, diagnostics = block_until_ready(self.calibrate_jit(
                    vis_model=full_vis_model,
                    vis_data=vis_data_avg,
                    weights=weights_avg,
                    flags=flags_avg,
                    freqs=model_freqs,
                    times=model_times,
                    antenna1=antenna1,
                    antenna2=antenna2,
                    state=previous_state.solver_state
                ))
                diagnostics: MultiStepLevenbergMarquardtDiagnostic
                # row 1: Plot error
                # row 2: Plot r
                # row 3: plot chi-2 (F_norm)
                # row 4: plot damping

                fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
                axs[0].plot(diagnostics.iteration, diagnostics.error)
                axs[0].set_title('Error')
                axs[1].plot(diagnostics.iteration, diagnostics.r)
                axs[1].set_title('r')
                axs[2].plot(diagnostics.iteration, diagnostics.F_norm)
                axs[2].set_title('|F|')
                axs[3].plot(diagnostics.iteration, diagnostics.damping)
                axs[3].set_title('Damping')
                plt.savefig(os.path.join(self.params.plot_folder, f'calibration_diagnostics_{sol_int_time_idx}_{sol_int_freq_idx}.png'))
                plt.close(fig)

            with TimerLog("Computing residuals..."):
                # We don't subtract background, just the bright stuff
                # Trim gains to the ones to be subtracted
                num_subtract = np.shape(vis_model)[0]
                subtract_gains = gains[:num_subtract, ...]
                vis_residual = np.asarray(
                    self._compute_residual_jit(
                        vis_model=vis_model,
                        vis_data=vis_data,
                        gains=subtract_gains,
                        antenna1=antenna1,
                        antenna2=antenna2
                    )
                )

            with TimerLog("Storing solving state..."):
                await self._calibration_solution_cache.store_calibration_solution(
                    sol_int_time_idx=sol_int_time_idx,
                    sol_int_freq_idx=sol_int_freq_idx,
                    solution=CalibrationSolution(
                        solver_state=solver_state,
                        gains=np.asarray(gains),
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


def compute_residual(vis_model, vis_data, gains, antenna1, antenna2):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    Args:
        vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
        vis_data: [Ts, B, Cs[,2,2]] the data visibilities, Ts = 0 mod Tm, Cs = 0 mod Cm i.e. Ts % Tm = 0, Cs % Cm = 0
        gains: [D, Tm, A, Cm[,2,2]] the gains
        antenna1: [B] the antenna1
        antenna2: [B] the antenna2

    Returns:
        [Ts, B, Cs[, 2, 2]] the residuals
    """

    def body_fn(accumulate, x):
        vis_model, gains = x

        g1 = gains[:, antenna1, :, ...]  # [Tm, B, Cm[, 2, 2]]
        g2 = gains[:, antenna2, :, ...]  # [Tm, B, Cm[, 2, 2]]

        @partial(
            simple_broadcast,  # [Tm,B,Cm,...]
            leading_dims=3
        )
        def apply_gains(g1, g2, vis):
            if np.shape(g1) != np.shape(g1):
                raise ValueError("Gains must have the same shape.")
            if np.shape(vis) != np.shape(g1):
                raise ValueError("Gains and visibilities must have the same shape.")
            if np.shape(g1) == (2, 2):
                return mp_policy.cast_to_vis(kron_product(g1, vis, g2.conj().T))
            elif np.shape(g1) == ():
                return mp_policy.cast_to_vis(g1 * vis * g2.conj())
            else:
                raise ValueError(f"Invalid shape: {np.shape(g1)}")

        delta_vis = apply_gains(g1, g2, vis_model)  # [Tm, B, Cm[, 2, 2]]
        return accumulate + delta_vis, ()

    accumulate = jnp.zeros(np.shape(vis_model)[1:], dtype=vis_model.dtype)
    accumulate, _ = jax.lax.scan(body_fn, accumulate, (vis_model, gains))

    # Invert average rule with tile
    time_rep = np.shape(vis_data)[0] // np.shape(accumulate)[0]  # Ts / Tm
    freq_rep = np.shape(vis_data)[2] // np.shape(accumulate)[2]  # Cs / Cm
    tile_reps = [1] * len(np.shape(accumulate))
    tile_reps[0] = time_rep
    tile_reps[2] = freq_rep
    if np.prod(tile_reps) > 1:
        accumulate = jnp.tile(accumulate, tile_reps)
    return vis_data - accumulate


@dataclasses.dataclass(eq=False)
class Calibration:
    gain_probabilistic_model: AbstractGainPriorModel
    full_stokes: bool
    num_ant: int
    verbose: bool = False

    def step(self,
             vis_model: ComplexArray,
             vis_data: ComplexArray,
             weights: FloatArray,
             flags: BoolArray,
             freqs: FloatArray,
             times: FloatArray,
             antenna1: IntArray,
             antenna2: IntArray,
             state: MultiStepLevenbergMarquardtState | None = None
             ) -> Tuple[ComplexArray, MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardtDiagnostic]:
        """
        Calibrate and subtract model visibilities from data visibilities.

        Args:
            vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
            vis_data: [Tm, B, Cm[,2,2]] the data visibilities
            weights: [Tm, B, Cm[,2,2]] the weights
            flags: [Tm, B, Cm[,2,2]] the flags
            freqs: [Cm] the frequencies
            times: [Tm] the times
            antenna1: [B] the antenna1
            antenna2: [B] the antenna2
            state: MultiStepLevenbergMarquardtState the state of the solver (optional)

        Returns:
            gains: [D, Tm, A, Cm[,2,2]] the gains
            state: MultiStepLevenbergMarquardtState the state of the solver
            diagnostics: the diagnostics of the solver
        """
        if np.shape(vis_model)[1:] != np.shape(vis_data):
            raise ValueError(
                f"Model visibilities and data visibilities must have the same shape, got {np.shape(vis_model)[1:]} "
                f"and {np.shape(vis_data)}")

        # calibrate and subtract
        key = jax.random.PRNGKey(0)

        D, Tm, B, Cm = np.shape(vis_model)[:4]

        # Create gain prior model
        def get_gains():
            # TODO: pass in probabilitic model
            prior_model = self.gain_probabilistic_model.build_prior_model(
                num_source=D,
                num_ant=self.num_ant,
                freqs=freqs,
                times=times
            )
            (gains,), _ = simulate_prior_model(key, prior_model)  # [D, Tm, A, Cm[,2,2]]
            return gains  # [D, Tm, A, Cm[,2,2]]

        get_gains_transformed = ctx.transform(get_gains)

        compute_residuals_fn = self.build_compute_residuals_fn()

        # Create residual_fn
        def residual_fn(params: ComplexArray) -> ComplexArray:
            gains = get_gains_transformed.apply(params, key).fn_val
            return compute_residuals_fn(vis_model, vis_data, weights, flags, gains, antenna1, antenna2)

        solver = MultiStepLevenbergMarquardt(
            residual_fn=residual_fn,
            num_approx_steps=0,
            num_iterations=100,
            verbose=self.verbose,
            gtol=1e-6
        )

        # Get solver state
        if state is None:
            init_params = get_gains_transformed.init(key).params
            state = solver.create_initial_state(init_params)
        else:
            # TODO: EKF forward update on data
            state = solver.update_initial_state(state)
        state, diagnostics = solver.solve(state)

        gains = get_gains_transformed.apply(state.x, key).fn_val

        return gains, state, diagnostics

    def build_compute_residuals_fn(self):
        def compute_residuals_fn(
                vis_model: ComplexArray,
                vis_data: ComplexArray,
                weights: FloatArray,
                flags: BoolArray,
                gains: ComplexArray,
                antenna1: IntArray,
                antenna2: IntArray
        ):
            """
            Compute the residual between the model visibilities and the observed visibilities.

            Args:
                vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
                vis_data: [Ts, B, Cs[,2,2]] the data visibilities
                weights: [Ts, B, Cs[,2,2]] the data weights
                flags: [Ts, B, Cs[,2,2]] the data flags
                gains: [D, Tm, A, Cm[,2, 2]] the gains
                antenna1: [B] the antenna1
                antenna2: [B] the antenna2

            Returns:
                residuals: [Ts, B, Cs[,2,2]]
            """

            if np.shape(weights) != np.shape(flags):
                raise ValueError(
                    f"Weights and flags must have the same shape, got {np.shape(weights)} and {np.shape(flags)}")
            if np.shape(vis_data) != np.shape(weights):
                raise ValueError(
                    f"Visibilities and weights must have the same shape, got {np.shape(vis_data)} and {np.shape(weights)}")

            residuals = compute_residual(
                vis_model=vis_model,
                vis_data=vis_data,
                gains=gains,
                antenna1=antenna1,
                antenna2=antenna2
            )
            weights *= jnp.logical_not(flags).astype(weights.dtype)  # [Tm, B, Cm[,2,2]]
            residuals *= jnp.sqrt(weights)  # [Tm, B, Cm[,2,2]]
            return residuals.real, residuals.imag

        return compute_residuals_fn
