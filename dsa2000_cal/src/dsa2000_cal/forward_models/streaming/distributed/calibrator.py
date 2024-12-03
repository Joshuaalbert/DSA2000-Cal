import asyncio
import dataclasses
import logging
import os
from functools import partial
from typing import NamedTuple, Type, Tuple, Dict, List

import jax
import jaxns.framework.context as ctx
import numpy as np
import ray
from jax import numpy as jnp
from jaxns.framework.ops import simulate_prior_model
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_cal.actors.namespace import NAMESPACE
from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardt, \
    MultiStepLevenbergMarquardtDiagnostic
from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import AbstractGainPriorModel, UnconstrainedGain
from dsa2000_cal.common.array_types import ComplexArray, FloatArray, BoolArray, IntArray
from dsa2000_cal.common.jax_utils import multi_vmap, block_until_ready
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_cal.common.ray_utils import get_head_node_id, TimerLog
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_cal.forward_models.streaming.distributed.data_streamer import DataStreamerResponse
from dsa2000_cal.forward_models.streaming.distributed.model_predictor import ModelPredictorResponse
from dsa2000_cal.forward_models.streaming.distributed.supervisor import Supervisor

logger = logging.getLogger('ray')


class CalibrationSolution(NamedTuple):
    solver_state: MultiStepLevenbergMarquardtState | None


class CalibrationSolutionCacheParams(SerialisableBaseModel):
    ...


def compuate_calibration_solution_cache_options(run_params: ForwardModellingRunParams):
    # memory os Tm * A * Cm * num_coh * itemsize(gains)
    num_coh = 4 if run_params.full_stokes else 1
    Tm = run_params.chunk_params.num_model_times_per_solution_interval
    Cm = run_params.chunk_params.num_model_freqs_per_solution_interval
    A = len(run_params.ms_meta.antennas)
    itemsize_gains = np.dtype(np.complex64).itemsize
    memory = Tm * A * Cm * num_coh * itemsize_gains
    return {
        'memory': 1.1 * memory
    }


class CalibrationSolutionCache:

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        return (self._deserialise, (self._serialised_data,))

    @classmethod
    def _deserialise(cls, kwargs):
        # Create a new instance, bypassing __init__ and setting the actor directly
        return cls(**kwargs)

    def __init__(self, params: CalibrationSolutionCacheParams | None = None, memory: int = 3 * 1024 ** 3):

        self._serialised_data = dict(
            params=None
        )
        actor_name = self.actor_name()

        try:
            actor = ray.get_actor(actor_name, namespace=NAMESPACE)
            logger.info(f"Connected to existing {actor_name}")
        except ValueError:
            if params is None:
                raise ValueError(f"Actor {actor_name} does not exist, and params is None")
            try:
                placement_node_id = get_head_node_id()
            except AssertionError as e:
                if "Cannot find alive head node." in str(e):
                    placement_node_id = ray.get_runtime_context().get_node_id()
                else:
                    raise e
            actor_options = {
                "num_cpus": 0,
                "num_gpus": 0,
                "memory": memory,  #
                "name": actor_name,
                "lifetime": "detached",
                "max_restarts": -1,
                "max_task_retries": -1,
                # Schedule the controller on the head node with a soft constraint. This
                # prefers it to run on the head node in most cases, but allows it to be
                # restarted on other nodes in an HA cluster.
                "scheduling_strategy": NodeAffinitySchedulingStrategy(placement_node_id, soft=True),
                "namespace": NAMESPACE,
                "max_concurrency": 15000  # Needs to be large, as there should be no limit.
            }

            dynamic_cls = self.dynamic_cls()

            actor_kwargs = dict(
                params=params
            )

            actor = ray.remote(dynamic_cls).options(**actor_options).remote(**actor_kwargs)
            ray.get(actor.health_check.remote())

        self._actor = actor

    @staticmethod
    def dynamic_cls() -> Type:
        """
        Create a dynamic class that will be parsed properly by ray dashboard, so that it has a nice class name.

        Returns:
            a dynamic class
        """
        # a dynamic class that will be parsed properly by ray dashboard, so that it has a nice class name.
        return type(
            f"CalibrationSolutionCache",
            (_CalibrationSolutionCache,),
            dict(_CalibrationSolutionCache.__dict__),
        )

    @staticmethod
    def actor_name() -> str:
        return "CALIBRATION_SOLUTION_CACHE"

    def store_calibration_solution(self, sol_int_time_idx: int, sol_int_freq_idx: int,
                                   solution: CalibrationSolution):
        """
        Store a calibration solution for a given frequency chunk.

        Args:
            sol_int_time_idx: The time index of the solution interval
            sol_int_freq_idx: The frequency index of the solution interval
            solution: The calibration solution
        """
        ray.get(self._actor.store_calibration_solution.remote(sol_int_time_idx, sol_int_freq_idx,
                                                              jax.tree.map(np.asarray, solution)))

    def get_calibration_solution_snapshot(self, sol_int_time_idx: int,
                                          sol_int_freq_idx: int) -> CalibrationSolution:
        """
        Get a snapshot of the calibration solution for a given frequency chunk.

        Args:
            sol_int_time_idx: The time index of the solution interval
            sol_int_freq_idx: The frequency index of the solution interval

        Returns:
            The calibration solution.
        """
        return jax.tree.map(jnp.asarray, ray.get(
            self._actor.get_calibration_solution_snapshot.remote(sol_int_time_idx, sol_int_freq_idx)))


class _CalibrationSolutionCache:
    """
    A cache for storing calibration, per frequency chunk.
    """

    def __init__(self, params: CalibrationSolutionCacheParams):
        self.params = params
        self.cache: Dict[
            Tuple[int, int], CalibrationSolution] = {}  # (sol_int_time_idx, sol_int_freq_idx) -> CalibrationSolution

    def health_check(self):
        """
        Announce health check.
        """
        logger.info(f"Healthy {self.__class__.__name__}")
        return

    def store_calibration_solution(self, sol_int_time_idx: int, sol_int_freq_idx: int,
                                   solution: CalibrationSolution):
        self.cache[(sol_int_time_idx, sol_int_freq_idx)] = solution

    def get_calibration_solution_snapshot(self, sol_int_time_idx: int,
                                          sol_int_freq_idx: int) -> CalibrationSolution:
        return self.cache.get((sol_int_time_idx - 1, sol_int_freq_idx), CalibrationSolution(solver_state=None))


def compute_residual(vis_model, vis_data, gains, antenna1, antenna2):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    Args:
        vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
        vis_data: [Ts, B, Cs[,2,2]] the data visibilities
        gains: [D, Tm, A, Cm[, 2, 2]] the gains
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
            multi_vmap,
            in_mapping="[Tm,B,Cm,...],[Tm,B,Cm,...],[Tm,B,Cm,...]",
            out_mapping="[T,B,F,...]",
            verbose=True
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
    time_rep = np.shape(vis_data)[0] // np.shape(accumulate)[0]
    freq_rep = np.shape(vis_data)[2] // np.shape(accumulate)[2]
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
    num_background_source_models: int = 0
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
            vis_model: [D, Tm, B, Cm, 2, 2] the model visibilities per direction
            vis_data: [Tm, B, Cm, 2, 2] the data visibilities
            weights: [Tm, B, Cm, 2, 2] the weights
            flags: [Tm, B, Cm, 2, 2] the flags
            freqs: [Cm] the frequencies
            times: [Tm] the times
            antenna1: [B] the antenna1
            antenna2: [B] the antenna2
            state: MultiStepLevenbergMarquardtState the state of the solver (optional)

        Returns:
            gains: [D, Tm, A, Cm, 2, 2] the gains
            state: MultiStepLevenbergMarquardtState the state of the solver
            diagnostics: the diagnostics of the solver
        """
        if np.shape(vis_model)[1:] != np.shape(vis_data):
            raise ValueError(
                f"Model visibilities and data visibilities must have the same shape, got {np.shape(vis_model)} "
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
            (gains,), _ = simulate_prior_model(key, prior_model)  # [D, Tm, A, Cm, 2, 2]
            return gains  # [D, Tm, A, Cm, 2, 2]

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
                gains: [D, Tm, A, Cm[, 2, 2]] the gains
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
            weights *= jnp.logical_not(flags).astype(weights.dtype)  # [Tm, B, Cm, 2, 2]
            residuals *= jnp.sqrt(weights)  # [Tm, B, Cm, 2, 2]
            return residuals.real, residuals.imag

        return compute_residuals_fn


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


def test_average_rule():
    array = np.arange(9)
    num_model_times = 3
    axis = 0
    result = average_rule(array, num_model_times, axis)
    assert np.allclose(result, np.array([1., 4., 7.]))

    # Tile the array
    n = 5
    array = np.tile(array[None, :], (n, 1))
    result = average_rule(array, num_model_times, 1)
    assert np.allclose(result, np.array([1., 4, 7])[None, :])

    array = array.T
    result = average_rule(array, num_model_times, 0)
    assert np.allclose(result, np.array([1., 4, 7])[:, None])


def compute_calibrator_options(run_params: ForwardModellingRunParams):
    # memory for inputs from stream:
    # Ts * B * Cs * num_coh * (itemsize(vis) + itemsize(weights) + itemsize(flags))

    # memory for averaged data:
    # Tm * B * Cm * num_coh * (itemsize(vis) + itemsize(weights) + itemsize(flags))

    # memory for model:
    # D * Tm * B * Cm * num_coh * itemsize(vis)

    # memory for solution:
    # D * Tm * A * Cm * num_coh * itemsize(gains)

    num_coh = 4 if run_params.full_stokes else 1
    Ts = run_params.chunk_params.num_times_per_sol_int
    B = run_params.chunk_params.num_baselines
    Cs = run_params.chunk_params.num_freqs_per_sol_int
    D = run_params.num_cal_facets
    Tm = 1
    Cm = 1
    A = len(run_params.ms_meta.antennas)
    itemsize_vis = np.dtype(np.complex64).itemsize
    itemsize_weights = np.dtype(np.float16).itemsize
    itemsize_flags = np.dtype(np.bool_).itemsize
    itemsize_gains = np.dtype(np.complex64).itemsize
    memory = Ts * B * Cs * num_coh * (itemsize_vis + itemsize_weights + itemsize_flags) + \
             Tm * B * Cm * num_coh * (itemsize_vis + itemsize_weights + itemsize_flags) + \
             D * Tm * B * Cm * num_coh * itemsize_vis + \
             D * Tm * A * Cm * num_coh * itemsize_gains
    # TODO: flip to GPU once we standardise the GPU's per device. This would need different resources for varying GPU's
    return {
        "num_cpus": 1,
        "num_gpus": 0,
        'memory': 1.1 * memory
    }


@ray.remote
class Calibrator:
    """
    Calibrates and subtracts model visibilities from data vis and streams results, per sol_idx.
    A sol_int corresponds to a chunk of time and frequency data of size (num_times_per_sol_int, num_freqs_per_sol_int).
    The total dataset is (num_times_per_obs, num_freqs_per_obs) or in terms of sol_ints (num_times_per_obs//num_times_per_sol_int, num_freqs_per_obs//num_freqs_per_sol_int).
    """

    def __init__(self, params: ForwardModellingRunParams, data_streamer: Supervisor[DataStreamerResponse],
                 model_predictor: Supervisor[ModelPredictorResponse],
                 calibration_solution_cache: CalibrationSolutionCache):
        self.params = params
        self._calibration_solution_cache = calibration_solution_cache
        self._data_streamer = data_streamer
        self._model_predictor = model_predictor
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'calibrator')
        os.makedirs(self.params.plot_folder, exist_ok=True)

        calibration = Calibration(
            full_stokes=self.params.full_stokes,
            num_ant=len(self.params.ms_meta.antennas),
            gain_probabilistic_model=UnconstrainedGain(
                full_stokes=self.params.full_stokes,
                gain_stddev=2.,
                dof=1
            )
        )

        self.calibrate_jit = jax.jit(calibration.step)

        self._compute_residual_jit = jax.jit(compute_residual)

    async def __call__(self, key, sol_int_time_idx: int, sol_int_freq_idx: int) -> CalibratorResponse:
        logger.info(f"Calibrating and subtracting model visibilities for sol_int_time_idx={sol_int_time_idx} and "
                    f"sol_int_freq_idx={sol_int_freq_idx}")

        time_idxs = sol_int_time_idx * self.params.chunk_params.num_times_per_sol_int + np.arange(
            self.params.chunk_params.num_times_per_sol_int)
        freq_idxs = sol_int_freq_idx * self.params.chunk_params.num_freqs_per_sol_int + np.arange(
            self.params.chunk_params.num_freqs_per_sol_int)
        logger.info(f"Time indices: {time_idxs}")
        logger.info(f"Freq indices: {freq_idxs}")

        times = time_to_jnp(self.params.ms_meta.times[time_idxs], self.params.ms_meta.ref_time)
        freqs = quantity_to_jnp(self.params.ms_meta.freqs[freq_idxs], 'Hz')
        model_times = average_rule(times, self.params.chunk_params.num_model_times_per_solution_interval, axis=0)
        model_freqs = average_rule(freqs, self.params.chunk_params.num_model_freqs_per_solution_interval, axis=0)

        async def gather_data(key, time_idxs, freq_idxs):
            Ts = len(time_idxs)
            Cs = len(freq_idxs)
            time_idxs, freq_idxs = np.meshgrid(time_idxs, freq_idxs, indexing='ij')
            time_idxs = time_idxs.flatten().tolist()
            freq_idxs = freq_idxs.flatten().tolist()

            keys = jax.random.split(key, len(time_idxs))
            # Submit them and get one at a time, to avoid memory issues.
            data_tasks = []
            for (key, time_idx, freq_idx) in zip(keys, time_idxs, freq_idxs):
                data_tasks.append(self._data_streamer(key, time_idx, freq_idx))
            data_gather: List[DataStreamerResponse] = await asyncio.gather(*data_tasks)
            # stack, reshape, and transpose to [Ts, B, Cs, 2, 2]

            vis_data = np.stack([data.vis for data in data_gather], axis=0)  # [Ts * Cs, B[2,2]]
            vis_data = np.reshape(vis_data, (Ts, Cs) + np.shape(vis_data)[1:])  # [Ts, Cs, B[2,2]]
            vis_data = np.moveaxis(vis_data, 1, 2)  # [Ts, B, Cs[, 2, 2]]

            weights = np.stack([data.weights for data in data_gather], axis=0)  # [Ts * Cs, B[2,2]]
            weights = np.reshape(weights, (Ts, Cs) + np.shape(weights)[1:])  # [Ts, Cs, B[2,2]]
            weights = np.moveaxis(weights, 1, 2)  # [Ts, B, Cs[, 2, 2]]

            flags = np.stack([data.flags for data in data_gather], axis=0)  # [Ts * Cs, B[2,2]]
            flags = np.reshape(flags, (Ts, Cs) + np.shape(flags)[1:])  # [Ts, Cs, B[2,2]]
            flags = np.moveaxis(flags, 1, 2)  # [Ts, B, Cs[, 2, 2]]

            uvw = np.stack([data.visibility_coords.uvw for data in data_gather], axis=0)  # [Ts * Cs, B, 3]
            uvw = np.reshape(uvw, (Ts, Cs) + np.shape(uvw)[1:])  # [Ts, Cs, B, 3]
            uvw = uvw[:, 0, :, :]

            freqs = np.stack([data.visibility_coords.freqs for data in data_gather], axis=0)  # [Ts * Cs]
            freqs = np.reshape(freqs, (Ts, Cs))  # [Ts, Cs]
            freqs = freqs[0, :]

            times = np.stack([data.visibility_coords.times for data in data_gather], axis=0)  # [Ts * Cs]
            times = np.reshape(times, (Ts, Cs))  # [Ts, Cs]
            times = times[:, 0]

            antenna1 = data_gather[0].visibility_coords.antenna1
            antenna2 = data_gather[0].visibility_coords.antenna2

            return vis_data, weights, flags, uvw, freqs, times, antenna1, antenna2

        async def model_gather(model_times, model_freqs):
            Tm = len(model_times)
            Cm = len(model_freqs)
            model_times, model_freqs = np.meshgrid(model_times, model_freqs, indexing='ij')
            model_times = model_times.flatten().tolist()
            model_freqs = model_freqs.flatten().tolist()
            model_tasks = []
            for (time, freq) in zip(model_times, model_freqs):
                model_tasks.append(self._model_predictor(time, freq))
            model_gather = await asyncio.gather(*model_tasks)

            vis_model = np.stack([data.vis for data in model_gather], axis=0)  # [Tm * Cm, B[2,2]]
            vis_model = np.reshape(vis_model, (Tm, Cm) + np.shape(vis_model)[1:])  # [Tm, Cm, B[2,2]]
            vis_model = np.moveaxis(vis_model, 1, 2)  # [Tm, B, Cm[, 2, 2]]

            return vis_model

        with TimerLog("Gathering data and model visibilities"):
            (vis_data, weights, flags, uvw, freqs, times, antenna1, antenna2), vis_model = await asyncio.gather(
                gather_data(key, time_idxs, freq_idxs),
                model_gather(model_times, model_freqs)
            )

        # Response generator can be used in an `async for` block.
        with TimerLog("Getting previous state..."):
            previous_state = self._calibration_solution_cache.get_calibration_solution_snapshot(
                sol_int_time_idx, sol_int_freq_idx)

        with TimerLog("Averaging data and model visibilities"):
            # vis_model, vis_data, weights, flags, freqs, times, antenna1, antenna2, state
            # average data to match model
            vis_data_avg = average_rule(
                average_rule(
                    vis_data,
                    num_model_size=self.params.chunk_params.num_model_times_per_solution_interval,
                    axis=0
                ),
                num_model_size=self.params.chunk_params.num_model_freqs_per_solution_interval,
                axis=0
            )
            weights_avg = 1. / average_rule(
                average_rule(
                    1. / weights,
                    num_model_size=self.params.chunk_params.num_model_times_per_solution_interval,
                    axis=0
                ),
                num_model_size=self.params.chunk_params.num_model_freqs_per_solution_interval,
                axis=0
            )
            flags_avg = average_rule(
                average_rule(
                    flags.astype(np.float16),
                    num_model_size=self.params.chunk_params.num_model_times_per_solution_interval,
                    axis=0
                ),
                num_model_size=self.params.chunk_params.num_model_freqs_per_solution_interval,
                axis=0
            ).astype(np.bool_)

        with TimerLog("Calibrating..."):
            gains, solver_state, diagnostics = block_until_ready(self.calibrate_jit(
                vis_model=vis_model,
                vis_data=vis_data_avg,
                weights=weights_avg,
                flags=flags_avg,
                freqs=model_freqs,
                times=model_times,
                antenna1=antenna1,
                antenna2=antenna2,
                state=previous_state.solver_state
            ))

        with TimerLog("Computing residuals..."):
            vis_residual = np.asarray(
                self._compute_residual_jit(
                    vis_model=vis_model,
                    vis_data=vis_data,
                    weights=weights,
                    flags=flags,
                    gains=gains,
                    antenna1=antenna1,
                    antenna2=antenna2
                )
            )
        with TimerLog("Storing solving state..."):
            self._calibration_solution_cache.store_calibration_solution(
                sol_int_time_idx, sol_int_freq_idx,
                CalibrationSolution(
                    solver_state=solver_state
                ))

        return CalibratorResponse(
            visibilities=np.asarray(vis_residual),
            weights=np.asarray(weights),
            flags=np.asarray(flags),
            uvw=np.asarray(uvw)
        )
