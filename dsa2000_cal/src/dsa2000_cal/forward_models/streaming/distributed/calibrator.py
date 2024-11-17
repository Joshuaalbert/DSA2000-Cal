import asyncio
import logging
import os
from functools import partial
from typing import NamedTuple, List, Type, Tuple, Dict

import jax
import jaxns.framework.context as ctx
import numpy as np
import ray
from jax import numpy as jnp
from jaxns.framework.ops import simulate_prior_model
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_cal.actors.namespace import NAMESPACE
from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardt
from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import UnconstrainedGain
from dsa2000_cal.common.array_types import ComplexArray, FloatArray, BoolArray, IntArray
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.ray_utils import get_head_node_id
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_cal.forward_models.streaming.distributed.data_streamer import DataStreamerResponse

logger = logging.getLogger('ray')


class CalibrationSolution(NamedTuple):
    solver_state: MultiStepLevenbergMarquardtState | None


class CalibrationSolutionCacheParams(SerialisableBaseModel):
    ...


class CalibrationSolutionCache:

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        return (self._deserialise, (self._serialised_data,))

    @classmethod
    def _deserialise(cls, kwargs):
        # Create a new instance, bypassing __init__ and setting the actor directly
        return cls(**kwargs)

    def __init__(self, params: CalibrationSolutionCacheParams | None = None):

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

    async def store_calibration_solution(self, freq_idxs: List[int], solution: CalibrationSolution):
        """
        Store a calibration solution for a given frequency chunk.

        Args:
            freq_idxs: List of frequency indices
            solution: The calibration solution
        """
        await self._actor.store_calibration_solution.remote(freq_idxs, solution)

    async def get_calibration_solution_snapshot(self, freq_idxs: List[int]) -> CalibrationSolution:
        """
        Get a snapshot of the calibration solution for a given frequency chunk.

        Args:
            freq_idxs: List of frequency indices

        Returns:
            The calibration solution.
        """
        return await self._actor.get_calibration_solution_snapshot.remote(freq_idxs)


class _CalibrationSolutionCache:
    """
    A cache for storing calibration, per frequency chunk.
    """

    def __init__(self, params: CalibrationSolutionCacheParams):
        self.params = params
        self.cache: Dict[Tuple[int, ...], CalibrationSolution] = {}  # tuple(freq_idxs) -> solution

    async def store_calibration_solution(self, freq_idxs: List[int], solution: CalibrationSolution):
        self.cache[tuple(freq_idxs)] = solution

    async def get_calibration_solution_snapshot(self, freq_idxs: List[int]) -> CalibrationSolution:
        return self.cache.get(tuple(freq_idxs), CalibrationSolution(solver_state=None))


def build_compute_residuals():
    def compute_residuals(
            gains: ComplexArray,
            vis_per_direction: ComplexArray,
            vis_data: ComplexArray,
            weights: FloatArray,
            flags: BoolArray,
            antenna1: IntArray,
            antenna2: IntArray,
            weighted: bool = True
    ):
        """
        Compute the residual between the model visibilities and the observed visibilities.

        Args:
            gains: [D, A, F, 2, 2]
            vis_per_direction: [D, T, B, F, 2, 2]
            vis_data: [T, B, F, 2, 2]
            weights: [T, B, F, 2, 2]
            flags: [T, B, F, 2, 2]
            antenna1: [B]
            antenna2: [B]

        Returns:
            residuals: [T, B, F, 2, 2]
        """

        if np.shape(weights) != np.shape(flags):
            raise ValueError(
                f"Visibilities shape {np.shape(vis_per_direction)} must match flags shape {np.shape(flags)}.")

        def body_fn(accumulate, x):
            gains, vis_per_direction = x
            # Compute the model visibilities
            g1 = gains[antenna1]  # [B, F, 2, 2]
            g2 = gains[antenna2]  # [B, F, 2, 2]

            @partial(
                multi_vmap,
                in_mapping="[B,F,2,2],[B,F,2,2],[T,B,F,2,2]",
                out_mapping="[T,B,F,~P,~Q]",
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

            delta_vis = apply_gains(g1, g2, vis_per_direction)  # [T, B, F, 2, 2]
            return accumulate + delta_vis, ()

        accumulate = jnp.zeros_like(vis_data)
        model_vis, _ = jax.lax.scan(body_fn, accumulate, (gains, vis_per_direction))
        residuals = model_vis - vis_data  # [T, B, F, 2, 2]
        if weighted:
            weights = jnp.where(flags, jnp.zeros_like(weights), weights)  # [T, B, F, 2, 2]
            residuals = residuals * weights  # [T, B, F, 2, 2]
        return residuals

    return compute_residuals


class CalibratorResponse(NamedTuple):
    visibilities: np.ndarray  # [T_sol, B, C_sol[, 2, 2]]
    weights: np.ndarray  # [T_sol, B, C_sol[, 2, 2]]
    flags: np.ndarray  # [T_sol, B, C_sol[, 2, 2]]
    uvw: np.ndarray  # [T_sol, B, 3]
    freqs: np.ndarray  # [C_sol]


@serve.deployment
class Calibrator:
    """
    Calibrates and subtracts model visibilities from data vis and streams results, per sol_idx.
    A sol_int corresponds to a chunk of time and frequency data of size (num_times_per_sol_int, num_freqs_per_sol_int).
    The total dataset is (num_times_per_obs, num_freqs_per_obs) or in terms of sol_ints (num_times_per_obs//num_times_per_sol_int, num_freqs_per_obs//num_freqs_per_sol_int).
    """

    def __init__(self, params: ForwardModellingRunParams, data_streamer: DeploymentHandle,
                 model_predictor: DeploymentHandle,
                 calibration_solution_cache: CalibrationSolutionCache):
        self.params = params
        self._calibration_solution_cache = calibration_solution_cache
        self._data_streamer = data_streamer
        self._model_predictor = model_predictor
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'calibrator')
        os.makedirs(self.params.plot_folder, exist_ok=True)

        vis_shape = (params.chunk_params.num_times_per_sol_int, params.chunk_params.num_baselines,
                     params.chunk_params.num_freqs_per_sol_int)
        if self.params.full_stokes:
            vis_shape = vis_shape + (2, 2)
        model_vis_shape = (params.num_facets,) + vis_shape

        self.vis_data = np.zeros(vis_shape, dtype=np.complex128)
        self.weights_data = np.zeros(vis_shape, dtype=np.float16)
        self.flags_data = np.zeros(vis_shape, dtype=np.bool_)
        self.vis_model = np.zeros(model_vis_shape, dtype=np.complex128)

        self.uvw = np.zeros((params.chunk_params.num_times_per_sol_int, params.chunk_params.num_baselines, 3),
                            dtype=np.float64)
        self.antenna1 = np.zeros((params.chunk_params.num_baselines,), dtype=mp_policy.index_dtype)
        self.antenna2 = np.zeros((params.chunk_params.num_baselines,), dtype=mp_policy.index_dtype)
        self.times = np.zeros((params.chunk_params.num_times_per_sol_int,), dtype=np.float64)
        self.freqs = np.zeros((params.chunk_params.num_freqs_per_sol_int,), dtype=np.float64)

        def calibrate(vis_model, vis_data, weights, flags, freqs, times, antenna1, antenna2, state):
            # calibrate and subtract
            key = jax.random.PRNGKey(0)

            D, T, B, C = np.shape(vis_model)[:4]

            # Create gain prior model
            def get_gains():
                gain_probabilistic_model = UnconstrainedGain(
                    full_stokes=self.params.full_stokes,
                    dof=2
                )
                mean_time = jnp.mean(times)
                prior_model = gain_probabilistic_model.build_prior_model(
                    num_source=D,
                    num_ant=len(self.params.ms_meta.antennas),
                    freqs=freqs,
                    times=mean_time[None]
                )
                (gains,), _ = simulate_prior_model(key, prior_model)  # [D, 1, A, F, 2, 2]
                return gains[:, 0]  # [D, A, F, 2, 2]

            get_gains_transformed = ctx.transform(get_gains)

            compute_residuals = build_compute_residuals()

            # Create residual_fn
            def residual_fn(params: ComplexArray) -> ComplexArray:
                gains = get_gains_transformed.apply(params, key).fn_val
                return compute_residuals(gains, vis_model, vis_data, weights, flags, antenna1, antenna2)

            solver = MultiStepLevenbergMarquardt(
                residual_fn=residual_fn,
                num_approx_steps=0,
                num_iterations=100,
                verbose=True,
                gtol=1e-4
            )

            # Get solver state
            if state is None:
                init_params = get_gains_transformed.init(key).params
                state = solver.create_initial_state(init_params)
            else:
                state = solver.update_initial_state(state)
            state, diagnostics = solver.solve(state)

            gains = get_gains_transformed.apply(state.x, key).fn_val

            vis_data_residuals = compute_residuals(gains, vis_model,
                                                   vis_data, weights, flags, antenna1,
                                                   antenna2, weighted=False)

            return gains, vis_data_residuals, state, diagnostics

        self.calibrate_jit = jax.jit(calibrate)

    async def __call__(self, sol_idx: int, key) -> CalibratorResponse:

        sol_int_time_idx, sol_int_freq_idx = np.unravel_index(
            sol_idx,
            (
                self.params.chunk_params.num_sol_ints_time_per_image,
                self.params.chunk_params.num_sol_ints_freq_per_image
            )
        )

        time_idxs = sol_int_time_idx * self.params.chunk_params.num_times_per_sol_int + np.arange(
            self.params.chunk_params.num_times_per_sol_int)
        freq_idxs = sol_int_freq_idx * self.params.chunk_params.num_freqs_per_sol_int + np.arange(
            self.params.chunk_params.num_freqs_per_sol_int)
        T, F = np.meshgrid(time_idxs, freq_idxs, indexing='ij')
        time_idxs = T.flatten().tolist()
        freq_idxs = F.flatten().tolist()

        async def get_data(time_idxs: List[int], freq_idxs: List[int], key):
            keys = jax.random.split(key, len(time_idxs))
            data_responses: List[DataStreamerResponse] = await asyncio.gather(
                *[
                    self._data_streamer.remote(time_idx, freq_idx, key)
                    for (time_idx, freq_idx, key) in zip(time_idxs, freq_idxs, keys)
                ]
            )
            # Request idxs
            for data_response, (time_idx, freq_idx) in zip(data_responses, zip(time_idxs, freq_idxs)):
                self.vis_data[time_idx, :, freq_idx, ...] = data_response.vis  # [B, 2, 2]
                self.weights_data[time_idx, :, freq_idx, ...] = data_response.weights
                self.flags_data[time_idx, :, freq_idx, ...] = data_response.flags
                self.freqs[freq_idx] = data_response.visibility_coords.freqs[0]
                self.times[time_idx] = data_response.visibility_coords.times[0]
                self.uvw[time_idx, :, :] = data_response.visibility_coords.uvw[0]
                self.antenna1 = data_response.visibility_coords.antenna_1
                self.antenna2 = data_response.visibility_coords.antenna_2

        async def get_model_data(time_idxs: List[int], freq_idxs: List[int]):
            model_responses = await asyncio.gather(
                *[
                    self._model_predictor.remote(time_idx, freq_idx)
                    for (time_idx, freq_idx) in zip(time_idxs, freq_idxs)
                ]
            )
            # Request idxs
            for model_response, (time_idx, freq_idx) in zip(model_responses, zip(time_idxs, freq_idxs)):
                self.vis_model[:, time_idx, :, freq_idx, ...] = model_response.vis

        # Response generator can be used in an `async for` block.
        _, _, previous_state = await asyncio.gather(
            get_data(time_idxs, freq_idxs, key),
            get_model_data(time_idxs, freq_idxs),
            self._calibration_solution_cache.get_calibration_solution_snapshot(freq_idxs)
        )
        previous_state = jax.tree.map(jnp.asarray, previous_state)
        # vis_model, vis_data, weights, flags, freqs, times, antenna1, antenna2, state
        gains, vis_data_residuals, solver_state, diagnostics = self.calibrate_jit(
            vis_model=jnp.asarray(self.vis_model, mp_policy.vis_dtype),
            vis_data=jnp.asarray(self.vis_data, mp_policy.vis_dtype),
            weights=jnp.asarray(self.weights_data, mp_policy.weight_dtype),
            flags=jnp.asarray(self.flags_data, mp_policy.flag_dtype),
            freqs=jnp.asarray(self.freqs, mp_policy.freq_dtype),
            times=jnp.asarray(self.times, mp_policy.time_dtype),
            antenna1=jnp.asarray(self.antenna1, mp_policy.length_dtype),
            antenna2=jnp.asarray(self.antenna2, mp_policy.length_dtype),
            state=previous_state.solver_state
        )
        solution = jax.tree.map(np.asarray, CalibrationSolution(solver_state=solver_state))
        await self._calibration_solution_cache.store_calibration_solution(freq_idxs, solution)
        return CalibratorResponse(
            visibilities=np.asarray(vis_data_residuals),
            weights=np.asarray(self.weights_data),
            flags=np.asarray(self.flags_data),
            uvw=np.asarray(self.uvw),
            freqs=np.asarray(self.freqs)
        )
