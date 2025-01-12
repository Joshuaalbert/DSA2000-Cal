import asyncio
import logging
from datetime import timedelta
from typing import NamedTuple, Type, Dict, Tuple

import jax
import numpy as np
import ray
from jax import numpy as jnp
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_cal.calibration.solvers.multi_step_lm import MultiStepLevenbergMarquardtState
from dsa2000_cal.common.array_types import ComplexArray, FloatArray
from dsa2000_cal.common.ray_utils import get_head_node_id, resource_logger
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_fm.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_rcp.actors.namespace import NAMESPACE

logger = logging.getLogger('ray')


class CalibrationSolution(NamedTuple):
    solver_state: MultiStepLevenbergMarquardtState | None
    gains: ComplexArray | None  # [D, Tm, A, Cm[, 2, 2]]
    model_times: FloatArray | None  # [Tm]
    model_freqs: FloatArray | None  # [Cm]


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

    async def store_calibration_solution(self, sol_int_time_idx: int, sol_int_freq_idx: int,
                                         solution: CalibrationSolution):
        """
        Store a calibration solution for a given frequency chunk.

        Args:
            sol_int_time_idx: The time index of the solution interval
            sol_int_freq_idx: The frequency index of the solution interval
            solution: The calibration solution
        """
        await self._actor.store_calibration_solution.remote(sol_int_time_idx, sol_int_freq_idx,
                                                            jax.tree.map(np.asarray, solution))

    async def get_calibration_solution_snapshot(self, sol_int_time_idx: int,
                                                sol_int_freq_idx: int) -> CalibrationSolution:
        """
        Get a snapshot of the calibration solution for a given frequency chunk.

        Args:
            sol_int_time_idx: The time index of the solution interval
            sol_int_freq_idx: The frequency index of the solution interval

        Returns:
            The calibration solution.
        """
        last_solution = await self._actor.get_calibration_solution_snapshot.remote(sol_int_time_idx, sol_int_freq_idx)
        return jax.tree.map(jnp.asarray, last_solution)


class _CalibrationSolutionCache:
    """
    A cache for storing calibration, per frequency chunk.
    """

    def __init__(self, params: CalibrationSolutionCacheParams):
        self.params = params
        self.cache: Dict[
            Tuple[int, int], CalibrationSolution] = {}  # (sol_int_time_idx, sol_int_freq_idx) -> CalibrationSolution
        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

    def health_check(self):
        """
        Announce health check.
        """
        logger.info(f"Healthy {self.__class__.__name__}")
        return

    async def init(self):
        if self._initialised:
            return
        self._initialised = True
        self._memory_logger_task = asyncio.create_task(
            resource_logger(task='calibration_solution_cache', cadence=timedelta(seconds=5)))

    async def store_calibration_solution(self, sol_int_time_idx: int, sol_int_freq_idx: int,
                                         solution: CalibrationSolution):
        await self.init()
        self.cache[(sol_int_time_idx, sol_int_freq_idx)] = solution

    async def get_calibration_solution_snapshot(self, sol_int_time_idx: int,
                                                sol_int_freq_idx: int) -> CalibrationSolution:
        await self.init()
        return self.cache.get(
            (sol_int_time_idx - 1, sol_int_freq_idx),
            CalibrationSolution(
                solver_state=None,
                gains=None,
                model_freqs=None,
                model_times=None
            )
        )


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
