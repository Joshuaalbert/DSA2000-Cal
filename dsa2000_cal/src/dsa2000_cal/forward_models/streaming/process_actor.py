import asyncio
import logging
from datetime import datetime
from typing import Type

import jax
import ray
import uvloop
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_cal.actors.namespace import NAMESPACE
from dsa2000_cal.common.ray_utils import get_head_node_id
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger("ray")


class SFMProcessParams(SerialisableBaseModel):
    num_processes: int
    process_id: int
    plot_folder: str


class SFMProcess:

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        return (self._deserialise, (self._serialised_data,))

    @classmethod
    def _deserialise(cls, kwargs):
        # Create a new instance, bypassing __init__ and setting the actor directly
        return cls(**kwargs)

    def __init__(self, worker_id: str, params: SFMProcessParams | None = None) -> None:
        self._serialised_data = dict(
            worker_id=worker_id,
            params=params
        )

        actor_name = self.actor_name(node_id=worker_id)

        try:
            actor = ray.get_actor(actor_name, namespace=NAMESPACE)
            logger.info(f"Connected to existing {actor_name}")
        except ValueError:
            if params is None:
                raise ValueError(f"Actor {actor_name} does not exist, and params is None")
            placement_node_id = ray.get_runtime_context().get_node_id()
            actor_options = {
                "num_cpus": 0,
                "name": actor_name,
                # "lifetime": "detached",
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
            logger.info(f"Created new {actor_name}")

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
            f"SFMProcess",
            (_SFMProcess,),
            dict(_SFMProcess.__dict__),
        )

    @staticmethod
    def actor_name(node_id: str) -> str:
        return f"SFM_PROCESS#{node_id}"

    def run(self):
        return ray.get(self._actor.run.remote())


def get_head_ip():
    head_node_id = get_head_node_id()
    logger.info(f"Attempting to connect to Ray head node {head_node_id}...")
    # Retrieve all node information
    nodes_info = ray.nodes()

    # Find the IP address corresponding to the actor's node ID
    head_node_ip = None
    for node in nodes_info:
        if node["NodeID"] == head_node_id:
            head_node_ip = node["NodeManagerAddress"]
            break
    if head_node_ip is None:
        raise RuntimeError("Head node IP address not found")
    return head_node_ip


class _SFMProcess:
    def __init__(self, params: SFMProcessParams):
        self.params = params

    async def health_check(self):
        """
        Announce health check.
        """
        logger.info(f"Healthy {self.__class__.__name__}")
        return

    async def run(self):
        coordinator_address = get_head_ip()

        print(f"Beginning multi-host initialisation at {datetime.now()}")
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=self.params.num_processes,
            process_id=self.params.process_id
        )
        print(f"Initialised at {datetime.now()}")

        # Must import only after jax.distributed.initialize to avoid issues with jax devices
        from dsa2000_cal.forward_models.streaming.process import process_start

        process_start(
            process_id=self.params.process_id,
            key=jax.random.PRNGKey(0),
            array_name="dsa2000_31b",
            full_stokes=True,
            plot_folder=self.params.plot_folder
        )
