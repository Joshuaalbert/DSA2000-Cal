import asyncio
import logging
import time
from typing import List
from typing import Type
from uuid import uuid4

import ray
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_cal.common.ray_utils import get_head_node_id
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_fm.namespace import NAMESPACE

logger = logging.getLogger('ray')


class NodeResourceMonitorParams(SerialisableBaseModel):
    name: str
    actors: List


class NodeResourceMonitor:
    """
    Distributes load among actors in a first-come-first-serve manner.
    """

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        return (self._deserialise, (self._serialised_data,))

    @classmethod
    def _deserialise(cls, kwargs):
        # Create a new instance, bypassing __init__ and setting the actor directly
        return cls(**kwargs)

    def __init__(self, worker_id: str, params: NodeResourceMonitorParams | None = None):

        self._serialised_data = dict(
            worker_id=worker_id,
            params=None
        )
        actor_name = self.actor_name(worker_id)

        try:
            actor = ray.get_actor(actor_name, namespace=NAMESPACE)
            logger.info(f"Connected to existing {actor_name}")
        except ValueError:
            if params is None:
                raise ValueError(f"Actor {actor_name} does not exist, and params is None")

            placement_node_id = get_head_node_id()

            actor_options = {
                "name": actor_name,
                # "lifetime": "detached",
                "max_restarts": -1,
                "max_task_retries": -1,
                # Schedule the controller on the same node with a soft constraint. This
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
            f"NodeResourceMonitor",
            (_NodeResourceMonitor,),
            dict(_NodeResourceMonitor.__dict__),
        )

    @staticmethod
    def actor_name(node_id: str) -> str:
        return f"NODE_RESOURCE_MONITOR#{node_id}"

    async def num_running(self) -> int:
        return await self._actor.num_running.remote()

    async def num_available(self) -> int:
        return await self._actor.num_available.remote()


class _NodeResourceMonitor:
    def __init__(self, params: NodeResourceMonitorParams):
        self.params = params
        self._actor_queue = asyncio.Queue()
        for i, _ in enumerate(self.params.actors):
            self._actor_queue.put_nowait(i)
        self._run_time_gauge = Gauge(
            name=f"run_time_gauge_s",
            description="The time taken to run a task",
            tag_keys=("task",)
        )
        self._queue_time_gauge = Gauge(
            name=f"queue_time_gauge_s",
            description="The time taken to queue a task",
            tag_keys=("task",)
        )
        self._num_running_gauge = Gauge(
            name=f"num_running_gauge",
            description="The number of running tasks",
            tag_keys=("task",)
        )
        self._run_time_gauge.set_default_tags({"task": params.name})
        self._queue_time_gauge.set_default_tags({"task": params.name})
        self._num_running_gauge.set_default_tags({"task": params.name})

    def health_check(self):
        """
        Announce health check.
        """
        logger.info(f"Healthy {self.__class__.__name__}")
        return

    def num_running(self) -> int:
        return len(self.params.actors) - self._actor_queue.qsize()

    def num_available(self) -> int:
        return self._actor_queue.qsize()

    async def call(self, *args, **kwargs):
        # Get the next available actor
        t0 = time.time()
        actor_idx = await self._actor_queue.get()
        t1 = time.time()
        self._queue_time_gauge.set(t1 - t0)

        # Call the actor
        actor = self.params.actors[actor_idx]
        self._num_running_gauge.set(self.num_running())
        t0 = time.time()
        response = await actor.__call__.remote(*args, **kwargs)
        t1 = time.time()
        self._actor_queue.put_nowait(actor_idx)
        self._num_running_gauge.set(self.num_running())
        self._run_time_gauge.set(t1 - t0)
        return response


def create_supervisor(remote: ray.actor.ActorClass, name: str, num_actors: int, *args, **kwargs) -> NodeResourceMonitor:
    """
    Create a supervisor.

    Args:
        remote: the actor class
        name: the name of the supervisor
        num_actors: the number of actors
        *args: the args passed to remote
        **kwargs: the kwargs passed to remote

    Returns:
        the supervisor
    """
    actors = [remote.remote(*args, **kwargs) for _ in range(num_actors)]
    return NodeResourceMonitor(worker_id=f"{name.upper()}-{str(uuid4())}",
                               params=NodeResourceMonitorParams(name=name, actors=actors))
