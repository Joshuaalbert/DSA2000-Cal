import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Generic, TypeVar, AsyncGenerator, Awaitable
from typing import Type
from uuid import uuid4

import ray
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_fm.namespace import NAMESPACE

logger = logging.getLogger('ray')


class SupervisorParams(SerialisableBaseModel):
    name: str
    actors: List


T = TypeVar('T')


class Supervisor(Generic[T]):
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

    def __init__(self, worker_id: str, params: SupervisorParams | None = None):

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

            placement_node_id = ray.get_runtime_context().get_node_id()

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
            f"Supervisor",
            (_Supervisor,),
            dict(_Supervisor.__dict__),
        )

    @staticmethod
    def actor_name(node_id: str) -> str:
        return f"SUPERVISOR#{node_id}"

    async def stream(self, *args, **kwargs) -> AsyncGenerator[Awaitable[T], None]:
        ref_gen = self._actor.stream.remote(*args, **kwargs)
        async for ref in ref_gen:
            yield await ref

    async def __call__(self, *args, **kwargs) -> T:
        obj_ref = await self._actor.call.remote(*args, **kwargs)
        return await obj_ref

    async def num_running(self) -> int:
        return await self._actor.num_running.remote()

    async def num_available(self) -> int:
        return await self._actor.num_available.remote()


class _Supervisor:
    def __init__(self, params: SupervisorParams):
        self.params = params
        self._actor_queue = asyncio.Queue()
        for i, _ in enumerate(self.params.actors):
            self._actor_queue.put_nowait(i)
        self._thread_pool = ThreadPoolExecutor(max_workers=min(32, len(self.params.actors)))
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
        self._num_queued_gauge = Gauge(
            name=f"num_queued_gauge",
            description="The number of queued tasks",
            tag_keys=("task",)
        )
        self._run_time_gauge.set_default_tags({"task": params.name})
        self._queue_time_gauge.set_default_tags({"task": params.name})
        self._num_running_gauge.set_default_tags({"task": params.name})
        self._num_queued_gauge.set_default_tags({"task": params.name})
        self._num_queued = 0

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

    async def stream(self, *args, **kwargs):
        # Get the next available actor
        t0 = time.time()
        self._num_queued += 1
        if self._actor_queue.empty():
            # If there are no available actors, report
            self._num_queued_gauge.set(self._num_queued)
        actor_idx = await self._actor_queue.get()
        self._num_queued -= 1
        self._num_queued_gauge.set(self._num_queued)
        t1 = time.time()
        self._queue_time_gauge.set(t1 - t0)

        # Call the actor
        actor = self.params.actors[actor_idx]
        self._num_running_gauge.set(self.num_running())
        t0 = time.time()
        response_obj_ref_gen = actor.__call__.remote(*args, **kwargs)
        async for response_obj_ref in response_obj_ref_gen:
            yield response_obj_ref
            t1 = time.time()
            self._run_time_gauge.set(t1 - t0)
            t0 = time.time()

        self._actor_queue.put_nowait(actor_idx)
        self._num_running_gauge.set(self.num_running())

    async def call(self, *args, **kwargs):
        # Get the next available actor
        t0 = time.time()
        self._num_queued += 1
        if self._actor_queue.empty():
            # If there are no available actors, report
            self._num_queued_gauge.set(self._num_queued)
        actor_idx = await self._actor_queue.get()
        self._num_queued -= 1
        self._num_queued_gauge.set(self._num_queued)
        t1 = time.time()
        self._queue_time_gauge.set(t1 - t0)

        # Call the actor
        actor = self.params.actors[actor_idx]
        self._num_running_gauge.set(self.num_running())
        t0 = time.time()
        response_obj_ref = actor.__call__.remote(*args, **kwargs)
        loop = asyncio.get_event_loop()
        [ready_response_obj_ref], _ = await loop.run_in_executor(
            self._thread_pool,
            lambda: ray.wait([response_obj_ref], num_returns=1, fetch_local=False)
        )
        t1 = time.time()
        self._actor_queue.put_nowait(actor_idx)
        self._num_running_gauge.set(self.num_running())
        self._run_time_gauge.set(t1 - t0)
        return ready_response_obj_ref


def create_supervisor(remote: ray.actor.ActorClass, name: str, num_actors: int, *args, **kwargs) -> Supervisor:
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
    return Supervisor(worker_id=f"{name.upper()}-{str(uuid4())}", params=SupervisorParams(name=name, actors=actors))
