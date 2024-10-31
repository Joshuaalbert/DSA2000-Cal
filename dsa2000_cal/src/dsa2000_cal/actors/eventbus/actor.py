import asyncio
import atexit
import logging
from typing import List, Type

import ray
from pydantic import Field
from ray.actor import exit_actor
from ray.serve._private.utils import get_head_node_id
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_cal.actors.eventbus.abc import AbstractEventBus
from dsa2000_cal.actors.eventbus.websocket_server import EventBusServer
from dsa2000_cal.actors.namespace import NAMESPACE
from dsa2000_cal.common.ray_utils import get_or_create_event_loop, LogErrors
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel

logger = logging.getLogger('ray')
__all__ = [
    'EventBus',
    'EventBusParams'
]


class EventBusParams(SerialisableBaseModel):
    topics: List[str] = Field(
        description="List of topics to subscribe to."
    )


class EventBus(AbstractEventBus):

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        return (self._deserialise, (self._serialised_data,))

    @classmethod
    def _deserialise(cls, kwargs):
        # Create a new instance, bypassing __init__ and setting the actor directly
        return cls(**kwargs)

    def __init__(self, params: EventBusParams | None = None):

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
            f"EventBus",
            (_EventBus,),
            dict(_EventBus.__dict__),
        )

    @staticmethod
    def actor_name() -> str:
        return "EVENT_BUS"

    async def shutdown(self):
        await self._actor.shutdown.remote()


class _EventBus:
    def __init__(self, params: EventBusParams):
        self.params = params
        self._server: EventBusServer | None = None

        # Start control loop on actor creation, and assign clean up atexit

        self._control_long_running_task: asyncio.Task | None = get_or_create_event_loop().create_task(
            self._run_control_loop()
        )

        atexit.register(self._atexit_clean_up)

    def _atexit_clean_up(self):
        # Stop control loop
        if self._control_long_running_task is not None:
            self._control_long_running_task.cancel()

    async def health_check(self):
        """
        Announce health check.
        """
        logger.info(f"Healthy {self.__class__.__name__}")
        return

    async def _run_control_loop(self):
        logger.info(f"Starting control loop!")
        try:
            await self.run()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(str(e))
            raise e

    async def shutdown(self):
        # Triggers atexit handlers
        if self._server is not None:
            await self._server.stop()
            await asyncio.sleep(3.)
        exit_actor()

    async def run(self):
        """
        Collects data into queue.
        """
        async with LogErrors():
            self._server = EventBusServer.start_on_ray_head(
                topics=self.params.topics
            )
            # Wait for the server to stop, only when
            await self._server.start()
