import logging
from typing import Type

import ray
from pydantic import Field
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_rcp.actors.jax_coordinator.abc import AbstractJaxCoordinator
from dsa2000_rcp.actors.namespace import NAMESPACE
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel

logger = logging.getLogger('ray')
__all__ = [
    'JaxCoordinator',
    'JaxCoordinatorParams'
]


class JaxCoordinatorParams(SerialisableBaseModel):
    port: int = Field(
        description="Port number to listen on."
    )


class JaxCoordinator(AbstractJaxCoordinator):

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        return (self._deserialise, (self._serialised_data,))

    @classmethod
    def _deserialise(cls, kwargs):
        # Create a new instance, bypassing __init__ and setting the actor directly
        return cls(**kwargs)

    def __init__(self, params: JaxCoordinatorParams | None = None):

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
            placement_node_id = ray.get_runtime_context().get_node_id()
            actor_options = {
                "num_cpus": 0,
                "name": actor_name,
                "lifetime": "detached",
                "max_restarts": -1,
                "max_task_retries": -1,
                # Schedule the controller on the head node with a soft constraint. This
                # prefers it to run on the head node in most cases, but allows it to be
                # restarted on other nodes in an HA cluster.
                "scheduling_strategy": NodeAffinitySchedulingStrategy(placement_node_id, soft=False),
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
            f"JaxCoordinator",
            (_JaxCoordinator,),
            dict(_JaxCoordinator.__dict__),
        )

    @staticmethod
    def actor_name() -> str:
        return "JAX_COORDINATOR"

    def get_coordinator_address(self):
        return ray.get(self._actor.get_port.remote())


class _JaxCoordinator:
    def __init__(self, params: JaxCoordinatorParams):
        self.params = params

    async def health_check(self):
        """
        Announce health check.
        """
        logger.info(f"Healthy {self.__class__.__name__}")
        return

    async def get_coordinator_address(self):
        # Node IP
        node_id = ray.get_runtime_context().get_node_id()
        nodes_info = ray.nodes()

        # Find the IP address corresponding to the actor's node ID
        head_node_ip = None
        for node in nodes_info:
            if node["NodeID"] == node_id:
                head_node_ip = node["NodeManagerAddress"]
                break
        if head_node_ip is None:
            raise RuntimeError("Head node IP address not found")
        logger.debug(f"Node IP address found: {head_node_ip}")

        return f"{head_node_ip}:{self.params.port}"


