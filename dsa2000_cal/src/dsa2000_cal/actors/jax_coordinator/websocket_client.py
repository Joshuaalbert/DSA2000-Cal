import asyncio
import base64
import dataclasses
import logging
import os
import pickle
from typing import Any, Callable, Iterable, Coroutine

import msgpack
import ray
import websockets
from ray import ObjectRef
from websockets import WebSocketServerProtocol

from dsa2000_cal.common.ray_utils import get_head_node_id

logger = logging.getLogger("ray")


class DoneListening(Exception):
    """
    Signal to stop listening for incoming messages, as a poison pill style.
    """
    pass


@dataclasses.dataclass(eq=False)
class EventBusClient:
    server_ip: str = "0.0.0.0"
    port: int = 8765

    def __post_init__(self):
        self._websocket: WebSocketServerProtocol | None = None
        self.packer = msgpack.Packer(
            use_bin_type=True,
            use_single_float=False,
            datetime=True
        )

    @staticmethod
    def connect_to_ray_head(port: int | None = None) -> 'EventBusClient':
        if port is None:
            # Get EVENT_BUS_PORT from environment, falling back to 8765
            port = int(os.environ.get("EVENT_BUS_PORT", 8765))
        head_node_id = get_head_node_id()
        logger.debug(f"Attempting to connect to Ray head node {head_node_id}...")
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
        logger.debug(f"Head node IP address found: {head_node_ip}")
        return EventBusClient(
            server_ip=head_node_ip,
            port=port
        )

    @staticmethod
    def wrap_handler(handler):
        """
        Wraps the handler coroutine to handle shutdown events, etc.

        Args:
            handler: A coroutine that processes incoming messages.

        Returns:
            A wrapped coroutine.
        """
        unpacker = msgpack.Unpacker()

        async def wrapper(websocket, message):
            # 1-1 feed unpack cycle
            unpacker.feed(message)
            try:
                msg = unpacker.unpack()
            except msgpack.OutOfData:
                raise RuntimeError("Message is incomplete")
            event = msg.get("event")
            if event == "shutdown":
                logger.info("Server shutting down...")
                await websocket.close()
            elif event == "subscribed":
                logger.info(f"Subscribed to {msg['topic']}")
            elif event == "unsubscribed":
                logger.info(f"Unsubscribed from {msg['topic']}")
            else:
                unpacker.feed(msg['data'])
                msg['data'] = unpacker.unpack()
                # logger.debug(f"Received message on {msg['topic']}: {msg}")
                return await handler(websocket, msg)
            return None

        return wrapper

    async def subscribe(self, websocket: WebSocketServerProtocol, topic: str):
        """Subscribes to a topic."""
        message = self.packer.pack({"event": f"subscribe:{topic}"})
        logger.debug(f"Subscribing to topic: {topic}")
        await websocket.send(message)

    async def unsubscribe(self, websocket: WebSocketServerProtocol, topic: str):
        """Unsubscribes from a topic."""
        message = self.packer.pack({"event": f"unsubscribe:{topic}"})
        logger.debug(f"Unsubscribing from topic: {topic}")
        await websocket.send(message)

    async def listen(self, handler: Callable[[WebSocketServerProtocol, Any], Coroutine],
                     topics: Iterable[str]):
        """
        Listens for incoming messages from the event bus.

        Args:
            handler: A coroutine that processes incoming messages.
        """
        wrapped_handler = self.wrap_handler(handler)
        uri = f"ws://{self.server_ip}:{self.port}"
        async for websocket in websockets.connect(uri, max_queue=None):
            logger.info(f"Connected to event bus at {uri}")
            for topic in topics:
                await self.subscribe(websocket, topic)
            try:
                async for message in websocket:
                    await wrapped_handler(websocket, message)
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                continue
            except asyncio.CancelledError:
                logger.info("Listening cancelled, exiting...")
                break
            except DoneListening:
                logger.info("Done listening, exiting...")
                break

    @property
    def websocket(self):
        if self._websocket is None:
            logger.error("WebSocket connection is not established")
            raise RuntimeError("WebSocket connection is not established")
        return self._websocket

    async def __aenter__(self):
        """Establish the connection when entering the context."""
        uri = f"ws://{self.server_ip}:{self.port}"
        logger.debug(f"Attempting to establish connection to {uri}")
        try:
            self._websocket = await websockets.connect(uri, max_queue=None)
            logger.info(f"Successfully connected to {uri}")
        except asyncio.CancelledError:
            logger.warning("Connection attempt cancelled.")
            raise  # Re-raise to propagate the cancellation
        except Exception as e:
            logger.error(f"Error while establishing WebSocket connection to {uri}: {e}", exc_info=True)
            raise  # Re-raise the exception after logging
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the connection when exiting the context."""
        if self._websocket:
            try:
                logger.debug("Closing WebSocket connection")
                await self._websocket.close()
                logger.info("WebSocket connection closed")
            except asyncio.CancelledError:
                logger.warning("WebSocket closure cancelled.")
                raise  # Re-raise the cancellation to propagate it
            except Exception as e:
                logger.error(f"Error while closing WebSocket connection: {e}", exc_info=True)
            finally:
                self._websocket = None  # Ensure that the WebSocket is set to None after closing

        if exc_type is not None:
            if exc_type is asyncio.CancelledError:
                # Log cancellation specifically
                logger.warning("Operation was cancelled during context management.")
                raise  # Re-raise to propagate cancellation

            # Log other exceptions that occurred during the context
            logger.error(f"Exception occurred during context management: {exc_type.__name__}, {exc_val}", exc_info=True)
            return False  # Re-raise the exception if needed, else return True to suppress it

        return True  # Returning True to suppress any exception if handling gracefully

    async def publish(self, topic: str, data: Any):
        """Publish data to a topic."""
        data_pack = self.packer.pack(data)
        message = {"event": f"publish:{topic}", "data": data_pack}
        await self.websocket.send(self.packer.pack(message))


def object_to_ref_string(obj_ref: ObjectRef) -> str:
    # Serialize the ObjectRef to a byte string
    ref_bytes = pickle.dumps(obj_ref)

    # Encode the byte string to a base64 string (to make it easily serializable)
    ref_string = base64.b64encode(ref_bytes).decode('utf-8')

    return ref_string


def ref_string_to_object_ref(ref_string: str):
    # Decode the base64 string back to bytes
    ref_bytes = base64.b64decode(ref_string.encode('utf-8'))

    # Deserialize the bytes back to an ObjectRef
    obj_ref = pickle.loads(ref_bytes)

    return obj_ref
