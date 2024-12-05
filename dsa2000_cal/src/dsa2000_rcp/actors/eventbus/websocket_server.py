import asyncio
import dataclasses
import logging
import os
from collections import defaultdict
from typing import Dict, Set, Any, Iterable

import msgpack
import ray
import websockets
from ray.serve._private.utils import get_head_node_id
from websockets import WebSocketServerProtocol

logger = logging.getLogger('ray')


@dataclasses.dataclass(eq=False)
class EventBusServer:
    topics: Iterable[str]
    host: str = "0.0.0.0"
    port: int = 8765

    def __post_init__(self):
        self.subscribers: Dict[str, Set[WebSocketServerProtocol]] = defaultdict(
            set)  # Maps topics to a set of subscriber client addresses
        self.topic_queues = defaultdict(asyncio.Queue)  # Maps topics to message queues
        self._stop = asyncio.Future()
        self.packer = msgpack.Packer()
        self.unpacker = msgpack.Unpacker()

    @staticmethod
    def start_on_ray_head(topics: Iterable[str], port: int | None = None) -> 'EventBusServer':
        if port is None:
            # Get EVENT_BUS_PORT from environment, falling back to 8765
            port = int(os.environ.get("EVENT_BUS_PORT", 8765))
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
        return EventBusServer(
            topics=topics,
            host=head_node_ip,
            port=port
        )

    async def stop(self):
        """Stops the WebSocket server."""
        # Tell all subscribers of imminent shutdown
        for topic in self.topics:
            message = self.packer.pack({"event": "shutdown"})
            await self.topic_queues[topic].put(message)
        if not self._stop.done():
            self._stop.set_result(None)
            logger.info("Stopping server...")

    async def handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handles incoming WebSocket connections."""
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.ConnectionClosed:
            # Handle client disconnection
            await self.cleanup(websocket)
            logger.info(f"Client {websocket.remote_address} disconnected")
        except Exception as e:
            logger.info(f"Error handling message: {e}")

    async def handle_message(self, websocket: WebSocketServerProtocol, message: bytes):
        """Processes incoming messages and routes them based on event type."""
        self.unpacker.feed(message)
        try:
            message_data = self.unpacker.unpack()
        except msgpack.OutOfData:
            raise RuntimeError("Message is incomplete")
        event = message_data.get("event")
        if event:
            action, topic = event.split(":")
            if action == "subscribe":
                await self.subscribe(websocket, topic)
            elif action == "unsubscribe":
                await self.unsubscribe(websocket, topic)
            elif action == "publish":
                data = message_data.get("data", None)
                if data is None:
                    data = self.packer.pack({})
                await self.publish(websocket, topic, data)

    async def subscribe(self, websocket: WebSocketServerProtocol, topic: str):
        """Subscribes a client to a topic."""
        if topic in self.topics:
            self.subscribers[topic].add(websocket)
            await websocket.send(self.packer.pack({"event": f"subscribed", "topic": topic}))
            logger.info(f"Client {websocket.remote_address} subscribed to {topic}")
        else:
            logger.info(f"Invalid topic: {topic}")

    async def unsubscribe(self, websocket: WebSocketServerProtocol, topic: str):
        """Unsubscribes a client from a topic."""
        if topic in self.topics:
            self.subscribers[topic].discard(websocket)
            await websocket.send(self.packer.pack({"event": "unsubscribed", "topic": topic}))
            logger.info(f"Client {websocket.remote_address} unsubscribed from {topic}")
        else:
            logger.info(f"Invalid topic: {topic}")

    async def publish(self, websocket: WebSocketServerProtocol, topic: str, data_pack: Any):
        """Publishes a message to a topic, notifying all subscribers."""
        if topic in self.topics:
            message = self.packer.pack({"event": "data", "topic": topic, "data": data_pack})
            await self.topic_queues[topic].put(message)
            logger.debug(f"Message from {websocket.remote_address} published to {topic}")
        else:
            logger.info(f"Invalid topic: {topic}")

    async def broadcast(self, topic: str):
        """Sends all messages in the topic queue to its subscribers."""

        while True:
            message = await self.topic_queues[topic].get()
            subscribers = self.subscribers[topic].copy()  # To prevent size changing during iteration
            for subscriber in subscribers:
                try:
                    await subscriber.send(message)
                except websockets.ConnectionClosed:
                    await self.cleanup(subscriber)

    async def cleanup(self, websocket: WebSocketServerProtocol):
        """Cleans up subscriber and publisher lists when a client disconnects."""
        for topic in self.topics:
            self.subscribers[topic].discard(websocket)

    async def start(self):
        """Starts the WebSocket server."""
        tasks = []
        for topic in self.topics:
            tasks.append(asyncio.create_task(self.broadcast(topic)))
        async with websockets.serve(self.handler, self.host, self.port):
            await self._stop
        # Stop all broadcast tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("Server successfully stopped.")

    def run(self):
        """Runs the WebSocket server."""
        asyncio.run(self.start())
