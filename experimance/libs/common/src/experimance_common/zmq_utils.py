"""
Common ZeroMQ utilities for Experimance services.
"""

import asyncio
import json
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import zmq
import zmq.asyncio

from experimance_common.constants import DEFAULT_PORTS, DEFAULT_TIMEOUT, HEARTBEAT_INTERVAL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Message types used in the Experimance system."""
    ERA_CHANGED = "EraChanged"
    RENDER_REQUEST = "RenderRequest"
    IDLE_STATUS = "IdleStatus"
    IMAGE_READY = "ImageReady"
    TRANSITION_READY = "TransitionReady"
    LOOP_READY = "LoopReady"
    AGENT_CONTROL_EVENT = "AgentControlEvent"
    TRANSITION_REQUEST = "TransitionRequest"
    LOOP_REQUEST = "LoopRequest"
    HEARTBEAT = "Heartbeat"

# Use the constants from constants.py
# No need to redefine here


class ZmqPublisher:
    """A ZeroMQ publisher that sends messages on a specific topic."""
    
    def __init__(self, address: str, topic: str = "", use_asyncio: bool = True):
        """Initialize a ZeroMQ publisher.
        
        Args:
            address: ZeroMQ address to bind to
            topic: Topic to publish on
            use_asyncio: Whether to use asyncio (default: True)
        """
        self.address = address
        self.topic = topic
        self.use_asyncio = use_asyncio
        
        if use_asyncio:
            self.context = zmq.asyncio.Context()
        else:
            self.context = zmq.Context()
            
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)  # Only keep latest message
        self.socket.setsockopt(zmq.HEARTBEAT_IVL, int(HEARTBEAT_INTERVAL * 1000))  # Convert to ms
        self.socket.bind(address)
        logger.info(f"Publisher bound to {address} with topic '{topic}'")
        
    async def publish_async(self, message: Dict[str, Any]) -> None:
        """Publish a message asynchronously.
        
        Args:
            message: Message to publish (will be JSON encoded)
        """
        if not self.use_asyncio:
            raise RuntimeError("Cannot use publish_async with use_asyncio=False")
        
        message_json = json.dumps(message).encode('utf-8')
        topic_bytes = self.topic.encode('utf-8') if self.topic else b''
        
        await self.socket.send_multipart([topic_bytes, message_json])
        logger.debug(f"Published message: {message}")
        
    def publish(self, message: Dict[str, Any]) -> None:
        """Publish a message synchronously.
        
        Args:
            message: Message to publish (will be JSON encoded)
        """
        if self.use_asyncio:
            raise RuntimeError("Cannot use publish with use_asyncio=True")
            
        message_json = json.dumps(message).encode('utf-8')
        topic_bytes = self.topic.encode('utf-8') if self.topic else b''
        
        self.socket.send_multipart([topic_bytes, message_json])
        logger.debug(f"Published message: {message}")
        
    def close(self) -> None:
        """Close the publisher socket."""
        self.socket.close()
        self.context.term()


class ZmqSubscriber:
    """A ZeroMQ subscriber that receives messages on specific topics."""
    
    def __init__(self, address: str, topics: List[str] = None, use_asyncio: bool = True):
        """Initialize a ZeroMQ subscriber.
        
        Args:
            address: ZeroMQ address to connect to
            topics: List of topics to subscribe to (empty list = all topics)
            use_asyncio: Whether to use asyncio (default: True)
        """
        self.address = address
        self.topics = topics or []
        self.use_asyncio = use_asyncio
        
        if use_asyncio:
            self.context = zmq.asyncio.Context()
        else:
            self.context = zmq.Context()
            
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)  # Only keep latest message
        self.socket.setsockopt(zmq.HEARTBEAT_IVL, int(HEARTBEAT_INTERVAL * 1000))  # Convert to ms
        
        # Subscribe to specified topics or all if none specified
        if not self.topics:
            self.socket.setsockopt(zmq.SUBSCRIBE, b'')
            logger.info(f"Subscriber connected to {address} with subscription to all topics")
        else:
            for topic in self.topics:
                self.socket.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
            topic_list = ", ".join(self.topics)
            logger.info(f"Subscriber connected to {address} with subscriptions to {topic_list}")
            
        self.socket.connect(address)
        
    async def receive_async(self) -> tuple[str, Dict[str, Any]]:
        """Receive a message asynchronously.
        
        Returns:
            Tuple of (topic, message)
        """
        if not self.use_asyncio:
            raise RuntimeError("Cannot use receive_async with use_asyncio=False")
            
        [topic_bytes, message_json] = await self.socket.recv_multipart()
        topic = topic_bytes.decode('utf-8')
        message = json.loads(message_json.decode('utf-8'))
        logger.debug(f"Received message on topic '{topic}': {message}")
        return topic, message
        
    def receive(self) -> tuple[str, Dict[str, Any]]:
        """Receive a message synchronously.
        
        Returns:
            Tuple of (topic, message)
        """
        if self.use_asyncio:
            raise RuntimeError("Cannot use receive with use_asyncio=True")
            
        [topic_bytes, message_json] = self.socket.recv_multipart()
        topic = topic_bytes.decode('utf-8')
        message = json.loads(message_json.decode('utf-8'))
        logger.debug(f"Received message on topic '{topic}': {message}")
        return topic, message
        
    def close(self) -> None:
        """Close the subscriber socket."""
        self.socket.close()
        self.context.term()


class ZmqPushSocket:
    """A ZeroMQ PUSH socket for distributing work."""
    
    def __init__(self, address: str, use_asyncio: bool = True):
        """Initialize a ZeroMQ PUSH socket.
        
        Args:
            address: ZeroMQ address to bind to
            use_asyncio: Whether to use asyncio (default: True)
        """
        self.address = address
        self.use_asyncio = use_asyncio
        
        if use_asyncio:
            self.context = zmq.asyncio.Context()
        else:
            self.context = zmq.Context()
            
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.HEARTBEAT_IVL, int(HEARTBEAT_INTERVAL * 1000))  # Convert to ms
        self.socket.bind(address)
        logger.info(f"PUSH socket bound to {address}")
        
    async def push_async(self, message: Dict[str, Any]) -> None:
        """Push a message asynchronously.
        
        Args:
            message: Message to push (will be JSON encoded)
        """
        if not self.use_asyncio:
            raise RuntimeError("Cannot use push_async with use_asyncio=False")
            
        message_json = json.dumps(message).encode('utf-8')
        await self.socket.send(message_json)
        logger.debug(f"Pushed message: {message}")
        
    def push(self, message: Dict[str, Any]) -> None:
        """Push a message synchronously.
        
        Args:
            message: Message to push (will be JSON encoded)
        """
        if self.use_asyncio:
            raise RuntimeError("Cannot use push with use_asyncio=True")
            
        message_json = json.dumps(message).encode('utf-8')
        self.socket.send(message_json)
        logger.debug(f"Pushed message: {message}")
        
    def close(self) -> None:
        """Close the PUSH socket."""
        self.socket.close()
        self.context.term()


class ZmqPullSocket:
    """A ZeroMQ PULL socket for receiving distributed work."""
    
    def __init__(self, address: str, use_asyncio: bool = True):
        """Initialize a ZeroMQ PULL socket.
        
        Args:
            address: ZeroMQ address to connect to
            use_asyncio: Whether to use asyncio (default: True)
        """
        self.address = address
        self.use_asyncio = use_asyncio
        
        if use_asyncio:
            self.context = zmq.asyncio.Context()
        else:
            self.context = zmq.Context()
            
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.HEARTBEAT_IVL, int(HEARTBEAT_INTERVAL * 1000))  # Convert to ms
        self.socket.connect(address)
        logger.info(f"PULL socket connected to {address}")
        
    async def pull_async(self) -> Dict[str, Any]:
        """Pull a message asynchronously.
        
        Returns:
            Received message
        """
        if not self.use_asyncio:
            raise RuntimeError("Cannot use pull_async with use_asyncio=False")
            
        message_json = await self.socket.recv()
        message = json.loads(message_json.decode('utf-8'))
        logger.debug(f"Pulled message: {message}")
        return message
        
    def pull(self) -> Dict[str, Any]:
        """Pull a message synchronously.
        
        Returns:
            Received message
        """
        if self.use_asyncio:
            raise RuntimeError("Cannot use pull with use_asyncio=True")
            
        message_json = self.socket.recv()
        message = json.loads(message_json.decode('utf-8'))
        logger.debug(f"Pulled message: {message}")
        return message
        
    def close(self) -> None:
        """Close the PULL socket."""
        self.socket.close()
        self.context.term()
