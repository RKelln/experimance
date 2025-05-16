"""
ZMQ utility enhancement module that avoids hanging on socket operations.
"""

import asyncio
import json
import logging
import selectors
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import zmq
import zmq.asyncio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from experimance_common.constants import (
    DEFAULT_TIMEOUT,
    HEARTBEAT_INTERVAL,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RECV_TIMEOUT,
)


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


class ZmqException(Exception):
    """Base exception for ZMQ-related errors."""
    pass


class ZmqTimeoutError(ZmqException):
    """Exception raised when a ZMQ operation times out."""
    pass


class ZmqBase:
    """Base class for ZMQ socket wrappers."""
    
    def __init__(self, address: str, use_asyncio: bool = True):
        """Initialize the base ZMQ object.
        
        Args:
            address: The ZMQ address to connect/bind to
            use_asyncio: Whether to use asyncio context
        """
        self.address = address
        self.use_asyncio = use_asyncio
        self.closed = False
        
        if use_asyncio:
            self.context = zmq.asyncio.Context()
        else:
            self.context = zmq.Context()
        
        # Set socket to None initially
        self.socket = None
        
    def close(self):
        """Close the socket and terminate the context."""
        if self.closed:
            return
            
        if self.socket:
            try:
                self.socket.close(linger=0)
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
                
        if self.context:
            try:
                self.context.term()
            except Exception as e:
                logger.error(f"Error terminating context: {e}")
                
        self.closed = True


class ZmqPublisher(ZmqBase):
    """A ZeroMQ publisher that sends messages on a specific topic."""
    
    def __init__(self, address: str, topic: str = "", use_asyncio: bool = True):
        """Initialize a ZeroMQ publisher.
        
        Args:
            address: ZeroMQ address to bind to
            topic: Topic to publish on
            use_asyncio: Whether to use asyncio
        """
        super().__init__(address, use_asyncio)
        self.topic = topic
        
        # Create the socket
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(address)
        
        # Set timeout for send operations
        if not use_asyncio:
            self.socket.setsockopt(zmq.SNDTIMEO, DEFAULT_TIMEOUT)
        
        logger.debug(f"Publisher bound to {address} on topic '{topic}'")
    
    def publish(self, message: Dict[str, Any]) -> bool:
        """Publish a message on the topic.
        
        Args:
            message: The message to publish (will be serialized to JSON)
            
        Returns:
            True if successful, False otherwise
        """
        if self.closed:
            logger.error("Attempted to publish on closed socket")
            return False
            
        try:
            json_message = json.dumps(message)
            self.socket.send_string(f"{self.topic} {json_message}")
            return True
        except zmq.error.Again:
            logger.warning("Publish operation timed out")
            return False
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    async def publish_async(self, message: Dict[str, Any]) -> bool:
        """Publish a message asynchronously on the topic.
        
        Args:
            message: The message to publish (will be serialized to JSON)
            
        Returns:
            True if successful, False otherwise
        """
        if self.closed:
            logger.error("Attempted to publish on closed socket")
            return False
            
        try:
            json_message = json.dumps(message)
            # Use wait_for to add a timeout
            await asyncio.wait_for(
                self.socket.send_string(f"{self.topic} {json_message}"), # type: ignore
                timeout=DEFAULT_RECV_TIMEOUT
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Async publish operation timed out")
            return False
        except Exception as e:
            logger.error(f"Error publishing message asynchronously: {e}")
            return False


class ZmqSubscriber(ZmqBase):
    """A ZeroMQ subscriber that receives messages on specific topics."""
    
    def __init__(self, address: str, topics: List[str], use_asyncio: bool = True):
        """Initialize a ZeroMQ subscriber.
        
        Args:
            address: ZeroMQ address to connect to
            topics: List of topics to subscribe to
            use_asyncio: Whether to use asyncio
        """
        super().__init__(address, use_asyncio)
        self.topics = topics
        
        # Create the socket
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(address)
        
        # Set timeout for receive operations
        if not use_asyncio:
            self.socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)
        
        # Subscribe to topics
        for topic in topics:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        
        logger.debug(f"Subscriber connected to {address} with topics {topics}")
    
    def receive(self) -> Tuple[str, Dict[str, Any]]:
        """Receive a message from the subscribed topics.
        
        Returns:
            Tuple of (topic, message)
            
        Raises:
            ZmqTimeoutError: If the receive operation times out
        """
        if self.closed:
            raise ZmqException("Attempted to receive on closed socket")
            
        try:
            message_str : str = self.socket.recv_string() # type: ignore
            space_index = message_str.find(" ") 
            
            if space_index == -1:
                logger.warning(f"Received malformed message: {message_str}")
                return "", {}
                
            topic = message_str[:space_index]
            message_json = message_str[space_index + 1:]
            message = json.loads(message_json)
            
            return topic, message
        except zmq.error.Again:
            raise ZmqTimeoutError("Receive operation timed out")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message: {e}")
            return "", {}
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return "", {}
    
    async def receive_async(self) -> Tuple[str, Dict[str, Any]]:
        """Receive a message asynchronously from the subscribed topics.
        
        Returns:
            Tuple of (topic, message)
            
        Raises:
            ZmqTimeoutError: If the receive operation times out
        """
        if self.closed:
            raise ZmqException("Attempted to receive on closed socket")
            
        try:
            # Use wait_for to add a timeout
            message_str = await asyncio.wait_for(
                self.socket.recv_string(), # type: ignore
                timeout=DEFAULT_RECV_TIMEOUT
            )
            
            space_index = message_str.find(" ")
            
            if space_index == -1:
                logger.warning(f"Received malformed message: {message_str}")
                return "", {}
                
            topic = message_str[:space_index]
            message_json = message_str[space_index + 1:]
            message = json.loads(message_json)
            
            return topic, message
        except asyncio.TimeoutError:
            raise ZmqTimeoutError("Async receive operation timed out")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message: {e}")
            return "", {}
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return "", {}


class ZmqPushSocket(ZmqBase):
    """A ZeroMQ push socket that sends messages to a pull socket."""
    
    def __init__(self, address: str, use_asyncio: bool = True):
        """Initialize a ZeroMQ push socket.
        
        Args:
            address: ZeroMQ address to bind to
            use_asyncio: Whether to use asyncio
        """
        super().__init__(address, use_asyncio)
        
        # Create the socket
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(address)
        
        # Set timeout for send operations
        if not use_asyncio:
            self.socket.setsockopt(zmq.SNDTIMEO, DEFAULT_TIMEOUT)
        
        logger.debug(f"Push socket bound to {address}")
    
    def push(self, message: Dict[str, Any]) -> bool:
        """Push a message to the socket.
        
        Args:
            message: The message to push (will be serialized to JSON)
            
        Returns:
            True if successful, False otherwise
        """
        if self.closed:
            logger.error("Attempted to push on closed socket")
            return False
            
        try:
            json_message = json.dumps(message)
            self.socket.send_string(json_message)
            return True
        except zmq.error.Again:
            logger.warning("Push operation timed out")
            return False
        except Exception as e:
            logger.error(f"Error pushing message: {e}")
            return False
    
    async def push_async(self, message: Dict[str, Any]) -> bool:
        """Push a message asynchronously to the socket.
        
        Args:
            message: The message to push (will be serialized to JSON)
            
        Returns:
            True if successful, False otherwise
        """
        if self.closed:
            logger.error("Attempted to push on closed socket")
            return False
            
        try:
            json_message = json.dumps(message)
            # Use wait_for to add a timeout
            await asyncio.wait_for(
                self.socket.send_string(json_message), # type: ignore
                timeout=DEFAULT_RECV_TIMEOUT
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Async push operation timed out")
            return False
        except Exception as e:
            logger.error(f"Error pushing message asynchronously: {e}")
            return False


class ZmqPullSocket(ZmqBase):
    """A ZeroMQ pull socket that receives messages from a push socket."""
    
    def __init__(self, address: str, use_asyncio: bool = True):
        """Initialize a ZeroMQ pull socket.
        
        Args:
            address: ZeroMQ address to connect to
            use_asyncio: Whether to use asyncio
        """
        super().__init__(address, use_asyncio)
        
        # Create the socket
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect(address)
        
        # Set timeout for receive operations
        if not use_asyncio:
            self.socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)
        
        logger.debug(f"Pull socket connected to {address}")
    
    def pull(self) -> Dict[str, Any]:
        """Pull a message from the socket.
        
        Returns:
            The received message
            
        Raises:
            ZmqTimeoutError: If the receive operation times out
        """
        if self.closed:
            raise ZmqException("Attempted to pull from closed socket")
            
        try:
            message_str : str = self.socket.recv_string() # type: ignore
            return json.loads(message_str)
        except zmq.error.Again:
            raise ZmqTimeoutError("Pull operation timed out")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error pulling message: {e}")
            return {}
    
    async def pull_async(self) -> Dict[str, Any]:
        """Pull a message asynchronously from the socket.
        
        Returns:
            The received message
            
        Raises:
            ZmqTimeoutError: If the receive operation times out
        """
        if self.closed:
            raise ZmqException("Attempted to pull from closed socket")
            
        try:
            # Use wait_for to add a timeout
            message_str = await asyncio.wait_for(
                self.socket.recv_string(), # type: ignore
                timeout=DEFAULT_RECV_TIMEOUT
            )
            return json.loads(message_str)
        except asyncio.TimeoutError:
            raise ZmqTimeoutError("Async pull operation timed out")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error pulling message: {e}")
            return {}
