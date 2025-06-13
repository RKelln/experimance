"""
ZMQ utility enhancement module that avoids hanging on socket operations.
"""

import asyncio
import json
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias, Union, cast

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
    HEARTBEAT_TOPIC,
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
    ALERT = "Alert"
    # Display service message types
    TEXT_OVERLAY = "TextOverlay"
    REMOVE_TEXT = "RemoveText"
    VIDEO_MASK = "VideoMask"
    # Add more message types as needed


class ZmqException(Exception):
    """Base exception for ZMQ-related errors."""
    pass


class ZmqTimeoutError(ZmqException):
    """Exception raised when a ZMQ operation times out."""
    pass

TopicType: TypeAlias = str | MessageType

class ZmqBase:
    """Base class for ZMQ socket wrappers."""
    
    address: str
    use_asyncio: bool
    closed: bool
    context: Optional[Union[zmq.Context, zmq.asyncio.Context]]
    socket: Optional[Union[zmq.Socket, zmq.asyncio.Socket]]
    
    def __init__(self, address: str, use_asyncio: bool = True):
        """Initialize the base ZMQ object.
        
        Args:
            address: The ZMQ address to connect/bind to
            use_asyncio: Whether to use asyncio context
        """
        self.address = address
        self.use_asyncio = use_asyncio
        self.closed = False
        
        # Initialize context
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
                # Setting linger to 0 ensures socket closes immediately without waiting
                # This is critical for clean shutdown with asyncio
                self.socket.close(linger=0)
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
                
        if self.context:
            try:
                # Setting the context to None before terminating helps prevent errors
                # during shutdown with asyncio
                context = self.context
                self.context = None
                context.term()
            except Exception as e:
                logger.error(f"Error terminating context: {e}")
                
        self.closed = True


class ZmqPublisher(ZmqBase):
    """A ZeroMQ publisher that sends messages on a specific topic."""
    topic: str

    def __init__(self, address: str, topic: TopicType = "", use_asyncio: bool = True):
        """Initialize a ZeroMQ publisher.
        
        Args:
            address: ZeroMQ address to bind to
            topic: Topic to publish on (can be a string or MessageType enum)
            use_asyncio: Whether to use asyncio
        """
        super().__init__(address, use_asyncio)
        self.topic = topic_to_str(topic)  # Convert topic to string
        
        assert self.context is not None, "ZMQ context was not properly initialized"
        
        # Create the socket
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(address)
        
        # Set timeout for send operations
        if not use_asyncio:
            self.socket.setsockopt(zmq.SNDTIMEO, DEFAULT_TIMEOUT)
        
        logger.debug(f"Publisher bound to {address} on topic '{self.topic}'")
    
    def publish(self, message: Dict[str, Any], topic: Optional[TopicType] = None) -> bool:
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
            assert self.socket is not None, "ZMQ socket was not properly initialized"
            json_message = json.dumps(message)
            topic = topic_to_str(topic) if topic else self.topic
            self.socket.send_string(f"{topic} {json_message}")
            return True
        except zmq.error.Again:
            logger.warning("Publish operation timed out")
            return False
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    async def publish_async(self, message: Dict[str, Any], topic: Optional[TopicType] = None) -> bool:
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
            topic = topic_to_str(topic) if topic else self.topic
            # Use wait_for to add a timeout
            await asyncio.wait_for(
                self.socket.send_string(f"{topic} {json_message}"), # type: ignore
                timeout=DEFAULT_RECV_TIMEOUT
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Async publish operation timed out")
            return False
        except Exception as e:
            logger.error(f"Error publishing message asynchronously: {e}")
            return False

    def __str__(self) -> str:
        """String representation of the publisher."""
        return f"ZmqPublisher(address={self.address}, topic={self.topic})"

class ZmqSubscriber(ZmqBase):
    """A ZeroMQ subscriber that receives messages on specific topics.

        subscriber = ZmqSubscriber("tcp://localhost:5555", ["status-updates"])
    """
    topics: List[str]

    def __init__(self, address: str, topics: List[str|MessageType] = ["*"], use_asyncio: bool = True):
        """Initialize a ZeroMQ subscriber.
        
        Args:
            address: ZeroMQ address to connect to
            topics: List of topics to subscribe to
            use_asyncio: Whether to use asyncio
        """
        super().__init__(address, use_asyncio)

        self.topics = topics_to_strs(topics)  # Convert topics to string representations

        # Create the socket
        assert self.context is not None, "ZMQ context was not properly initialized"
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(address)
        
        # Set timeout for receive operations
        if not use_asyncio:
            self.socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)
        
        # Subscribe to topics
        for topic in self.topics:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        
        logger.debug(f"Subscriber connected to {address} with topics {self.topics}")
    
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
        except asyncio.CancelledError:
            # Quietly handle task cancellation during shutdown
            logger.debug("Receive operation was cancelled (likely during shutdown)")
            return "", {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message: {e}")
            return "", {}
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return "", {}
        
    def __str__(self) -> str:
        """String representation of the subscriber."""
        return f"ZmqSubscriber(address={self.address}, topics={self.topics})"


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
        assert self.context is not None, "ZMQ context was not properly initialized"
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
            assert self.socket is not None, "ZMQ socket was not properly initialized"
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
        assert self.context is not None, "ZMQ context was not properly initialized"
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
        except asyncio.CancelledError:
            # Quietly handle task cancellation during shutdown
            logger.debug("Pull operation was cancelled (likely during shutdown)")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error pulling message: {e}")
            return {}


class ZmqBindingPullSocket(ZmqBase):
    """A ZeroMQ pull socket that binds instead of connects.
    
    This is needed for controllers or central services that need to receive messages 
    from multiple workers. Unlike the standard ZmqPullSocket which connects to a 
    bound socket, this socket binds to an address and allows multiple clients to 
    connect to it.
    
    Key differences from ZmqPullSocket:
    - Uses bind() instead of connect()
    - Suitable for fan-in communication patterns where multiple senders report to one receiver
    - Typically used in controller/coordinator services to collect results from workers
    """
    
    def __init__(self, address: str, use_asyncio: bool = True):
        """Initialize a ZeroMQ pull socket that binds.
        
        Args:
            address: ZeroMQ address to bind to
            use_asyncio: Whether to use asyncio
        """
        super().__init__(address, use_asyncio)
        
        # Create the socket
        assert self.context is not None, "ZMQ context was not properly initialized"
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(address)  # bind instead of connect
        
        # Set timeout for receive operations
        if not use_asyncio:
            self.socket.setsockopt(zmq.RCVTIMEO, DEFAULT_TIMEOUT)
        
        logger.debug(f"Pull socket bound to {address}")
    
    # Re-use the pull and pull_async methods from ZmqPullSocket
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
        except asyncio.CancelledError:
            # Quietly handle task cancellation during shutdown
            logger.debug("Pull operation was cancelled (likely during shutdown)")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error pulling message: {e}")
            return {}


class ZmqConnectingPushSocket(ZmqBase):
    """A ZeroMQ push socket that explicitly connects instead of binds.
    
    This is needed for workers to connect to a controller's binding pull socket.
    Unlike the standard ZmqPushSocket which binds to an address, this socket 
    connects to a bound socket, allowing multiple instances to connect to 
    the same endpoint.
    
    Key differences from ZmqPushSocket:
    - Uses connect() instead of bind()
    - Suitable for fan-in communication patterns where multiple senders report to one receiver
    - Typically used in worker services to send results back to a controller
    """
    
    def __init__(self, address: str, use_asyncio: bool = True):
        """Initialize a ZeroMQ push socket that connects.
        
        Args:
            address: ZeroMQ address to connect to
            use_asyncio: Whether to use asyncio
        """
        super().__init__(address, use_asyncio)
        
        # Create the socket
        assert self.context is not None, "ZMQ context was not properly initialized"
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(address)  # connect instead of bind
        
        # Set timeout for send operations
        if not use_asyncio:
            self.socket.setsockopt(zmq.SNDTIMEO, DEFAULT_TIMEOUT)
        
        logger.debug(f"Push socket connected to {address}")
    
    def push(self, message: Dict[str, Any]) -> bool:
        """Push a message to the socket.
        
        Args:
            message: The message to push (will be serialized to JSON)
            
        Returns:
            True if successful, False otherwise
        """
        if self.closed:
            logger.error("Attempted to push to closed socket")
            return False
            
        try:
            message_str = json.dumps(message)
            self.socket.send_string(message_str) # type: ignore
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
            logger.error("Attempted to push to closed socket")
            return False
            
        try:
            message_str = json.dumps(message)
            # Use wait_for to add a timeout
            await asyncio.wait_for(
                self.socket.send_string(message_str), # type: ignore
                timeout=DEFAULT_RECV_TIMEOUT
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Async push operation timed out")
            return False
        except Exception as e:
            logger.error(f"Error pushing message: {e}")
            return False


def topic_to_str(topic: str | MessageType) -> str:
    """Convert a topic to their string representation."""
    if isinstance(topic, MessageType):
        return topic.value
    return str(topic)

def topics_to_strs(topics: List[str | MessageType]) -> List[str]:
    """Convert a list of topics to their string representations.
    
    Args:
        topics: List of topics to convert (can contain strings or MessageType enums)
        
    Returns:
        List of string representations of the topics
    """
    return [topic_to_str(topic) for topic in topics]
