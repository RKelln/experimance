"""
ZMQ Components - Composition-Based Architecture

Production-ready ZMQ components using composition instead of inheritance.
These components are designed to be robust, error-free, and integrate seamlessly
with the existing Experimance service architecture.

Key Features:
- Pattern 1: Direct config injection for type safety
- Robust async context management with guaranteed cleanup
- Comprehensive error handling and recovery
- Proper ZMQ socket lifecycle management
- Integration with existing logging system
- Message handling with error recovery
- Type-safe throughout
"""

import asyncio
import logging
from experimance_common.constants import DEFAULT_RECV_TIMEOUT
from experimance_common.schemas import MessageBase
import zmq
import zmq.asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Callable, Any, Union, Awaitable
from contextlib import asynccontextmanager

from experimance_common.zmq.config import (
    MessageDataType, TopicType, ZmqSocketConfig, PublisherConfig, SubscriberConfig, 
    PushConfig, PullConfig, MessageType
)

logger = logging.getLogger(__name__)


class ZmqComponentError(Exception):
    """Base exception for ZMQ component errors."""
    pass


class ComponentNotRunningError(ZmqComponentError):
    """Raised when attempting to use a component that isn't running."""
    pass


class BaseZmqComponent(ABC):
    """
    Abstract base class for all ZMQ components.
    
    Provides bulletproof async lifecycle management, error handling,
    and socket cleanup. All subclasses get robust foundation.
    """
    
    def __init__(self, config: ZmqSocketConfig, context: Optional[zmq.asyncio.Context] = None):
        """
        Initialize ZMQ component.
        
        Args:
            config: Socket configuration (validated by Pydantic)
            context: ZMQ context (creates new one if None)
        """
        self.config = config
        self.context = context or zmq.asyncio.Context()
        self.socket: Optional[zmq.asyncio.Socket] = None
        self.running = False
        
        # Component identification for logging
        self.component_type = self.__class__.__name__
        self.component_id = f"{self.component_type}_{id(self)}"
        
        # Setup component-specific logger
        self.logger = logging.getLogger(f"zmq.{self.component_type}")
        
        # Task tracking for cleanup
        self._tasks: Set[asyncio.Task] = set()
        
        self.logger.debug(f"Initialized {self.component_id} for {config.full_address}")
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with guaranteed cleanup."""
        await self.stop()
        
    async def start(self) -> None:
        """
        Start the component and create ZMQ socket.
        
        Raises:
            ZmqComponentError: If component is already running
            zmq.ZMQError: If socket creation or binding/connecting fails
        """
        if self.running:
            raise ZmqComponentError(f"{self.component_id} is already running")
            
        try:
            self.logger.info(f"Starting {self.component_id}")
            
            # Create socket with proper type
            socket_type = self._get_socket_type()
            self.socket = self.context.socket(socket_type)
            
            # Apply socket options from config
            self._apply_socket_options()
            
            # Bind or connect based on config
            if self.config.bind:
                self.socket.bind(self.config.full_address)
                self.logger.debug(f"{self.component_id} bound to {self.config.full_address}")
            else:
                self.socket.connect(self.config.full_address)
                self.logger.debug(f"{self.component_id} connected to {self.config.full_address}")
                
            # Component-specific startup
            await self._component_start()
            
            self.running = True
            self.logger.info(f"{self.component_id} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start {self.component_id}: {e}")
            await self._cleanup_socket()  # Ensure cleanup on failure
            raise ZmqComponentError(f"Failed to start {self.component_id}: {e}") from e
            
    async def stop(self) -> None:
        """
        Stop the component and cleanup all resources.
        
        This method is safe to call multiple times and guarantees cleanup.
        """
        if not self.running:
            return
            
        self.logger.info(f"Stopping {self.component_id}")
        
        try:
            # Component-specific cleanup first
            await self._component_stop()
            
            # Cancel and wait for all tasks
            await self._cleanup_tasks()
            
            # Clean up socket
            await self._cleanup_socket()
            
            self.running = False
            self.logger.info(f"{self.component_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping {self.component_id}: {e}")
            # Continue cleanup even if there are errors
            self.running = False
            
    @abstractmethod
    def _get_socket_type(self) -> int:
        """Return the ZMQ socket type for this component."""
        pass
        
    async def _component_start(self) -> None:
        """Component-specific startup logic. Override in subclasses."""
        pass
        
    async def _component_stop(self) -> None:
        """Component-specific cleanup logic. Override in subclasses."""
        pass
        
    def _apply_socket_options(self) -> None:
        """Apply socket options from config."""
        if not self.socket:
            return
            
        for option_name, value in self.config.socket_options.items():
            try:
                # Convert string option name to ZMQ constant
                option = getattr(zmq, option_name.upper())
                self.socket.setsockopt(option, value)
                self.logger.debug(f"Set socket option {option_name}={value}")
            except AttributeError:
                self.logger.warning(f"Unknown socket option: {option_name}")
            except Exception as e:
                self.logger.warning(f"Failed to set socket option {option_name}={value}: {e}")
                
    async def _cleanup_socket(self) -> None:
        """Clean up the ZMQ socket with error handling."""
        if self.socket and not self.socket.closed:
            try:
                # Close socket gracefully
                self.socket.close()
                self.logger.debug(f"{self.component_id} socket closed")
            except Exception as e:
                self.logger.warning(f"Error closing socket for {self.component_id}: {e}")
            finally:
                self.socket = None
                
    async def _cleanup_tasks(self) -> None:
        """Cancel all component tasks and wait for completion."""
        if not self._tasks:
            return
            
        self.logger.debug(f"Cancelling {len(self._tasks)} tasks for {self.component_id}")
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                
        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for {self.component_id} tasks to cancel")
            
        # Clear task set
        self._tasks.clear()
        
    def _add_task(self, task: asyncio.Task) -> None:
        """Add a task to be tracked for cleanup."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        
    def _ensure_running(self) -> None:
        """Ensure component is running, raise error if not."""
        if not self.running or not self.socket:
            raise ComponentNotRunningError(f"{self.component_id} is not running")
            
    @property
    def is_running(self) -> bool:
        """Check if component is running and has valid socket."""
        return self.running and self.socket is not None and not self.socket.closed


class PublisherComponent(BaseZmqComponent):
    """
    Publisher component for PUB socket.
    
    Handles message publishing with error recovery and proper formatting.
    """
    
    def __init__(self, config: PublisherConfig, context: Optional[zmq.asyncio.Context] = None):
        super().__init__(config, context)
        self._pub_config = config
        
    def _get_socket_type(self) -> int:
        return zmq.PUB
        
    async def publish(self, message: MessageDataType, topic: Optional[TopicType] = None) -> str:
        """
        Publish a message to the specified topic or default topic.
        
        Args:
            message: Message data (will be serialized)
            topic: Message topic (optional, uses default_topic or extracts from message)
        
        Returns: The topic published on or raises an error if no topic can be resolved.

        Raises:
            ComponentNotRunningError: If component is not running
            ZmqComponentError: If publish fails
        """
        self._ensure_running()
        
        try:
            # Resolve the topic using flexible topic handling
            final_topic = self._resolve_topic(message, topic)
            
            # Serialize message to JSON (handle both MessageBase and dict)
            if isinstance(message, MessageBase):
                json_message = message.model_dump_json()
            else:
                import json
                json_message = json.dumps(message)
            
            # Send in same format as ZmqPublisher: "topic json_message"
            if self.socket and not self.socket.closed:
                await self.socket.send_string(f"{final_topic} {json_message}")
                self.logger.debug(f"Published message to topic '{final_topic}': {len(json_message)} bytes")
            
            return final_topic
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            raise ZmqComponentError(f"Publish failed: {e}") from e
            
    def _resolve_topic(self, message: MessageDataType, topic: Optional[TopicType]) -> str:
        """
        Resolve the topic for a message using flexible topic handling.
        Priority: explicit topic -> message type -> default topic -> empty
        """
        # 1. Use explicit topic if provided
        if topic is not None:
            return str(topic)
            
        # 2. Extract topic from message if it has a message_type
        if isinstance(message, dict) and 'type' in message:
            return message['type']
        if isinstance(message, MessageBase):
            return str(message.type)
            
        # 3. Use default topic if configured
        if hasattr(self._pub_config, 'default_topic') and self._pub_config.default_topic is not None:
            return str(self._pub_config.default_topic)
            
        raise ZmqComponentError(
            f"No topic provided and no default topic configured for {self.component_id}. "  
            "Please provide a topic or configure a default topic."
        )


class SubscriberComponent(BaseZmqComponent):
    """
    Subscriber component for SUB socket.
    
    Handles topic subscription and message receiving with async message handling.
    """
    
    def __init__(self, config: SubscriberConfig, context: Optional[zmq.asyncio.Context] = None):
        super().__init__(config, context)
        self._sub_config = config
        
        # Per-topic handlers and default handler for unmatched topics
        self.topic_handlers: Dict[str, Callable[[MessageDataType], Union[None, Awaitable[None]]]] = {}
        self.default_handler: Optional[Callable[[str, MessageDataType], Union[None, Awaitable[None]]]] = None
        
        self._listen_task: Optional[asyncio.Task] = None
        
    def _get_socket_type(self) -> int:
        return zmq.SUB
        
    async def _component_start(self) -> None:
        """Subscribe to topics and start message listening."""
        # Subscribe to all configured topics
        for topic in self._sub_config.topics:
            if self.socket and not self.socket.closed:
                self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
                self.logger.debug(f"Subscribed to topic: {topic}")
            
        # Start listening task
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._add_task(self._listen_task)
        
    async def _component_stop(self) -> None:
        """Stop listening for messages."""
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            
    def register_handler(self, topic: str, handler: Callable[[MessageDataType], Union[None, Awaitable[None]]]) -> None:
        """
        Register a handler for a specific topic.
        
        Args:
            topic: Topic to handle messages for
            handler: Function that takes (message) parameter where message is MessageDataType.
                    Can be sync or async.
        """
        self.topic_handlers[topic] = handler
        self.logger.debug(f"Registered handler for topic '{topic}'")
        
    def unregister_handler(self, topic: str) -> None:
        """Remove a handler for a specific topic."""
        if topic in self.topic_handlers:
            del self.topic_handlers[topic]
            self.logger.debug(f"Unregistered handler for topic '{topic}'")
        else:
            self.logger.warning(f"No handler found for topic '{topic}' to unregister")
            
    def set_default_handler(self, handler: Callable[[str, MessageDataType], Union[None, Awaitable[None]]]) -> None:
        """
        Set a default handler for topics that don't have specific handlers.
        
        Args:
            handler: Function that takes (topic, message) parameters.
                    Can be sync or async.
        """
        self.default_handler = handler
        self.logger.debug("Default handler set")
        
    def clear_handlers(self) -> None:
        """Clear all topic handlers and default handler."""
        self.topic_handlers.clear()
        self.default_handler = None
        self.logger.debug("All handlers cleared")
        
    async def _listen_loop(self) -> None:
        """
        Main message listening loop with error recovery.
        
        Continues listening even if individual message processing fails.
        """
        self.logger.debug(f"Starting message listen loop for {self.component_id}")
        
        while self.running:
            try:
                # Parse message (expecting "topic json_message" format)
                try:
                    if not self.socket or self.socket.closed:
                        break
                    
                    # Receive as string (matches publisher format)
                    # Use timeout to allow graceful shutdown but don't log timeout as error
                    try:
                        raw_message = await asyncio.wait_for(
                            self.socket.recv_string(),
                            timeout=DEFAULT_RECV_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        # Normal timeout - no messages available, continue loop
                        continue
                    
                    # Split topic and JSON message
                    if ' ' in raw_message:
                        topic, json_message = raw_message.split(' ', 1)
                    else:
                        topic = ""
                        json_message = raw_message
                    
                    # Parse JSON to dict
                    import json
                    message_dict = json.loads(json_message)
                    
                    # Convert to MessageBase if possible, otherwise keep as dict
                    message = MessageBase.from_dict(message_dict)
                    
                    self.logger.debug(f"Received message on topic '{topic}': {len(json_message)} bytes")
                    
                    # Call per-topic handler if registered
                    if topic in self.topic_handlers:
                        handler = self.topic_handlers[topic]
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message)
                            else:
                                handler(message)
                        except Exception as e:
                            self.logger.error(f"Error in handler for topic '{topic}': {e}")
                    
                    # Call default handler if no specific handler found
                    elif self.default_handler:
                        try:
                            if asyncio.iscoroutinefunction(self.default_handler):
                                await self.default_handler(topic, message)
                            else:
                                self.default_handler(topic, message)
                        except Exception as e:
                            self.logger.error(f"Error in default handler for topic '{topic}': {e}")
                    
                    else:
                        self.logger.warning(f"No handler registered for topic '{topic}'")
                            
                except Exception as e:
                    self.logger.error(f"Error parsing message: {e}")
                    
            except asyncio.CancelledError:
                self.logger.debug(f"Listen loop cancelled for {self.component_id}")
                break
            except Exception as e:
                self.logger.error(f"Error in listen loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retrying


class PushComponent(BaseZmqComponent):
    """
    Push component for PUSH socket.
    
    Handles work distribution with load balancing across connected workers.
    """
    
    def __init__(self, config: PushConfig, context: Optional[zmq.asyncio.Context] = None):
        super().__init__(config, context)
        self._push_config = config
        
    def _get_socket_type(self) -> int:
        return zmq.PUSH
        
    async def push(self, message: MessageDataType) -> None:
        """
        Push a work message to connected workers.
        
        Args:
            message: Work message data (MessageDataType: Dict or MessageBase)
            
        Raises:
            ComponentNotRunningError: If component is not running
            ZmqComponentError: If push fails
        """
        self._ensure_running()
        
        try:
            # Serialize message to JSON (PUSH/PULL doesn't use topics, just the message)
            if isinstance(message, MessageBase):
                json_message = message.model_dump_json()
            else:
                import json
                json_message = json.dumps(message)
            
            if self.socket and not self.socket.closed:
                # Send just the JSON message (no topic prefix for PUSH/PULL)
                await self.socket.send_string(json_message)
            
                self.logger.debug(f"Pushed work message: {len(json_message)} bytes")
            
        except Exception as e:
            self.logger.error(f"Failed to push work message: {e}")
            raise ZmqComponentError(f"Push failed: {e}") from e


class PullComponent(BaseZmqComponent):
    """
    Pull component for PULL socket.
    
    Handles work reception with async work processing.
    """
    
    def __init__(self, config: PullConfig, context: Optional[zmq.asyncio.Context] = None):
        super().__init__(config, context)
        self._pull_config = config
        self.work_handler: Optional[Callable[[MessageDataType], Union[None, Awaitable[None]]]] = None
        self._listen_task: Optional[asyncio.Task] = None
        
    def _get_socket_type(self) -> int:
        return zmq.PULL
        
    async def _component_start(self) -> None:
        """Start work listening."""
        self._listen_task = asyncio.create_task(self._work_loop())
        self._add_task(self._listen_task)
        
    async def _component_stop(self) -> None:
        """Stop work listening."""
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            
    def set_work_handler(self, handler: Callable[[MessageDataType], Union[None, Awaitable[None]]]) -> None:
        """
        Set the work handler function.
        
        Args:
            handler: Function that takes (message) parameter where message is MessageDataType.
                    Can be sync or async.
        """
        self.work_handler = handler
        self.logger.debug("Work handler set")
        
    async def _work_loop(self) -> None:
        """
        Main work listening loop with error recovery.
        
        Continues processing even if individual work items fail.
        """
        self.logger.debug(f"Starting work loop for {self.component_id}")
        
        while self.running:
            try:
                # Parse work message (just JSON, no topic for PUSH/PULL)
                try:
                    if not self.socket or self.socket.closed:
                        break

                    # Receive as string (matches push format)
                    # Use timeout to allow graceful shutdown but don't log timeout as error
                    try:
                        json_message = await asyncio.wait_for(
                            self.socket.recv_string(),
                            timeout=DEFAULT_RECV_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        # Normal timeout - no work available, continue loop
                        continue
                    
                    # Parse JSON to dict
                    import json
                    message_dict = json.loads(json_message)
                    
                    # Convert to MessageBase if possible, otherwise keep as dict
                    message = MessageBase.from_dict(message_dict)
                    
                    self.logger.debug(f"Received work message: {len(json_message)} bytes")
                    
                    # Call work handler if set
                    if self.work_handler:
                        try:
                            if asyncio.iscoroutinefunction(self.work_handler):
                                await self.work_handler(message)
                            else:
                                self.work_handler(message)
                        except Exception as e:
                            self.logger.error(f"Error in work handler: {e}")
                            
                except Exception as e:
                    self.logger.error(f"Error parsing work message: {e}")
                    
            except asyncio.CancelledError:
                self.logger.debug(f"Work loop cancelled for {self.component_id}")
                break
            except Exception as e:
                self.logger.error(f"Error in work loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retrying


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

async def test_components():
    """Quick test to ensure components work correctly with MessageBase."""
    print("🔧 Testing ZMQ components with MessageBase...")
    
    try:
        # Test Publisher with MessageBase
        pub_config = PublisherConfig(address="tcp://*", port=5555, default_topic="test")
        pub = PublisherComponent(pub_config)
        async with pub:
            # Test with MessageBase
            from experimance_common.schemas import EraChanged, Era, Biome
            era_msg = EraChanged(era=Era.CURRENT, biome=Biome.RAINFOREST)
            await pub.publish(era_msg)
            
            # Test with dict
            dict_msg = {"type": "ImageReady", "image_id": "test", "uri": "file:///test.png"}
            await pub.publish(dict_msg)
            
            print("✅ Publisher works with MessageBase and dict")
            
        # Test Subscriber (just creation, no listening to avoid hanging)
        sub_config = SubscriberConfig(address="tcp://localhost", port=5556, topics=["test"])
        sub = SubscriberComponent(sub_config)
        await sub.start()
        await sub.stop()
        print("✅ Subscriber creation and lifecycle works")
            
        # Test Push with MessageBase
        push_config = PushConfig(address="tcp://*", port=5557)
        push = PushComponent(push_config)
        async with push:
            await push.push(era_msg)
            await push.push(dict_msg)
            print("✅ Push works with MessageBase and dict")
            
        # Test Pull (just creation, no listening to avoid hanging)
        pull_config = PullConfig(address="tcp://*", port=5558)
        pull = PullComponent(pull_config)
        await pull.start()
        await pull.stop()
        print("✅ Pull creation and lifecycle works")
            
        print("🎉 All components test successfully with MessageBase!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        raise


if __name__ == "__main__":
    # Test components when run directly
    asyncio.run(test_components())
