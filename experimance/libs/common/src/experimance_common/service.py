"""
Base service classes for Experimance services with and without ZeroMQ communication.

This module provides a hierarchy of service classes that standardize service 
behavior across the Experimance system. It includes:

1. BaseService: Core functionality for all services with lifecycle management
2. BaseZmqService: ZeroMQ-specific functionality extending BaseService
3. Role-specific ZMQ services (Publisher, Subscriber, Push, Pull)
4. Combined services for common patterns (PublisherSubscriber, Controller, Worker)

All services share common functionality like:
- Graceful shutdown handling with signal trapping
- Standard lifecycle methods (start, stop, run)
- Error handling with proper recovery
- Statistics tracking

ZeroMQ services additionally include:
- Proper initialization and cleanup of ZMQ sockets
- Standard communication patterns
"""

import asyncio
import json
import logging
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast, Coroutine

import zmq
import zmq.asyncio

from experimance_common.constants import (
    DEFAULT_PORTS,
    DEFAULT_TIMEOUT,
    HEARTBEAT_INTERVAL,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RECV_TIMEOUT,
)

from experimance_common.zmq_utils import (
    ZmqPublisher,
    ZmqSubscriber,
    ZmqPushSocket,
    ZmqPullSocket,
    MessageType,
    ZmqTimeoutError,
    ZmqException
)

# Configure logging
logger = logging.getLogger(__name__)


class ServiceState(str, Enum):
    """Service lifecycle states."""
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BaseService:
    """Base class for all services in the Experimance system.
    
    This class provides common functionality for all services:
    - Service lifecycle management (start, stop, run)
    - Signal handling for graceful shutdown
    - Statistics tracking for monitoring
    - Error handling and recovery
    
    Subclasses should implement their specific functionality
    by extending this class and implementing the necessary methods.
    """
    
    def __init__(self, service_name: str, service_type: str = "generic"):
        """Initialize the base service.
        
        Args:
            service_name: Unique name for this service instance
            service_type: Type of service (for logging and monitoring)
        """
        self.service_name = service_name
        self.service_type = service_type
        self.state = ServiceState.INITIALIZED
        self.running = False
        self.tasks = []
        
        # Statistics
        self.start_time = time.monotonic()
        self.messages_sent = 0
        self.messages_received = 0
        self.errors = 0
        self.last_stats_time = self.start_time
        
        # State control
        self._stopping = False  # Flag to prevent multiple stop() calls
        
        # Set up signal handlers for graceful shutdown
        # We'll only use these for non-asyncio contexts
        # Asyncio signal handlers will be set up in the run() method
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals in non-asyncio contexts.
        
        This ensures proper cleanup on service termination.
        """
        # Only process the signal if we're not already stopping
        if self._stopping:
            logger.debug(f"Signal handler called while already stopping, ignoring")
            return
            
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name} ({signum}), shutting down gracefully...")
        
        # Set stopping flag and update state
        self._stopping = True
        self.running = False
        self.state = ServiceState.STOPPING
    
    def _register_task(self, task_coroutine: Coroutine):
        """Register a task to be executed in the service's run loop.
        
        Args:
            task_coroutine: Coroutine function to execute
        """
        # Store the coroutine without awaiting it yet
        self.tasks.append(task_coroutine)
    
    async def display_stats(self):
        """Periodically display service statistics."""
        while self.running:
            await asyncio.sleep(10)  # Update stats every 10 seconds
            
            now = time.monotonic()
            elapsed = now - self.start_time
            elapsed_since_last = now - self.last_stats_time
            
            # Calculate message rates
            sent_rate = self.messages_sent / elapsed if elapsed > 0 else 0
            received_rate = self.messages_received / elapsed if elapsed > 0 else 0
            
            # Format uptime as hours:minutes:seconds
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            
            # Prepare and log statistics
            stats = {
                "service": self.service_name,
                "type": self.service_type,
                "state": self.state,
                "uptime": uptime_str,
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "errors": self.errors,
                "msg_send_rate": f"{sent_rate:.2f}/s",
                "msg_recv_rate": f"{received_rate:.2f}/s"
            }
            
            logger.info(f"Stats for {self.service_name}: {stats}")
            self.last_stats_time = now
    
    async def start(self):
        """Start the service.
        
        This method should be extended by subclasses to initialize 
        their specific components before calling super().start().
        """
        if self.state not in (ServiceState.INITIALIZED, ServiceState.STOPPED):
            raise RuntimeError(f"Cannot start service in state {self.state}")
        
        logger.info(f"Starting {self.service_type} service: {self.service_name}")
        self.running = True
        self.state = ServiceState.STARTING
        self.start_time = time.monotonic()
        
        # Always include the stats display task
        self._register_task(self.display_stats())
    
    async def stop(self):
        """Stop the service and clean up resources.
        
        This method ensures all tasks are properly cancelled
        and resources are cleaned up.
        """
        # Prevent multiple simultaneous calls to stop()
        if self._stopping:
            logger.debug(f"Service {self.service_name} already stopping, ignoring duplicate stop call")
            # If we're already in STOPPED state, don't proceed further
            if self.state == ServiceState.STOPPED:
                return
            # Otherwise continue with the cleanup but don't log duplicate messages
        else:
            self._stopping = True
            logger.info(f"Stopping {self.service_name}...")
            self.running = False
            self.state = ServiceState.STOPPING
        
        # Cancel any running tasks
        tasks = [task for task in asyncio.all_tasks() 
                if task is not asyncio.current_task() and not task.done()]
        
        if tasks:
            logger.info(f"Cancelling {len(tasks)} pending tasks")
            for task in tasks:
                task.cancel()
            
            try:
                # Wait for tasks to be cancelled with a timeout
                await asyncio.wait(tasks, timeout=1.0)
            except asyncio.CancelledError:
                logger.debug("Task cancellation was itself cancelled")
            except Exception as e:
                logger.warning(f"Error while waiting for tasks to cancel: {e}")
        
        # Clean up any unrun coroutines to prevent 'coroutine was never awaited' warnings
        # When a coroutine object goes out of scope without being awaited, Python warns
        # Here we explicitly close the coroutines
        for task in self.tasks:
            try:
                task.close()  # Close the coroutine to prevent the warning
            except Exception:
                pass  # Ignore any errors when closing
        self.tasks = []
        
        self.state = ServiceState.STOPPED
        logger.info(f"Service {self.service_name} stopped")
    
    async def run(self):
        """Run the service until stopped.
        
        This method executes all registered tasks concurrently and
        handles proper cleanup on termination.
        """
        if not self.tasks:
            raise RuntimeError("No tasks registered for service")
        
        self.state = ServiceState.RUNNING
        logger.info(f"Service {self.service_name} running")
        
        try:
            # Register asyncio-specific signal handlers
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._handle_signal_async(s))
                )
            
            # Run all tasks concurrently
            # Use asyncio.gather with return_exceptions=True to allow proper cleanup
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except asyncio.CancelledError:
            logger.info(f"Service {self.service_name} tasks cancelled")
            # Don't re-raise, this is handled gracefully
        except Exception as e:
            logger.error(f"Error in service {self.service_name}: {e}")
            logger.debug(traceback.format_exc())
            self.errors += 1
            self.state = ServiceState.ERROR
            raise
        finally:
            # Even if we're cancelled, ensure service is stopped properly
            if not self._stopping:  # Only call stop() if we haven't started stopping already
                try:
                    await self.stop()
                except Exception as e:
                    logger.error(f"Error during service shutdown: {e}")
                    # Don't re-raise, we're in cleanup mode
    
    async def _handle_signal_async(self, sig):
        """Handle signals in the asyncio event loop."""
        # Only process the signal if we're not already stopping
        if self._stopping:
            logger.debug("Signal handler called while already stopping, ignoring")
            return
            
        signal_name = signal.Signals(sig).name
        logger.info(f"Received signal {signal_name} in asyncio event loop")
        
        # Mark as stopping before doing anything to prevent recursion
        self._stopping = True
        self.running = False
        self.state = ServiceState.STOPPING
        
        # We don't need to call stop() from here - the main loop's finally block
        # will handle that, and we've already set the proper flags
        # Just log that we're shutting down
        logger.info(f"Service {self.service_name} shutting down due to signal {signal_name}")

        # Don't stop the event loop immediately - let the tasks clean up properly
        # and let the main loop handle the exit


class BaseZmqService(BaseService):
    """Base class for ZeroMQ-based services in the Experimance system.
    
    This class extends BaseService with ZeroMQ-specific functionality:
    - ZMQ socket registration and cleanup
    - Common ZMQ communication patterns
    
    Subclasses should implement their specific communication patterns
    by extending this class and implementing the necessary methods.
    """
    
    def __init__(self, service_name: str, service_type: str = "zmq-service"):
        """Initialize the base ZMQ service.
        
        Args:
            service_name: Unique name for this service instance
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        
        # ZMQ sockets - to be initialized by subclasses
        self._sockets = []
    
    def register_socket(self, socket):
        """Register a ZMQ socket for automatic cleanup.
        
        Args:
            socket: ZMQ socket wrapper to register
        """
        self._sockets.append(socket)
    
    async def stop(self):
        """Stop the service and clean up ZMQ resources.
        
        This method ensures all ZMQ sockets are properly closed
        in addition to the standard service cleanup.
        """
        # If we're already stopping, let the parent handle it - but close sockets first
        if self._stopping and self.state == ServiceState.STOPPED:
            logger.debug(f"ZMQ Service {self.service_name} already fully stopped")
            return
            
        # Set stopping flags even if already stopping - we still need to close sockets
        self._stopping = True
        self.running = False
        self.state = ServiceState.STOPPING
        
        # First close all sockets in reverse order of registration
        # We do this BEFORE calling super().stop() to avoid task cancellation issues
        socket_errors = 0
        for socket in reversed(self._sockets):
            if socket:
                try:
                    logger.debug(f"Closing socket: {type(socket).__name__}")
                    socket.close()
                except Exception as e:
                    logger.warning(f"Error closing socket: {e}")
                    socket_errors += 1
        
        # Clear the sockets list to prevent double-closure
        self._sockets = []
        
        if socket_errors > 0:
            logger.warning(f"Encountered {socket_errors} errors while closing ZMQ sockets")
        
        # Call the parent class stop method to handle task cancellation
        try:
            await super().stop()
        except Exception as e:
            logger.error(f"Error in parent stop method: {e}")
            # Still mark as stopped even if there's an error
            self.state = ServiceState.STOPPED


class ZmqPublisherService(BaseZmqService):
    """Service that publishes messages on specific topics.
    
    This service type establishes a ZeroMQ PUB socket to broadcast
    messages to subscribing services.
    """
    
    def __init__(self, service_name: str, 
                 pub_address: str, 
                 heartbeat_topic: str = "heartbeat",
                 service_type: str = "publisher"):
        """Initialize a publisher service.
        
        Args:
            service_name: Unique name for this service instance
            pub_address: ZeroMQ address to bind publisher to
            heartbeat_topic: Topic for heartbeat messages
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.pub_address = pub_address
        self.heartbeat_topic = heartbeat_topic
        self.publisher:Optional[ZmqPublisher] = None
    
    async def start(self):
        """Start the publisher service."""
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.heartbeat_topic)
        self.register_socket(self.publisher)
        
        # Register heartbeat task
        self._register_task(self.send_heartbeat())
        
        await super().start()
    
    async def send_heartbeat(self, interval: float = HEARTBEAT_INTERVAL):
        """Send periodic heartbeat messages.
        
        Args:
            interval: Time between heartbeats in seconds
        """
        while self.running:
            try:
                heartbeat = {
                    "type": MessageType.HEARTBEAT,
                    "timestamp": time.time(),
                    "service": self.service_name,
                    "state": self.state
                }
                
                success = await self.publisher.publish_async(heartbeat) # type: ignore
                if success:
                    logger.debug(f"Sent heartbeat: {self.service_name}")
                    self.messages_sent += 1
                else:
                    logger.warning("Failed to send heartbeat")
                    self.errors += 1
            
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                self.errors += 1
            
            await asyncio.sleep(interval)
    
    async def publish_message(self, message: Dict[str, Any], topic: Optional[str] = None) -> bool:
        """Publish a message to subscribers.
        
        Args:
            message: Message to publish
            topic: Topic to publish on (if None, uses the default heartbeat topic)
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.publisher:
            logger.error("Cannot publish message: publisher not initialized")
            self.errors += 1
            return False
        
        # If topic provided, create a new publisher or use existing one with that topic
        publisher = self.publisher
        if topic is not None and topic != self.heartbeat_topic:
            publisher = ZmqPublisher(self.pub_address, topic)
            try:
                success = await publisher.publish_async(message)
                if success:
                    self.messages_sent += 1
                else:
                    self.errors += 1
                return success
            finally:
                publisher.close()
        else:
            # Use the default publisher
            try:
                success = await self.publisher.publish_async(message)
                if success:
                    self.messages_sent += 1
                else:
                    self.errors += 1
                return success
            except Exception as e:
                logger.error(f"Error publishing message: {e}")
                self.errors += 1
                return False


class ZmqSubscriberService(BaseZmqService):
    """Service that subscribes to messages on specific topics.
    
    This service type establishes a ZeroMQ SUB socket to receive
    broadcasts from publishing services.
    """
    
    def __init__(self, service_name: str, 
                 sub_address: str, 
                 topics: List[str],
                 service_type: str = "subscriber"):
        """Initialize a subscriber service.
        
        Args:
            service_name: Unique name for this service instance
            sub_address: ZeroMQ address to connect subscriber to
            topics: List of topics to subscribe to
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.sub_address = sub_address
        self.topics = topics
        self.subscriber = None
        self.message_handlers = {}
    
    async def start(self):
        """Start the subscriber service."""
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Register message listening task
        self._register_task(self.listen_for_messages())
        
        await super().start()
    
    def register_handler(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for a specific topic.
        
        Args:
            topic: Topic to handle messages for
            handler: Function to call with message data
        """
        if topic not in self.topics:
            logger.warning(f"Registering handler for topic {topic} which is not in subscription list")
        
        self.message_handlers[topic] = handler
    
    async def listen_for_messages(self):
        """Listen for incoming messages on subscribed topics."""
        if not self.subscriber:
            logger.error("Cannot listen for messages: subscriber not initialized")
            return
            
        while self.running:
            try:
                topic, message = await self.subscriber.receive_async()
                logger.debug(f"Received message on {topic}: {message}")
                self.messages_received += 1
                
                # Process message with registered handler if any
                if topic in self.message_handlers:
                    try:
                        self.message_handlers[topic](message)
                    except Exception as e:
                        logger.error(f"Error in message handler for topic {topic}: {e}")
                        self.errors += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                self.errors += 1
            
            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning


class ZmqPushService(BaseZmqService):
    """Service that pushes tasks to workers.
    
    This service type establishes a ZeroMQ PUSH socket to distribute
    tasks to pulling workers.
    """
    
    def __init__(self, service_name: str, 
                 push_address: str,
                 service_type: str = "push"):
        """Initialize a push service.
        
        Args:
            service_name: Unique name for this service instance
            push_address: ZeroMQ address to bind push socket to
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.push_address = push_address
        self.push_socket = None
    
    async def start(self):
        """Start the push service."""
        logger.info(f"Initializing push socket on {self.push_address}")
        self.push_socket = ZmqPushSocket(self.push_address)
        self.register_socket(self.push_socket)
        
        await super().start()
    
    async def push_task(self, task: Dict[str, Any]) -> bool:
        """Push a task to workers.
        
        Args:
            task: Task data to send
            
        Returns:
            True if task was sent successfully, False otherwise
        """
        if not self.push_socket:
            logger.error("Cannot push task: push socket not initialized")
            self.errors += 1
            return False
        
        try:
            success = await self.push_socket.push_async(task)
            if success:
                logger.debug(f"Pushed task: {task.get('id', 'unknown')}")
                self.messages_sent += 1
            else:
                logger.warning(f"Failed to push task: {task.get('id', 'unknown')}")
                self.errors += 1
            return success
        except Exception as e:
            logger.error(f"Error pushing task: {e}")
            self.errors += 1
            return False


class ZmqPullService(BaseZmqService):
    """Service that pulls tasks from pushers.
    
    This service type establishes a ZeroMQ PULL socket to receive
    tasks from pushing services.
    """
    
    def __init__(self, service_name: str, 
                 pull_address: str,
                 service_type: str = "pull"):
        """Initialize a pull service.
        
        Args:
            service_name: Unique name for this service instance
            pull_address: ZeroMQ address to connect pull socket to
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        self.pull_address = pull_address
        self.pull_socket = None
        self.task_handler = None
    
    async def start(self):
        """Start the pull service."""
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Register message listening task
        self._register_task(self.pull_tasks())
        
        await super().start()
    
    def register_task_handler(self, handler: Callable[[Dict[str, Any]], Coroutine]):
        """Register a handler for incoming tasks.
        
        Args:
            handler: Async function to call with task data
        """
        self.task_handler = handler
    
    async def pull_tasks(self):
        """Pull and process tasks."""
        if not self.pull_socket:
            logger.error("Cannot pull tasks: pull socket not initialized")
            return
            
        while self.running:
            try:
                task = await self.pull_socket.pull_async()
                if task:
                    logger.debug(f"Received task: {task.get('id', 'unknown')}")
                    self.messages_received += 1
                    
                    # Process task with registered handler if any
                    if self.task_handler:
                        try:
                            await self.task_handler(task)
                        except Exception as e:
                            logger.error(f"Error in task handler: {e}")
                            self.errors += 1
            
            except ZmqTimeoutError:
                # This is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error pulling task: {e}")
                self.errors += 1
            
            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning


class ZmqPublisherSubscriberService(ZmqPublisherService, ZmqSubscriberService):
    """Service that both publishes and subscribes to messages.
    
    This combined service type is suitable for services that need to
    both broadcast their state and listen for events from other services.
    """
    
    def __init__(self, service_name: str,
                 pub_address: str,
                 sub_address: str,
                 topics: List[str],
                 heartbeat_topic: str = "heartbeat",
                 service_type: str = "pubsub"):
        """Initialize a publisher-subscriber service.
        
        Args:
            service_name: Unique name for this service instance
            pub_address: ZeroMQ address to bind publisher to
            sub_address: ZeroMQ address to connect subscriber to
            topics: List of topics to subscribe to
            heartbeat_topic: Topic for heartbeat messages
            service_type: Type of service (for logging and monitoring)
        """
        BaseZmqService.__init__(self, service_name, service_type)
        self.pub_address = pub_address
        self.sub_address = sub_address
        self.topics = topics
        self.heartbeat_topic = heartbeat_topic
        self.publisher = None
        self.subscriber = None
        self.message_handlers = {}
    
    async def start(self):
        """Start the publisher-subscriber service."""
        # Initialize publisher
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.heartbeat_topic)
        self.register_socket(self.publisher)
        
        # Initialize subscriber
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Register tasks
        self._register_task(self.send_heartbeat())
        self._register_task(self.listen_for_messages())
        
        await BaseZmqService.start(self)


class ZmqControllerService(ZmqPublisherSubscriberService, ZmqPushService):
    """Controller service that publishes events, listens for responses, and pushes tasks.
    
    This combined service is suitable for central coordinator services that
    need to broadcast messages, listen for responses, and distribute tasks.
    """
    
    def __init__(self, service_name: str,
                 pub_address: str,
                 sub_address: str,
                 push_address: str,
                 pull_address: str,
                 topics: List[str],
                 heartbeat_topic: str = "heartbeat",
                 service_type: str = "controller"):
        """Initialize a controller service.
        
        Args:
            service_name: Unique name for this service instance
            pub_address: ZeroMQ address to bind publisher to
            sub_address: ZeroMQ address to connect subscriber to
            push_address: ZeroMQ address to bind push socket to
            pull_address: ZeroMQ address to connect pull socket to
            topics: List of topics to subscribe to
            heartbeat_topic: Topic for heartbeat messages
            service_type: Type of service (for logging and monitoring)
        """
        BaseZmqService.__init__(self, service_name, service_type)
        self.pub_address = pub_address
        self.sub_address = sub_address
        self.push_address = push_address
        self.pull_address = pull_address
        self.topics = topics
        self.heartbeat_topic = heartbeat_topic
        
        # Initialize all sockets as None - they will be created in start()
        self.publisher = None
        self.subscriber = None
        self.push_socket = None
        self.pull_socket = None
        
        # Handlers
        self.message_handlers = {}
        self.task_handler = None
    
    async def start(self):
        """Start the controller service."""
        # Initialize publisher for broadcasting
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.heartbeat_topic)
        self.register_socket(self.publisher)
        
        # Initialize subscriber for receiving responses
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Initialize push socket for distributing tasks
        logger.info(f"Initializing push socket on {self.push_address}")
        self.push_socket = ZmqPushSocket(self.push_address)
        self.register_socket(self.push_socket)
        
        # Initialize pull socket for receiving worker responses
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Register tasks
        self._register_task(self.send_heartbeat())
        self._register_task(self.listen_for_messages())
        
        await BaseZmqService.start(self)


class ZmqWorkerService(ZmqSubscriberService, ZmqPullService):
    """Worker service that subscribes to events and pulls tasks.
    
    This combined service is suitable for worker services that
    need to listen for control messages and receive tasks to process.
    """
    
    def __init__(self, service_name: str,
                 sub_address: str,
                 pull_address: str,
                 push_address: Optional[str] = None,
                 topics: Optional[List[str]] = None,
                 service_type: str = "worker"):
        """Initialize a worker service.
        
        Args:
            service_name: Unique name for this service instance
            sub_address: ZeroMQ address to connect subscriber to
            pull_address: ZeroMQ address to connect pull socket to
            push_address: Optional ZeroMQ address to bind push socket for responses
            topics: List of topics to subscribe to
            service_type: Type of service (for logging and monitoring)
        """
        topics = topics or ["heartbeat"]
        BaseZmqService.__init__(self, service_name, service_type)
        self.sub_address = sub_address
        self.pull_address = pull_address
        self.push_address = push_address
        self.topics = topics
        
        # Initialize sockets as None - they will be created in start()
        self.subscriber = None
        self.pull_socket = None
        self.push_socket = None
        
        # Handlers
        self.message_handlers = {}
        self.task_handler = None
    
    async def start(self):
        """Start the worker service."""
        # Initialize subscriber for receiving control messages
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.topics)
        self.register_socket(self.subscriber)
        
        # Initialize pull socket for receiving tasks
        logger.info(f"Initializing pull socket on {self.pull_address}")
        self.pull_socket = ZmqPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Initialize push socket for sending responses back (if address provided)
        if self.push_address:
            logger.info(f"Initializing push socket on {self.push_address}")
            self.push_socket = ZmqPushSocket(self.push_address)
            self.register_socket(self.push_socket)
        
        # Register tasks
        self._register_task(self.listen_for_messages())
        self._register_task(self.pull_tasks())
        
        await BaseZmqService.start(self)
    
    async def send_response(self, response: Dict[str, Any]) -> bool:
        """Send a response back to the controller.
        
        Args:
            response: Response data to send
            
        Returns:
            True if response was sent successfully, False otherwise
        """
        if not self.push_socket:
            logger.error("Cannot send response: push socket not initialized")
            self.errors += 1
            return False
        
        try:
            success = await self.push_socket.push_async(response)
            if success:
                logger.debug(f"Sent response: {response.get('type', 'unknown')}")
                self.messages_sent += 1
            else:
                logger.warning(f"Failed to send response: {response.get('type', 'unknown')}")
                self.errors += 1
            return success
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            self.errors += 1
            return False
