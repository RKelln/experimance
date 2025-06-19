"""
ZeroMQ Controller Service for Experimance.

This module provides the ZmqControllerService class which combines
publisher, subscriber, push, and pull patterns for centralized
control of a distributed system using ZeroMQ.

It also provides ZmqMultiControllerService for handling multiple worker types.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Coroutine, Union

from pydantic import BaseModel, Field, field_validator

from experimance_common.constants import HEARTBEAT_TOPIC, TICK, DEFAULT_PORTS
from experimance_common.zmq.base_zmq import BaseZmqService
from experimance_common.zmq.zmq_utils import ZmqPublisher, ZmqSubscriber, ZmqBindingPullSocket, ZmqPushSocket, MessageDataType, MessageType
from experimance_common.zmq.pubsub import ZmqPublisherSubscriberService
from experimance_common.zmq.push import ZmqPushService
from experimance_common.zmq.pull import ZmqPullService

logger = logging.getLogger(__name__)


# Worker Configuration Schemas
class WorkerConnectionConfig(BaseModel):
    """Configuration for a single worker connection.
    
    This class defines the networking and message type configuration for a worker.
    The worker type is inferred from the key in the MultiControllerWorkerConfig.workers dict.
    """
    
    push_address: str = Field(
        description="Address for controller to bind PUSH socket (tasks to workers)"
    )
    
    pull_address: str = Field(
        description="Address for controller to bind PULL socket (results from workers)"
    )
    
    push_message_types: List[MessageType] = Field(
        default_factory=list,
        description="Message types this worker accepts via PUSH (tasks)"
    )
    
    pull_message_types: List[MessageType] = Field(
        default_factory=list, 
        description="Message types this worker sends via PULL (results)"
    )
    
    @field_validator('push_address', 'pull_address')
    @classmethod
    def validate_address_format(cls, v: str) -> str:
        """Validate ZMQ address format."""
        if not v.startswith('tcp://'):
            raise ValueError(f"Address must start with 'tcp://': {v}")
        return v
    
    def get_push_port(self) -> int:
        """Extract port number from push address."""
        try:
            return int(self.push_address.split(":")[-1])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot extract port from push address: {self.push_address}")
    
    def get_pull_port(self) -> int:
        """Extract port number from pull address.""" 
        try:
            return int(self.pull_address.split(":")[-1])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot extract port from pull address: {self.pull_address}")


class ControllerMultiWorkerConfig(BaseModel):
    """Configuration for all workers in a multi-controller setup."""
    
    workers: Dict[str, WorkerConnectionConfig] = Field(
        description="Worker configurations keyed by worker type"
    )
    
    def get_worker_for_message_type(self, message_type: MessageType) -> Optional[str]:
        """Get the worker type that handles a specific message type.
        
        Searches through both push and pull message types to find which worker
        can handle the given message type.
        
        Args:
            message_type: The message type to find a handler for
            
        Returns:
            Worker type name if found, None if no worker handles this message type
        """
        for worker_type, config in self.workers.items():
            if message_type in config.push_message_types:
                return worker_type
            if message_type in config.pull_message_types:
                return worker_type
        return None
    
    def get_routing_map(self) -> Dict[str, str]:
        """Get a mapping of message types to worker types.
        
        Creates a dictionary where keys are message type strings and values
        are the worker type names that handle those message types.
        
        Returns:
            Dictionary mapping message type strings to worker type names
            
        Example:
            routing_map = config.get_routing_map()
            # Returns: {"RenderRequest": "image", "TransitionRequest": "transition"}
        """
        routing_map = {}
        for worker_type, config in self.workers.items():
            for msg_type in config.push_message_types:
                routing_map[str(msg_type)] = worker_type
        return routing_map
    
    def check_port_conflicts(self) -> None:
        """Check for port conflicts between workers.
        
        Raises:
            ValueError: If port conflicts are detected
        """
        used_ports = set()
        
        for worker_type, config in self.workers.items():
            try:
                push_port = config.get_push_port()
                pull_port = config.get_pull_port()
                
                if push_port in used_ports:
                    raise ValueError(f"Port conflict: {worker_type} push port {push_port} already in use")
                if pull_port in used_ports:
                    raise ValueError(f"Port conflict: {worker_type} pull port {pull_port} already in use")
                    
                used_ports.update([push_port, pull_port])
                
            except ValueError as e:
                if "Port conflict" in str(e):
                    raise
                # Re-raise port extraction errors
                raise ValueError(f"Invalid ports for worker {worker_type}: {e}")
    
    def validate_message_type_routing(self) -> None:
        """Validate that each message type is handled by only one worker.
        
        Raises:
            ValueError: If multiple workers handle the same message type
        """
        message_type_map = {}
        
        for worker_type, config in self.workers.items():
            for msg_type in config.push_message_types:
                if msg_type in message_type_map:
                    raise ValueError(
                        f"Message type {msg_type.value} is handled by both "
                        f"'{message_type_map[msg_type]}' and '{worker_type}' workers. "
                        f"Each message type can only be handled by one worker."
                    )
                message_type_map[msg_type] = worker_type


class WorkerConnection:
    """Represents a PUSH/PULL connection pair to a specific worker type.
    
    This class encapsulates the socket management for communication with
    a specific type of worker (e.g., image processing, transition rendering).
    
    ## Socket Pattern
    
    - **PUSH Socket (Controller → Worker)**: Controller binds, workers connect
      - Used for distributing tasks from controller to workers
      - Multiple workers can connect to receive tasks (load balancing)
      - Address format: "tcp://*:port" (controller binds to all interfaces)
      
    - **PULL Socket (Worker → Controller)**: Controller binds, workers connect  
      - Used for collecting results from workers back to controller
      - Multiple workers can send results to the same controller
      - Address format: "tcp://*:port" (controller binds to all interfaces)
    
    ## Lifecycle
    
    1. **Creation**: WorkerConnection created with addresses and message types
    2. **Initialization**: Sockets created and bound during controller.start()
    3. **Operation**: Tasks pushed, responses pulled during service runtime
    4. **Cleanup**: Sockets automatically closed during controller.stop()
    
    ## Socket Ownership
    
    The WorkerConnection owns the socket objects but delegates cleanup to the
    parent service via register_socket_func. This ensures proper shutdown order
    and prevents resource leaks.
    """
    
    def __init__(self, 
                 worker_type: str,
                 push_address: str, 
                 pull_address: str,
                 push_message_types: Optional[List[MessageType]] = None,
                 pull_message_types: Optional[List[MessageType]] = None):
        """Initialize a worker connection.
        
        Args:
            worker_type: Type of worker (e.g., 'image', 'transition')
            push_address: Address for controller to bind PUSH socket (sends tasks to workers)
            pull_address: Address for controller to bind PULL socket (receives results from workers)
            push_message_types: Message types this worker accepts via PUSH (tasks from controller)
            pull_message_types: Message types this worker sends via PULL (results to controller)
                          If None or empty, worker can handle any message type on that direction
        
        Example:
            connection = WorkerConnection(
                worker_type="image",
                push_address="tcp://*:5563",  # Controller binds here, workers connect
                pull_address="tcp://*:5564",  # Controller binds here, workers connect
                push_message_types=[MessageType.RENDER_REQUEST],  # What controller sends to worker
                pull_message_types=[MessageType.IMAGE_READY]      # What worker sends to controller
            )
        """
        self.worker_type = worker_type
        self.push_address = push_address
        self.pull_address = pull_address
        self.push_message_types = set(push_message_types or [])  # Messages controller sends TO worker
        self.pull_message_types = set(pull_message_types or [])  # Messages worker sends TO controller
        
        # Sockets are created during initialize() and managed by parent service
        self.push_socket: Optional[ZmqPushSocket] = None
        self.pull_socket: Optional[ZmqBindingPullSocket] = None
        
    async def initialize(self, register_socket_func: Callable):
        """Initialize the push/pull sockets for this worker connection.
        
        Args:
            register_socket_func: Function to register sockets with parent service for cleanup
            
        Raises:
            ValueError: If addresses are invalid
            Exception: If socket creation fails
        """
        # Validate addresses before creating sockets
        self.validate_addresses()
        
        logger.info(f"Initializing {self.worker_type} worker: PUSH={self.push_address}, PULL={self.pull_address}")
        
        try:
            # Controller binds push socket for distributing tasks to workers
            self.push_socket = ZmqPushSocket(self.push_address)
            # Controller binds pull socket to receive responses from workers  
            self.pull_socket = ZmqBindingPullSocket(self.pull_address)
            
            register_socket_func(self.push_socket)
            register_socket_func(self.pull_socket)
            
        except Exception as e:
            logger.error(f"Failed to initialize sockets for {self.worker_type} worker: {e}")
            # Clean up any partially created sockets
            self.cleanup()
            raise
    
    def validate_addresses(self) -> None:
        """Validate that addresses are properly formatted.
        
        Raises:
            ValueError: If addresses are invalid
        """
        if not self.push_address.startswith("tcp://"):
            raise ValueError(f"Invalid push address format: {self.push_address}")
        if not self.pull_address.startswith("tcp://"):
            raise ValueError(f"Invalid pull address format: {self.pull_address}")
        
        # Extract ports and check for conflicts
        try:
            push_port = int(self.push_address.split(":")[-1])
            pull_port = int(self.pull_address.split(":")[-1])
            if push_port == pull_port:
                raise ValueError(f"Push and pull ports cannot be the same: {push_port}")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Cannot parse ports from addresses: {e}")
    
    def cleanup(self) -> None:
        """Clean up sockets (called by parent service during shutdown)."""
        # Note: Actual socket cleanup is handled by the parent service's socket management
        # This method exists for explicit cleanup if needed
        self.push_socket = None
        self.pull_socket = None
        
    def can_handle_message_type(self, message_type: MessageType) -> bool:
        """Check if this worker can handle the given message type (for outgoing PUSH messages).
        
        Args:
            message_type: The message type to check
            
        Returns:
            True if this worker can handle the message type via PUSH, False otherwise
        """
        return message_type in self.push_message_types if self.push_message_types else True
        
    async def push_task(self, message: MessageDataType) -> bool:
        """Push a task to this worker."""
        if not self.push_socket:
            logger.error(f"Cannot push task to {self.worker_type}: socket not initialized")
            return False
            
        return await self.push_socket.push_async(message)
        
    async def pull_response(self) -> MessageDataType:
        """Pull a response from this worker."""
        if not self.pull_socket:
            raise RuntimeError(f"Cannot pull response from {self.worker_type}: socket not initialized")
            
        return await self.pull_socket.pull_async()
    
    async def pull_response_nonblocking(self) -> Optional[MessageDataType]:
        """Try to pull a response from this worker without blocking.
        
        Returns:
            Message if available, None if no message is ready
            
        Raises:
            RuntimeError: If socket is not initialized
        """
        if not self.pull_socket:
            raise RuntimeError(f"Cannot pull response from {self.worker_type}: socket not initialized")
            
        try:
            # Use NOBLOCK flag to return immediately if no message available
            import zmq
            import json
            message_str: str = self.pull_socket.socket.recv_string(zmq.NOBLOCK)  # type: ignore
            return json.loads(message_str)
        except zmq.Again:
            # No message available
            return None
        except Exception as e:
            logger.error(f"Error in non-blocking pull from {self.worker_type}: {e}")
            return None
        
    # ...existing code...


class ZmqControllerService(ZmqPublisherSubscriberService, ZmqPushService, ZmqPullService):
    """Controller service that publishes events, listens for responses, and pushes tasks.
    
    This combined service is suitable for central coordinator services that
    need to broadcast messages, listen for responses, and distribute tasks.
    It also pulls responses from workers.
    """
    
    def __init__(self, service_name: str,
                 pub_address: str,
                 sub_address: str,
                 push_address: str,
                 pull_address: str,
                 topics: list,
                 heartbeat_topic: str = HEARTBEAT_TOPIC,
                 service_type: str = "controller"):
        ZmqPublisherSubscriberService.__init__(
            self,
            service_name=service_name,
            pub_address=pub_address,
            sub_address=sub_address,
            subscribe_topics=topics,
            publish_topic=heartbeat_topic,
            service_type=service_type
        )
        ZmqPushService.__init__(self, service_name=service_name, push_address=push_address, service_type=service_type)
        ZmqPullService.__init__(self, service_name=service_name, pull_address=pull_address, service_type=service_type)

        # Register the handler for messages from the PULL socket
        #self.register_task_handler(self._handle_worker_response)
    
    async def start(self):
        """Start the controller service."""
        # Initialize publisher for broadcasting
        logger.info(f"Initializing publisher on {self.pub_address}")
        self.publisher = ZmqPublisher(self.pub_address, self.publish_topic)
        self.register_socket(self.publisher)
        
        # Initialize subscriber for receiving responses
        logger.info(f"Initializing subscriber on {self.sub_address} with topics {self.subscribe_topics}")
        self.subscriber = ZmqSubscriber(self.sub_address, self.subscribe_topics)
        self.register_socket(self.subscriber)
        
        # Initialize push socket for distributing tasks
        logger.info(f"Initializing push socket on {self.push_address}")
        self.push_socket = ZmqPushSocket(self.push_address)
        self.register_socket(self.push_socket)
        
        # Initialize pull socket for receiving worker responses (BIND to allow multiple workers to connect)
        logger.info(f"Initializing binding pull socket on {self.pull_address}")
        self.pull_socket = ZmqBindingPullSocket(self.pull_address)
        self.register_socket(self.pull_socket)
        
        # Register tasks
        self.add_task(self.send_heartbeat())
        self.add_task(self.listen_for_messages())
        self.add_task(self.pull_tasks())
        
        await BaseZmqService.start(self)


class ZmqControllerMultiWorkerService(ZmqPublisherSubscriberService):
    """High-performance controller for managing multiple worker types with automatic message routing.
    
    Manages multiple worker types, each handling specific message types through dedicated 
    PUSH/PULL socket pairs. Provides automatic message routing, type-safe configuration,
    and optimized performance for high-throughput scenarios.
    
    ## Key Features
    - **Automatic routing**: Messages routed to workers based on message type
    - **Type-safe config**: Pydantic-based configuration with validation
    - **Multiple workers**: Each worker type handles specific message types  
    - **Performance options**: Event-driven or non-blocking message handling
    - **Flexible access**: Access workers by type or message type
    
    ## Basic Usage
    
    ```python
    # 1. Configure workers with type safety
    config = MultiControllerWorkerConfig(workers={
        "image": WorkerConnectionConfig(
            push_address="tcp://*:5563",
            pull_address="tcp://*:5564",
            push_message_types=[MessageType.RENDER_REQUEST],
            pull_message_types=[MessageType.IMAGE_READY]
        )
    })
    
    # 2. Create and configure controller
    controller = ZmqMultiControllerService(
        service_name="core",
        pub_address="tcp://*:5555",
        sub_address="tcp://localhost:5556",
        topics=["heartbeat"],
        use_nonblocking_handler=True  # For high performance
    )
    controller.register_workers_from_config(config)
    
    # 3. Start and use
    await controller.start()
    
    # Automatic routing by message type
    await controller.push_task({
        "type": "RENDER_REQUEST",  # Routes to "image" worker
        "prompt": "sunset over mountains"
    })
    
    # Flexible worker access
    response = await controller.pull_response("image")
    # OR: response = await controller.pull_response(MessageType.IMAGE_READY)
    ```
    
    ## Architecture
    - **PUB/SUB**: Event broadcasting to all services
    - **PUSH/PULL**: Task distribution and result collection per worker type
    - **Binding**: Controller binds sockets, workers connect (enables scaling)
    - **Routing**: Automatic message-to-worker routing with validation
    """
    
    def __init__(self, service_name: str,
                 pub_address: str,
                 sub_address: str,
                 topics: list,
                 heartbeat_topic: str = HEARTBEAT_TOPIC,
                 service_type: str = "multi_controller",
                 use_nonblocking_handler: bool = False):
        """Initialize the multi-controller service.
        
        Args:
            service_name: Name of this service
            pub_address: Address to publish events on
            sub_address: Address to subscribe to events on  
            topics: List of topics to subscribe to
            heartbeat_topic: Topic for heartbeat messages
            service_type: Type identifier for this service
            use_nonblocking_handler: If True, use the high-performance non-blocking 
                                   message handler instead of the asyncio.wait() approach
            
        Note: Worker connections are registered separately via register_worker()
              or register_workers_from_config()
        """
        ZmqPublisherSubscriberService.__init__(
            self,
            service_type=service_type,
            service_name=service_name,
            pub_address=pub_address,
            sub_address=sub_address,
            subscribe_topics=topics,
            publish_topic=heartbeat_topic,
        )
        self.worker_connections: Dict[str, WorkerConnection] = {}
        # Message handlers should be async coroutines for consistency with service patterns
        # TODO: create a MessageHandlerType and see if it can be reconciled with ZMQSubscriber  handlers
        self.worker_message_handlers: Dict[str, Callable[[MessageDataType], Coroutine[Any, Any, None]]] = {}
        self.use_nonblocking_handler = use_nonblocking_handler
        
    def register_worker(self, worker_type: str, push_address: str, pull_address: str, 
                       push_message_types: Optional[List[MessageType]] = None,
                       pull_message_types: Optional[List[MessageType]] = None):
        """Register a new worker type with its associated PUSH/PULL addresses and message types.
        
        Args:
            worker_type: Type of worker (e.g., 'image', 'transition')
            push_address: Address for controller to bind PUSH socket
            pull_address: Address for controller to bind PULL socket
            push_message_types: Message types this worker accepts via PUSH (tasks)
            pull_message_types: Message types this worker sends via PULL (results)
        """
        if worker_type in self.worker_connections:
            raise ValueError(f"Worker type {worker_type} is already registered")
        
        worker_connection = WorkerConnection(
            worker_type, push_address, pull_address, push_message_types, pull_message_types
        )
        self.worker_connections[worker_type] = worker_connection
        
        # Register default handler for this worker type (can be overridden by user)
        self.register_worker_handler(worker_type, self._create_default_worker_handler(worker_type))
        
    def register_worker_handler(self, worker_type: str, handler: Callable[[MessageDataType], Coroutine[Any, Any, None]]):
        """Register a custom message handler for a specific worker type.
        
        Args:
            worker_type: The worker type this handler is for
            handler: Async coroutine that accepts a MessageDataType and returns None
                    
        Example:
            async def my_handler(message: MessageDataType) -> None:
                msg_type = message.get("type")
                if msg_type == "IMAGE_READY":
                    # Process completed image
                    await self.handle_image_ready(message)
            
            controller.register_worker_handler("image", my_handler)
        """
        if worker_type not in self.worker_connections:
            raise ValueError(f"Worker type {worker_type} is not registered")
        
        if worker_type in self.worker_message_handlers:
            raise ValueError(f"Handler for worker type {worker_type} is already registered")

        self.worker_message_handlers[worker_type] = handler

    async def initialize_workers(self):
        """Initialize all registered workers.
        
        Raises:
            ValueError: If port conflicts are detected
            Exception: If worker initialization fails
        """
        # Check for port conflicts before initializing any sockets
        self._check_port_conflicts()
        
        logger.info(f"Initializing {len(self.worker_connections)} worker types")
        
        for worker_connection in self.worker_connections.values():
            await worker_connection.initialize(self.register_socket)
    
    async def default_worker_handler(self, worker_type: str, message: MessageDataType):
        """Default message handler for processing messages from workers."""
        logger.info(f"Received message from {worker_type} worker: {message}")
        
    def _create_default_worker_handler(self, worker_type: str) -> Callable:
        """Create a default handler function for a worker type."""
        async def handler(message: MessageDataType):
            await self.default_worker_handler(worker_type, message)
        return handler
        
    async def start(self):
        """Start the multi-controller service."""
        await super().start()
        await self.initialize_workers()
        
        # Add task to handle worker responses - choose handler based on configuration
        if self.use_nonblocking_handler:
            self.add_task(self.handle_worker_responses_nonblocking())
        else:
            self.add_task(self.handle_worker_responses())
        
        logger.info(f"Multi-controller service started with {len(self.worker_connections)} worker types"
                   f" using {'non-blocking' if self.use_nonblocking_handler else 'asyncio.wait()'} handler")
    
    async def handle_worker_responses(self):
        """Continuously handle responses from all workers - runs as a background task.
        
        Uses asyncio.wait() to efficiently wait for any worker to have a message ready,
        avoiding nested timeouts and busy polling. This approach scales well with 
        many workers and provides optimal performance.
        """
        logger.info("Starting worker response handler task")
        
        while self.running:
            try:
                if not self.worker_connections:
                    # No workers registered, sleep and continue
                    await asyncio.sleep(0.1)
                    continue
                
                # Create pull tasks for all workers
                pull_tasks = {}
                for worker_type, worker_connection in self.worker_connections.items():
                    if worker_connection.pull_socket and not worker_connection.pull_socket.closed:
                        task = asyncio.create_task(
                            worker_connection.pull_response(),
                            name=f"pull_{worker_type}"
                        )
                        pull_tasks[task] = worker_type
                
                if not pull_tasks:
                    # No active worker sockets, sleep and continue
                    await asyncio.sleep(0.1)
                    continue
                
                # Wait for any worker to have a message (or timeout after reasonable period)
                try:
                    done_tasks, pending_tasks = await asyncio.wait(
                        pull_tasks.keys(),
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=0.1  # Reasonable timeout to allow service shutdown checks
                    )
                    
                    # Process completed tasks
                    for task in done_tasks:
                        worker_type = pull_tasks[task]
                        try:
                            response = await task  # Get the result
                            
                            # Call the registered handler for this worker type
                            if worker_type in self.worker_message_handlers:
                                await self.worker_message_handlers[worker_type](response)
                            else:
                                await self.default_worker_handler(worker_type, response)
                                
                        except Exception as e:
                            logger.error(f"Error handling response from {worker_type} worker: {e}")
                    
                    # Cancel any pending tasks to avoid resource leaks
                    for task in pending_tasks:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logger.debug(f"Exception while cancelling task: {e}")
                
                except asyncio.TimeoutError:
                    # No messages from any worker within timeout - this is normal
                    # Cancel all pending tasks
                    for task in pull_tasks.keys():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logger.debug(f"Exception while cancelling task: {e}")
                
            except Exception as e:
                logger.error(f"Error in worker response handler: {e}")
                # Use service sleep for state-aware sleeping during error recovery
                if not await self._sleep_if_running(0.1):
                    break
                    
        logger.info("Worker response handler task stopped")

    async def handle_worker_responses_nonblocking(self):
        """Alternative implementation using non-blocking socket operations.
        
        This approach uses zmq.NOBLOCK to check for messages without any timeouts,
        providing maximum performance for high-throughput scenarios.
        """
        logger.info("Starting non-blocking worker response handler task")
        
        while self.running:
            try:
                if not self.worker_connections:
                    await asyncio.sleep(0.01)  # Brief sleep when no workers
                    continue
                
                messages_received = 0
                
                # Check all workers for available messages without blocking
                for worker_type, worker_connection in self.worker_connections.items():
                    if not self.running:
                        break
                        
                    try:
                        response = await worker_connection.pull_response_nonblocking()
                        if response is not None:
                            messages_received += 1
                            
                            # Call the registered handler for this worker type
                            if worker_type in self.worker_message_handlers:
                                await self.worker_message_handlers[worker_type](response)
                            else:
                                await self.default_worker_handler(worker_type, response)
                    
                    except Exception as e:
                        logger.error(f"Error handling response from {worker_type} worker: {e}")
                
                # Adaptive sleep: shorter when busy, longer when idle
                if messages_received == 0:
                    await asyncio.sleep(0.01)  # 10ms when idle
                else:
                    await asyncio.sleep(0.001)  # 1ms when busy
                
            except Exception as e:
                logger.error(f"Error in non-blocking worker response handler: {e}")
                if not await self._sleep_if_running(0.1):
                    break
                    
        logger.info("Non-blocking worker response handler task stopped")

    async def push_task(self, message: MessageDataType) -> bool:
        """Push a task to the appropriate worker based on the message type.
        
        Automatically routes the message to the correct worker based on the 'type' field
        in the message and the configured message type routing.
        
        Args:
            message: The task message to send. Must contain a 'type' field that matches
                    a configured message type for one of the registered workers.
        
        Returns:
            True if task was successfully sent to a worker, False otherwise
            
        Example:
            # Message will be routed to the worker configured to handle RENDER_REQUEST
            success = await controller.push_task({
                "type": "RENDER_REQUEST",
                "prompt": "a beautiful landscape",
                "style": "photorealistic"
            })
        """
        # get worker from message type
        message_type = message.get("type", None)
        if message_type is None:
            logger.error(f"Cannot push task: message does not contain 'type' field: {message}")
            return False
        worker_type = str(message_type)

        worker_connection = self.get_worker_connection(worker_type)
        return await worker_connection.push_task(message)
    
    async def pull_response(self, worker_or_message_type: Union[str, MessageType]) -> MessageDataType:
        """Pull a response from a worker by worker type or message type.
        
        Args:
            worker_or_message_type: Either a worker type name (e.g., 'image') or
                                   a MessageType enum that identifies which worker to pull from
        
        Returns:
            The received message from the specified worker
            
        Raises:
            ValueError: If the worker type or message type is not registered
            RuntimeError: If the worker's socket is not initialized
            
        Example:
            # Pull from worker by type
            response = await controller.pull_response("image")
            
            # Pull from worker by message type it handles
            response = await controller.pull_response(MessageType.IMAGE_READY)
        """

        worker_connection = self.get_worker_connection(worker_or_message_type)
        return await worker_connection.pull_response()

    def register_workers_from_config(self, config: ControllerMultiWorkerConfig):
        """Register workers from a validated Pydantic configuration object.
        
        This method provides type-safe configuration with automatic validation.
        
        Args:
            config: MultiControllerWorkerConfig object with validated worker configurations
            
        Example:
            from experimance_common.zmq.controller import MultiControllerWorkerConfig, WorkerConnectionConfig
            
            config = MultiControllerWorkerConfig(workers={
                "image": WorkerConnectionConfig(
                    push_address="tcp://*:5563",
                    pull_address="tcp://*:5564",
                    push_message_types=[MessageType.RENDER_REQUEST],
                    pull_message_types=[MessageType.IMAGE_READY]
                ),
                "transition": WorkerConnectionConfig(
                    push_address="tcp://*:5561",
                    pull_address="tcp://*:5565",
                    push_message_types=[MessageType.TRANSITION_REQUEST],
                    pull_message_types=[MessageType.TRANSITION_READY]
                )
            })
            
            controller.register_workers_from_config(config)
        """
        # Validate configuration and check for conflicts before registering any workers
        config.check_port_conflicts()
        config.validate_message_type_routing()
        self.worker_config = config
        self.worker_message_routing_map = config.get_routing_map()

        for worker_type, worker_config in config.workers.items():
            self.register_worker(
                worker_type=worker_type,  # Use the key from the dictionary as worker_type
                push_address=worker_config.push_address,
                pull_address=worker_config.pull_address,
                push_message_types=worker_config.push_message_types,
                pull_message_types=worker_config.pull_message_types
            )
            logger.info(f"Registered worker type '{worker_type}' from Pydantic config")
        
        logger.info(f"Successfully registered {len(config.workers)} worker types from configuration")

    def setup_workers_from_config_provider(self, config_provider):
        """Set up worker connections from a configuration provider.
        
        This is a convenient method for services to automatically set up workers
        from their configuration. The config_provider must have a get_controller_config()
        method that returns a ControllerMultiWorkerConfig.
        
        Args:
            config_provider: Object with get_controller_config() method
            
        Raises:
            AttributeError: If config_provider doesn't have get_controller_config() method
            Exception: If worker setup fails
        """
        try:
            # Get the ControllerMultiWorkerConfig from the provider
            if not hasattr(config_provider, 'get_controller_config'):
                raise AttributeError(
                    f"Config provider {type(config_provider)} must have get_controller_config() method"
                )
            
            controller_config = config_provider.get_controller_config()
            
            # Register workers with the new configuration
            self.register_workers_from_config(controller_config)
            
            logger.info(f"Configured {len(controller_config.workers)} worker types: {list(controller_config.workers.keys())}")
            
            # Log worker details for debugging
            for worker_type, config in controller_config.workers.items():
                logger.info(f"Worker '{worker_type}': push={config.push_address}, pull={config.pull_address}")
                logger.info(f"  - Accepts: {[mt.value for mt in config.push_message_types]}")
                logger.info(f"  - Sends: {[mt.value for mt in config.pull_message_types]}")
                
        except Exception as e:
            logger.error(f"Failed to setup workers from config: {e}")
            raise

    def get_worker_connection(self, worker_or_message_type: Union[str, MessageType]) -> WorkerConnection:
        """Get the WorkerConnection object for a worker by type or message type.

        This method allows flexible access to worker connections using either:
        - Direct worker type names (e.g., 'image', 'transition')  
        - Message types that the workers handle (e.g., MessageType.RENDER_REQUEST)

        Args:
            worker_or_message_type: Either a worker type name or a MessageType enum
                                   that identifies which worker connection to retrieve

        Returns:
            WorkerConnection object for the specified worker

        Raises:
            ValueError: If the worker type or message type is not registered
            
        Example:
            # Get connection by worker type
            connection = controller.get_worker_connection("image")
            
            # Get connection by message type the worker handles
            connection = controller.get_worker_connection(MessageType.RENDER_REQUEST)
        """
        if worker_or_message_type in self.worker_connections:
            return self.worker_connections[worker_or_message_type]
        if worker_or_message_type in self.worker_message_routing_map:
            try:
                # If it's a message type, find the worker type that handles it
                worker_type = self.worker_message_routing_map[worker_or_message_type]
                return self.worker_connections[worker_type]  
            except KeyError:
                raise ValueError(f"Message type {worker_or_message_type} is not handled by any registered worker")
        raise ValueError(f"Worker type {worker_or_message_type} is not registered")

    def get_worker_types(self) -> List[str]:
        """Get list of registered worker types."""
        return list(self.worker_connections.keys())
    
    def has_worker(self, worker_or_message_type: Union[str, MessageType]) -> bool:
        """Check if a worker is registered by worker type or message type.
        
        Args:
            worker_or_message_type: Either a worker type name or a MessageType enum
        
        Returns:
            True if a worker is registered that matches the type or handles the message type
            
        Example:
            # Check by worker type
            if controller.has_worker("image"):
                print("Image worker is available")
                
            # Check by message type
            if controller.has_worker(MessageType.RENDER_REQUEST):
                print("A worker can handle render requests")
        """
        try:
            self.get_worker_connection(worker_or_message_type)
            return True
        except ValueError:
            return False # If we get a ValueError, the worker type is not registered
    
    def get_worker_info(self, worker_type: str) -> Dict[str, Any]:
        """Get information about a specific worker."""
        if worker_type not in self.worker_connections:
            raise ValueError(f"Worker type {worker_type} is not registered")
        
        connection = self.worker_connections[worker_type]
        return {
            "worker_type": connection.worker_type,
            "push_address": connection.push_address,
            "pull_address": connection.pull_address,
            "push_message_types": [msg_type.value for msg_type in connection.push_message_types],
            "pull_message_types": [msg_type.value for msg_type in connection.pull_message_types],
            "sockets_initialized": connection.push_socket is not None and connection.pull_socket is not None
        }
    
    def list_workers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered workers."""
        return {worker_type: self.get_worker_info(worker_type) 
                for worker_type in self.worker_connections.keys()}
    
    #def 

    def _check_port_conflicts(self) -> None:
        """Check for port conflicts between workers and with the service itself.
        
        Raises:
            ValueError: If port conflicts are detected
        """
        used_ports = set()
        
        # Check service's own ports
        try:
            pub_port = int(self.pub_address.split(":")[-1])
            sub_port = int(self.sub_address.split(":")[-1])
            used_ports.update([pub_port, sub_port])
        except (ValueError, IndexError):
            pass  # Address might not be in expected format
        
        # Check worker ports
        for worker_type, connection in self.worker_connections.items():
            try:
                push_port = int(connection.push_address.split(":")[-1])
                pull_port = int(connection.pull_address.split(":")[-1])
                
                if push_port in used_ports:
                    raise ValueError(f"Port conflict: {worker_type} push port {push_port} already in use")
                if pull_port in used_ports:
                    raise ValueError(f"Port conflict: {worker_type} pull port {pull_port} already in use")
                    
                used_ports.update([push_port, pull_port])
                
            except (ValueError, IndexError) as e:
                if "Port conflict" in str(e):
                    raise
                # Ignore address parsing errors for now
                continue