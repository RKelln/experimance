"""
ZMQ Composed Services for Experimance Project

This module provides high-level, composed ZMQ services that combine multiple
socket components to handle common messaging patterns in the Experimance system.

Services implemented:
- PubSubService: Publisher + Subscriber for bidirectional communication
- WorkerService: Pull worker with optional Push responses  
- ControllerService: Push commands + Pull responses (controller pattern)

All services use composition with the BaseZmqComponent classes and integrate
with the existing BaseService pattern for graceful shutdown and logging.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable
from contextlib import AsyncExitStack

from experimance_common.config import BaseConfig
from .components import (
    BaseZmqComponent,
    PublisherComponent,
    SubscriberComponent, 
    PushComponent,
    PullComponent
)
from .config import (
    PubSubServiceConfig,
    WorkerServiceConfig,
    ControllerServiceConfig,
    PublisherConfig,
    SubscriberConfig,
    PushConfig,
    PullConfig,
    MessageDataType,
    TopicType
)


class BaseZmqService:
    """
    Base class for all composed ZMQ services.
    
    Provides common functionality:
    - Component lifecycle management
    - Error handling and recovery
    - Graceful shutdown
    - Health monitoring
    """
    
    def __init__(self, name: str = "ZmqService"):
        self.name = name
        self.logger = logging.getLogger(f"experimance.zmq.{name}")
        self._components: List[BaseZmqComponent] = []
        self._exit_stack: Optional[AsyncExitStack] = None
        self._running = False
        self._health_check_interval = 30.0  # seconds
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        
    async def start(self) -> None:
        """Start all components in the service."""
        if self._running:
            self.logger.warning(f"{self.name} is already running")
            return
            
        self.logger.info(f"Starting {self.name}")
        self._exit_stack = AsyncExitStack()
        
        try:
            # Start all components
            for component in self._components:
                await self._exit_stack.enter_async_context(component)
                self.logger.debug(f"Started component: {component.__class__.__name__}")
                
            self._running = True
            self.logger.info(f"{self.name} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start {self.name}: {e}")
            await self.stop()
            raise
            
    async def stop(self) -> None:
        """Stop all components gracefully."""
        if not self._running:
            return
            
        self.logger.info(f"Stopping {self.name}")
        self._running = False
        
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
                self.logger.info(f"{self.name} stopped successfully")
            except Exception as e:
                self.logger.error(f"Error stopping {self.name}: {e}")
            finally:
                self._exit_stack = None
                
    def is_running(self) -> bool:
        """Check if the service is running."""
        return self._running
        
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Dict with health status and component details
        """
        health = {
            "service": self.name,
            "running": self._running,
            "components": {},
            "overall_healthy": True
        }
        
        for component in self._components:
            component_name = component.__class__.__name__
            try:
                # Check if component socket is still valid
                component_healthy = component.socket and not component.socket.closed
                health["components"][component_name] = {
                    "healthy": component_healthy,
                    "address": getattr(component.config, 'address', 'unknown')
                }
                if not component_healthy:
                    health["overall_healthy"] = False
                    
            except Exception as e:
                health["components"][component_name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health["overall_healthy"] = False
                
        return health


class PubSubService(BaseZmqService):
    """
    Publisher + Subscriber service for bidirectional communication.
    
    This service provides both publishing and subscribing capabilities,
    commonly used for services that need to both broadcast state updates
    and receive updates from other services.
    """
    
    def __init__(self, config: PubSubServiceConfig, name: Optional[str] = None):
        # Build service name based on available components
        service_parts = []
        if config.publisher:
            service_parts.append(f"Pub-{config.publisher.port}")
        if config.subscriber:
            service_parts.append(f"Sub-{config.subscriber.port}")
        service_name = name or f"PubSub-{'-'.join(service_parts) if service_parts else 'NoSockets'}"
        
        super().__init__(service_name)
        self.config = config
        
        # Create components only if configs are provided
        self.publisher = PublisherComponent(config.publisher) if config.publisher else None
        self.subscriber = SubscriberComponent(config.subscriber) if config.subscriber else None
        
        # Register components for lifecycle management
        self._components = []
        if self.publisher:
            self._components.append(self.publisher)
        if self.subscriber:
            self._components.append(self.subscriber)
        
        # Validate that at least one component is configured
        if not self._components:
            raise ValueError("PubSubService requires at least one of publisher or subscriber configuration")
        
        # Message handlers
        self._message_handlers: Dict[bytes, Callable] = {}
        self._default_handler: Optional[Callable] = None
        
    async def publish(self, data: MessageDataType, topic: Optional[TopicType] = None) -> None:
        """
        Publish a message to a topic or default topic.
        
        Args:
            data: The data to publish (MessageDataType: Dict or MessageBase)
            topic: The topic to publish to (optional, uses default_topic or extracts from message)
        """
        if not self.publisher:
            raise RuntimeError("No publisher configured for this service")
            
        try:
            resolved_topic = await self.publisher.publish(data, topic)
            self.logger.debug(f"Published to topic '{resolved_topic}'")
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            raise
            
    def _get_resolved_topic(self, data: MessageDataType) -> str:
        """Get the resolved topic for debugging/logging purposes."""
        # This mirrors the logic in PublisherComponent._resolve_topic
        if isinstance(data, dict) and 'type' in data:
            return str(data['type'])
        elif hasattr(data, 'type'):  # MessageBase
            return str(data.type)  # type: ignore
        elif self.config.publisher and self.config.publisher.default_topic:
            return str(self.config.publisher.default_topic)
        return ""
            
    def add_message_handler(self, topic: TopicType, handler: Callable[[MessageDataType], Union[None, Awaitable[None]]]) -> None:
        """
        Add a message handler for a specific topic.
        
        Args:
            topic: The topic to handle
            handler: Function to handle messages for this topic (message: MessageDataType). Can be sync or async.
        """
        if not self.subscriber:
            raise RuntimeError("No subscriber configured for this service")
            
        topic_str = str(topic)
        self.subscriber.register_handler(topic_str, handler)
        self.logger.debug(f"Added handler for topic '{topic_str}'")
        
    def set_default_handler(self, handler: Callable[[str, MessageDataType], Union[None, Awaitable[None]]]) -> None:
        """
        Set a default handler for unmatched topics.
        
        Args:
            handler: Function to handle unmatched messages (topic: str, message: MessageDataType). Can be sync or async.
        """
        self._default_handler = handler
        self.logger.debug("Set default message handler")
        
    async def start(self) -> None:
        """Start the service and begin message processing."""
        await super().start()
        
        # Set up default handler on subscriber to route to our internal handlers
        if self.subscriber:
            self.subscriber.set_default_handler(self._handle_message)
        
    async def stop(self) -> None:
        """Stop the service and message processing."""        
        await super().stop()
        
    async def _handle_message(self, topic: str, message: MessageDataType) -> None:
        """Handle a received message by routing to appropriate handler."""
        try:
            topic_bytes = topic.encode() if isinstance(topic, str) else topic
            
            # Find appropriate handler
            handler = self._message_handlers.get(topic_bytes, self._default_handler)
            
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(topic, message)
                else:
                    handler(topic, message)
            else:
                self.logger.warning(f"No handler for topic '{topic}'")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")


class WorkerService(BaseZmqService):
    """
    Worker service with pubsub + push + pull combined.
    
    This service has full communication capabilities:
    - Publisher/Subscriber for status and coordination  
    - Pull for receiving work
    - Push for sending results
    Commonly used for distributed work processing with coordination.
    """
    
    def __init__(self, config: WorkerServiceConfig, name: Optional[str] = None):
        super().__init__(name or f"Worker-{config.pull.port}")
        self.config = config
        
        # Create all components according to plan
        self.publisher = PublisherComponent(config.publisher)
        self.subscriber = SubscriberComponent(config.subscriber)
        self.puller = PullComponent(config.pull)
        self.pusher = PushComponent(config.push)
        
        # Register all components for lifecycle management
        self._components = [self.publisher, self.subscriber, self.puller, self.pusher]
        
        # Message handlers for subscriber
        self._message_handlers: Dict[bytes, Callable] = {}
        self._default_handler: Optional[Callable] = None
        
        # Work handler
        self._work_handler: Optional[Callable] = None
        self._max_concurrent_tasks = config.max_concurrent_tasks
        self._current_tasks: Set[asyncio.Task] = set()
        
    def set_work_handler(self, handler: Callable) -> None:
        """
        Set the work handler function.
        
        Args:
            handler: Async callable that processes work items
                    Should accept (work_data) and optionally return response
        """
        self._work_handler = handler
        self.logger.debug("Set work handler")
        
    async def publish(self, data: MessageDataType, topic: Optional[TopicType] = None) -> None:
        """Publish a message via the publisher component."""
        try:
            resolved_topic = await self.publisher.publish(data, topic)
            self.logger.debug(f"Published message to topic '{resolved_topic}'")
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            raise
            
    def add_message_handler(self, topic: TopicType, handler: Callable[[str, MessageDataType], None]) -> None:
        """Add a message handler for subscriber component."""
        topic_str = str(topic)
        topic_bytes = topic_str.encode() if isinstance(topic_str, str) else topic_str
        self._message_handlers[topic_bytes] = handler
        self.logger.debug(f"Added handler for topic '{topic_str}'")
        
    def set_default_handler(self, handler: Callable[[str, MessageDataType], None]) -> None:
        """Set a default handler for unmatched subscriber topics."""
        self._default_handler = handler
        self.logger.debug("Set default message handler")
        
    async def send_response(self, response: Any) -> None:
        """
        Send a response (requires push component configured).
        
        Args:
            response: Response data to send
        """
        if not self.pusher:
            raise RuntimeError("No push component configured for responses")
            
        try:
            await self.pusher.push(response)
            self.logger.debug("Sent response")
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")
            raise
            
    async def start(self) -> None:
        """Start the service and begin work processing."""
        if not self._work_handler:
            raise RuntimeError("Work handler must be set before starting")
            
        await super().start()
        
        # Set up work handler on puller
        self.puller.set_work_handler(self._handle_work)
        
        # Set up subscriber default handler for routing
        self.subscriber.set_default_handler(self._handle_subscriber_message)
        
    async def stop(self) -> None:
        """Stop the service and finish current work."""
        # Wait for current tasks to complete
        if self._current_tasks:
            self.logger.info(f"Waiting for {len(self._current_tasks)} tasks to complete")
            await asyncio.gather(*self._current_tasks, return_exceptions=True)
            
        await super().stop()
        
    async def _handle_work(self, work_data: MessageDataType) -> None:
        """Handle a work item."""
        # Check if we can accept more work
        if len(self._current_tasks) >= self._max_concurrent_tasks:
            self.logger.warning("Max concurrent tasks reached, dropping work item")
            return
            
        # Create task for work processing
        task = asyncio.create_task(self._process_work_item(work_data))
        self._current_tasks.add(task)
        
        # Clean up completed tasks
        done_tasks = [task for task in self._current_tasks if task.done()]
        for task in done_tasks:
            self._current_tasks.discard(task)
            if task.exception():
                self.logger.error(f"Work task failed: {task.exception()}")
        
    async def _process_work_item(self, work_data: MessageDataType) -> None:
        """Process a single work item."""
        try:
            self.logger.debug("Processing work item")
            
            if self._work_handler is None:
                self.logger.error("No work handler set")
                return
                
            if asyncio.iscoroutinefunction(self._work_handler):
                result = await self._work_handler(work_data)
            else:
                result = self._work_handler(work_data)
                
            # Send response if we have a result and push component
            if result is not None and self.pusher:
                await self.send_response(result)
                
        except Exception as e:
            self.logger.error(f"Error processing work item: {e}")
        finally:
            # Remove from current tasks when done
            current_task = asyncio.current_task()
            if current_task:
                self._current_tasks.discard(current_task)

    async def _handle_subscriber_message(self, topic: str, data: MessageDataType) -> None:
        """Handle messages received via subscriber component."""
        try:
            # Find appropriate handler
            topic_bytes = topic.encode() if isinstance(topic, str) else topic
            handler = self._message_handlers.get(topic_bytes, self._default_handler)
            
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(topic, data)
                else:
                    handler(topic, data)
            else:
                self.logger.warning(f"No handler for subscriber topic '{topic}'")
                
        except Exception as e:
            self.logger.error(f"Error handling subscriber message: {e}")


class ControllerService(BaseZmqService):
    """
    Controller service with pubsub + multi push/pull workers.
    
    This service provides coordination and work distribution:
    - Publisher/Subscriber for status and coordination
    - Multiple Push/Pull worker connections for distributed work
    Commonly used for coordinating distributed work across multiple workers.
    """
    
    def __init__(self, config: ControllerServiceConfig, name: Optional[str] = None):
        super().__init__(name or f"Controller-{config.publisher.port}")
        self.config = config
        
        # Create pubsub components
        self.publisher = PublisherComponent(config.publisher)
        self.subscriber = SubscriberComponent(config.subscriber)
        self._components = [self.publisher, self.subscriber]
        
        # Create worker push/pull components
        self.workers: Dict[str, Dict[str, Any]] = {}
        for worker_name, worker_config in config.workers.items():
            push_comp = PushComponent(worker_config.push_config)
            pull_comp = PullComponent(worker_config.pull_config)
            
            self.workers[worker_name] = {
                'push': push_comp,
                'pull': pull_comp
            }
            self._components.extend([push_comp, pull_comp])
            
        # Message handlers for subscriber
        self._message_handlers: Dict[bytes, Callable] = {}
        self._default_handler: Optional[Callable] = None
        
        # Response handlers for workers
        self._response_handlers: List[Callable] = []
        self._pending_responses: Dict[str, asyncio.Future] = {}
        
    async def publish(self, data: MessageDataType, topic: Optional[TopicType] = None) -> None:
        """Publish a message via the publisher component."""
        try:
            resolved_topic = await self.publisher.publish(data, topic)
            self.logger.debug(f"Published message to: {resolved_topic}")
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            raise
            
    def add_message_handler(self, topic: TopicType, handler: Callable[[str, MessageDataType], None]) -> None:
        """Add a message handler for subscriber component."""
        topic_str = str(topic)
        topic_bytes = topic_str.encode() if isinstance(topic_str, str) else topic_str
        self._message_handlers[topic_bytes] = handler
        self.logger.debug(f"Added handler for topic '{topic_str}'")
        
    def set_default_handler(self, handler: Callable[[str, MessageDataType], None]) -> None:
        """Set a default handler for unmatched subscriber topics."""
        self._default_handler = handler
        self.logger.debug("Set default message handler")
        
    async def send_work_to_worker(self, worker_name: str, work_data: Any) -> None:
        """
        Send work to a specific worker.
        
        Args:
            worker_name: Name of the worker to send work to
            work_data: Work data to send
        """
        if worker_name not in self.workers:
            raise ValueError(f"Unknown worker: {worker_name}")
            
        try:
            push_component = self.workers[worker_name]['push']
            await push_component.push(work_data)
            self.logger.debug(f"Sent work to worker '{worker_name}'")
        except Exception as e:
            self.logger.error(f"Failed to send work to worker '{worker_name}': {e}")
            raise
            
    async def send_work_to_all_workers(self, work_data: Any) -> None:
        """Send work to all workers (broadcast)."""
        for worker_name in self.workers:
            try:
                await self.send_work_to_worker(worker_name, work_data)
            except Exception as e:
                self.logger.error(f"Failed to send work to worker '{worker_name}': {e}")
                
    def add_response_handler(self, handler: Callable) -> None:
        """Add a response handler for worker responses."""
        self._response_handlers.append(handler)
        self.logger.debug("Added response handler")
        
    async def start(self) -> None:
        """Start the service and message/response processing."""
        await super().start()
        
        # Set up subscriber default handler for routing
        self.subscriber.set_default_handler(self._handle_subscriber_message)
        
        # Set up worker response handlers
        for worker_name, worker_components in self.workers.items():
            pull_component = worker_components['pull']
            # Create a wrapper function to capture worker_name
            def create_handler(name: str):
                async def handler(data: MessageDataType) -> None:
                    await self._handle_worker_response(name, data)
                return handler
            pull_component.set_work_handler(create_handler(worker_name))
            
    async def stop(self) -> None:
        """Stop the service gracefully."""
        await super().stop()
        
    async def _handle_subscriber_message(self, topic: str, data: MessageDataType) -> None:
        """Handle messages received via subscriber component."""
        try:
            # Find appropriate handler
            topic_bytes = topic.encode() if isinstance(topic, str) else topic
            handler = self._message_handlers.get(topic_bytes, self._default_handler)
            
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(topic, data)
                else:
                    handler(topic, data)
            else:
                self.logger.warning(f"No handler for subscriber topic '{topic}'")
                
        except Exception as e:
            self.logger.error(f"Error handling subscriber message: {e}")
            
    async def _handle_worker_response(self, worker_name: str, response_data: MessageDataType) -> None:
        """Handle responses from worker pull components."""
        try:
            self.logger.debug(f"Processing response from worker '{worker_name}'")
            
            # Call all response handlers
            for handler in self._response_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(worker_name, response_data)
                    else:
                        handler(worker_name, response_data)
                except Exception as e:
                    self.logger.error(f"Error in response handler: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling worker response: {e}")


# # Quick service creation from BaseConfig

# def create_pubsub_from_config(base_config: BaseConfig, 
#                              pub_port: int, sub_port: int,
#                              sub_topics: Optional[List[str]] = None,
#                              name: Optional[str] = None) -> PubSubService:
#     """
#     Create a PubSubService from BaseConfig with specific ports.
    
#     Args:
#         base_config: Base configuration object
#         pub_port: Publisher port
#         sub_port: Subscriber port  
#         sub_topics: Topics to subscribe to
#         name: Service name
#     """
#     config = PubSubServiceConfig(
#         publisher=PublisherConfig(
#             address=f"tcp://*:{pub_port}",
#             port=pub_port
#         ),
#         subscriber=SubscriberConfig(
#             address=f"tcp://localhost:{sub_port}",
#             port=sub_port,
#             topics=sub_topics or []
#         )
#     )
#     return PubSubService(config, name)

# def create_worker_from_config(base_config: BaseConfig,
#                              pull_port: int,
#                              push_port: Optional[int] = None,
#                              max_concurrent: int = 10,
#                              name: Optional[str] = None) -> WorkerService:
#     """
#     Create a WorkerService from BaseConfig with specific ports.
    
#     Args:
#         base_config: Base configuration object
#         pull_port: Pull port for receiving work
#         push_port: Optional push port for responses
#         max_concurrent: Maximum concurrent tasks
#         name: Service name
#     """
#     push_config = None
#     if push_port:
#         push_config = PushConfig(
#             address=f"tcp://*",
#             port=push_port
#         )
#     else:
#         # Provide a default push config if none specified
#         push_config = PushConfig(
#             address=f"tcp://*",
#             port=pull_port + 300
#         )
        
#     config = WorkerServiceConfig(
#         publisher=PublisherConfig(address="tcp://*", port=pull_port + 100),
#         subscriber=SubscriberConfig(address="tcp://localhost", port=pull_port + 200, topics=["status"]),
#         pull=PullConfig(
#             address=f"tcp://localhost:{pull_port}",
#             port=pull_port
#         ),
#         push=push_config,
#         max_concurrent_tasks=max_concurrent
#     )
#     return WorkerService(config, name)


async def test_services():
    """Basic test function for the services."""
    import tempfile
    import json
    
    print("Testing ZMQ Services...")
    
    # Test PubSubService
    try:
        config = PubSubServiceConfig(
            publisher=PublisherConfig(address="tcp://*", port=9901),
            subscriber=SubscriberConfig(
                address="tcp://localhost", 
                port=9902,
                topics=["test"]
            )
        )
        
        async with PubSubService(config, "TestPubSub") as service:
            print("✓ PubSubService created and started")
            
            # Test health check
            health = await service.health_check()
            print(f"✓ Health check: {health['overall_healthy']}")
            
    except Exception as e:
        print(f"✗ PubSubService test failed: {e}")
        
    # Test WorkerService  
    try:
        config = WorkerServiceConfig(
            publisher=PublisherConfig(address="tcp://*", port=9903),
            subscriber=SubscriberConfig(address="tcp://localhost", port=9904, topics=["status"]),
            pull=PullConfig(address="tcp://*", port=9905, bind=True),
            push=PushConfig(address="tcp://localhost", port=9906, bind=False),
            max_concurrent_tasks=5
        )
        
        worker = WorkerService(config, "TestWorker")
        
        # Set a simple work handler
        async def work_handler(data):
            return f"processed: {data}"
            
        worker.set_work_handler(work_handler)
        
        async with worker:
            print("✓ WorkerService created and started")
            
            health = await worker.health_check() 
            print(f"✓ Health check: {health['overall_healthy']}")
            
    except Exception as e:
        print(f"✗ WorkerService test failed: {e}")
        
    # Test ControllerService
    try:
        from experimance_common.zmq.config import create_local_controller_config
        
        config = create_local_controller_config("test_controller")
        
        async with ControllerService(config, "TestController") as service:
            print("✓ ControllerService created and started")
            
            health = await service.health_check()
            print(f"✓ Health check: {health['overall_healthy']}")
            
    except Exception as e:
        print(f"✗ ControllerService test failed: {e}")
        
    print("Service tests completed!")


if __name__ == "__main__":
    asyncio.run(test_services())
