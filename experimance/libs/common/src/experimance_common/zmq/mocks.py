"""
Mock implementations for ZMQ services.

This module provides reusable mock implementations for testing ZMQ-based services
without requiring actual ZMQ sockets. All mocks follow the frozen config pattern
and provide the same interface as the real ZMQ services.

Key Features:
- Global message bus for cross-service testing
- Async-friendly mock services
- Message history tracking for assertions
- Drop-in replacements for real ZMQ services
- Follows frozen config + mutable state pattern

Usage Examples:

    # Recommended: Use inline config (minimal, clean)
    from experimance_common.zmq.config import (
        PubSubServiceConfig, PublisherConfig, SubscriberConfig
    )
    from experimance_common.zmq.mocks import MockPubSubService
    
    config = PubSubServiceConfig(
        name="test",
        publisher=PublisherConfig(address="tcp://*", port=5555),
        subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=["test"])
    )
    
    async with MockPubSubService(config) as service:
        await service.publish("test", {"message": "hello"})
        messages = service.get_published_messages()
    
    # Alternative: Use BaseConfig.from_overrides() for production-like testing
    base_config = PubSubServiceConfig(name="test")
    config = PubSubServiceConfig.from_overrides(
        default_config=base_config,
        override_config={"subscriber": {"topics": ["custom.topic"]}}
    )
"""

import asyncio
import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Callable, Deque, Union, Awaitable
from unittest.mock import AsyncMock, MagicMock
import json
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

try:
    # Relative import when used as a module
    from .config import (
        PublisherConfig, SubscriberConfig, PushConfig, PullConfig,
        PubSubServiceConfig, WorkerServiceConfig, ControllerServiceConfig,
        WorkerConfig, TopicType, MessageDataType
    )
except ImportError:
    # Absolute import when run directly
    from experimance_common.zmq.config import (
        PublisherConfig, SubscriberConfig, PushConfig, PullConfig,
        PubSubServiceConfig, WorkerServiceConfig, ControllerServiceConfig,
        WorkerConfig, TopicType, MessageDataType
    )

logger = logging.getLogger(__name__)

# =============================================================================
# MOCK MESSAGE STORAGE
# =============================================================================

@dataclass
class MockMessage:
    """Represents a captured message in testing."""
    topic: str
    content: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    sender_id: str = ""


class MockMessageBus:
    """Global message bus for inter-service communication during testing."""
    
    def __init__(self):
        self.messages: List[MockMessage] = []
        self.subscribers: Dict[str, List[Callable]] = {}
        self.workers: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
    
    async def publish(self, topic: str, content: Dict[str, Any], sender_id: str = ""):
        """Publish a message to all subscribers."""
        message = MockMessage(topic=topic, content=content, sender_id=sender_id)
        
        async with self._lock:
            self.messages.append(message)
            
            # Notify subscribers
            if topic in self.subscribers:
                for handler in self.subscribers[topic]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(topic, content)
                        else:
                            handler(topic, content)
                    except Exception as e:
                        logger.error(f"Error in mock subscriber handler: {e}")
    
    async def push_to_worker(self, worker_name: str, content: Dict[str, Any], sender_id: str = ""):
        """Push work to a specific worker."""
        message = MockMessage(topic=f"worker.{worker_name}", content=content, sender_id=sender_id)
        
        async with self._lock:
            self.messages.append(message)
            
            # Notify worker handlers
            if worker_name in self.workers:
                for handler in self.workers[worker_name]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(content)
                        else:
                            handler(content)
                    except Exception as e:
                        logger.error(f"Error in mock worker handler: {e}")
    
    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(handler)
    
    def register_worker(self, worker_name: str, handler: Callable):
        """Register a worker handler."""
        if worker_name not in self.workers:
            self.workers[worker_name] = []
        self.workers[worker_name].append(handler)
    
    def get_messages(self, topic: Optional[str] = None) -> List[MockMessage]:
        """Get captured messages, optionally filtered by topic."""
        if topic is None:
            return self.messages.copy()
        return [msg for msg in self.messages if msg.topic == topic]
    
    def clear(self):
        """Clear all messages and handlers."""
        self.messages.clear()
        self.subscribers.clear()
        self.workers.clear()


# Global message bus for testing
mock_message_bus = MockMessageBus()

# =============================================================================
# MOCK SERVICES
# =============================================================================

class MockPubSubService:
    """Mock PubSub service following the frozen config pattern."""
    
    def __init__(self, config: PubSubServiceConfig):
        # Store immutable config
        self.config = config
        self.name = config.name
        
        # Mutable runtime state
        self.running = False
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None
        self.error_count = 0
        self.message_count = 0
        
        # Message storage (mutable)
        self.published_messages: List[MockMessage] = []
        self.received_messages: List[MockMessage] = []
        
        # Message handlers (mutable)
        self.message_handlers: List[Callable] = []
        
        # Track subscriptions
        self._subscribed_topics: Set[str] = set()
        
        logger.debug(f"Created MockPubSubService '{self.name}'")
    
    async def start(self):
        """Start the mock service."""
        if self.running:
            raise RuntimeError(f"MockPubSubService '{self.name}' already running")
        
        self.running = True
        self.started_at = asyncio.get_event_loop().time()
        
        # Subscribe to topics if subscriber config exists
        if self.config.subscriber and self.config.subscriber.topics:
            for topic in self.config.subscriber.topics:
                mock_message_bus.subscribe(topic, self._handle_message)
                self._subscribed_topics.add(topic)
        
        logger.debug(f"MockPubSubService '{self.name}' started")
    
    async def stop(self):
        """Stop the mock service."""
        if not self.running:
            return
        
        self.running = False
        self.stopped_at = asyncio.get_event_loop().time()
        
        # Clear subscriptions
        self._subscribed_topics.clear()
        
        logger.debug(f"MockPubSubService '{self.name}' stopped")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def publish(self, data: "MessageDataType", topic: Optional["TopicType"] = None) -> None:
        """Publish a message (matching real PubSubService interface)."""
        if not self.running:
            raise RuntimeError(f"MockPubSubService '{self.name}' not running")
        
        if not self.config.publisher:
            raise RuntimeError(f"MockPubSubService '{self.name}' has no publisher config")
        
        # Resolve topic like the real implementation
        if topic is None:
            if isinstance(data, dict) and 'type' in data:
                resolved_topic = str(data['type'])
            elif hasattr(data, 'type'):  # MessageBase
                resolved_topic = str(data.type)  # type: ignore
            elif self.config.publisher and self.config.publisher.default_topic:
                resolved_topic = str(self.config.publisher.default_topic)
            else:
                resolved_topic = "default"
        else:
            resolved_topic = str(topic)
        
        # Convert data to dict if needed
        if isinstance(data, dict):
            message = data
        else:
            # Assume it's a MessageBase or similar with a dict() method
            message = data.dict() if hasattr(data, 'dict') else dict(data)
        
        # Store message
        mock_msg = MockMessage(topic=resolved_topic, content=message, sender_id=self.name)
        self.published_messages.append(mock_msg)
        self.message_count += 1
        
        # Send to global message bus
        await mock_message_bus.publish(resolved_topic, message, self.name)
        
        logger.debug(f"MockPubSubService '{self.name}' published to {resolved_topic}: {message}")
    
    def set_message_handler(self, handler: Callable):
        """Set message handler for received messages."""
        self.message_handlers.append(handler)
    
    def add_message_handler(self, topic: "TopicType", handler: Callable[["MessageDataType"], Union[None, Awaitable[None]]]) -> None:
        """Add a message handler for a specific topic (matching real PubSubService interface)."""
        topic_str = str(topic)
        # Store topic-specific handlers
        if not hasattr(self, '_topic_handlers'):
            self._topic_handlers = {}
        self._topic_handlers[topic_str] = handler
        logger.debug(f"MockPubSubService '{self.name}' added handler for topic '{topic_str}'")
    
    async def _handle_message(self, topic: str, message: Dict[str, Any]):
        """Handle incoming message."""
        if not self.running:
            return
        
        # Store received message
        mock_msg = MockMessage(topic=topic, content=message, sender_id="message_bus")
        self.received_messages.append(mock_msg)
        self.message_count += 1
        
        # Call topic-specific handlers first (matching real PubSubService behavior)
        if hasattr(self, '_topic_handlers') and topic in self._topic_handlers:
            handler = self._topic_handlers[topic]
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)  # Real interface passes only message
                else:
                    handler(message)  # Real interface passes only message
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in MockPubSubService '{self.name}' topic handler for '{topic}': {e}")
        
        # Call general handlers (legacy behavior)
        for handler in self.message_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(topic, message)
                else:
                    handler(topic, message)
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in MockPubSubService '{self.name}' handler: {e}")
    
    def get_published_messages(self, topic: Optional[str] = None) -> List[MockMessage]:
        """Get published messages, optionally filtered by topic."""
        if topic is None:
            return self.published_messages.copy()
        return [msg for msg in self.published_messages if msg.topic == topic]
    
    def get_received_messages(self, topic: Optional[str] = None) -> List[MockMessage]:
        """Get received messages, optionally filtered by topic."""
        if topic is None:
            return self.received_messages.copy()
        return [msg for msg in self.received_messages if msg.topic == topic]
    
    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self.running
    
    @property
    def uptime(self) -> Optional[float]:
        """Get uptime in seconds."""
        if not self.started_at:
            return None
        end_time = self.stopped_at or asyncio.get_event_loop().time()
        return end_time - self.started_at


class MockWorkerService:
    """Mock Worker service with PubSub and Push/Pull functionality."""
    
    def __init__(self, config: WorkerServiceConfig):
        # Store immutable config
        self.config = config
        self.name = config.name
        
        # Mutable runtime state
        self.running = False
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None
        self.error_count = 0
        self.message_count = 0
        
        # Message storage (mutable)
        self.published_messages: List[MockMessage] = []
        self.received_messages: List[MockMessage] = []
        self.pushed_work: List[MockMessage] = []
        self.pulled_work: List[MockMessage] = []
        
        # Message handlers (mutable)
        self.message_handlers: List[Callable] = []
        self.work_handlers: List[Callable] = []
        
        # Track subscriptions
        self._subscribed_topics: Set[str] = set()
        
        logger.debug(f"Created MockWorkerService '{self.name}'")
    
    async def start(self):
        """Start the mock service."""
        if self.running:
            raise RuntimeError(f"MockWorkerService '{self.name}' already running")
        
        self.running = True
        self.started_at = asyncio.get_event_loop().time()
        
        # Subscribe to topics if subscriber config exists
        if self.config.subscriber and self.config.subscriber.topics:
            for topic in self.config.subscriber.topics:
                mock_message_bus.subscribe(topic, self._handle_message)
                self._subscribed_topics.add(topic)
        
        # Register as worker
        mock_message_bus.register_worker(self.name, self._handle_work)
        
        logger.debug(f"MockWorkerService '{self.name}' started")
    
    async def stop(self):
        """Stop the mock service."""
        if not self.running:
            return
        
        self.running = False
        self.stopped_at = asyncio.get_event_loop().time()
        
        # Clear subscriptions
        self._subscribed_topics.clear()
        
        logger.debug(f"MockWorkerService '{self.name}' stopped")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def publish(self, data: "MessageDataType", topic: Optional["TopicType"] = None) -> None:
        """Publish a message (matching real WorkerService interface)."""
        if not self.running:
            raise RuntimeError(f"MockWorkerService '{self.name}' not running")
        
        # Resolve topic like the real implementation
        if topic is None:
            if isinstance(data, dict) and 'type' in data:
                resolved_topic = str(data['type'])
            elif hasattr(data, 'type'):  # MessageBase
                resolved_topic = str(data.type)  # type: ignore
            else:
                resolved_topic = "default"
        else:
            resolved_topic = str(topic)
        
        # Convert data to dict if needed
        if isinstance(data, dict):
            message = data
        else:
            # Assume it's a MessageBase or similar with a dict() method
            message = data.dict() if hasattr(data, 'dict') else dict(data)
        
        # Store message
        mock_msg = MockMessage(topic=resolved_topic, content=message, sender_id=self.name)
        self.published_messages.append(mock_msg)
        self.message_count += 1
        
        # Send to global message bus
        await mock_message_bus.publish(resolved_topic, message, self.name)
        
        logger.debug(f"MockWorkerService '{self.name}' published to {resolved_topic}: {message}")
    
    async def push(self, message: Dict[str, Any]):
        """Push work."""
        if not self.running:
            raise RuntimeError(f"MockWorkerService '{self.name}' not running")
        
        # Store pushed work
        mock_msg = MockMessage(topic="work", content=message, sender_id=self.name)
        self.pushed_work.append(mock_msg)
        self.message_count += 1
        
        logger.debug(f"MockWorkerService '{self.name}' pushed work: {message}")
    
    def set_message_handler(self, handler: Callable):
        """Set message handler for received messages."""
        self.message_handlers.append(handler)
    
    def add_message_handler(self, topic: "TopicType", handler: Callable[[str, "MessageDataType"], None]) -> None:
        """Add a message handler for a specific topic (matching real WorkerService interface)."""
        # For mocks, we'll add to the general handlers but with topic filtering
        def topic_filter_handler(msg_topic: str, message: Dict[str, Any]):
            if str(topic) == msg_topic:
                handler(msg_topic, message)
        self.message_handlers.append(topic_filter_handler)
    
    def set_work_handler(self, handler: Callable):
        """Set work handler for pulled work."""
        self.work_handlers.append(handler)
    
    async def _handle_message(self, topic: str, message: Dict[str, Any]):
        """Handle incoming message."""
        if not self.running:
            return
        
        # Store received message
        mock_msg = MockMessage(topic=topic, content=message, sender_id="message_bus")
        self.received_messages.append(mock_msg)
        self.message_count += 1
        
        # Call handlers
        for handler in self.message_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(topic, message)
                else:
                    handler(topic, message)
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in MockWorkerService '{self.name}' message handler: {e}")
    
    async def _handle_work(self, message: Dict[str, Any]):
        """Handle incoming work."""
        if not self.running:
            return
        
        # Store pulled work
        mock_msg = MockMessage(topic="work", content=message, sender_id="message_bus")
        self.pulled_work.append(mock_msg)
        self.message_count += 1
        
        # Call work handlers
        for handler in self.work_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in MockWorkerService '{self.name}' work handler: {e}")


class MockControllerService:
    """Mock Controller service with PubSub and multiple workers."""
    
    def __init__(self, config: ControllerServiceConfig):
        # Store immutable config
        self.config = config
        self.name = config.name
        
        # Mutable runtime state
        self.running = False
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None
        self.error_count = 0
        self.message_count = 0
        
        # Message storage (mutable)
        self.published_messages: List[MockMessage] = []
        self.received_messages: List[MockMessage] = []
        
        # Worker state (mutable)
        self.worker_messages: Dict[str, List[MockMessage]] = {}
        for worker_name in config.workers.keys():
            self.worker_messages[worker_name] = []
        
        # Message handlers (mutable)
        self.message_handlers: List[Callable] = []
        self.worker_handlers: Dict[str, List[Callable]] = {}
        for worker_name in config.workers.keys():
            self.worker_handlers[worker_name] = []
        
        # Track subscriptions
        self._subscribed_topics: Set[str] = set()
        
        logger.debug(f"Created MockControllerService '{self.name}' with workers: {list(config.workers.keys())}")
    
    async def start(self):
        """Start the mock service."""
        if self.running:
            raise RuntimeError(f"MockControllerService '{self.name}' already running")
        
        self.running = True
        self.started_at = asyncio.get_event_loop().time()
        
        # Subscribe to topics if subscriber config exists
        if self.config.subscriber and self.config.subscriber.topics:
            for topic in self.config.subscriber.topics:
                mock_message_bus.subscribe(topic, self._handle_message)
                self._subscribed_topics.add(topic)
        
        # Register workers
        for worker_name in self.config.workers.keys():
            mock_message_bus.register_worker(worker_name, self._create_worker_handler(worker_name))
        
        logger.debug(f"MockControllerService '{self.name}' started")
    
    async def stop(self):
        """Stop the mock service."""
        if not self.running:
            return
        
        self.running = False
        self.stopped_at = asyncio.get_event_loop().time()
        
        # Clear subscriptions
        self._subscribed_topics.clear()
        
        logger.debug(f"MockControllerService '{self.name}' stopped")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def publish(self, data: "MessageDataType", topic: Optional["TopicType"] = None) -> None:
        """Publish a message (matching real ControllerService interface)."""
        if not self.running:
            raise RuntimeError(f"MockControllerService '{self.name}' not running")
        
        # Resolve topic like the real implementation
        if topic is None:
            if isinstance(data, dict) and 'type' in data:
                resolved_topic = str(data['type'])
            elif hasattr(data, 'type'):  # MessageBase
                resolved_topic = str(data.type)  # type: ignore
            else:
                resolved_topic = "default"
        else:
            resolved_topic = str(topic)
        
        # Convert data to dict if needed
        if isinstance(data, dict):
            message = data
        else:
            # Assume it's a MessageBase or similar with a dict() method
            message = data.dict() if hasattr(data, 'dict') else dict(data)
        
        # Store message
        mock_msg = MockMessage(topic=resolved_topic, content=message, sender_id=self.name)
        self.published_messages.append(mock_msg)
        self.message_count += 1
        
        # Send to global message bus
        await mock_message_bus.publish(resolved_topic, message, self.name)
        
        logger.debug(f"MockControllerService '{self.name}' published to {resolved_topic}: {message}")
    
    async def send_work_to_worker(self, worker_name: str, work_data: Any) -> None:
        """Send work to a specific worker (matching real ControllerService interface)."""
        if not self.running:
            raise RuntimeError(f"MockControllerService '{self.name}' not running")
        
        if worker_name not in self.config.workers:
            raise ValueError(f"Unknown worker: {worker_name}")
        
        # Convert work_data to dict if needed
        if isinstance(work_data, dict):
            message = work_data
        else:
            # Assume it's a MessageBase or similar with a dict() method
            message = work_data.dict() if hasattr(work_data, 'dict') else dict(work_data)
        
        # Store message for testing
        mock_msg = MockMessage(topic=f"worker.{worker_name}", content=message, sender_id=self.name)
        self.worker_messages[worker_name].append(mock_msg)
        
        # Send via message bus for cross-service testing
        await mock_message_bus.push_to_worker(worker_name, message, self.name)
        
        logger.debug(f"MockControllerService '{self.name}' sent work to worker {worker_name}: {message}")
    
    async def send_work_to_all_workers(self, work_data: Any) -> None:
        """Send work to all workers (matching real ControllerService interface)."""
        for worker_name in self.config.workers:
            await self.send_work_to_worker(worker_name, work_data)
    
    async def push_to_worker(self, worker_name: str, message: Dict[str, Any]):
        """Push message to specific worker (legacy method name for compatibility)."""
        await self.send_work_to_worker(worker_name, message)
    
    def set_message_handler(self, handler: Callable):
        """Set message handler for received messages."""
        self.message_handlers.append(handler)
    
    def add_message_handler(self, topic: "TopicType", handler: Callable[[str, "MessageDataType"], None]) -> None:
        """Add a message handler for a specific topic (matching real ControllerService interface)."""
        # For mocks, we'll add to the general handlers but with topic filtering
        def topic_filter_handler(msg_topic: str, message: Dict[str, Any]):
            if str(topic) == msg_topic:
                handler(msg_topic, message)
        self.message_handlers.append(topic_filter_handler)
    
    def set_worker_handler(self, worker_name: str, handler: Callable):
        """Set handler for specific worker."""
        if worker_name not in self.config.workers:
            raise ValueError(f"Unknown worker: {worker_name}")
        
        self.worker_handlers[worker_name].append(handler)
    
    def add_response_handler(self, handler: Callable) -> None:
        """Add a response handler for worker responses (matching real ControllerService interface)."""
        if not hasattr(self, '_response_handlers'):
            self._response_handlers = []
        self._response_handlers.append(handler)
        logger.debug(f"MockControllerService '{self.name}' added response handler")
    
    def set_default_handler(self, handler: Callable[[str, "MessageDataType"], None]) -> None:
        """Set default handler for unmatched subscriber topics (matching real ControllerService interface)."""
        self._default_handler = handler
        logger.debug(f"MockControllerService '{self.name}' set default message handler")
    
    async def _handle_message(self, topic: str, message: Dict[str, Any]):
        """Handle incoming message."""
        if not self.running:
            return
        
        # Store received message
        mock_msg = MockMessage(topic=topic, content=message, sender_id="message_bus")
        self.received_messages.append(mock_msg)
        self.message_count += 1
        
        # Call handlers
        for handler in self.message_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(topic, message)
                else:
                    handler(topic, message)
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in MockControllerService '{self.name}' message handler: {e}")
    
    def _create_worker_handler(self, worker_name: str):
        """Create a worker handler for the given worker name."""
        async def handler(message: Dict[str, Any]):
            await self._handle_worker_message(worker_name, message)
        return handler
    
    async def _handle_worker_message(self, worker_name: str, message: Dict[str, Any]):
        """Handle incoming worker message."""
        if not self.running:
            return
        
        # Store worker message
        mock_msg = MockMessage(topic=f"worker.{worker_name}", content=message, sender_id="message_bus")
        self.worker_messages[worker_name].append(mock_msg)
        self.message_count += 1
        
        # Call worker handlers
        for handler in self.worker_handlers[worker_name]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in MockControllerService '{self.name}' worker handler: {e}")
    
    def get_worker_messages(self, worker_name: str) -> List[MockMessage]:
        """Get messages for a specific worker."""
        if worker_name not in self.config.workers:
            raise ValueError(f"Unknown worker: {worker_name}")
        
        return self.worker_messages[worker_name].copy()

# =============================================================================
# TESTING UTILITIES
# =============================================================================

@asynccontextmanager
async def mock_environment():
    """Context manager for clean test environment."""
    # Clear message bus
    mock_message_bus.clear()
    
    try:
        yield mock_message_bus
    finally:
        # Clean up after test
        mock_message_bus.clear()


async def wait_for_messages(service, count: int, timeout: float = 1.0, message_type: str = "published") -> bool:
    """Wait for a service to have a certain number of messages."""
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        if message_type == "published":
            current_count = len(service.published_messages)
        elif message_type == "received":
            current_count = len(service.received_messages)
        else:
            current_count = service.message_count
            
        if current_count >= count:
            return True
        await asyncio.sleep(0.01)
    
    return False


if __name__ == "__main__":
    # Demo of mock system with clean inline configs
    async def demo():
        print("=== ZMQ Mock System Demo ===")
        
        async with mock_environment() as bus:
            # Simple inline config - no factory functions needed
            pubsub_config = PubSubServiceConfig(
                name="demo_pubsub",
                publisher=PublisherConfig(address="tcp://*", port=5555),
                subscriber=SubscriberConfig(
                    address="tcp://localhost", 
                    port=5556, 
                    topics=["demo.topic"]
                )
            )
            
            # Test PubSub
            print("\n1. Testing MockPubSubService:")
            async with MockPubSubService(pubsub_config) as pubsub:
                received_messages = []
                
                def handler(topic, message):
                    received_messages.append((topic, message))
                
                pubsub.set_message_handler(handler)
                
                await pubsub.publish({"test": "message"}, "demo.topic")
                await asyncio.sleep(0.1)  # Allow message processing
                
                print(f"   Published: {len(pubsub.published_messages)} messages")
                print(f"   Received: {len(received_messages)} messages")
            
            # Test Controller with inline worker configs
            controller_config = ControllerServiceConfig(
                name="demo_controller",
                publisher=PublisherConfig(address="tcp://*", port=5557),
                subscriber=SubscriberConfig(
                    address="tcp://localhost",
                    port=5558,
                    topics=["heartbeat"]
                ),
                workers={
                    "demo_worker": WorkerConfig(
                        name="demo_worker",
                        push_config=PushConfig(address="tcp://localhost", port=5559),
                        pull_config=PullConfig(address="tcp://*", port=5560),
                        message_types=["demo.work"]
                    )
                }
            )
            
            print("\n2. Testing MockControllerService:")
            async with MockControllerService(controller_config) as controller:
                work_received = []
                
                def work_handler(message):
                    work_received.append(message)
                
                controller.set_worker_handler("demo_worker", work_handler)
                
                await controller.publish({"status": "alive"}, "heartbeat")
                await controller.push_to_worker("demo_worker", {"action": "process"})
                await asyncio.sleep(0.1)
                
                print(f"   Published: {len(controller.published_messages)} messages")
                print(f"   Worker received: {len(work_received)} work items")
                print(f"   Global messages: {len(bus.get_messages())}")
        
        print("\n✅ Mock system working correctly!")
    
    import asyncio
    asyncio.run(demo())
