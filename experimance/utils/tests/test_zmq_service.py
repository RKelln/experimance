#!/usr/bin/env python3
"""
Tests for the ZeroMQ service classes in experimance_common.service module.

This test suite validates:
1. ZMQ socket registration and cleanup
2. Proper inheritance from BaseService
3. Publisher/Subscriber functionality
4. Push/Pull functionality
5. Graceful shutdown with ZMQ resources
6. Error handling in ZMQ operations

Run with:
    uv run -m pytest utils/tests/test_zmq_service.py -v
"""

import asyncio
import logging
import signal
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from experimance_common.service import (
    BaseZmqService, ServiceState,
    ZmqPublisherService, ZmqSubscriberService,
    ZmqPushService, ZmqPullService,
    ZmqPublisherSubscriberService, ZmqControllerService, ZmqWorkerService
)
from experimance_common.zmq_utils import MessageType

# Configure test logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockZmqSocketBase:
    """Base class for mock ZMQ sockets with common setup logic but no behavior."""
    
    def __init__(self, *args, **kwargs):
        self.closed = False
        self.messages = []
        
        # Store address properly from args or kwargs
        if args and isinstance(args[0], str):
            self.address = args[0]
        else:
            self.address = kwargs.get('address', 'mock-address')
            
        # Store topic or topics from kwargs
        self.topic = kwargs.get('topic', None)
        if not self.topic and kwargs.get('topics'):
            # If 'topics' is provided, use the first one as the main topic
            self.topic = kwargs.get('topics', [])[0] if kwargs.get('topics') else 'mock-topic'
        elif not self.topic:
            # Default if neither topic nor topics was provided
            self.topic = 'mock-topic'
            
        # Store all topics for subscriber mocks
        self.topics = kwargs.get('topics', [self.topic] if self.topic else [])
    
    def close(self):
        """Close the socket."""
        self.closed = True


class MockZmqSocketTimeout(MockZmqSocketBase):
    """Mock ZMQ socket that simulates timeouts for all operations."""
    
    async def publish_async(self, message):
        """Mock publishing a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(0.01)
        
        from experimance_common.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock publishing timeout")
    
    async def receive_async(self):
        """Mock receiving a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(0.01)
        
        from experimance_common.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock receiving timeout")
    
    async def push_async(self, message):
        """Mock pushing a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(0.01)
        
        from experimance_common.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock pushing timeout")
    
    async def pull_async(self):
        """Mock pulling a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(0.01)
        
        from experimance_common.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock pulling timeout")


class MockZmqSocketWorking(MockZmqSocketBase):
    """Mock ZMQ socket that works for all operations."""
    
    async def publish_async(self, message):
        """Mock publishing a message successfully."""
        self.messages.append(message)
        return True
    
    async def receive_async(self):
        """Mock receiving a message successfully."""
        # Add a small delay to simulate network
        await asyncio.sleep(0.01)
        
        if not self.messages:
            return self.topic, {"type": "test"}
        return self.topic, self.messages.pop(0)
    
    async def push_async(self, message):
        """Mock pushing a message successfully."""
        self.messages.append(message)
        return True
    
    async def pull_async(self):
        """Mock pulling a message successfully."""
        # Add a small delay to simulate network
        await asyncio.sleep(0.01)
        
        if not self.messages:
            return {"id": "test-id"}
        return self.messages.pop(0)


# For backward compatibility with existing tests
class MockZmqSocket(MockZmqSocketBase):
    """Legacy mock ZMQ socket for backward compatibility.
    
    This class retains the original behavior:
    - Working publish/push operations
    - Timeout receive/pull operations
    """
    
    async def publish_async(self, message):
        """Mock publishing a message successfully."""
        self.messages.append(message)
        return True
    
    async def receive_async(self):
        """Mock receiving a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(0.01)
        
        from experimance_common.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock socket timeout")
    
    async def push_async(self, message):
        """Mock pushing a message successfully."""
        self.messages.append(message)
        return True
    
    async def pull_async(self):
        """Mock pulling a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(0.01)
        
        from experimance_common.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock socket timeout")


# For backward compatibility with existing tests
class MockZmqSocketWithMessages(MockZmqSocketWorking):
    """Legacy mock ZMQ socket that returns messages for specific tests."""
    pass


class MockPublisher(MockZmqSocketWorking):
    """Special mock for publisher tests that preserves the heartbeat topic."""
    
    def __init__(self, address, topic=None):
        super().__init__(address=address, topic=topic)
        # Specifically preserve the topic passed in constructor
        self.topic = topic


class MockSubscriber(MockZmqSocketWorking):
    """Special mock for subscriber tests that preserves the address."""
    
    def __init__(self, address, topics=None):
        super().__init__(address=address, topics=topics)


class MockPullSocket(MockZmqSocketWorking):
    """Special mock for pull socket tests that preserves the address."""
    
    def __init__(self, address):
        super().__init__(address=address)


class TestBaseZmqService:
    """Tests for the BaseZmqService class."""
    
    @pytest.fixture
    async def zmq_service(self):
        """Create a BaseZmqService instance for testing."""
        service = BaseZmqService(service_name="test-zmq-service", service_type="test-zmq")
        yield service
        # Clean up after test
        if service.state != ServiceState.STOPPED:
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_initialization(self, zmq_service):
        """Test that BaseZmqService initializes correctly."""
        assert zmq_service.service_name == "test-zmq-service"
        assert zmq_service.service_type == "test-zmq"
        assert zmq_service.state == ServiceState.INITIALIZED
        assert zmq_service._sockets == []
    
    @pytest.mark.asyncio
    async def test_socket_registration(self, zmq_service):
        """Test socket registration and cleanup."""
        # Create and register mock sockets
        socket1 = MockZmqSocketBase(address="tcp://localhost:5555")
        socket2 = MockZmqSocketBase(address="tcp://localhost:5556")
        
        zmq_service.register_socket(socket1)
        zmq_service.register_socket(socket2)
        
        assert len(zmq_service._sockets) == 2
        assert zmq_service._sockets[0] == socket1
        assert zmq_service._sockets[1] == socket2
        
        # Test socket cleanup on stop
        await zmq_service.stop()
        
        assert socket1.closed is True
        assert socket2.closed is True
        assert zmq_service._sockets == []
    
    @pytest.mark.asyncio
    async def test_stop_with_socket_error(self):
        """Test that socket errors during stop are handled gracefully."""
        service = BaseZmqService(service_name="error-socket-service")
        
        # Create a socket that raises an exception when closed
        bad_socket = MagicMock()
        bad_socket.close.side_effect = Exception("Socket close error")
        
        # Register the bad socket
        service.register_socket(bad_socket)
        
        # Stop should handle the exception
        await service.stop()
        
        # Verify socket was attempted to be closed
        bad_socket.close.assert_called_once()
        assert service.state == ServiceState.STOPPED


@pytest.mark.asyncio
@patch('experimance_common.service.ZmqPublisher', MockPublisher)
class TestPublisherService:
    """Tests for ZmqPublisherService."""
    
    async def test_publisher_initialization(self):
        """Test publisher initialization and start."""
        service = ZmqPublisherService(
            service_name="test-publisher", 
            pub_address="tcp://*:5555",
            heartbeat_topic="test.heartbeat"
        )
        
        # Start the service
        await service.start()
        
        try:
            # Check publisher was created
            assert service.publisher is not None
            assert service.publisher.address == "tcp://*:5555"
            assert service.publisher.topic == "test.heartbeat"
            
            # Should have at least one task (heartbeat)
            assert len(service.tasks) > 0
        finally:
            # Clean up
            await service.stop()
    
    async def test_publish_message(self):
        """Test publishing messages."""
        service = ZmqPublisherService(
            service_name="test-publisher", 
            pub_address="tcp://*:5555"
        )
        
        # Start the service
        await service.start()
        
        try:
            # Publish a test message
            message = {"type": "TEST", "content": "test-data"}
            success = await service.publish_message(message)
            
            # Check message was published
            assert success is True
            assert service.messages_sent == 1
            
            # Check message is in the publisher's message list
            assert len(service.publisher.messages) == 1  # type: ignore
            assert service.publisher.messages[0] == message  # type: ignore
        finally:
            # Clean up
            await service.stop()


@pytest.mark.asyncio
@patch('experimance_common.service.ZmqSubscriber', MockSubscriber)
class TestSubscriberService:
    """Tests for ZmqSubscriberService."""
    
    async def test_subscriber_initialization(self):
        """Test subscriber initialization and start."""
        service = ZmqSubscriberService(
            service_name="test-subscriber", 
            sub_address="tcp://localhost:5555",
            topics=["test.topic1", "test.topic2"]
        )
        
        # Start the service
        await service.start()
        
        # Check subscriber was created
        assert service.subscriber is not None
        assert service.subscriber.address == "tcp://localhost:5555"
        
        # Should have at least one task (message listener)
        assert len(service.tasks) > 0
        
        # Clean up
        await service.stop()
    
    async def test_message_handler_registration(self):
        """Test registering and calling message handlers."""
        service = ZmqSubscriberService(
            service_name="test-subscriber", 
            sub_address="tcp://localhost:5555",
            topics=["test.topic"]
        )
        
        # Create a mock handler
        handler_mock = MagicMock()
        
        # Register the handler
        service.register_handler("test.topic", handler_mock)
        
        # Check handler was registered
        assert "test.topic" in service.message_handlers
        assert service.message_handlers["test.topic"] == handler_mock
        
        # Create a test message manually and call the handler directly
        test_message = {"type": "TEST", "content": "test-data"}
        service.message_handlers["test.topic"](test_message)
        
        # Check handler was called
        handler_mock.assert_called_once_with(test_message)
        
        # Start the service - but we don't need to wait for messages
        await service.start()
        
        # Clean up
        await service.stop()


@pytest.mark.asyncio
@patch('experimance_common.service.ZmqPushSocket', MockZmqSocketWithMessages)
class TestPushService:
    """Tests for ZmqPushService."""
    
    async def test_push_service_initialization(self):
        """Test push service initialization and start."""
        service = ZmqPushService(
            service_name="test-push", 
            push_address="tcp://*:5555"
        )
        
        # Start the service
        await service.start()
        
        # Check push socket was created
        assert service.push_socket is not None
        assert service.push_socket.address == "tcp://*:5555"
        
        # Clean up
        await service.stop()
    
    async def test_push_task(self):
        """Test pushing tasks."""
        service = ZmqPushService(
            service_name="test-push", 
            push_address="tcp://*:5555"
        )
        
        # Start the service
        await service.start()
        
        # Push a test task
        task = {"id": "task-1", "command": "test-command"}
        success = await service.push_task(task)
        
        # Check task was pushed
        assert success is True
        assert service.messages_sent == 1
        
        # Check task is in the socket's message list
        assert len(service.push_socket.messages) == 1  # type: ignore
        assert service.push_socket.messages[0] == task  # type: ignore
        
        # Clean up
        await service.stop()


@pytest.mark.asyncio
@patch('experimance_common.service.ZmqPullSocket', MockPullSocket)
class TestPullService:
    """Tests for ZmqPullService."""
    
    async def test_pull_service_initialization(self):
        """Test pull service initialization and start."""
        service = ZmqPullService(
            service_name="test-pull", 
            pull_address="tcp://localhost:5555"
        )
        
        # Start the service
        await service.start()
        
        # Check pull socket was created
        assert service.pull_socket is not None
        assert service.pull_socket.address == "tcp://localhost:5555"
        
        # Should have at least one task (task puller)
        assert len(service.tasks) > 0
        
        # Clean up
        await service.stop()
    
    async def test_task_handler_registration(self):
        """Test registering task handlers."""
        service = ZmqPullService(
            service_name="test-pull", 
            pull_address="tcp://localhost:5555"
        )
        
        # Create a mock handler
        handler_mock = AsyncMock()
        
        # Register the handler
        service.register_task_handler(handler_mock)
        
        # Check handler was registered
        assert service.task_handler == handler_mock
        
        # Start the service
        await service.start()
        
        # Manually call the task handler with a test task
        test_task = {"id": "test-task", "data": "test"}
        if service.task_handler:  # Check if task_handler is not None
            await service.task_handler(test_task)
            
            # Check handler was called with the task
            handler_mock.assert_called_once_with(test_task)
        
        # Clean up
        await service.stop()


@pytest.mark.asyncio
@patch('experimance_common.service.ZmqPublisher', MockPublisher)
@patch('experimance_common.service.ZmqSubscriber', MockSubscriber)
@patch('experimance_common.service.ZmqPushSocket', MockZmqSocketWithMessages)
@patch('experimance_common.service.ZmqPullSocket', MockPullSocket)
class TestCombinedServices:
    """Tests for combined ZMQ service classes."""
    
    async def test_pubsub_service(self):
        """Test ZmqPublisherSubscriberService."""
        service = ZmqPublisherSubscriberService(
            service_name="test-pubsub",
            pub_address="tcp://*:5555",
            sub_address="tcp://localhost:5556",
            topics=["test.topic"],
            heartbeat_topic="test.heartbeat"
        )
        
        # Start the service
        await service.start()
        
        # Check both publisher and subscriber were created
        assert service.publisher is not None
        assert service.subscriber is not None
        
        # Should have multiple tasks (heartbeat and message listener)
        assert len(service.tasks) >= 2
        
        # Test publishing a message
        message = {"type": "TEST", "content": "test-data"}
        success = await service.publish_message(message)
        assert success is True
        
        # Clean up
        await service.stop()
    
    async def test_controller_service(self):
        """Test ZmqControllerService."""
        service = ZmqControllerService(
            service_name="test-controller",
            pub_address="tcp://*:5555",
            sub_address="tcp://localhost:5556",
            push_address="tcp://*:5557",
            pull_address="tcp://localhost:5558",
            topics=["test.topic"],
            heartbeat_topic="test.heartbeat"
        )
        
        # Start the service
        await service.start()
        
        # Check all sockets were created
        assert service.publisher is not None
        assert service.subscriber is not None
        assert service.push_socket is not None
        assert service.pull_socket is not None
        
        # Should have multiple tasks
        assert len(service.tasks) >= 2
        
        # Test publishing a message
        message = {"type": "TEST", "content": "test-data"}
        success = await service.publish_message(message)
        assert success is True
        
        # Clean up
        await service.stop()
    
    async def test_worker_service(self):
        """Test ZmqWorkerService."""
        service = ZmqWorkerService(
            service_name="test-worker",
            sub_address="tcp://localhost:5555",
            pull_address="tcp://localhost:5556",
            push_address="tcp://*:5557",
            topics=["test.topic"]
        )
        
        # Start the service
        await service.start()
        
        # Check all sockets were created
        assert service.subscriber is not None
        assert service.pull_socket is not None
        assert service.push_socket is not None
        
        # Should have multiple tasks
        assert len(service.tasks) >= 2
        
        # Test sending a response
        response = {"type": "RESPONSE", "content": "test-data"}
        success = await service.send_response(response)
        assert success is True
        
        # Clean up
        await service.stop()
