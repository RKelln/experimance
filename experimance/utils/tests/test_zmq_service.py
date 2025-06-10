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
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from experimance_common.base_service import ServiceState
from experimance_common.zmq.base_zmq import BaseZmqService
from experimance_common.zmq.publisher import ZmqPublisherService
from experimance_common.zmq.subscriber import ZmqSubscriberService
from experimance_common.zmq.push import ZmqPushService
from experimance_common.zmq.pull import ZmqPullService
from experimance_common.zmq.pubsub import ZmqPublisherSubscriberService
from experimance_common.zmq.controller import ZmqControllerService
from experimance_common.zmq.worker import ZmqWorkerService
from experimance_common.zmq.zmq_utils import MessageType
from utils.tests.test_utils import active_service, wait_for_service_state, debug_service_tasks, SIMULATE_NETWORK_DELAY

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
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)
        
        from experimance_common.zmq.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock publishing timeout")
    
    async def receive_async(self):
        """Mock receiving a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)
        
        from experimance_common.zmq.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock receiving timeout")
    
    async def push_async(self, message):
        """Mock pushing a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)
        
        from experimance_common.zmq.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock pushing timeout")
    
    async def pull_async(self):
        """Mock pulling a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)
        
        from experimance_common.zmq.zmq_utils import ZmqTimeoutError
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
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)
        
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
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)
        
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
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)
        
        from experimance_common.zmq.zmq_utils import ZmqTimeoutError
        raise ZmqTimeoutError("Mock socket timeout")
    
    async def push_async(self, message):
        """Mock pushing a message successfully."""
        self.messages.append(message)
        return True
    
    async def pull_async(self):
        """Mock pulling a message with timeout."""
        # Simulate a timeout after a short delay, to avoid hanging
        await asyncio.sleep(SIMULATE_NETWORK_DELAY)
        
        from experimance_common.zmq.zmq_utils import ZmqTimeoutError
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
        
        # Wait for service to reach STOPPED state
        await wait_for_service_state(service, ServiceState.STOPPED)
        
        # Verify socket was attempted to be closed
        bad_socket.close.assert_called_once()
        assert service.state == ServiceState.STOPPED


@pytest.mark.asyncio
@patch('experimance_common.zmq.publisher.ZmqPublisher', MockPublisher)
class TestPublisherService:
    """Tests for ZmqPublisherService."""
    
    async def test_publisher_initialization(self):
        """Test publisher initialization and start."""
        service = ZmqPublisherService(
            service_name="test-publisher", 
            pub_address="tcp://*:5555",
            topic="test.heartbeat"
        )
        
        async with active_service(service) as s:
            assert s.publisher is not None
            assert s.publisher.address == "tcp://*:5555"
            assert s.publisher.topic == "test.heartbeat"
            
            # Should have at least one task (heartbeat)
            assert len(s.tasks) > 0

    async def test_publish_message(self):
        """Test publishing messages."""
        service = ZmqPublisherService(
            service_name="test-publisher", 
            pub_address="tcp://*:5555"
        )
        
        async with active_service(service) as s:
            # Publish a test message
            message = {"type": "TEST", "content": "test-data"}
            success = await s.publish_message(message)
            
            # Check message was published
            assert success is True
            assert s.messages_sent == 2  # Expect heartbeat + published message
            
            # Check message is in the publisher's message list
            assert len(s.publisher.messages) == 2  # type: ignore
            # The first message should be the heartbeat
            assert s.publisher.messages[0]["type"] == MessageType.HEARTBEAT.value  # type: ignore
            # The second message should be the one explicitly published by the test
            assert s.publisher.messages[1] == message  # type: ignore


@pytest.mark.asyncio
@patch('experimance_common.zmq.subscriber.ZmqSubscriber', MockSubscriber)
class TestSubscriberService:
    """Tests for ZmqSubscriberService."""
    
    async def test_subscriber_initialization(self):
        """Test subscriber initialization and start."""
        service = ZmqSubscriberService(
            service_name="test-subscriber", 
            sub_address="tcp://localhost:5555",
            topics=["test.topic1", "test.topic2"]
        )
        
        async with active_service(service) as s:
            assert s.subscriber is not None
            assert s.subscriber.address == "tcp://localhost:5555"
            
            # Should have at least one task (message listener)
            assert len(s.tasks) > 0

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
@patch('experimance_common.zmq.push.ZmqPushSocket', MockZmqSocketWithMessages)
class TestPushService:
    """Tests for ZmqPushService."""
    
    async def test_push_service_initialization(self):
        """Test push service initialization and start."""
        service = ZmqPushService(
            service_name="test-push", 
            push_address="tcp://*:5555"
        )
        
        async with active_service(service) as s:
            assert s.push_socket is not None
            assert s.push_socket.address == "tcp://*:5555"
            assert len(s.tasks) > 0, "Expected at least one task to be created"
    
    async def test_push_task(self):
        """Test pushing tasks."""
        service = ZmqPushService(
            service_name="test-push", 
            push_address="tcp://*:5555"
        )
        
        async with active_service(service) as s:
            task = {"id": "task-1", "command": "test-command"}
            success = await s.push_task(task)
            
            assert success is True
            assert s.messages_sent == 1
            assert len(s.push_socket.messages) == 1  # type: ignore
            assert s.push_socket.messages[0] == task  # type: ignore
            assert len(s.tasks) > 0, "Expected at least one task to be created"


@pytest.mark.asyncio
@patch('experimance_common.zmq.pull.ZmqPullSocket', MockPullSocket)
class TestPullService:
    """Tests for ZmqPullService."""
    
    async def test_pull_service_initialization(self):
        """Test pull service initialization and start."""
        service = ZmqPullService(
            service_name="test-pull", 
            pull_address="tcp://localhost:5555"
        )
        # Register a dummy handler so _task_puller_task is created by service.start()
        dummy_handler = AsyncMock()
        service.register_task_handler(dummy_handler)
        
        async with active_service(service) as s:
            assert s.pull_socket is not None
            assert s.pull_socket.address == "tcp://localhost:5555"
            assert len(s.tasks) >= 2 # run_task + task_puller
    
    async def test_task_handler_registration(self):
        """Test registering task handlers."""
        service = ZmqPullService(
            service_name="test-pull", 
            pull_address="tcp://localhost:5555"
        )
        
        handler_mock = AsyncMock()
        # Register the handler directly on the service instance before active_service starts it
        service.register_task_handler(handler_mock)
        
        async with active_service(service) as s:
            assert s.task_handler == handler_mock
            
            test_task = {"id": "test-task", "data": "test"}
            # Manually call the handler to test registration, not the pull loop
            if s.task_handler:
                await s.task_handler(test_task)
                # Check that the handler was called with test_task at least once,
                # acknowledging that the service's pull loop might also call it.
                handler_mock.assert_any_call(test_task)
            
            # Verify tasks are running
            assert len(s.tasks) >= 2


@pytest.mark.asyncio
@patch('experimance_common.zmq.publisher.ZmqPublisher', MockPublisher)
@patch('experimance_common.zmq.subscriber.ZmqSubscriber', MockSubscriber)
@patch('experimance_common.zmq.push.ZmqPushSocket', MockZmqSocketWithMessages)
@patch('experimance_common.zmq.pull.ZmqPullSocket', MockPullSocket)
@patch('experimance_common.zmq.worker.ZmqPushSocket', MockZmqSocketWithMessages)  # Explicitly patch worker's push socket
class TestCombinedServices:
    """Tests for combined ZMQ service classes."""
    
    async def test_pubsub_service(self):
        """Test ZmqPublisherSubscriberService."""
        service = ZmqPublisherSubscriberService(
            service_name="test-pubsub",
            pub_address="tcp://*:5555",
            sub_address="tcp://localhost:5556",
            subscribe_topics=["test.topic"],
            publish_topic="test.heartbeat"
        )
        
        async with active_service(service) as s:
            assert s.publisher is not None
            assert s.subscriber is not None

            assert len(s.tasks) >= 3
            task_names = []
            for task in s.tasks:
                if hasattr(task, '__name__'):
                    task_names.append(task.__name__)
                elif hasattr(task, 'get_name'):
                    name = task.get_name()
                    # Extract just the function name part if the task was named with service_name-function_name pattern
                    if '-' in name:
                        task_names.append(name.split('-', 1)[1])
            
            # Check for required task names
            assert any("send_heartbeat" in name for name in task_names), f"send_heartbeat not found in {task_names}"
            assert any("listen_for_messages" in name for name in task_names), f"listen_for_messages not found in {task_names}"
            assert any("display_stats" in name for name in task_names), f"display_stats not found in {task_names}"
            
            message = {"type": "TEST", "content": "test-data"}
            success = await s.publish_message(message)
            assert success is True
    
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
        # ZmqControllerService auto-registers a pull handler (_handle_worker_response)
        
        async with active_service(service) as s:
            assert s.publisher is not None
            assert s.subscriber is not None
            assert s.push_socket is not None
            assert s.pull_socket is not None

            debug_service_tasks(s)
            assert len(s.tasks) >= 4
            
            # Use the new helper method to get task names
            task_names = s.get_task_names()
            
            # Check for required task names
            assert any("send_heartbeat" in name for name in task_names), f"send_heartbeat not found in {task_names}"
            assert any("pull_tasks" in name for name in task_names), f"pull_tasks not found in {task_names}"
            assert any("listen_for_messages" in name for name in task_names), f"listen_for_messages not found in {task_names}"
            assert any("display_stats" in name for name in task_names), f"display_stats not found in {task_names}"
            
            message = {"type": "TEST", "content": "test-data"}
            success = await s.publish_message(message)
            assert success is True
    
    async def test_worker_service(self):
        """Test ZmqWorkerService."""
        service = ZmqWorkerService(
            service_name="test-worker",
            sub_address="tcp://localhost:5555",
            pull_address="tcp://localhost:5556",
            push_address="tcp://*:5557",
            topics=["test.topic"]
        )
        # ZmqWorkerService auto-registers _handle_task for pull and _handle_message for sub
        
        async with active_service(service) as s:
            assert s.subscriber is not None
            assert s.pull_socket is not None
            assert s.push_socket is not None
            
            assert len(s.tasks) >= 3
            
            task_names = []
            for task in s.tasks:
                if hasattr(task, '__name__'):
                    task_names.append(task.__name__)
                elif hasattr(task, 'get_name'):
                    name = task.get_name()
                    # Extract just the function name part if the task was named with service_name-function_name pattern
                    if '-' in name:
                        task_names.append(name.split('-', 1)[1])
            
            # Check for required task names
            assert any("pull_tasks" in name for name in task_names), f"pull_tasks not found in {task_names}"
            assert any("listen_for_messages" in name for name in task_names), f"listen_for_messages not found in {task_names}"
            assert any("display_stats" in name for name in task_names), f"display_stats not found in {task_names}"
            
            response = {"type": "RESPONSE", "content": "test-data"}
            success = await s.send_response(response)
            assert success is True
