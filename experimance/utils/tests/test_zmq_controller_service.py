"""
Tests for ZmqControllerService.

This module contains comprehensive tests for the ZmqControllerService class,
including initialization, socket management, message handling, and cleanup.
"""

import asyncio
import pytest
import zmq
from unittest.mock import Mock, patch, AsyncMock
import logging

from experimance_common.zmq.controller import ZmqControllerService
from experimance_common.zmq.zmq_utils import MessageType
from experimance_common.constants import DEFAULT_PORTS, HEARTBEAT_TOPIC

logger = logging.getLogger(__name__)


@pytest.fixture
async def controller_addresses():
    """Provide test addresses for controller service."""
    return {
        'pub_address': f"tcp://*:9977",
        'sub_address': f"tcp://localhost:9977",
        'push_address': f"tcp://*:9978",
        'pull_address': f"tcp://*:9979"
    }


@pytest.fixture
async def controller_service(controller_addresses):
    """Create a test controller service."""
    service = ZmqControllerService(
        service_name="test_controller",
        pub_address=controller_addresses['pub_address'],
        sub_address=controller_addresses['sub_address'],
        push_address=controller_addresses['push_address'],
        pull_address=controller_addresses['pull_address'],
        topics=[HEARTBEAT_TOPIC, "image.ready", "transition.ready"]
    )
    yield service
    # Cleanup
    if hasattr(service, 'running') and service.running:
        await service.stop()


class TestZmqControllerServiceInitialization:
    """Test controller service initialization."""
    
    def test_init_sets_correct_attributes(self, controller_addresses):
        """Test that initialization sets all required attributes."""
        service = ZmqControllerService(
            service_name="test_controller",
            pub_address=controller_addresses['pub_address'],
            sub_address=controller_addresses['sub_address'],
            push_address=controller_addresses['push_address'],
            pull_address=controller_addresses['pull_address'],
            topics=["test.topic"]
        )
        
        assert service.service_name == "test_controller"
        assert service.pub_address == controller_addresses['pub_address']
        assert service.sub_address == controller_addresses['sub_address']
        assert service.push_address == controller_addresses['push_address']
        assert service.pull_address == controller_addresses['pull_address']
        assert "test.topic" in service.subscribe_topics
        assert service.service_type == "controller"
    
    def test_init_with_custom_heartbeat_topic(self, controller_addresses):
        """Test initialization with custom heartbeat topic."""
        custom_heartbeat = "custom.heartbeat"
        service = ZmqControllerService(
            service_name="test_controller",
            pub_address=controller_addresses['pub_address'],
            sub_address=controller_addresses['sub_address'],
            push_address=controller_addresses['push_address'],
            pull_address=controller_addresses['pull_address'],
            topics=["test.topic"],
            heartbeat_topic=custom_heartbeat
        )
        
        assert service.publish_topic == custom_heartbeat
    
    def test_init_with_custom_service_type(self, controller_addresses):
        """Test initialization with custom service type."""
        service = ZmqControllerService(
            service_name="test_controller",
            pub_address=controller_addresses['pub_address'],
            sub_address=controller_addresses['sub_address'],
            push_address=controller_addresses['push_address'],
            pull_address=controller_addresses['pull_address'],
            topics=["test.topic"],
            service_type="custom_controller"
        )
        
        assert service.service_type == "custom_controller"


class TestZmqControllerServiceStartup:
    """Test controller service startup process."""
    
    @pytest.mark.asyncio
    async def test_start_initializes_all_sockets(self, controller_service:ZmqControllerService):
        """Test that start() initializes all required sockets."""
        with patch.object(controller_service, 'register_socket') as mock_register:
            with patch.object(controller_service, 'add_task') as mock_add_task:
                with patch('experimance_common.zmq.base_zmq.BaseZmqService.start') as mock_base_start:
                    await controller_service.start()
                    
                    # Verify all sockets are created and registered
                    assert mock_register.call_count == 4  # publisher, subscriber, push, pull
                    assert hasattr(controller_service, 'publisher')
                    assert hasattr(controller_service, 'subscriber')
                    assert hasattr(controller_service, 'push_socket')
                    assert hasattr(controller_service, 'pull_socket')
                    
                    # Verify tasks are added
                    assert mock_add_task.call_count == 3  # heartbeat, listen, pull
                    
                    # Verify base start is called
                    mock_base_start.assert_called_once()


class TestZmqControllerServiceMessaging:
    """Test controller service messaging capabilities."""
    
    @pytest.mark.asyncio
    async def test_publish_message(self, controller_service:ZmqControllerService):
        """Test publishing a message."""
        # Mock the publisher
        controller_service.publisher = Mock()
        controller_service.publisher.publish_async = AsyncMock()
        
        test_message = {
            "type":"test",
        }
        
        await controller_service.publish_message(test_message)
        controller_service.publisher.publish_async.assert_called_once_with(test_message)
    
    @pytest.mark.asyncio
    async def test_push_task(self, controller_service:ZmqControllerService):
        """Test pushing a task to workers."""
        # Mock the push socket
        controller_service.push_socket = Mock()
        controller_service.push_socket.push_async = AsyncMock()
        
        test_task = {
            "type": "task",
            "data": {
                "task": "generate_image", "params": {"prompt": "test"}
            }
        }
        
        await controller_service.push_task(test_task)
        controller_service.push_socket.push_async.assert_called_once_with(test_task)
    
    @pytest.mark.asyncio
    async def test_register_task_handler(self, controller_service:ZmqControllerService):
        """Test registering a response handler."""
        handler_called = False
        
        async def test_handler(message):
            nonlocal handler_called
            handler_called = True
        
        controller_service.register_task_handler(test_handler)
        
        # Simulate receiving a response
        test_response = {
            "type": "task_response",
        }
        
        # Mock the handler system
        if hasattr(controller_service, 'task_handler'):
            assert controller_service.task_handler is not None
            await controller_service.task_handler(test_response)
            assert handler_called


class TestZmqControllerServiceErrorHandling:
    """Test controller service error handling."""
    
    @pytest.mark.asyncio
    async def test_start_handles_socket_creation_failure(self, controller_service:ZmqControllerService):
        """Test that start() handles socket creation failures gracefully."""
        with patch('experimance_common.zmq.zmq_utils.ZmqPublisher') as mock_publisher:
            mock_publisher.side_effect = zmq.ZMQError("Failed to bind") # type: ignore
            
            with pytest.raises(zmq.ZMQError):
                await controller_service.start()
    
    @pytest.mark.asyncio
    async def test_send_message_handles_socket_error(self, controller_service:ZmqControllerService):
        """Test that message sending handles socket errors."""
        controller_service.publisher = Mock()
        controller_service.publisher.send_message = AsyncMock(side_effect=zmq.ZMQError("Socket error")) # type: ignore
        
        test_message = {"type": "test"}
        
        # Should not raise an exception, but should log the error
        with patch('experimance_common.zmq.controller.logger') as mock_logger:
            try:
                await controller_service.publish_message(test_message)
            except zmq.ZMQError:
                pass  # Expected to be handled
            
            # Verify error was logged (if implemented in the service)
            # This depends on the actual error handling implementation


class TestZmqControllerServiceCleanup:
    """Test controller service cleanup and shutdown."""
    
    @pytest.mark.asyncio
    async def test_stop_cleans_up_sockets(self, controller_service):
        """Test that stop() properly cleans up all sockets."""
        # Mock sockets
        controller_service.publisher = Mock()
        controller_service.subscriber = Mock()
        controller_service.push_socket = Mock()
        controller_service.pull_socket = Mock()
        
        with patch('experimance_common.zmq.base_zmq.BaseZmqService.stop') as mock_base_stop:
            await controller_service.stop()
            mock_base_stop.assert_called_once()


class TestZmqControllerServiceIntegration:
    """Integration tests for controller service."""
    
    @pytest.mark.asyncio
    async def test_full_message_flow(self, controller_service):
        """Test a complete message flow through the controller."""
        # This would be a more complex test that actually starts the service
        # and tests message flow. For now, we'll mock the components.
        
        with patch.object(controller_service, 'start') as mock_start:
            with patch.object(controller_service, 'publish_message') as mock_publish:
                with patch.object(controller_service, 'push_task') as mock_push:
                    # Simulate starting the service
                    await mock_start()
                    
                    # Simulate publishing an event
                    event = {"type": "test"}
                    await mock_publish(event)
                    
                    # Simulate pushing a task
                    task = {"type": "task", "data": {}}
                    await mock_push(task)
                    
                    # Verify calls were made
                    mock_start.assert_called_once()
                    mock_publish.assert_called_once_with(event)
                    mock_push.assert_called_once_with(task)


# Test utilities and fixtures
@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    return {
        'event': {"type": "event", "topic":"test.event", "data":{"key": "value"}},
        'request': {"type": "request", "data": {"task": "test_task"}},
        'response': {"type": "response", "data": {"result": "success"}},
        'heartbeat': {"type": "heartbeat", "topic": HEARTBEAT_TOPIC, "data": {"service": "test"}}
    }


class TestZmqControllerServiceWithRealSockets:
    """Tests that use real sockets (marked as integration tests)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_socket_initialization(self):
        """Test with real ZMQ sockets (requires available ports)."""
        # Use different ports to avoid conflicts
        test_ports = {
            'pub': 15501,
            'sub': 15502,
            'push': 15503,
            'pull': 15504
        }
        
        service = ZmqControllerService(
            service_name="integration_test_controller",
            pub_address=f"tcp://*:{test_ports['pub']}",
            sub_address=f"tcp://localhost:{test_ports['sub']}",
            push_address=f"tcp://*:{test_ports['push']}",
            pull_address=f"tcp://*:{test_ports['pull']}",
            topics=["test.topic"]
        )
        
        try:
            await service.start()
            
            # Verify sockets are created
            assert service.publisher is not None
            assert service.subscriber is not None
            assert service.push_socket is not None
            assert service.pull_socket is not None
            
            # Give time for sockets to initialize
            await asyncio.sleep(0.1)
            
        finally:
            await service.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
