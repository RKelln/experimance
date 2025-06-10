"""
Tests for the non-hanging ZeroMQ utilities module.
"""

import asyncio
import logging
import time

import pytest

from experimance_common.zmq.zmq_utils import (
    ZmqPublisher, 
    ZmqSubscriber, 
    ZmqPushSocket, 
    ZmqPullSocket,
    ZmqBindingPullSocket,
    ZmqConnectingPushSocket,
    MessageType,
    ZmqTimeoutError
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
TEST_PUB_SUB_PORT = 5566
TEST_PUSH_PULL_PORT = 5567
TEST_BINDING_PULL_PORT = 5568
TEST_CONNECTING_PUSH_PORT = TEST_BINDING_PULL_PORT  # Same port for binding pull and connecting push


class TestZmqPubSub:
    """Tests for Publisher-Subscriber pattern."""

    @pytest.fixture
    async def setup_pubsub_async(self, request):
        """Setup and teardown for async pub/sub tests."""
        # Get topic from request.param if available, otherwise use default
        topic = getattr(request, "param", "test-topic")
        
        # Create publisher and subscriber
        publisher = ZmqPublisher(f"tcp://*:{TEST_PUB_SUB_PORT}", topic)
        subscriber = ZmqSubscriber(f"tcp://localhost:{TEST_PUB_SUB_PORT}", [topic])
        
        # Allow time for connection to establish
        await asyncio.sleep(0.5)
        
        yield publisher, subscriber
        
        # Cleanup
        logger.debug("Cleaning up publisher and subscriber")
        publisher.close()
        subscriber.close()
    
    @pytest.mark.asyncio
    async def test_pub_sub_async(self, setup_pubsub_async):
        """Test asynchronous publish-subscribe communication."""
        publisher, subscriber = setup_pubsub_async
        
        # Test message
        test_message = {"type": MessageType.HEARTBEAT, "timestamp": time.time()}
        
        # Publish message
        success = await publisher.publish_async(test_message)
        assert success, "Failed to publish message"
        
        # Give some time for message to be delivered
        await asyncio.sleep(0.5)
        
        try:
            # Receive message with timeout built into the implementation
            topic, message = await subscriber.receive_async()
            
            # Verify
            assert topic == "test-topic", f"Unexpected topic: {topic}"
            assert message["type"] == test_message["type"], "Message content mismatch"
        except ZmqTimeoutError:
            pytest.fail("Timed out waiting for message")
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("setup_pubsub_async", [MessageType.RENDER_REQUEST], indirect=True)
    async def test_pub_sub_messagetype_async(self, setup_pubsub_async):
        """Test asynchronous publish-subscribe communication with MessageType topic."""
        publisher, subscriber = setup_pubsub_async
        
        # Test message
        test_message = {"type": MessageType.HEARTBEAT, "timestamp": time.time()}
        
        # Publish message
        success = await publisher.publish_async(test_message)
        assert success, "Failed to publish message"
        
        # Give some time for message to be delivered
        await asyncio.sleep(0.5)
        
        try:
            # Receive message with timeout built into the implementation
            topic, message = await subscriber.receive_async()
            
            # Verify
            assert topic == MessageType.RENDER_REQUEST.value, f"Unexpected topic: {topic}"
            assert message["type"] == test_message["type"], "Message content mismatch"
        except ZmqTimeoutError:
            pytest.fail("Timed out waiting for message")

    @pytest.fixture
    def setup_pubsub_sync(self):
        """Setup and teardown for sync pub/sub tests."""
        # Create publisher and subscriber
        publisher = ZmqPublisher(f"tcp://*:{TEST_PUB_SUB_PORT}", "test-topic", use_asyncio=False)
        subscriber = ZmqSubscriber(f"tcp://localhost:{TEST_PUB_SUB_PORT}", ["test-topic"], use_asyncio=False)
        
        # Allow time for connection to establish
        time.sleep(0.5)
        
        yield publisher, subscriber
        
        # Cleanup
        logger.debug("Cleaning up publisher and subscriber")
        publisher.close()
        subscriber.close()
    
    def test_pub_sub_sync(self, setup_pubsub_sync):
        """Test synchronous publish-subscribe communication."""
        publisher, subscriber = setup_pubsub_sync
        
        # Test message
        test_message = {"type": MessageType.HEARTBEAT, "timestamp": time.time()}
        
        # Publish message
        success = publisher.publish(test_message)
        assert success, "Failed to publish message"
        
        # Give some time for message to be delivered
        time.sleep(0.5)
        
        try:
            # Receive message with timeout built into the implementation
            topic, message = subscriber.receive()
            
            # Verify
            assert topic == "test-topic", f"Unexpected topic: {topic}"
            assert message["type"] == test_message["type"], "Message content mismatch"
        except ZmqTimeoutError:
            pytest.fail("Timed out waiting for message")


class TestZmqPushPull:
    """Tests for Push-Pull pattern."""

    @pytest.fixture
    async def setup_pushpull_async(self):
        """Setup and teardown for async push/pull tests."""
        # Create push and pull sockets
        push_socket = ZmqPushSocket(f"tcp://*:{TEST_PUSH_PULL_PORT}")
        pull_socket = ZmqPullSocket(f"tcp://localhost:{TEST_PUSH_PULL_PORT}")
        
        # Allow time for connection to establish
        await asyncio.sleep(0.5)
        
        yield push_socket, pull_socket
        
        # Cleanup
        logger.debug("Cleaning up push and pull sockets")
        push_socket.close()
        pull_socket.close()

    @pytest.mark.asyncio
    async def test_push_pull_async(self, setup_pushpull_async):
        """Test asynchronous push-pull communication."""
        push_socket, pull_socket = setup_pushpull_async
        
        # Test message
        test_message = {"type": MessageType.HEARTBEAT, "timestamp": time.time()}
        
        # Push message
        success = await push_socket.push_async(test_message)
        assert success, "Failed to push message"
        
        # Give some time for message to be delivered
        await asyncio.sleep(0.5)
        
        try:
            # Receive message with timeout built into the implementation
            message = await pull_socket.pull_async()
            
            # Verify
            assert message["type"] == test_message["type"], "Message content mismatch"
        except ZmqTimeoutError:
            pytest.fail("Timed out waiting for message")
    
    @pytest.fixture
    def setup_pushpull_sync(self):
        """Setup and teardown for sync push/pull tests."""
        # Create push and pull sockets
        push_socket = ZmqPushSocket(f"tcp://*:{TEST_PUSH_PULL_PORT}", use_asyncio=False)
        pull_socket = ZmqPullSocket(f"tcp://localhost:{TEST_PUSH_PULL_PORT}", use_asyncio=False)
        
        # Allow time for connection to establish
        time.sleep(0.5)
        
        yield push_socket, pull_socket
        
        # Cleanup
        logger.debug("Cleaning up push and pull sockets")
        push_socket.close()
        pull_socket.close()
    
    def test_push_pull_sync(self, setup_pushpull_sync):
        """Test synchronous push-pull communication."""
        push_socket, pull_socket = setup_pushpull_sync
        
        # Test message
        test_message = {"type": MessageType.HEARTBEAT, "timestamp": time.time()}
        
        # Push message
        success = push_socket.push(test_message)
        assert success, "Failed to push message"
        
        # Give some time for message to be delivered
        time.sleep(0.5)
        
        try:
            # Receive message with timeout built into the implementation
            message = pull_socket.pull()
            
            # Verify
            assert message["type"] == test_message["type"], "Message content mismatch"
        except ZmqTimeoutError:
            pytest.fail("Timed out waiting for message")


class TestZmqCustomSockets:
    """Tests for custom socket patterns (ZmqBindingPullSocket and ZmqConnectingPushSocket)."""
    
    @pytest.fixture
    async def setup_binding_connecting_async(self):
        """Setup and teardown for async binding pull and connecting push test."""
        # Create binding pull and connecting push sockets
        binding_pull = ZmqBindingPullSocket(f"tcp://*:{TEST_BINDING_PULL_PORT}")
        connecting_push = ZmqConnectingPushSocket(f"tcp://localhost:{TEST_CONNECTING_PUSH_PORT}")
        
        # Allow time for connection to establish
        await asyncio.sleep(0.5)
        
        yield binding_pull, connecting_push
        
        # Cleanup
        logger.debug("Cleaning up binding pull and connecting push sockets")
        binding_pull.close()
        connecting_push.close()
    
    @pytest.mark.asyncio
    async def test_binding_pull_connecting_push_async(self, setup_binding_connecting_async):
        """Test asynchronous binding pull and connecting push communication."""
        binding_pull, connecting_push = setup_binding_connecting_async
        
        # Test message
        test_message = {"type": MessageType.RENDER_REQUEST, "timestamp": time.time(), "test_id": "controller-worker"}
        
        # Push message using the connecting push socket
        success = await connecting_push.push_async(test_message)
        assert success, "Failed to push message from connecting push socket"
        
        # Give some time for message to be delivered
        await asyncio.sleep(0.5)
        
        try:
            # Receive message with the binding pull socket
            message = await binding_pull.pull_async()
            
            # Verify
            assert message["type"] == test_message["type"], "Message content mismatch"
            assert message["test_id"] == test_message["test_id"], "Message ID mismatch"
        except ZmqTimeoutError:
            pytest.fail("Timed out waiting for message")
    
    @pytest.fixture
    def setup_binding_connecting_sync(self):
        """Setup and teardown for sync binding pull and connecting push test."""
        # Create binding pull and connecting push sockets
        binding_pull = ZmqBindingPullSocket(f"tcp://*:{TEST_BINDING_PULL_PORT}", use_asyncio=False)
        connecting_push = ZmqConnectingPushSocket(f"tcp://localhost:{TEST_CONNECTING_PUSH_PORT}", use_asyncio=False)
        
        # Allow time for connection to establish
        time.sleep(0.5)
        
        yield binding_pull, connecting_push
        
        # Cleanup
        logger.debug("Cleaning up binding pull and connecting push sockets")
        binding_pull.close()
        connecting_push.close()
    
    def test_binding_pull_connecting_push_sync(self, setup_binding_connecting_sync):
        """Test synchronous binding pull and connecting push communication."""
        binding_pull, connecting_push = setup_binding_connecting_sync
        
        # Test message 
        test_message = {"type": MessageType.RENDER_REQUEST, "timestamp": time.time(), "test_id": "controller-worker-sync"}
        
        # Push message using the connecting push socket
        success = connecting_push.push(test_message)
        assert success, "Failed to push message from connecting push socket"
        
        # Give some time for message to be delivered
        time.sleep(0.5)
        
        try:
            # Receive message with the binding pull socket
            message = binding_pull.pull()
            
            # Verify
            assert message["type"] == test_message["type"], "Message content mismatch"
            assert message["test_id"] == test_message["test_id"], "Message ID mismatch"
        except ZmqTimeoutError:
            pytest.fail("Timed out waiting for message")


if __name__ == "__main__":
    pytest.main(["-v", "test_zmq_utils.py"])
