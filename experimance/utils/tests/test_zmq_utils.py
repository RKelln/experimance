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
    ZmqTimeoutError,
    prepare_image_message,
    IMAGE_TRANSPORT_MODES
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


class TestPrepareImageMessage:
    """Tests for prepare_image_message function with numpy arrays and PIL images."""
    
    def test_numpy_array_file_uri_mode(self):
        """Test prepare_image_message with numpy array in file_uri mode."""
        import numpy as np
        from experimance_common.zmq.zmq_utils import prepare_image_message, IMAGE_TRANSPORT_MODES
        from experimance_common.image_utils import cleanup_temp_file
        from pathlib import Path
        
        # Create test numpy array
        array = np.zeros((32, 32, 3), dtype=np.uint8)
        array[:, :, 0] = 255  # Red channel
        
        message = prepare_image_message(
            image_data=array,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["FILE_URI"],
            mask_id="test_numpy_file_uri"
        )
        
        # Should have URI but no image_data for FILE_URI mode
        assert "uri" in message, "Should have URI field"
        assert "image_data" not in message, "Should not have image_data field in FILE_URI mode"
        assert message["mask_id"] == "test_numpy_file_uri"
        
        # Verify URI format
        uri = message["uri"]
        assert uri.startswith("file://"), f"URI should start with file://, got: {uri}"
        
        # Check that temp file exists
        temp_path = Path(uri[7:])  # Remove "file://" prefix
        assert temp_path.exists(), f"Temporary file should exist: {temp_path}"
        assert temp_path.suffix == ".png", f"Should be PNG file: {temp_path}"
        
        # Cleanup
        cleanup_temp_file(temp_path)
    
    def test_numpy_array_hybrid_mode(self):
        """Test prepare_image_message with numpy array in hybrid mode."""
        import numpy as np
        from experimance_common.zmq.zmq_utils import prepare_image_message, IMAGE_TRANSPORT_MODES
        from experimance_common.image_utils import cleanup_temp_file
        from pathlib import Path
        
        # Create test numpy array
        array = np.random.randint(0, 256, (24, 24, 3), dtype=np.uint8)
        
        message = prepare_image_message(
            image_data=array,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["HYBRID"],
            mask_id="test_numpy_hybrid"
        )
        
        # Should have both URI and image_data for HYBRID mode
        assert "uri" in message, "Should have URI field"
        assert "image_data" in message, "Should have image_data field"
        assert message["mask_id"] == "test_numpy_hybrid"
        
        # Verify URI format
        uri = message["uri"]
        assert uri.startswith("file://"), f"URI should start with file://"
        
        # Verify base64 data
        image_data = message["image_data"]
        assert image_data.startswith("data:image/png;base64,"), "Should be base64 data URL"
        assert len(image_data) > 100, "Should have substantial base64 data"
        
        # Check that temp file exists
        temp_path = Path(uri[7:])  # Remove "file://" prefix
        assert temp_path.exists(), f"Temporary file should exist: {temp_path}"
        
        # Cleanup
        cleanup_temp_file(temp_path)
    
    def test_pil_image_file_uri_mode(self):
        """Test prepare_image_message with PIL image in file_uri mode."""
        from PIL import Image, ImageDraw
        from experimance_common.zmq.zmq_utils import prepare_image_message, IMAGE_TRANSPORT_MODES
        from experimance_common.image_utils import cleanup_temp_file
        from pathlib import Path
        
        # Create test PIL image
        img = Image.new('RGB', (48, 48), color='blue')
        draw = ImageDraw.Draw(img)
        draw.rectangle([12, 12, 36, 36], fill='yellow')
        
        message = prepare_image_message(
            image_data=img,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["FILE_URI"],
            mask_id="test_pil_file_uri"
        )
        
        # Should have URI but no image_data
        assert "uri" in message, "Should have URI field"
        assert "image_data" not in message, "Should not have image_data field in FILE_URI mode"
        
        # Verify URI and cleanup
        uri = message["uri"]
        temp_path = Path(uri[7:])  # Remove "file://" prefix
        assert temp_path.exists(), f"Temporary file should exist: {temp_path}"
        
        # Cleanup
        cleanup_temp_file(temp_path)
    
    def test_numpy_array_auto_mode_local_vs_remote(self):
        """Test that auto mode chooses appropriate transport for numpy arrays."""
        import numpy as np
        from experimance_common.zmq.zmq_utils import prepare_image_message, IMAGE_TRANSPORT_MODES
        
        # Create test numpy array
        array = np.ones((16, 16), dtype=np.uint8) * 128
        
        # Test with local target (should prefer file_uri but arrays force base64)
        local_message = prepare_image_message(
            image_data=array,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["AUTO"],
            mask_id="test_auto_local"
        )
        
        # Test with remote target (should use base64)
        remote_message = prepare_image_message(
            image_data=array,
            target_address="tcp://192.168.1.100:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["AUTO"],
            mask_id="test_auto_remote"
        )
        
        # Both should use base64 since numpy arrays can't use file_uri without temp files
        # and AUTO mode for arrays defaults to base64
        assert "image_data" in local_message, "Local should have image_data"
        assert "image_data" in remote_message, "Remote should have image_data"
        
        # Neither should have URI in AUTO mode for arrays (unless hybrid behavior)
        # This depends on the implementation - arrays in AUTO might create temp files
    
    def test_numpy_array_base64_mode(self):
        """Test prepare_image_message with numpy array in base64 mode."""
        import numpy as np
        from experimance_common.zmq.zmq_utils import prepare_image_message, IMAGE_TRANSPORT_MODES
        
        # Create test numpy array
        array = np.zeros((20, 20, 4), dtype=np.uint8)  # RGBA
        array[:, :, 0] = 255  # Red
        array[:, :, 3] = 128  # 50% alpha
        
        message = prepare_image_message(
            image_data=array,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["BASE64"],
            mask_id="test_numpy_base64"
        )
        
        # Should only have image_data, no URI
        assert "image_data" in message, "Should have image_data field"
        assert "uri" not in message, "Should not have URI field in BASE64 mode"
        
        # Verify base64 format
        image_data = message["image_data"]
        assert image_data.startswith("data:image/png;base64,"), "Should be base64 data URL"
        assert len(image_data) > 100, "Should have substantial base64 data"


class TestTransportModeSelection:
    """Test cases for image transport mode selection functions."""
    
    def test_is_local_address(self):
        """Test address detection for local vs remote."""
        from experimance_common.zmq.zmq_utils import is_local_address
        
        # Test local addresses
        assert is_local_address("tcp://localhost:5555") == True
        assert is_local_address("tcp://127.0.0.1:5555") == True
        # Note: 0.0.0.0 is not considered local by this function
        
        # Test remote addresses (assuming these are remote)
        assert is_local_address("tcp://192.168.1.100:5555") == False
        assert is_local_address("tcp://10.0.0.1:5555") == False
        assert is_local_address("tcp://example.com:5555") == False
        
    def test_choose_image_transport_mode(self):
        """Test transport mode selection logic."""
        from experimance_common.zmq.zmq_utils import choose_image_transport_mode, IMAGE_TRANSPORT_MODES
        
        # Test AUTO mode with local address but no file (should prefer BASE64)
        mode = choose_image_transport_mode(
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["AUTO"]
        )
        assert mode == IMAGE_TRANSPORT_MODES["BASE64"]
        
        # Test AUTO mode with remote address (should prefer BASE64)
        mode = choose_image_transport_mode(
            target_address="tcp://192.168.1.100:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["AUTO"]
        )
        assert mode == IMAGE_TRANSPORT_MODES["BASE64"]
        
        # Test explicit modes are preserved
        for explicit_mode in IMAGE_TRANSPORT_MODES.values():
            if explicit_mode != IMAGE_TRANSPORT_MODES["AUTO"]:
                mode = choose_image_transport_mode(
                    target_address="tcp://localhost:5555",
                    transport_mode=explicit_mode
                )
                assert mode == explicit_mode
                
    def test_message_type_constants(self):
        """Test that required MessageType constants exist."""
        from experimance_common.zmq.zmq_utils import MessageType
        
        # Test that key message types exist
        assert hasattr(MessageType, 'CHANGE_MAP')
        assert MessageType.CHANGE_MAP == "ChangeMap"
        
        # Test a few other expected message types
        expected_types = ['ERA_CHANGED', 'RENDER_REQUEST', 'IDLE_STATUS']
        for msg_type in expected_types:
            assert hasattr(MessageType, msg_type)
            assert isinstance(getattr(MessageType, msg_type), str)


if __name__ == "__main__":
    pytest.main(["-v", "test_zmq_utils.py"])
