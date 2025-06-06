#!/usr/bin/env python3
"""
Unit tests for the ImageServerService using TDD approach.

This test file follows the established patterns from the experimance common library
and tests the image server service implementation.
"""

import asyncio
import logging
import pytest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq.zmq_utils import MessageType
from experimance_common.base_service import ServiceState

# Import reusable test utilities and mocks
from utils.tests.test_utils import (
    MockZmqPublisher, MockZmqSubscriber,
    wait_for_service_state
)

# Import the classes we'll test
from image_server.image_service import ImageServerService
from image_server.generators import ImageGenerator, MockImageGenerator

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestImageServerService:
    """Test cases for the main ImageServerService class."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for test configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_config_dir):
        """Create a mock configuration for testing."""
        return {
            "cache_dir": str(temp_config_dir / "images"),
            "max_cache_size_gb": 1.0,
            "zmq": {
                "events_sub_address": "tcp://localhost:5555",
                "images_pub_address": "tcp://*:5558"
            },
            "generator": {
                "default_strategy": "mock",
                "timeout_seconds": 5
            }
        }

    @pytest.fixture
    async def image_service(self, mock_config):
        """Create an ImageServerService instance for testing."""
        service = ImageServerService(
            service_name="test-image-server",
            config=mock_config
        )
        yield service
        # Cleanup
        if service.state != ServiceState.STOPPED:
            await service.stop()

    @pytest.mark.asyncio
    async def test_service_initialization(self, image_service):
        """Test that the service initializes correctly."""
        assert image_service.service_name == "test-image-server"
        assert image_service.service_type == "image-server"
        assert image_service.state == ServiceState.INITIALIZED
        assert image_service._default_strategy == "mock"

    @pytest.mark.asyncio
    async def test_service_lifecycle(self, image_service):
        """Test the service start/stop lifecycle."""
        # Service should start in STOPPED state
        assert image_service.state == ServiceState.INITIALIZED
        
        # Start the service
        await image_service.start()
        assert image_service.state == ServiceState.STARTED

        # Create a task to run the service - this will move it to RUNNING state
        run_task = asyncio.create_task(image_service.run())
        
        # Wait for the service to be fully running
        await wait_for_service_state(image_service, ServiceState.RUNNING)
        assert image_service.state == ServiceState.RUNNING

        # Stop the service
        await image_service.stop()
        assert image_service.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_render_request_handler_registration(self, image_service):
        """Test that RenderRequest handler is properly registered."""
        await image_service.start()
        
        # Check that the handler is registered for RenderRequest
        assert MessageType.RENDER_REQUEST in image_service.message_handlers
        handler = image_service.message_handlers[MessageType.RENDER_REQUEST]
        assert callable(handler)
        
        await image_service.stop()

    @pytest.mark.asyncio
    @patch('experimance_common.zmq.publisher.ZmqPublisher', MockZmqPublisher)
    @patch('experimance_common.zmq.subscriber.ZmqSubscriber', MockZmqSubscriber)
    async def test_zmq_socket_initialization(self, image_service):
        """Test that ZMQ sockets are properly initialized."""
        await image_service.start()
        
        # Check that sockets were created and registered
        assert image_service.publisher is not None
        assert image_service.subscriber is not None
        
        # Verify the addresses are set correctly
        assert image_service.publisher.address == "tcp://*:5558"
        assert image_service.subscriber.address == "tcp://localhost:5555"
        
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_render_request_handling(self, image_service):
        """Test handling of RenderRequest messages."""
        await image_service.start()
        
        # Create a mock render request
        request_id = str(uuid.uuid4())
        render_request = {
            "type": MessageType.RENDER_REQUEST,
            "request_id": request_id,
            "era": "modern",
            "biome": "coastal",
            "prompt": "A beautiful coastal city from above",
            "depth_map_png": "mock_base64_data"
        }
        
        # Mock the image generation
        with patch.object(image_service, '_generate_image') as mock_generate:
            mock_generate.return_value = "/tmp/generated_image.png"
            
            # Handle the request
            await image_service._handle_render_request(render_request)
            
            # Give time for the task to complete since _handle_render_request creates an async task
            await asyncio.sleep(0.1)
            
            # Verify image generation was called
            mock_generate.assert_called_once()
            
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_image_ready_publishing(self, image_service):
        """Test publishing of ImageReady messages."""
        await image_service.start()
        
        # Mock the publisher
        with patch.object(image_service, 'publish_message') as mock_publish:
            request_id = str(uuid.uuid4())
            image_path = "/tmp/test_image.png"
            
            await image_service._publish_image_ready(request_id, image_path)
            
            # Verify the message was published
            mock_publish.assert_called_once()
            call_args = mock_publish.call_args[0][0]
            
            assert call_args["type"] == MessageType.IMAGE_READY
            assert call_args["request_id"] == request_id
            assert call_args["uri"].endswith("test_image.png")
            assert "image_id" in call_args
            
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_generator_strategy_selection(self, image_service):
        """Test that the correct generator strategy is selected."""
        await image_service.start()
        
        # Test default strategy
        generator = image_service._get_generator()
        assert isinstance(generator, MockImageGenerator)
        
        # Test strategy override
        generator = image_service._get_generator("mock")
        assert isinstance(generator, MockImageGenerator)
        
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_error_handling_invalid_message(self, image_service):
        """Test error handling for invalid messages."""
        await image_service.start()
        
        # Create an invalid message (missing required fields)
        invalid_request = {
            "type": MessageType.RENDER_REQUEST,
            # Missing request_id, era, biome, prompt
        }
        
        # This should not raise an exception
        try:
            await image_service._handle_render_request(invalid_request)
        except Exception as e:
            pytest.fail(f"Handler should not raise exception for invalid message: {e}")
            
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_cache_directory_creation(self, temp_config_dir):
        """Test that cache directory is created if it doesn't exist."""
        # Use a different subdirectory to avoid conflicts with other tests
        cache_dir = temp_config_dir / "test_cache"
        assert not cache_dir.exists()
        
        # Create service with this specific cache directory
        config = {
            "cache_dir": str(cache_dir),
            "max_cache_size_gb": 1.0,
            "zmq": {
                "events_sub_address": "tcp://localhost:5555",
                "images_pub_address": "tcp://*:5558"
            },
            "generator": {
                "default_strategy": "mock",
                "timeout_seconds": 5
            }
        }
        
        service = ImageServerService(
            service_name="test-cache-service",
            config=config
        )
        
        try:
            await service.start()
            
            # Cache directory should be created
            assert cache_dir.exists()
            assert cache_dir.is_dir()
            
        finally:
            await service.stop()


class TestImageGenerators:
    """Test cases for image generator implementations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_mock_generator_initialization(self):
        """Test MockImageGenerator initialization."""
        generator = MockImageGenerator()
        assert isinstance(generator, ImageGenerator)

    @pytest.mark.asyncio
    async def test_mock_generator_generate_image(self, temp_dir):
        """Test image generation with MockImageGenerator."""
        generator = MockImageGenerator(output_dir=str(temp_dir))
        
        prompt = "A test image"
        depth_map = "mock_base64_data"
        
        image_path = await generator.generate_image(prompt, depth_map)
        
        # Verify image was "generated" (mock implementation)
        assert image_path is not None
        assert Path(image_path).exists()
        assert Path(image_path).suffix in ['.png', '.jpg', '.jpeg']

    @pytest.mark.asyncio
    async def test_generator_error_handling(self, temp_dir):
        """Test error handling in generators."""
        generator = MockImageGenerator(output_dir=str(temp_dir))
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            await generator.generate_image("", "")  # Empty prompt


class TestImageServerIntegration:
    """Integration tests for the image server service."""

    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
        return {
            "cache_dir": "/tmp/test_images",
            "max_cache_size_gb": 0.1,  # Small cache for testing
            "zmq": {
                "events_sub_address": "tcp://localhost:15555",  # Use different ports for testing
                "images_pub_address": "tcp://*:15558"
            },
            "generator": {
                "default_strategy": "mock",
                "timeout_seconds": 2
            }
        }

    @pytest.mark.asyncio
    async def test_end_to_end_image_generation(self, integration_config):
        """Test end-to-end image generation flow."""
        service = ImageServerService(
            service_name="integration-test-server",
            config=integration_config
        )
        
        try:
            await service.start()
            
            # Simulate a render request
            request_id = str(uuid.uuid4())
            render_request = {
                "type": MessageType.RENDER_REQUEST,
                "request_id": request_id,
                "era": "ai_future",
                "biome": "desert",
                "prompt": "Futuristic desert landscape with AI structures",
                "depth_map_png": "mock_depth_data"
            }
            
            # Mock the publish method to capture the output
            published_messages = []
            
            async def mock_publish(message, topic=None):
                published_messages.append(message)
                return True
            
            service.publish_message = mock_publish
            
            # Process the request
            await service._handle_render_request(render_request)
            
            # Give more time for processing since _handle_render_request creates an async task
            await asyncio.sleep(0.5)
            
            # Verify an ImageReady message was published
            assert len(published_messages) > 0
            image_ready_msg = published_messages[0]
            assert image_ready_msg["type"] == MessageType.IMAGE_READY
            assert image_ready_msg["request_id"] == request_id
            
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, integration_config):
        """Test handling multiple concurrent requests."""
        service = ImageServerService(
            service_name="concurrent-test-server",
            config=integration_config
        )
        
        try:
            await service.start()
            
            # Create multiple requests
            requests = []
            for i in range(3):
                request_id = str(uuid.uuid4())
                requests.append({
                    "type": MessageType.RENDER_REQUEST,
                    "request_id": request_id,
                    "era": "modern",
                    "biome": "forest",
                    "prompt": f"Forest landscape {i}",
                    "depth_map_png": f"mock_depth_data_{i}"
                })
            
            published_messages = []
            
            async def mock_publish(message, topic=None):
                published_messages.append(message)
                return True
            
            service.publish_message = mock_publish
            
            # Process all requests concurrently
            tasks = [
                service._handle_render_request(req) 
                for req in requests
            ]
            await asyncio.gather(*tasks)
            
            # Give some time for processing
            await asyncio.sleep(0.2)
            
            # Verify all requests were processed
            assert len(published_messages) == 3
            request_ids = {msg["request_id"] for msg in published_messages}
            expected_ids = {req["request_id"] for req in requests}
            assert request_ids == expected_ids
            
        finally:
            await service.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
