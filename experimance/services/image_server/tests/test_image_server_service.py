#!/usr/bin/env python3
"""
Tests for the Image Server Service using the new ZMQ architecture.

These tests verify the functionality of the ImageServerService including:
- Service lifecycle management
- ZMQ message handling
- Image generation pipeline
- Error handling and recovery
"""

import pytest
import asyncio
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from experimance_common.service_state import ServiceState
from experimance_common.base_service import ServiceStatus
from experimance_common.zmq.config import MessageType
from experimance_common.zmq.mocks import MockWorkerService, mock_message_bus

from image_server.image_service import ImageServerService
from image_server.config import ImageServerConfig

# Import our custom mocks
from tests.mocks import (
    create_mock_image_server_config,
    create_mock_image_server_service,
    mock_zmq_for_image_server,
    mock_generator_factory,
    SAMPLE_RENDER_REQUEST,
    SAMPLE_IMAGE_READY_RESPONSE,
    MockImageServerTestCase
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestImageServerService(MockImageServerTestCase):
    """Test cases for the main ImageServerService class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration using the new ZMQ architecture."""
        return create_mock_image_server_config(
            service_name="test-image-server",
            cache_dir=self.test_cache_dir,
            strategy="mock"
        )

    @pytest.fixture
    def mock_service(self, mock_config):
        """Create a mock ImageServerService for testing."""
        return create_mock_image_server_service(config=mock_config)

    @pytest.fixture  
    async def image_service(self, mock_config: ImageServerConfig):
        """Create an ImageServerService instance for testing with mocked ZMQ."""
        # Clear the mock message bus before each test
        mock_message_bus.clear()
        
        with mock_zmq_for_image_server(), mock_generator_factory():
            service = ImageServerService(config=mock_config)
            yield service
            # Cleanup
            if hasattr(service, 'zmq_service'):
                await service.zmq_service.stop()

    @pytest.mark.asyncio
    async def test_service_initialization(self, image_service: ImageServerService):
        """Test that the service initializes correctly."""
        assert image_service.service_name == "test-image-server"
        assert image_service.config.service_name == "test-image-server"
        assert image_service._default_strategy == "mock"
        assert hasattr(image_service, 'zmq_service')
        assert hasattr(image_service, 'generator')

    @pytest.mark.asyncio
    async def test_service_lifecycle(self, image_service: ImageServerService):
        """Test the service start/stop lifecycle."""
        # Service should start in INITIALIZED state  
        assert image_service.state == ServiceState.INITIALIZED
        
        # Start the service
        await image_service.start()
        assert image_service.state == ServiceState.STARTED

        # Stop the service
        await image_service.stop()
        assert image_service.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_render_request_handler_registration(self, image_service: ImageServerService):
        """Test that RenderRequest handler is properly registered."""
        await image_service.start()
        
        # Check that the ZMQ service has handlers set up
        # With mocked service, we verify the mock was called correctly
        image_service.zmq_service.set_work_handler.assert_called_once()
        image_service.zmq_service.add_message_handler.assert_called_once()
        
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_zmq_service_initialization(self, image_service: ImageServerService):
        """Test that ZMQ service is properly initialized."""
        # Check that ZMQ service was created 
        assert image_service.zmq_service is not None
        # With mocked ZMQ service, we verify it's a Mock object
        assert isinstance(image_service.zmq_service, Mock)

    @pytest.mark.asyncio
    async def test_render_request_handling(self, image_service: ImageServerService):
        """Test handling of RenderRequest messages."""
        await image_service.start()
        
        # Create a copy of our sample render request
        render_request = SAMPLE_RENDER_REQUEST.copy()
        
        # Mock the image generation process
        with patch.object(image_service, '_process_render_request') as mock_process:
            mock_process.return_value = AsyncMock()
            
            # Handle the request
            await image_service._handle_render_request(render_request)
            
            # Verify the process was called
            mock_process.assert_called_once()
        
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_image_generation_mock_strategy(self, image_service: ImageServerService):
        """Test image generation using mock strategy."""
        await image_service.start()
        
        # Test image generation
        image_path = await image_service._generate_image(
            prompt="Test image",
            era="current",
            biome="urban"
        )
        
        # Verify the image was "generated" (mock should create a file)
        assert isinstance(image_path, str)
        assert len(image_path) > 0
        
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_message_publishing(self, image_service: ImageServerService):
        """Test publishing ImageReady messages."""
        await image_service.start()
        
        request_id = "test_request_001"
        test_image_path = "/tmp/test_image.png"
        
        # Mock the image path existence
        with patch('pathlib.Path.exists', return_value=True):
            
            await image_service._publish_image_ready(
                request_id=request_id,
                image_path=test_image_path,
                prompt="Test prompt"
            )
            
            # Check that send_response was called on the ZMQ service
            image_service.zmq_service.send_response.assert_called_once()
            
            # Get the message that was sent
            call_args = image_service.zmq_service.send_response.call_args[0]
            message_content = call_args[0]
            assert message_content["type"] == str(MessageType.IMAGE_READY)
            assert message_content["request_id"] == request_id
        
        await image_service.stop()

    @pytest.mark.asyncio 
    async def test_cache_cleanup(self, image_service: ImageServerService):
        """Test cache cleanup functionality."""
        await image_service.start()
        
        # Mock cache directory with some files
        cache_dir = image_service.config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some test files
        test_files = []
        for i in range(3):
            test_file = cache_dir / f"test_image_{i}.png"
            test_file.write_text("mock image data")
            test_files.append(test_file)
        
        # Run cache cleanup
        await image_service._cleanup_cache()
        
        # Verify files still exist (since we're under the size limit)
        for test_file in test_files:
            assert test_file.exists()
        
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_error_handling_invalid_request(self, image_service: ImageServerService):
        """Test error handling for invalid render requests."""
        await image_service.start()
        
        # Create an invalid render request (missing required fields)
        invalid_request = {
            "type": str(MessageType.RENDER_REQUEST),
            # Missing request_id and prompt
        }
        
        # Mock the record_error method to capture error calls
        image_service.record_error = Mock()
        
        # Handle the invalid request
        await image_service._handle_render_request(invalid_request)
        
        # Verify error was recorded
        image_service.record_error.assert_called()
        
        await image_service.stop()

    @pytest.mark.asyncio
    async def test_generator_strategy_selection(self, image_service: ImageServerService):
        """Test that the correct generator strategy is selected."""
        # Verify the mock generator was created
        assert image_service.generator is not None
        assert image_service._default_strategy == "mock"

    @pytest.mark.asyncio
    async def test_error_handling_invalid_message(self, image_service: ImageServerService):
        """Test that invalid messages don't crash the service.""" 
        await image_service.start()
        
        # Create completely invalid message
        invalid_request = {"invalid": "data"}
        
        # This should not raise an exception
        try:
            await image_service._handle_render_request(invalid_request)
        except Exception as e:
            pytest.fail(f"Handler should not raise exception for invalid message: {e}")
            
        await image_service.stop()

    @pytest.mark.asyncio  
    async def test_cache_directory_creation(self, mock_config):
        """Test that cache directory is created if it doesn't exist."""
        # Create service 
        service = create_mock_image_server_service(config=mock_config)
        
        # Cache directory should be created during initialization
        assert mock_config.cache_dir.exists()
        assert mock_config.cache_dir.is_dir()


class TestImageGenerators:
    """Test cases for image generator implementations."""

    def test_mock_generator_config(self):
        """Test mock generator configuration."""
        from image_server.generators.mock.mock_generator_config import MockGeneratorConfig
        config = MockGeneratorConfig()
        assert config.use_existing_images is False
        assert config.image_size == (1024, 1024)

    def test_mock_generator_creation(self):
        """Test creating a mock generator."""
        from image_server.generators.mock.mock_generator import MockImageGenerator
        from image_server.generators.mock.mock_generator_config import MockGeneratorConfig
        
        config = MockGeneratorConfig()
        generator = MockImageGenerator(output_dir="/tmp", config=config)
        assert generator is not None
        assert generator.config.image_size == (1024, 1024)


if __name__ == "__main__":
    pytest.main([__file__])
