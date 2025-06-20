"""
Mock utilities for testing the Image Server Service.

This module provides mock classes and factory functions specifically for testing
the image server service without requiring real image generation or network dependencies.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

from experimance_common.zmq.services import WorkerService
from experimance_common.zmq.config import WorkerServiceConfig
from image_server.config import ImageServerConfig
from image_server.image_service import ImageServerService
from image_server.generators.mock.mock_generator import MockImageGenerator
from image_server.generators.mock.mock_generator_config import MockGeneratorConfig


def create_mock_image_server_config(
    service_name: str = "test-image-server",
    cache_dir: Optional[Path] = None,
    strategy: str = "mock"
) -> ImageServerConfig:
    """
    Create a minimal ImageServerConfig for testing.
    
    Args:
        service_name: Name of the service instance
        cache_dir: Directory for image cache (defaults to tmp)
        strategy: Image generation strategy to use
        
    Returns:
        ImageServerConfig instance suitable for testing
    """
    if cache_dir is None:
        cache_dir = Path("/tmp/test_images")
        
    return ImageServerConfig.from_overrides(
        default_config={
            "service_name": service_name,
            "cache_dir": str(cache_dir),
            "generator": {
                "default_strategy": strategy,
                "timeout": 30
            },
            "mock": {
                "use_existing_images": False,
                "image_size": [512, 512]
            }
        }
    )


def create_mock_zmq_service() -> Mock:
    """
    Create a mock WorkerService for testing.
    
    Returns:
        Mock WorkerService with common methods mocked
    """
    mock_service = Mock(spec=WorkerService)
    mock_service.start = AsyncMock()
    mock_service.stop = AsyncMock()
    mock_service.set_work_handler = Mock()
    mock_service.add_message_handler = Mock()
    mock_service.publish = AsyncMock()
    mock_service.send_response = AsyncMock()
    return mock_service


def create_mock_image_generator() -> Mock:
    """
    Create a mock image generator for testing.
    
    Returns:
        Mock generator with generate_image method
    """
    mock_generator = Mock(spec=MockImageGenerator)
    mock_generator.generate_image = AsyncMock(return_value="/tmp/test_image.png")
    mock_generator.stop = AsyncMock()
    return mock_generator


def create_mock_image_server_service(
    config: Optional[ImageServerConfig] = None,
    mock_zmq: bool = True,
    mock_generator: bool = True
) -> ImageServerService:
    """
    Create a mock ImageServerService for testing.
    
    Args:
        config: Optional config to use (creates default if None)
        mock_zmq: Whether to mock the ZMQ service
        mock_generator: Whether to mock the image generator
        
    Returns:
        ImageServerService with mocked dependencies
    """
    if config is None:
        config = create_mock_image_server_config()
        
    # Create service instance
    service = ImageServerService(config)
    
    # Mock ZMQ service if requested
    if mock_zmq:
        service.zmq_service = create_mock_zmq_service()
        
    # Mock generator if requested
    if mock_generator:
        service.generator = create_mock_image_generator()
        
    return service


def mock_zmq_for_image_server():
    """
    Context manager to mock ZMQ components for image server testing.
    
    Returns:
        Context manager that patches WorkerService
    """
    return patch('image_server.image_service.WorkerService', return_value=create_mock_zmq_service())


def mock_generator_factory():
    """
    Context manager to mock the generator factory.
    
    Returns:
        Context manager that patches create_generator_from_config
    """
    return patch(
        'image_server.image_service.create_generator_from_config',
        return_value=create_mock_image_generator()
    )


# Sample test messages for use in tests
SAMPLE_RENDER_REQUEST = {
    "type": "RenderRequest",
    "request_id": "test_request_001",
    "prompt": "A beautiful landscape with mountains",
    "era": "current",
    "biome": "mountain",
    "seed": 12345,
    "negative_prompt": "blurry, low quality",
    "style": "photorealistic"
}

SAMPLE_IMAGE_READY_RESPONSE = {
    "type": "ImageReady",
    "request_id": "test_request_001",
    "image_path": "/tmp/test_image.png",
    "prompt": "A beautiful landscape with mountains",
    "era": "current",
    "biome": "mountain"
}

SAMPLE_ERROR_MESSAGE = {
    "type": "Alert",
    "request_id": "test_request_001",
    "severity": "error",
    "message": "Image generation failed: Mock error"
}


class MockImageServerTestCase:
    """
    Base test case class with common setup for image server tests.
    
    Provides common fixtures and helper methods for testing image server functionality.
    """
    
    def setup_method(self):
        """Set up method called before each test."""
        self.test_cache_dir = Path("/tmp/test_image_cache")
        self.test_cache_dir.mkdir(exist_ok=True)
        
    def teardown_method(self):
        """Cleanup method called after each test."""
        # Clean up test cache directory
        if self.test_cache_dir.exists():
            import shutil
            shutil.rmtree(self.test_cache_dir, ignore_errors=True)
            
    def create_test_config(self, **overrides) -> ImageServerConfig:
        """Create a test configuration with optional overrides."""
        default_overrides = {
            "cache_dir": str(self.test_cache_dir),
            **overrides
        }
        return create_mock_image_server_config(**default_overrides)
        
    def create_test_service(self, config: Optional[ImageServerConfig] = None) -> ImageServerService:
        """Create a test service instance."""
        if config is None:
            config = self.create_test_config()
        return create_mock_image_server_service(config)
