#!/usr/bin/env python3
"""
Tests for the DisplayService headless mode.

This test suite uses the built-in headless mode to test the display service
without requiring a display or extensive mocking.
"""

import asyncio
import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the display service to the path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.display.src.experimance_display.config import DisplayServiceConfig
from services.display.src.experimance_display.display_service import DisplayService


@pytest.fixture
def headless_config():
    """Create a test configuration for headless mode."""
    return DisplayServiceConfig.from_overrides(
        override_config={
            'service_name': 'test-display-headless',
            'display': {
                'fullscreen': False,
                'resolution': [800, 600],
                'debug_overlay': True,
                'headless': True  # Enable headless mode
            },
            'zmq': {
                'images_sub_address': 'tcp://localhost:15558',  # Different port for testing
                'events_sub_address': 'tcp://localhost:15555'
            }
        }
    )


@pytest.fixture
def mock_zmq_base():
    """Mock the ZMQ base service to avoid actual network connections."""
    with patch('experimance_common.zmq.subscriber.ZmqSubscriberService') as mock_class:
        
        # Create a mock instance
        mock_instance = Mock()
        mock_instance.service_name = "test-display-headless"
        mock_instance.running = False
        mock_instance.state = "STOPPED"
        mock_instance.handlers = {}
        
        # Create async mock methods
        mock_instance.start = AsyncMock()
        mock_instance.stop = AsyncMock()
        mock_instance.run = AsyncMock()
        mock_instance.register_handler = Mock()
        mock_instance.add_task = Mock()
        mock_instance.record_error = Mock()
        mock_instance.request_stop = AsyncMock()
        
        # Make the class return our mock instance
        mock_class.return_value = mock_instance
        
        yield mock_instance


class TestHeadlessMode:
    """Test the headless mode functionality."""
    
    def test_headless_config(self, headless_config):
        """Test that headless configuration is set correctly."""
        assert headless_config.display.headless is True
        assert headless_config.display.fullscreen is False
        assert headless_config.display.resolution == (800, 600)
    
    @pytest.mark.asyncio
    async def test_headless_service_creation(self, headless_config, mock_zmq_base):
        """Test creating a service in headless mode."""
        service = DisplayService(config=headless_config)
        
        # Check configuration
        assert service.config.display.headless is True
        assert service.window is None  # Not created yet
    
    @pytest.mark.asyncio
    async def test_headless_window_creation(self, headless_config, mock_zmq_base):
        """Test that headless mode creates a mock window."""
        service = DisplayService(config=headless_config)
        
        # Initialize window (this happens in start())
        service._initialize_window()
        
        # Should have created a headless window
        assert service.window is not None
        assert hasattr(service.window, 'width')
        assert hasattr(service.window, 'height')
        assert service.window.width == 800
        assert service.window.height == 600
        assert hasattr(service.window, 'has_exit')
    
    @pytest.mark.asyncio
    async def test_headless_start_stop(self, headless_config, mock_zmq_base):
        """Test starting and stopping service in headless mode."""
        service = DisplayService(config=headless_config)
        
        # Mock the renderers to avoid import issues
        with patch('services.display.src.experimance_display.display_service.LayerManager') as mock_layer, \
             patch('services.display.src.experimance_display.display_service.ImageRenderer') as mock_image, \
             patch('services.display.src.experimance_display.display_service.VideoOverlayRenderer') as mock_video, \
             patch('services.display.src.experimance_display.display_service.TextOverlayManager') as mock_text, \
             patch('services.display.src.experimance_display.display_service.clock') as mock_clock:
            
            # Mock clock functions
            mock_clock.schedule_interval = Mock()
            mock_clock.unschedule = Mock()
            
            # Mock renderer instances
            mock_layer.return_value = Mock()
            mock_image.return_value = Mock() 
            mock_video.return_value = Mock()
            mock_text.return_value = Mock()
            
            # Add async cleanup method to layer manager
            mock_layer.return_value.cleanup = AsyncMock()
            
            await service.start()
            
            # Check that window was created in headless mode
            assert service.window is not None
            assert not hasattr(service.window, 'on_draw')  # Headless window won't have event handlers
            
            # Check that renderers were created
            assert service.layer_manager is not None
            assert service.image_renderer is not None
            assert service.video_overlay_renderer is not None
            assert service.text_overlay_manager is not None
            
            # Stop the service
            await service.stop()
            
            # Verify cleanup was called
            mock_clock.unschedule.assert_called()
            service.layer_manager.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_headless_message_handling(self, headless_config, mock_zmq_base):
        """Test message handling in headless mode."""
        service = DisplayService(config=headless_config)
        
        # Mock renderers
        service.image_renderer = Mock()
        service.image_renderer.handle_image_ready = AsyncMock()
        service.text_overlay_manager = Mock()
        service.text_overlay_manager.handle_text_overlay = AsyncMock()
        
        # Test image message
        image_message = {
            'image_id': 'test-image-1',
            'uri': 'file:///tmp/test.jpg'
        }
        
        # Test message handling
        await service._handle_image_ready(image_message)
        service.image_renderer.handle_image_ready.assert_called_once_with(image_message)
        
        # Test text message
        text_message = {
            'text_id': 'test-text-1',
            'content': 'Hello, headless world!'
        }
        
        await service._handle_text_overlay(text_message)
        service.text_overlay_manager.handle_text_overlay.assert_called_once_with(text_message)
    
    @pytest.mark.asyncio
    async def test_headless_direct_interface(self, headless_config, mock_zmq_base):
        """Test the direct interface in headless mode."""
        service = DisplayService(config=headless_config)
        
        # Register direct handlers
        service._register_direct_handlers()
        
        # Mock clock to capture scheduled tasks
        with patch('services.display.src.experimance_display.display_service.clock') as mock_clock:
            mock_clock.schedule_once = Mock()
            
            # Test direct update
            test_data = {'test': 'data'}
            service.trigger_display_update('image_ready', test_data)
            
            # Should have scheduled a task
            mock_clock.schedule_once.assert_called_once()
            
            # Test unknown update type
            mock_clock.reset_mock()
            service.trigger_display_update('unknown_type', test_data)
            # Should not schedule anything for unknown types
            mock_clock.schedule_once.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_headless_error_handling(self, headless_config, mock_zmq_base):
        """Test error handling in headless mode."""
        service = DisplayService(config=headless_config)
        
        # Test that validation errors are handled gracefully
        invalid_message = {'missing': 'required_fields'}
        
        # These should not raise exceptions, and should return without processing
        with patch.object(service, '_validate_image_ready', return_value=False):
            await service._handle_image_ready(invalid_message)
        
        with patch.object(service, '_validate_text_overlay', return_value=False):
            await service._handle_text_overlay(invalid_message)
        
        # Test that the validation methods properly detect missing fields
        assert service._validate_image_ready(invalid_message) is False
        assert service._validate_text_overlay(invalid_message) is False


class TestHeadlessValidation:
    """Test message validation in headless mode."""
    
    def test_image_validation(self, headless_config, mock_zmq_base):
        """Test ImageReady message validation."""
        service = DisplayService(config=headless_config)
        
        # Valid message
        valid_msg = {'image_id': 'test-1', 'uri': 'file:///tmp/test.jpg'}
        assert service._validate_image_ready(valid_msg) is True
        
        # Invalid messages
        assert service._validate_image_ready({'uri': 'file:///tmp/test.jpg'}) is False
        assert service._validate_image_ready({'image_id': 'test-1'}) is False
        assert service._validate_image_ready({}) is False
    
    def test_text_validation(self, headless_config, mock_zmq_base):
        """Test TextOverlay message validation."""
        service = DisplayService(config=headless_config)
        
        # Valid message
        valid_msg = {'text_id': 'test-1', 'content': 'Hello!'}
        assert service._validate_text_overlay(valid_msg) is True
        
        # Invalid messages
        assert service._validate_text_overlay({'content': 'Hello!'}) is False
        assert service._validate_text_overlay({'text_id': 'test-1'}) is False
        assert service._validate_text_overlay({}) is False


class TestHeadlessMainFunction:
    """Test the main function with headless mode."""
    
    @pytest.mark.asyncio
    async def test_headless_argument_parsing(self):
        """Test that --headless argument is parsed correctly."""
        # This is more of an integration test
        import argparse
        
        # Simulate the argument parser from main()
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--windowed", "-w", action="store_true")
        parser.add_argument("--debug", action="store_true")
        
        # Test headless flag
        args = parser.parse_args(["--headless"])
        assert args.headless is True
        
        # Test combined flags
        args = parser.parse_args(["--headless", "--debug"])
        assert args.headless is True
        assert args.debug is True


if __name__ == "__main__":
    # Run a simple test to verify everything works
    pytest.main([__file__, "-v"])
