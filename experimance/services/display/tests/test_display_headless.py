#!/usr/bin/env python3
"""
Headless test suite for the Experimance Display Service.

This test suite uses mocked pyglet components to avoid requiring a display,
making it suitable for CI/CD environments and automated testing.
"""

import asyncio
import pytest
import logging
import tempfile
import json
import base64
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
import numpy as np

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the display service to the path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.display.src.experimance_display.config import DisplayServiceConfig
from experimance_common.test_utils import active_service


@pytest.fixture
def mock_pyglet():
    """Mock pyglet components to avoid creating real windows."""
    with patch('services.display.src.experimance_display.display_service.pyglet') as mock_pyglet_module, \
         patch('services.display.src.experimance_display.display_service.clock') as mock_clock, \
         patch('services.display.src.experimance_display.display_service.key') as mock_key:
        
        # Mock window class and instance
        mock_window_class = Mock()
        mock_window = Mock()
        mock_window.width = 800
        mock_window.height = 600
        mock_window.fullscreen = False
        mock_window.close = Mock()
        mock_window.clear = Mock()
        mock_window.flip = Mock()
        mock_window.switch_to = Mock()
        mock_window.dispatch_events = Mock()
        mock_window.dispatch_event = Mock()
        mock_window.has_exit = False
        mock_window.set_fullscreen = Mock()
        
        # Set up window class to return our mock window
        mock_window_class.return_value = mock_window
        
        # Mock pyglet module structure
        mock_pyglet_module.window.Window = mock_window_class
        mock_pyglet_module.display.get_display.return_value.get_default_screen.return_value = Mock()
        mock_pyglet_module.app.windows = [mock_window]
        mock_pyglet_module.app.exit = Mock()
        mock_pyglet_module.clock.tick = Mock()
        
        # Mock clock functions  
        mock_clock.unschedule = Mock()
        mock_clock.tick = Mock()
        mock_clock.schedule_once = Mock()
        
        # Mock key constants
        mock_key.ESCAPE = 65307  # Standard escape key code
        mock_key.Q = 113  # Standard Q key code  
        mock_key.F11 = 65480  # Standard F11 key code
        mock_key.F1 = 65470  # Standard F1 key code
        
        
        yield {
            'pyglet_module': mock_pyglet_module,
            'window_class': mock_window_class,
            'window': mock_window,
            'clock': mock_clock,
            'key': mock_key
        }


@pytest.fixture
def mock_zmq_subscriber():
    """Mock the ZMQ subscriber base class."""
    with patch('experimance_common.zmq.subscriber.ZmqSubscriberService') as mock_class:
        
        def create_mock_instance(service_name, subscribe_topics, **kwargs):
            """Create a properly configured mock instance."""
            mock_instance = Mock()
            
            # Set attributes that the display service expects
            mock_instance.service_name = service_name
            mock_instance.subscribe_topics = subscribe_topics if isinstance(subscribe_topics, list) else [subscribe_topics]
            mock_instance.handlers = {}
            mock_instance.running = False
            mock_instance.state = "STOPPED"  # Use state instead of _shutdown_requested
            
            # Create async mock methods
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            mock_instance.run = AsyncMock()
            mock_instance.register_handler = Mock()
            
            return mock_instance
        
        # Configure the mock class to return our custom mock instances
        mock_class.side_effect = create_mock_instance
        
        yield mock_class


@pytest.fixture
def test_config():
    """Create a test configuration suitable for headless testing."""
    return DisplayServiceConfig.from_overrides(
        override_config={
            'service_name': 'test-display',
            'display': {
                'fullscreen': False,
                'resolution': [800, 600],
                'debug_overlay': True
            },
            'zmq': {
                'subscriber': {
                    'address': 'tcp://localhost',
                    'port': 15555,
                    'topics': ['image.ready', 'text.overlay']
                }
            }
        }
    )


class TestDisplayServiceBasic:
    """Basic tests for display service initialization and lifecycle."""
    
    def test_config_creation(self, test_config):
        """Test that test configuration is created correctly."""
        assert test_config.service_name == 'test-display'
        assert test_config.display.fullscreen is False
        assert test_config.display.resolution == (800, 600)
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test service initialization without creating real windows."""
        # Import here after mocking is set up
        from services.display.src.experimance_display.display_service import DisplayService
        
        # Create service
        service = DisplayService(config=test_config)
        
        # Check initial state
        assert service.config.service_name == 'test-display'
        assert service.window is None
        assert hasattr(service, 'running')  # Check service has running property
        assert service.state  # Check service has state property
    
    @pytest.mark.asyncio
    async def test_service_start_stop(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test service start and stop lifecycle."""
        # Import here after mocking is set up
        from services.display.src.experimance_display.display_service import DisplayService
        from experimance_common.service_state import ServiceState
        
        service = DisplayService(config=test_config)
        
        # Use active_service for proper lifecycle management
        async with active_service(service) as active:
            # Verify mocks were called
            mock_pyglet['window_class'].assert_called_once()
            
            # Verify service state - when using active_service, it should be RUNNING
            assert active.state == ServiceState.RUNNING
            assert active.window is not None
            assert active.layer_manager is not None
        
        # Verify cleanup - check that service is stopped
        assert service.state == ServiceState.STOPPED


class TestDisplayServiceMessaging:
    """Test message handling and direct interface."""
    
    @pytest.mark.asyncio
    async def test_text_overlay_direct(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test text overlay functionality via direct interface."""
        from services.display.src.experimance_display.display_service import DisplayService
        from experimance_common.test_utils import active_service
        
        service = DisplayService(config=test_config)
        async with active_service(service) as active:
            # Mock the text overlay manager
            active.text_overlay_manager = Mock()
            active.text_overlay_manager.handle_text_overlay = AsyncMock()
            
            # Test text overlay message
            text_message = {
                'text_id': 'test-1',
                'content': 'Hello, World!',
                'speaker': 'agent',
                'duration': 5.0
            }
            
            # Trigger via direct interface
            active.trigger_display_update('text_overlay', text_message)
            
            # Wait a bit for the scheduled task
            await asyncio.sleep(0.1)
            
            # Verify the handler was called (would be in next frame)
            # Note: In real implementation, this would be scheduled via pyglet clock
    
    @pytest.mark.asyncio
    async def test_display_media_message(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test image ready message handling."""
        from services.display.src.experimance_display.display_service import DisplayService
        from experimance_common.test_utils import active_service
        
        service = DisplayService(config=test_config)
        async with active_service(service) as active:
            # Mock the image renderer
            active.image_renderer = Mock()
            active.image_renderer.handle_display_media = AsyncMock()
            
            # Test image message
            image_message = {
                'image_id': 'test-image-1',
                'uri': 'file:///tmp/test.jpg',
                'image_type': 'satellite_landscape'
            }
            
            # Test validation
            assert active._validate_image_ready(image_message) is True
            
            # Test handling
            await active._handle_display_media(image_message)
            
            # Verify the handler was called
            active.image_renderer.handle_display_media.assert_called_once_with(image_message)
    
    @pytest.mark.asyncio
    async def test_video_mask_message(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test video mask message handling."""
        from services.display.src.experimance_display.display_service import DisplayService
        from experimance_common.test_utils import active_service
        
        service = DisplayService(config=test_config)
        async with active_service(service) as active:
            # Mock the video overlay renderer
            active.video_overlay_renderer = Mock()
            active.video_overlay_renderer.handle_video_mask = AsyncMock()
            
            # Create a simple test mask (8x8 grayscale)
            test_mask = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
            
            # Test video mask message
            mask_message = {
                'mask_data': base64.b64encode(test_mask.tobytes()).decode(),
                'mask_width': 8,
                'mask_height': 8,
                'fade_in_duration': 0.5
            }
            
            # Test handling
            await active._handle_video_mask(mask_message)
            
            # Verify the handler was called
            active.video_overlay_renderer.handle_video_mask.assert_called_once_with(mask_message)


class TestMessageValidation:
    """Test message validation functions."""
    
    def test_image_ready_validation(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test ImageReady message validation."""
        from services.display.src.experimance_display.display_service import DisplayService
        
        service = DisplayService(config=test_config)
        
        # Valid message
        valid_message = {
            'image_id': 'test-1',
            'uri': 'file:///tmp/test.jpg'
        }
        assert service._validate_image_ready(valid_message) is True
        
        # Missing image_id
        invalid_message1 = {
            'uri': 'file:///tmp/test.jpg'
        }
        assert service._validate_image_ready(invalid_message1) is False
        
        # Missing uri
        invalid_message2 = {
            'image_id': 'test-1'
        }
        assert service._validate_image_ready(invalid_message2) is False
    
    def test_text_overlay_validation(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test TextOverlay message validation."""
        from services.display.src.experimance_display.display_service import DisplayService
        
        service = DisplayService(config=test_config)
        
        # Valid message
        valid_message = {
            'text_id': 'test-1',
            'content': 'Hello!'
        }
        assert service._validate_text_overlay(valid_message) is True
        
        # Missing text_id
        invalid_message1 = {
            'content': 'Hello!'
        }
        assert service._validate_text_overlay(invalid_message1) is False
        
        # Missing content
        invalid_message2 = {
            'text_id': 'test-1'
        }
        assert service._validate_text_overlay(invalid_message2) is False


class TestConfigurationLoading:
    """Test configuration loading from various sources."""
    
    def test_default_config_values(self):
        """Test that default configuration has expected values."""
        config = DisplayServiceConfig()
        
        # Service settings
        assert config.service_name == "display-service"
        
        # Display settings
        assert config.display.fullscreen is True
        assert config.display.resolution == (1920, 1080)
        assert config.display.fps_limit == 60
        assert config.display.vsync is True
        
        # Text styles
        assert config.text_styles.agent.font_size == 28
        assert config.text_styles.agent.position == "bottom_center"
        assert config.text_styles.system.font_size == 24
        assert config.text_styles.debug.font_size == 16
    
    def test_config_override(self):
        """Test configuration override functionality."""
        override_config = {
            'display': {
                'fullscreen': False,
                'resolution': [800, 600]
            },
            'text_styles': {
                'agent': {
                    'font_size': 32
                }
            }
        }
        
        config = DisplayServiceConfig.from_overrides(
            override_config=override_config
        )
        
        # Check overrides were applied
        assert config.display.fullscreen is False
        assert config.display.resolution == (800, 600)
        assert config.text_styles.agent.font_size == 32
        
        # Check non-overridden values remain default
        assert config.display.fps_limit == 60
        assert config.text_styles.system.font_size == 24
    
    def test_config_from_toml_structure(self):
        """Test configuration loading with proper TOML structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[display]
fullscreen = false
resolution = [1280, 720]
fps_limit = 30

[text_styles.agent]
font_size = 36
color = [255, 0, 0, 255]

[zmq.subscriber]
address = "tcp://localhost"
port = 9999
topics = ["image.ready", "text.overlay"]
""")
            f.flush()
            
            config = DisplayServiceConfig.from_overrides(config_file=f.name)
            
            # Check TOML values were loaded
            assert config.display.fullscreen is False
            assert config.display.resolution == (1280, 720)
            assert config.display.fps_limit == 30
            assert config.text_styles.agent.font_size == 36
            assert config.text_styles.agent.color == (255, 0, 0, 255)
            assert config.zmq.subscriber.address == "tcp://localhost"
            assert config.zmq.subscriber.port == 9999
        
        # Clean up
        Path(f.name).unlink()


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_message_handling(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test that invalid messages are handled gracefully."""
        from services.display.src.experimance_display.display_service import DisplayService
        from experimance_common.test_utils import active_service
        
        service = DisplayService(config=test_config)
        async with active_service(service) as active:
            # Test with completely invalid message
            invalid_message = {"invalid": "data"}
            
            # These should not raise exceptions
            await active._handle_display_media(invalid_message)
            await active._handle_text_overlay(invalid_message)
            await active._handle_video_mask(invalid_message)
    
    @pytest.mark.asyncio
    async def test_missing_renderer_handling(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test handling when renderers are not initialized."""
        from services.display.src.experimance_display.display_service import DisplayService
        
        service = DisplayService(config=test_config)
        
        # Don't call start() so renderers are None
        service.image_renderer = None
        service.text_overlay_manager = None
        service.video_overlay_renderer = None
        
        # Test that these don't crash
        valid_image_msg = {'image_id': 'test', 'uri': 'file:///tmp/test.jpg'}
        valid_text_msg = {'text_id': 'test', 'content': 'Hello'}
        valid_mask_msg = {'mask_data': 'dGVzdA=='}  # base64 'test'
        
        await service._handle_display_media(valid_image_msg)
        await service._handle_text_overlay(valid_text_msg)
        await service._handle_video_mask(valid_mask_msg)


class TestDirectInterface:
    """Test the direct interface for non-ZMQ testing."""
    
    @pytest.mark.asyncio
    async def test_trigger_display_update(self, test_config, mock_pyglet, mock_zmq_subscriber):
        """Test direct interface for triggering updates."""
        from services.display.src.experimance_display.display_service import DisplayService
        from experimance_common.test_utils import active_service
        
        service = DisplayService(config=test_config)
        async with active_service(service) as active:
            # Mock handlers
            active._direct_handlers = {
                'test_update': AsyncMock()
            }
            
            test_data = {'test': 'data'}
            
            # This should schedule the handler
            active.trigger_display_update('test_update', test_data)
            
            # Verify scheduling was attempted
            mock_pyglet['clock'].schedule_once.assert_called()
            
            # Test unknown update type
            active.trigger_display_update('unknown_type', test_data)


if __name__ == "__main__":
    # Run a simple test to verify everything works
    pytest.main([__file__, "-v"])
