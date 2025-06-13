#!/usr/bin/env python3
"""
Test script for the Display Service's direct interface.

This test demonstrates how to use the direct interface for testing the display service
without requiring ZMQ infrastructure or actual windows.
"""

import asyncio
import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add the project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.display.src.experimance_display.config import DisplayServiceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        # Create a callable for the schedule_once that correctly executes the lambda function
        def schedule_callback(func, delay, *args, **kwargs):
            # The lambda in trigger_display_update looks like:
            # lambda dt: asyncio.create_task(self._direct_handlers[update_type](data))
            # We need to execute it with the dt parameter and let it create the asyncio task
            func(0.0, *args, **kwargs)
        
        # Set the side_effect to call our handler
        mock_clock.schedule_once.side_effect = schedule_callback
        
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
            mock_instance._running = False
            mock_instance.config = Mock()
            mock_instance.config.display = Mock()
            mock_instance.config.display.headless = True
            
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
                'images_sub_address': 'tcp://localhost:15558',  # Different port for testing
                'events_sub_address': 'tcp://localhost:15555'
            }
        }
    )


@pytest.mark.asyncio
async def test_direct_interface_sequence(test_config, mock_pyglet, mock_zmq_subscriber):
    """Test the sequence of direct interface calls."""
    # Import here after mocking is set up
    from services.display.src.experimance_display.display_service import DisplayService
    
    service = DisplayService(config=test_config)
    
    # Start service first, which will initialize the real renderers and handlers
    await service.start()
    
    # Now replace the handlers with our mocks after service initialization
    # We need to create proper mocks for the handlers
    handle_image_ready_mock = AsyncMock()
    handle_text_overlay_mock = AsyncMock()
    handle_video_mask_mock = AsyncMock()
    
    # Replace the handlers with our mocks
    service._direct_handlers["image_ready"] = handle_image_ready_mock
    service._direct_handlers["text_overlay"] = handle_text_overlay_mock
    service._direct_handlers["video_mask"] = handle_video_mask_mock
    
    # Test text overlay
    service.trigger_display_update("text_overlay", {
        "text_id": "test_text_1",
        "content": "Hello, Experimance!",
        "speaker": "agent",
        "duration": 5.0
    })
    
    # Test image update with mock path - we're mocking the handler so the file doesn't need to exist
    service.trigger_display_update("image_ready", {
        "image_id": "test_image_1",
        "uri": "file:///mock/path/test_image.png" 
    })
    
    # Test another text overlay
    service.trigger_display_update("text_overlay", {
        "text_id": "test_text_2", 
        "content": "This is a system message",
        "speaker": "system",
        "duration": 3.0
    })
    
    # Wait a bit to ensure any async tasks are processed
    await asyncio.sleep(0.1)
    
    # Verify that schedule_once was called for each trigger
    assert mock_pyglet['clock'].schedule_once.call_count >= 3
    
    # Verify that our mock handlers were called directly
    assert handle_text_overlay_mock.call_count > 0, "Text overlay handler wasn't called"
    assert handle_image_ready_mock.call_count > 0, "Image ready handler wasn't called"
    
    # Stop service
    await service.stop()


if __name__ == "__main__":
    # Run a simple test to verify everything works
    pytest.main([__file__, "-v"])
