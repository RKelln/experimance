#!/usr/bin/env python3
"""
Test suite for the Display Service.

This module contains comprehensive tests for the display service using the new
composition-based ZMQ architecture and proper mocking patterns.
"""

import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from experimance_display.display_service import DisplayService
from experimance_display.config import DisplayServiceConfig
from experimance_common.test_utils import active_service
from experimance_common.zmq.config import MessageType

from mocks import (
    create_mock_display_service,
    create_mock_display_config,
    create_test_message,
    mock_pyglet_components
)

logger = logging.getLogger(__name__)

class TestDisplayService:
    """Test suite for DisplayService using modern testing patterns."""
    
    @pytest.fixture
    def display_config(self):
        """Create a test configuration."""
        return create_mock_display_config()
    
    @pytest.fixture
    def mock_service(self, display_config):
        """Create a mocked display service."""
        return create_mock_display_service()
    
    def test_service_initialization(self, display_config):
        """Test that the service initializes correctly with mocked components."""
        with mock_pyglet_components():
            service = DisplayService(config=display_config)
            
            assert service.config == display_config
            assert service.target_fps == 30
            assert service.frame_timer == 0.0
            assert service.frame_count == 0
    
    def test_config_from_overrides(self):
        """Test that config overrides work correctly."""
        overrides = {
            "service_name": "custom_display",
            "display": {
                "resolution": [1920, 1080],
                "debug_overlay": True
            }
        }
        
        config = create_mock_display_config(overrides)
        
        assert config.service_name == "custom_display"
        assert config.display.resolution == (1920, 1080)  # Resolution is stored as tuple
        assert config.display.debug_overlay is True
        assert config.display.headless is True  # From defaults
    
    @pytest.mark.asyncio
    async def test_zmq_message_handlers_registration(self, mock_service):
        """Test that ZMQ message handlers are properly registered."""
        # The service should have registered handlers during initialization
        zmq_service = mock_service.zmq_service
        
        # Verify that add_message_handler was called for each message type
        expected_calls = [
            MessageType.IMAGE_READY,
            MessageType.TRANSITION_READY,
            MessageType.LOOP_READY,
            MessageType.TEXT_OVERLAY,
            MessageType.REMOVE_TEXT,
            MessageType.CHANGE_MAP,
            MessageType.ERA_CHANGED
        ]
        
        # Check that message handlers were registered
        assert hasattr(zmq_service, 'add_message_handler')
    
    @pytest.mark.asyncio
    async def test_handle_image_ready(self, mock_service):
        """Test handling of ImageReady messages."""
        message = create_test_message("image_ready")
        
        await mock_service._handle_image_ready(message)
        
        # Verify the image renderer was called
        mock_service.image_renderer.handle_image_ready.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_handle_text_overlay(self, mock_service):
        """Test handling of TextOverlay messages."""
        message = create_test_message("text_overlay")
        
        await mock_service._handle_text_overlay(message)
        
        # Verify the text overlay manager was called
        mock_service.text_overlay_manager.handle_text_overlay.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_handle_remove_text(self, mock_service):
        """Test handling of RemoveText messages."""
        message = create_test_message("remove_text")
        
        await mock_service._handle_remove_text(message)
        
        # Verify the text overlay manager was called
        mock_service.text_overlay_manager.handle_remove_text.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_handle_video_mask(self, mock_service):
        """Test handling of VideoMask messages."""
        message = create_test_message("video_mask")
        
        await mock_service._handle_video_mask(message)
        
        # Verify the video overlay renderer was called
        mock_service.video_overlay_renderer.handle_video_mask.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_handle_transition_ready(self, mock_service):
        """Test handling of TransitionReady messages."""
        message = create_test_message("transition_ready")
        
        await mock_service._handle_transition_ready(message)
        
        # Verify the image renderer was called
        mock_service.image_renderer.handle_transition_ready.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_handle_era_changed(self, mock_service):
        """Test handling of EraChanged messages."""
        message = create_test_message("era_changed")
        
        # This should not raise an error
        await mock_service._handle_era_changed(message)
    
    def test_validate_image_ready_valid(self, mock_service):
        """Test validation of valid ImageReady messages."""
        valid_message = create_test_message("image_ready")
        
        assert mock_service._validate_image_ready(valid_message) is True
    
    def test_validate_image_ready_invalid(self, mock_service):
        """Test validation of invalid ImageReady messages."""
        invalid_message = {"image_id": "test"}  # Missing uri
        
        assert mock_service._validate_image_ready(invalid_message) is False
    
    def test_validate_text_overlay_valid(self, mock_service):
        """Test validation of valid TextOverlay messages."""
        valid_message = create_test_message("text_overlay")
        
        assert mock_service._validate_text_overlay(valid_message) is True
    
    def test_validate_text_overlay_invalid(self, mock_service):
        """Test validation of invalid TextOverlay messages."""
        invalid_message = {"text_id": "test"}  # Missing content
        
        assert mock_service._validate_text_overlay(invalid_message) is False
    
    def test_direct_interface_handlers(self, mock_service):
        """Test that direct interface handlers are properly registered."""
        expected_handlers = [
            "image_ready",
            "text_overlay", 
            "remove_text",
            "video_mask",
            "transition_ready",
            "loop_ready",
            "era_changed"
        ]
        
        for handler_name in expected_handlers:
            assert handler_name in mock_service._direct_handlers
    
    def test_trigger_display_update(self, mock_service):
        """Test the direct interface for triggering display updates."""
        # Mock pyglet clock to avoid actual scheduling
        with patch('experimance_display.display_service.clock') as mock_clock:
            message = create_test_message("text_overlay")
            
            mock_service.trigger_display_update("text_overlay", message)
            
            # Verify clock.schedule_once was called
            mock_clock.schedule_once.assert_called_once()
    
    def test_trigger_display_update_unknown_type(self, mock_service):
        """Test triggering display update with unknown type."""
        with patch('experimance_display.display_service.clock'):
            mock_service.trigger_display_update("unknown_type", {})
            
            # Should not raise an error, just log a warning
    
    @pytest.mark.asyncio
    async def test_service_startup_and_shutdown(self):
        """Test service startup and shutdown sequence."""
        config = create_mock_display_config()
        
        with mock_pyglet_components():
            service = DisplayService(config=config)
            
            # Use active_service for proper lifecycle management
            async with active_service(service) as active:
                # Mock the components that would be initialized
                active.zmq_service = AsyncMock()
                active.layer_manager = AsyncMock()
                
                # The service is already started by active_service
                # Verify ZMQ service was started
                active.zmq_service.start.assert_called_once()
                
                # Test that layer_manager cleanup is available
                assert hasattr(active.layer_manager, 'cleanup')
                
                # Manually test cleanup
                await active.layer_manager.cleanup()
                active.layer_manager.cleanup.assert_called_once()


class TestDisplayServiceIntegration:
    """Integration tests for DisplayService with more realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_message_processing_sequence(self):
        """Test a sequence of different message types."""
        service = create_mock_display_service()
        
        # Process a sequence of messages
        messages = [
            ("image_ready", create_test_message("image_ready")),
            ("text_overlay", create_test_message("text_overlay")),
            ("video_mask", create_test_message("video_mask")),
            ("remove_text", create_test_message("remove_text"))
        ]
        
        for msg_type, message in messages:
            handler = service._direct_handlers[msg_type]
            await handler(message)
        
        # Verify all handlers were called
        service.image_renderer.handle_image_ready.assert_called_once()
        service.text_overlay_manager.handle_text_overlay.assert_called_once()
        service.video_overlay_renderer.handle_video_mask.assert_called_once()
        service.text_overlay_manager.handle_remove_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_message_processing(self):
        """Test that errors in message processing are handled gracefully."""
        service = create_mock_display_service()
        
        # Make the text overlay manager raise an exception
        service.text_overlay_manager.handle_text_overlay.side_effect = Exception("Test error")
        
        message = create_test_message("text_overlay")
        
        # This should not raise an exception
        await service._handle_text_overlay(message)
        
        # Verify error was recorded
        service.record_error.assert_called_once()


@pytest.mark.skip(reason="Visual test requires a display. Run manually with: pytest -xvs tests/test_display_service.py::test_visual_display_service")
async def test_visual_display_service():
    """Visual test for the display service (requires a display)."""
    # Create test configuration for visual testing
    config = DisplayServiceConfig.from_overrides(override_config={
        "service_name": "test_visual_display",
        "display": {
            "headless": False,  # Enable visual display
            "fullscreen": False,
            "resolution": [800, 600],
            "debug_overlay": True,
            "debug_text": True
        },
        "title_screen": {
            "enabled": True,
            "text": "Visual Test Mode",
            "duration": 2.0
        }
    })
    
    service = DisplayService(config=config)
    
    async with active_service(service):
        # Test text overlay
        logger.info("Testing text overlay...")
        text_message = create_test_message("text_overlay", 
            content="Hello! Welcome to Experimance Visual Test.",
            duration=3.0
        )
        service.trigger_display_update("text_overlay", text_message)
        
        # Wait to see the display
        await asyncio.sleep(5)


if __name__ == "__main__":
    # Allow running this test file directly for visual testing
    import sys
    if "--visual" in sys.argv:
        asyncio.run(test_visual_display_service())
    else:
        pytest.main([__file__])
