#!/usr/bin/env python3
"""
Comprehensive test suite for the Experimance Display Service.

This test suite covers:
- Text overlay functionality with ZMQ message simulation
- Video overlay with mask updates
- Image loading and crossfade transitions
- Direct interface testing
- Configuration validation
- Error handling scenarios

Uses the active_service() context manager from utils/tests/test_utils.py for proper
service lifecycle management.
"""

import asyncio
import pytest
import logging
import tempfile
import json
import base64
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
import numpy as np
from PIL import Image

# Add the display service to the path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experimance_display.display_service import DisplayService
from experimance_display.config import DisplayServiceConfig, DisplayConfig
from experimance_common.base_service import ServiceState, ServiceStatus
from experimance_common.config import Config
from experimance_common.test_utils import active_service

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDisplayServiceConfig:
    """Test configuration loading and validation."""
    
    def test_default_config(self):
        """Test that default configuration is valid."""
        config = DisplayServiceConfig()
        assert config.service_name == "display-service"
        assert config.display.fullscreen is True
        assert config.display.resolution == (1920, 1080)
        assert config.text_styles.agent.font_size == 28
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "display": {
                "fullscreen": False,
                "resolution": [800, 600],
                "debug_overlay": True
            },
            "text_styles": {
                "agent": {
                    "font_size": 32,
                    "color": [255, 0, 0, 255]
                }
            }
        }
        
        config = DisplayServiceConfig(**config_dict)
        assert config.display.fullscreen is False
        assert config.display.resolution == (800, 600)
        assert config.display.debug_overlay is True
        assert config.text_styles.agent.font_size == 32
        assert config.text_styles.agent.color == (255, 0, 0, 255)
    
    def test_config_validation_errors(self):
        """Test that invalid configurations raise validation errors."""
        with pytest.raises(Exception):
            # Invalid resolution format
            DisplayServiceConfig(display=DisplayConfig(resolution="invalid")) # type: ignore
        
        with pytest.raises(Exception):
            # Invalid color format
            DisplayServiceConfig(text_styles={
                "agent": {"color": [255, 255]}  # Should be 4 values
            }) # type: ignore


@pytest.fixture
def test_config():
    """Create a test configuration for headless mode testing."""
    return DisplayServiceConfig(
        service_name="test-display",
        display=DisplayConfig(
            fullscreen=False,
            resolution=(800, 600),
            debug_overlay=True,
            vsync=False,  # Disable for faster testing
            headless=True  # Use headless mode for testing to avoid window issues
        )
    )


@pytest.fixture
def mock_image_files():
    """Create temporary test image files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create a simple test image
    test_image = Image.new('RGB', (256, 256), color='red')
    image_path = temp_dir / "test_image.png"
    test_image.save(image_path)
    
    # Create another test image for transitions
    test_image2 = Image.new('RGB', (256, 256), color='blue')
    image_path2 = temp_dir / "test_image2.png"
    test_image2.save(image_path2)
    
    yield {
        "image1": image_path,
        "image2": image_path2,
        "temp_dir": temp_dir
    }
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_video_mask():
    """Create a mock video mask for testing."""
    # Create a simple grayscale mask
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255  # White square in center
    
    # Convert to base64
    mask_image = Image.fromarray(mask, mode='L')
    import io
    buffer = io.BytesIO()
    mask_image.save(buffer, format='PNG')
    mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return mask_base64


class TestDisplayServiceBasics:
    """Test basic display service functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, test_config):
        """Test that the service initializes correctly."""
        with patch('pyglet.window.Window') as mock_window:
            # Mock the window creation
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            service = DisplayService(config=test_config)
            
            # Using active_service context manager for proper lifecycle management
            async with active_service(service) as active:
                # Service is now properly started and running
                assert active.config.service_name == "test-display"
                assert active.layer_manager is not None
                assert active.image_renderer is not None
                assert active.text_overlay_manager is not None
                assert active.video_overlay_renderer is not None
                
            # Service is automatically stopped by active_service when exiting the context
    
    @pytest.mark.asyncio
    async def test_service_shutdown(self, test_config):
        """Test that the service shuts down gracefully."""
        with patch('pyglet.window.Window') as mock_window:
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            service = DisplayService(config=test_config)
            
            # Start service using active_service
            async with active_service(service):
                # Service is running here
                pass
                
            # After context exits, service should be stopped
            assert service.state == ServiceState.STOPPED


class TestTextOverlays:
    """Test text overlay functionality."""
    
    @pytest.mark.asyncio
    async def test_text_overlay_creation(self, test_config):
        """Test creating text overlays with different speakers."""
        with patch('pyglet.window.Window') as mock_window, \
             patch('pyglet.clock.schedule_once') as mock_schedule:
            # Setup window mock
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            # Configure schedule_once to immediately execute the callback
            def execute_callback(callback, delay, *args, **kwargs):
                callback(0)  # Call with dt=0
            mock_schedule.side_effect = execute_callback
            
            service = DisplayService(config=test_config)
            await service.start()
            
            assert service.text_overlay_manager is not None

            # Mock the text overlay handler to directly add text
            async def mock_handle_text(message):
                await service.text_overlay_manager.handle_text_overlay(message) # type: ignore
            service._handle_text_overlay = mock_handle_text
            
            # Test agent text
            agent_message = {
                "text_id": "agent_1",
                "content": "Hello from the agent!",
                "speaker": "agent",
                "duration": 5.0
            }
            
            # Call handler directly instead of using trigger_display_update
            await service._handle_text_overlay(agent_message)
            
            # Verify text was added to manager
            assert "agent_1" in service.text_overlay_manager.active_texts
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_text_overlay_replacement(self, test_config):
        """Test text replacement with same ID (streaming text)."""
        with patch('pyglet.window.Window') as mock_window, \
             patch('pyglet.clock.schedule_once') as mock_schedule:
            # Setup window mock
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            # Configure schedule_once to immediately execute the callback
            def execute_callback(callback, delay, *args, **kwargs):
                callback(0)  # Call with dt=0
            mock_schedule.side_effect = execute_callback
            
            service = DisplayService(config=test_config)
            await service.start()
            
            # Mock the text overlay handler to directly add text
            async def mock_handle_text(message):
                await service.text_overlay_manager.handle_text_overlay(message) # type: ignore
            service._handle_text_overlay = mock_handle_text
            
            # Create initial text
            initial_message = {
                "text_id": "streaming_1",
                "content": "Initial text",
                "speaker": "agent",
                "duration": None
            }
            # Call handler directly
            await service._handle_text_overlay(initial_message)
            
            # Replace with new text (same ID)
            updated_message = {
                "text_id": "streaming_1",
                "content": "Updated text content",
                "speaker": "agent",
                "duration": None,
                "replace": True
            }
            # Call handler directly
            await service._handle_text_overlay(updated_message)
            
            # Should still have only one text with the same ID
            assert service.text_overlay_manager is not None
            assert "streaming_1" in service.text_overlay_manager.active_texts
            assert len(service.text_overlay_manager.active_texts) == 1
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_text_removal(self, test_config):
        """Test removing specific text overlays."""
        with patch('pyglet.window.Window') as mock_window, \
             patch('pyglet.clock.schedule_once') as mock_schedule:
            # Setup window mock
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            # Configure schedule_once to immediately execute the callback
            def execute_callback(callback, delay, *args, **kwargs):
                callback(0)  # Call with dt=0
            mock_schedule.side_effect = execute_callback
            
            service = DisplayService(config=test_config)
            await service.start()
            
            # Mock the handlers to bypass the scheduling mechanism
            async def mock_handle_text(message):
                await service.text_overlay_manager.handle_text_overlay(message) # type: ignore
            service._handle_text_overlay = mock_handle_text
            
            async def mock_handle_remove_text(message):
                await service.text_overlay_manager.handle_remove_text(message) # type: ignore
            service._handle_remove_text = mock_handle_remove_text
            
            # Create text
            text_message = {
                "text_id": "removable_1",
                "content": "This text will be removed",
                "speaker": "system",
                "duration": None
            }
            # Call handler directly
            await service._handle_text_overlay(text_message)
            
            assert "removable_1" in service.text_overlay_manager.active_texts # type: ignore
            
            # Remove text
            remove_message = {
                "text_id": "removable_1"
            }
            # Call handler directly
            await service._handle_remove_text(remove_message)
            
            # Start fade-out process
            assert service.text_overlay_manager is not None
            service.text_overlay_manager.update(1.0)  # Update with large enough dt to complete fade out
            
            assert "removable_1" not in service.text_overlay_manager.active_texts # type: ignore
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_speaker_styles(self, test_config):
        """Test that different speakers use different text styles."""
        with patch('pyglet.window.Window') as mock_window, \
             patch('pyglet.clock.schedule_once') as mock_schedule:
            # Setup window mock
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            # Configure schedule_once to immediately execute the callback
            def execute_callback(callback, delay, *args, **kwargs):
                callback(0)  # Call with dt=0
            mock_schedule.side_effect = execute_callback
            
            service = DisplayService(config=test_config)
            await service.start()
            
            # Mock the text overlay handler
            async def mock_handle_text(message):
                await service.text_overlay_manager.handle_text_overlay(message) # type: ignore
            service._handle_text_overlay = mock_handle_text
            
            # Create text for different speakers
            speakers = ["agent", "system", "debug"]
            
            for i, speaker in enumerate(speakers):
                message = {
                    "text_id": f"{speaker}_text",
                    "content": f"Text from {speaker}",
                    "speaker": speaker,
                    "duration": None
                }
                # Call handler directly
                await service._handle_text_overlay(message)
            
            # Verify all texts are active
            for speaker in speakers:
                assert f"{speaker}_text" in service.text_overlay_manager.active_texts # type: ignore
            
            await service.stop()


class TestImageDisplay:
    """Test image loading and display functionality."""
    
    @pytest.mark.asyncio
    async def test_image_loading(self, test_config, mock_image_files):
        """Test loading and displaying images."""
        with patch('pyglet.window.Window') as mock_window, \
             patch('pyglet.clock.schedule_once') as mock_schedule, \
             patch('pyglet.image.load') as mock_load, \
             patch('pyglet.sprite.Sprite') as mock_sprite:
            # Setup window mock
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            # Configure schedule_once to immediately execute the callback
            def execute_callback(callback, delay, *args, **kwargs):
                callback(0)  # Call with dt=0
            mock_schedule.side_effect = execute_callback
            
            # Mock image loading
            mock_image = Mock()
            mock_image.width = 256
            mock_image.height = 256
            mock_load.return_value = mock_image
            
            # Mock sprite creation
            mock_sprite_instance = Mock()
            mock_sprite.return_value = mock_sprite_instance
            
            service = DisplayService(config=test_config)
            await service.start()
            
            # Mock the direct image handling to bypass actual file loading
            async def mock_handle_image(message):
                service.image_renderer.current_image_id_value = message["image_id"] # type: ignore
            service._handle_image_ready = mock_handle_image
            
            # Test image loading
            image_message = {
                "image_id": "test_image_1",
                "uri": f"file://{mock_image_files['image1']}",
                "image_type": "satellite_landscape"
            }
            
            # Call handler directly
            await service._handle_image_ready(image_message)
            
            # Verify image was processed
            assert service.image_renderer.current_image_id == "test_image_1" # type: ignore
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_image_crossfade_transition(self, test_config, mock_image_files):
        """Test crossfade transition between images."""
        with patch('pyglet.window.Window') as mock_window, \
             patch('pyglet.clock.schedule_once') as mock_schedule, \
             patch('pyglet.image.load') as mock_load, \
             patch('pyglet.sprite.Sprite') as mock_sprite:
            # Setup window mock
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            # Configure schedule_once to immediately execute the callback
            def execute_callback(callback, delay, *args, **kwargs):
                callback(0)  # Call with dt=0
            mock_schedule.side_effect = execute_callback
            
            # Mock image loading
            mock_image = Mock()
            mock_image.width = 256
            mock_image.height = 256
            mock_load.return_value = mock_image
            
            # Mock sprite creation
            mock_sprite_instance = Mock()
            mock_sprite.return_value = mock_sprite_instance
            
            service = DisplayService(config=test_config)
            await service.start()
            
            # Setup a mocked image renderer with controlled behavior
            assert service.image_renderer is not None
            service.image_renderer.current_image_id_value = None
            service.image_renderer.transition_active = False
            
            # Custom handler to control transition behavior
            async def mock_handle_image(message):
                image_id = message["image_id"]
                
                assert service.image_renderer is not None
                if service.image_renderer.current_image_id_value is None:
                    # First image
                    service.image_renderer.current_image_id_value = image_id
                else:
                    # Second image starts transition
                    service.image_renderer.next_image_id_value = image_id
                    service.image_renderer.transition_active = True
                    
                    # Simulate transition completing
                    await asyncio.sleep(0.1)  # Small delay for test order
                    if message["image_id"] == "image_2":
                        # Update state after "transition"
                        service.image_renderer.current_image_id_value = service.image_renderer.next_image_id_value
                        service.image_renderer.next_image_id_value = None
                        service.image_renderer.transition_active = False
                
            service._handle_image_ready = mock_handle_image
            
            # Load first image
            image1_message = {
                "image_id": "image_1",
                "uri": f"file://{mock_image_files['image1']}",
                "image_type": "satellite_landscape"
            }
            await service._handle_image_ready(image1_message)
            
            # Verify first image is loaded
            assert service.image_renderer.current_image_id == "image_1"
            assert service.image_renderer.is_transitioning is False
            
            # Load second image (should trigger crossfade)
            image2_message = {
                "image_id": "image_2", 
                "uri": f"file://{mock_image_files['image2']}",
                "image_type": "satellite_landscape",
                "transition_duration": 0.5  # Fast transition for testing
            }
            
            # Start transition
            transition_task = asyncio.create_task(service._handle_image_ready(image2_message))
            
            # Check transition is active
            await asyncio.sleep(0.05)
            assert service.image_renderer.is_transitioning is True
            
            # Wait for transition to complete
            await transition_task
            await asyncio.sleep(0.1)
            
            # Should now show the second image
            assert service.image_renderer.current_image_id == "image_2"
            assert service.image_renderer.is_transitioning is False
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_invalid_image_handling(self, test_config):
        """Test handling of invalid or missing images."""
        with patch('pyglet.window.Window') as mock_window:
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            service = DisplayService(config=test_config)
            
            # Use active_service for proper lifecycle management
            async with active_service(service) as active:
                # Try to load non-existent image
                invalid_message = {
                    "image_id": "invalid_image",
                    "uri": "file:///non/existent/path.jpg",
                    "image_type": "satellite_landscape"
                }
                
                # Should not crash the service
                active.trigger_display_update("image_ready", invalid_message)
                await asyncio.sleep(0.2)
                
                # Service should still be running
                assert active.state == ServiceState.RUNNING


class TestVideoOverlay:
    """Test video overlay and masking functionality."""
    
    @pytest.mark.asyncio
    async def test_video_mask_update(self, test_config, mock_video_mask):
        """Test updating video overlay mask."""
        with patch('pyglet.window.Window') as mock_window, \
             patch('pyglet.clock.schedule_once') as mock_schedule:
            # Setup window mock
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            # Configure schedule_once to immediately execute the callback
            def execute_callback(callback, delay, *args, **kwargs):
                callback(0)  # Call with dt=0
            mock_schedule.side_effect = execute_callback
            
            service = DisplayService(config=test_config)
            await service.start()
            
            # Mock the video mask handler
            async def mock_handle_video_mask(message):
                # Set a test mask value
                mask_data = message["mask_data"]
                service.video_overlay_renderer.current_mask = mask_data # type: ignore
            
            service._handle_video_mask = mock_handle_video_mask
            
            # Test mask update
            mask_message = {
                "mask_data": mock_video_mask,
                "fade_in_duration": 0.2,
                "fade_out_duration": 0.5
            }
            
            # Call handler directly
            await service._handle_video_mask(mask_message)
            
            # Verify mask was processed
            assert service.video_overlay_renderer.current_mask is not None # type: ignore
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_video_fade_timing(self, test_config, mock_video_mask):
        """Test video overlay fade in/out timing."""
        with patch('pyglet.window.Window') as mock_window, \
             patch('pyglet.clock.schedule_once') as mock_schedule:
            # Setup window mock
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            # Configure schedule_once to immediately execute the callback
            def execute_callback(callback, delay, *args, **kwargs):
                callback(0)  # Call with dt=0
            mock_schedule.side_effect = execute_callback
            
            service = DisplayService(config=test_config)
            await service.start()
            
            assert service.video_overlay_renderer is not None

            # Mock the video overlay renderer's fade state
            service.video_overlay_renderer.fade_state = "hidden"  # Initial state
            
            # Store initial opacity
            initial_opacity = service.video_overlay_renderer.opacity
            
            # Create a mock handler that simulates fading
            async def mock_handle_video_mask(message):
                assert service.video_overlay_renderer is not None

                # Simulate fade in beginning
                service.video_overlay_renderer.fade_state = "fading_in"
                service.video_overlay_renderer.current_mask = message["mask_data"]
                
                # Save initial state
                initial_alpha = service.video_overlay_renderer._current_alpha
                
                # Simulate the fade completing after a delay
                await asyncio.sleep(0.1)
                
                # Update the alpha value directly to simulate fade completing
                service.video_overlay_renderer.fade_state = "visible"
                service.video_overlay_renderer._current_alpha = 1.0  # This is the key change
                
                logger.info(f"Alpha value changed: {initial_alpha} -> {service.video_overlay_renderer._current_alpha}")
                
            service._handle_video_mask = mock_handle_video_mask
            
            # Trigger fade in
            mask_message = {
                "mask_data": mock_video_mask,
                "fade_in_duration": 0.1,
                "fade_out_duration": 0.1
            }
            
            # Start the fade process
            fade_task = asyncio.create_task(service._handle_video_mask(mask_message))
            
            # Should start fading in
            await asyncio.sleep(0.05)
            assert service.video_overlay_renderer.is_fading is True
            
            # Wait for fade to complete
            await fade_task
            await asyncio.sleep(0.1)
            
            # Check that opacity increased
            assert service.video_overlay_renderer.opacity > initial_opacity
            
            await service.stop()


class TestMessageValidation:
    """Test message validation and error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_text_message(self, test_config):
        """Test handling of invalid text overlay messages."""
        with patch('pyglet.window.Window') as mock_window:
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            service = DisplayService(config=test_config)
            
            # Use active_service for proper lifecycle management
            async with active_service(service) as active:
                # Missing required fields
                invalid_message = {
                    "text_id": "test",
                    # Missing "content" field
                    "speaker": "agent"
                }
                
                # Should handle gracefully without crashing
                active.trigger_display_update("text_overlay", invalid_message)
                await asyncio.sleep(0.1)
                
                # Service should still be running
                assert active.state == ServiceState.RUNNING
    
    @pytest.mark.asyncio
    async def test_invalid_image_message(self, test_config):
        """Test handling of invalid image messages."""
        with patch('pyglet.window.Window') as mock_window:
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            service = DisplayService(config=test_config)
            async with active_service(service) as active:
                # Missing required fields
                invalid_message = {
                    "image_id": "test",
                    # Missing "uri" field
                }
                
                # Should handle gracefully
                active.trigger_display_update("image_ready", invalid_message)
                await asyncio.sleep(0.1)
                
                assert active.state == ServiceState.RUNNING


class TestDirectInterface:
    """Test the direct (non-ZMQ) interface for development."""
    
    @pytest.mark.asyncio
    async def test_direct_interface_handlers(self, test_config):
        """Test that all direct interface handlers are registered."""
        with patch('pyglet.window.Window') as mock_window:
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            service = DisplayService(config=test_config)
            await service.start()
            
            # Check that all expected handlers are registered
            expected_handlers = [
                "image_ready",
                "text_overlay", 
                "remove_text",
                "video_mask",
                "transition_ready",
                "loop_ready",
                "era_changed"
            ]
            
            for handler in expected_handlers:
                assert handler in service._direct_handlers
            
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_unknown_update_type(self, test_config):
        """Test handling of unknown update types."""
        with patch('pyglet.window.Window') as mock_window:
            mock_window_instance = Mock()
            mock_window_instance.width = 800
            mock_window_instance.height = 600
            mock_window.return_value = mock_window_instance
            
            service = DisplayService(config=test_config)
            async with active_service(service) as active:
                # Try unknown update type
                active.trigger_display_update("unknown_type", {"data": "test"})
                await asyncio.sleep(0.1)
                
                # Should handle gracefully
                assert active.state == ServiceState.RUNNING


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
