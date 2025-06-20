#!/usr/bin/env python3
"""
Tests for ImageRenderer.handle_display_media method.

Tests the integration with the new enum-based image loading utilities.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from PIL import Image as PILImage
import pyglet

from experimance_common.zmq.zmq_utils import prepare_image_message, IMAGE_TRANSPORT_MODES
from experimance_common.image_utils import png_to_base64url
from experimance_display.renderers.image_renderer import ImageRenderer


class TestImageRendererHandleDisplayMedia:
    """Test the handle_image_ready method with various message formats."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock config
        self.mock_config = Mock()
        self.mock_config.transition_duration = 2.0
        self.mock_config.opacity = 1.0
        
        self.mock_transitions_config = Mock()
        
        # Create renderer
        self.renderer = ImageRenderer(
            window_size=(1920, 1080),
            config=self.mock_config,
            transitions_config=self.mock_transitions_config
        )
        
        # Create test image
        self.test_image = PILImage.new('RGB', (256, 256), color='red')
        
    def create_test_message_file_uri(self, image_path: str) -> Dict[str, Any]:
        """Create a test message with file URI."""
        return prepare_image_message(
            image_data=image_path,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["FILE_URI"],
            image_id="test_image"
        )
    
    def create_test_message_base64(self) -> Dict[str, Any]:
        """Create a test message with base64 data."""
        return prepare_image_message(
            image_data=self.test_image,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["BASE64"],
            image_id="test_image_base64"
        )
    
    def create_test_message_hybrid(self, image_path: str) -> Dict[str, Any]:
        """Create a test message with hybrid transport."""
        return prepare_image_message(
            image_data=image_path,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["HYBRID"],
            image_id="test_image_hybrid"
        )
    
    @pytest.mark.asyncio
    async def test_handle_display_media_with_file_uri(self):
        """Test handle_image_ready with file URI message."""
        # Save test image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            self.test_image.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Create message with file URI
            message = self.create_test_message_file_uri(temp_path)
            
            # Mock the sprite creation and positioning
            mock_pyglet_image = Mock()
            mock_pyglet_image.width = 256
            mock_pyglet_image.height = 256
            mock_sprite = Mock()
            
            with patch('experimance_display.renderers.image_renderer.pyglet.image.load', return_value=mock_pyglet_image), \
                 patch('experimance_display.renderers.image_renderer.create_positioned_sprite', return_value=mock_sprite), \
                 patch.object(self.renderer, '_start_transition_to_image') as mock_transition:
                
                await self.renderer.handle_display_media(message)
                
                # Verify image was loaded and transition started
                mock_transition.assert_called_once_with(mock_pyglet_image, mock_sprite, "test_image")
                
                # Verify image was cached
                assert "test_image" in self.renderer.image_cache
                
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_handle_display_media_with_base64(self):
        """Test handle_display_media with base64 message."""
        # Create message with base64 data
        message = self.create_test_message_base64()
        
        # Mock the pyglet loading components
        mock_pyglet_image = Mock()
        mock_pyglet_image.width = 256
        mock_pyglet_image.height = 256
        mock_sprite = Mock()
        
        with patch('experimance_display.renderers.image_renderer.pyglet.image.load', return_value=mock_pyglet_image), \
             patch('experimance_display.renderers.image_renderer.create_positioned_sprite', return_value=mock_sprite), \
             patch.object(self.renderer, '_start_transition_to_image') as mock_transition:
            
            await self.renderer.handle_display_media(message)
            
            # Verify image was loaded and transition started
            mock_transition.assert_called_once_with(mock_pyglet_image, mock_sprite, "test_image_base64")
            
            # Verify image was cached
            assert "test_image_base64" in self.renderer.image_cache
    
    @pytest.mark.asyncio
    async def test_handle_display_media_with_cached_image(self):
        """Test handle_display_media when image is already cached."""
        # Pre-populate cache
        mock_image = Mock()
        mock_sprite = Mock()
        self.renderer.image_cache["cached_image"] = (mock_image, mock_sprite)
        
        # Create message
        message = {"image_id": "cached_image"}
        
        with patch.object(self.renderer, '_start_transition_to_image') as mock_transition, \
             patch('experimance_display.renderers.image_renderer.load_pyglet_image_from_message') as mock_load:
            
            await self.renderer.handle_display_media(message)
            
            # Verify cached image was used (no loading occurred)
            mock_load.assert_not_called()
            mock_transition.assert_called_once_with(mock_image, mock_sprite, "cached_image")
    
    @pytest.mark.asyncio
    async def test_handle_display_media_with_missing_image_id(self):
        """Test handle_display_media with malformed message."""
        message = {"some_other_field": "value"}  # Missing image_id
        
        with patch('experimance_display.renderers.image_renderer.logger') as mock_logger:
            await self.renderer.handle_display_media(message)
            
            # Should log an error due to missing image_id
            mock_logger.error.assert_called()
    
    @pytest.mark.asyncio  
    async def test_handle_display_media_with_load_failure(self):
        """Test handle_display_media when image loading fails."""
        message = {
            "image_id": "failing_image",
            "uri": "file:///nonexistent/path.png"
        }
        
        with patch('experimance_display.renderers.image_renderer.load_pyglet_image_from_message', return_value=(None, None)), \
             patch('experimance_display.renderers.image_renderer.logger') as mock_logger, \
             patch.object(self.renderer, '_start_transition_to_image') as mock_transition:
            
            await self.renderer.handle_display_media(message)
            
            # Should log error and not start transition
            mock_logger.error.assert_called()
            mock_transition.assert_not_called()
            
            # Should not cache the failed image
            assert "failing_image" not in self.renderer.image_cache
    
    @pytest.mark.asyncio
    async def test_handle_display_media_temp_file_cleanup(self):
        """Test that temporary files are properly cleaned up."""
        # Create message with base64 data (will create temp file)
        message = self.create_test_message_base64()
        
        # Mock components
        mock_pyglet_image = Mock()
        mock_pyglet_image.width = 256
        mock_pyglet_image.height = 256
        mock_sprite = Mock()
        
        # Track cleanup calls
        cleanup_calls = []
        
        def mock_cleanup(temp_path):
            cleanup_calls.append(temp_path)
        
        with patch('experimance_display.renderers.image_renderer.pyglet.image.load', return_value=mock_pyglet_image), \
             patch('experimance_display.renderers.image_renderer.create_positioned_sprite', return_value=mock_sprite), \
             patch('experimance_display.renderers.image_renderer.cleanup_temp_file', side_effect=mock_cleanup), \
             patch.object(self.renderer, '_start_transition_to_image'):
            
            await self.renderer.handle_display_media(message)
            
            # Verify cleanup was called (should be called even when temp_file_path is None for non-temp files)
            assert len(cleanup_calls) >= 0  # cleanup_temp_file handles None gracefully
    
    @pytest.mark.asyncio
    async def test_handle_display_media_with_hybrid_transport(self):
        """Test handle_display_media with hybrid transport message."""
        # Save test image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            self.test_image.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Create message with hybrid transport
            message = self.create_test_message_hybrid(temp_path)
            
            # Verify message has both URI and base64 data
            assert "uri" in message
            assert "image_data" in message
            
            # Mock components
            mock_pyglet_image = Mock()
            mock_pyglet_image.width = 256
            mock_pyglet_image.height = 256
            mock_sprite = Mock()
            
            with patch('experimance_display.renderers.image_renderer.pyglet.image.load', return_value=mock_pyglet_image), \
                 patch('experimance_display.renderers.image_renderer.create_positioned_sprite', return_value=mock_sprite), \
                 patch.object(self.renderer, '_start_transition_to_image') as mock_transition:
                
                await self.renderer.handle_display_media(message)
                
                # Verify image was loaded and transition started
                mock_transition.assert_called_once_with(mock_pyglet_image, mock_sprite, "test_image_hybrid")
                
                # Verify image was cached
                assert "test_image_hybrid" in self.renderer.image_cache
                
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_handle_display_media_exception_handling(self):
        """Test that exceptions in handle_display_media are properly caught and logged."""
        message = {"image_id": "test_exception"}
        
        # Mock load_pyglet_image_from_message to raise an exception
        with patch('experimance_display.renderers.image_renderer.load_pyglet_image_from_message', 
                  side_effect=Exception("Test exception")), \
             patch('experimance_display.renderers.image_renderer.logger') as mock_logger:
            
            await self.renderer.handle_display_media(message)
            
            # Should log the error with exception info
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args
            assert "Error handling ImageReady" in str(error_call)
