"""
Test module for ZMQ utilities, particularly image transport and message conversion.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from experimance_common.schemas import ImageReady, DisplayMedia, MessageBase, ContentType
from experimance_common.zmq.zmq_utils import (
    image_ready_to_display_media,
    create_display_media_message,
    choose_image_transport_mode,
    is_local_address
)
from experimance_common.constants import IMAGE_TRANSPORT_MODES


class TestMessageConversion:
    """Test message type conversion utilities."""
    
    def test_message_base_to_message_type_success(self):
        """Test successful conversion from dict to ImageReady."""
        # Create test data
        image_ready_data = {
            "type": "ImageReady",
            "request_id": "test-123",
            "uri": "file:///tmp/test.png"
        }
        
        # Test conversion
        result = MessageBase.to_message_type(image_ready_data, ImageReady)
        
        assert result is not None
        assert isinstance(result, ImageReady)
        assert result.request_id == "test-123"
        assert result.uri == "file:///tmp/test.png"
    
    def test_message_base_to_message_type_invalid_data(self):
        """Test conversion with invalid data returns None."""
        invalid_data = {
            "type": "ImageReady",
            # Missing required fields
        }
        
        result = MessageBase.to_message_type(invalid_data, ImageReady)
        assert result is None
    
    def test_message_base_to_message_type_already_correct_type(self):
        """Test conversion when object is already the correct type."""
        image_ready = ImageReady(
            request_id="test-123",
            uri="file:///tmp/test.png"
        )
        
        result = MessageBase.to_message_type(image_ready, ImageReady)
        assert result is image_ready  # Should return the same object


class TestImageTransport:
    """Test image transport mode selection."""
    
    def test_is_local_address_localhost(self):
        """Test detection of localhost addresses."""
        assert is_local_address("tcp://localhost:5555")
        assert is_local_address("tcp://127.0.0.1:5555")
    
    def test_is_local_address_remote(self):
        """Test detection of remote addresses."""
        assert not is_local_address("tcp://192.168.1.100:5555")
        assert not is_local_address("tcp://example.com:5555")
    
    def test_choose_image_transport_mode_force(self):
        """Test forced transport mode selection."""
        result = choose_image_transport_mode(force_mode="base64")
        assert result == IMAGE_TRANSPORT_MODES["BASE64"]
    
    def test_choose_image_transport_mode_file_uri(self):
        """Test file URI mode selection."""
        result = choose_image_transport_mode(
            transport_mode=IMAGE_TRANSPORT_MODES["FILE_URI"]
        )
        assert result == IMAGE_TRANSPORT_MODES["FILE_URI"]
    
    @patch('experimance_common.zmq.zmq_utils.get_file_size')
    def test_choose_image_transport_mode_auto_large_file(self, mock_file_size):
        """Test auto mode with large file prefers file URI for local."""
        mock_file_size.return_value = 2 * 1024 * 1024  # 2MB
        
        result = choose_image_transport_mode(
            file_path="/tmp/large_image.png",
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["AUTO"]
        )
        assert result == IMAGE_TRANSPORT_MODES["FILE_URI"]
    
    @patch('experimance_common.zmq.zmq_utils.get_file_size')  
    def test_choose_image_transport_mode_auto_small_file(self, mock_file_size):
        """Test auto mode with small file prefers base64."""
        mock_file_size.return_value = 100 * 1024  # 100KB
        
        result = choose_image_transport_mode(
            file_path="/tmp/small_image.png",
            target_address="tcp://localhost:5555", 
            transport_mode=IMAGE_TRANSPORT_MODES["AUTO"]
        )
        assert result == IMAGE_TRANSPORT_MODES["BASE64"]


class TestDisplayMediaCreation:
    """Test DisplayMedia message creation."""
    
    def test_image_ready_to_display_media_basic(self):
        """Test basic conversion from ImageReady to DisplayMedia."""
        image_ready = ImageReady(
            request_id="test-123",
            uri="file:///tmp/test.png"
        )
        
        result = image_ready_to_display_media(image_ready)
        
        assert result["type"] == "DisplayMedia"
        assert result["content_type"] == "image"
        assert result["uri"] == "file:///tmp/test.png"
        assert result["request_id"] == "test-123"
    
    def test_image_ready_to_display_media_with_temp_file(self):
        """Test conversion preserves temp file info."""
        # Create ImageReady with temp file info (as dict to simulate ZMQ message)
        image_ready_data = {
            "type": "ImageReady",
            "request_id": "test-123", 
            "uri": "file:///tmp/test.png",
            "_temp_file": "/tmp/experimance_img_123.png"
        }
        
        image_ready = ImageReady(**image_ready_data)
        # Add temp file info manually (since it's not part of schema)
        image_ready._temp_file = image_ready_data["_temp_file"]
        
        result = image_ready_to_display_media(image_ready)
        
        assert result.get("_temp_file") == "/tmp/experimance_img_123.png"
    
    def test_create_display_media_message_image(self):
        """Test creating DisplayMedia message for image content."""
        result = create_display_media_message(
            content_type="image",
            uri="file:///tmp/test.png",
            era="current",
            biome="temperate_forest"
        )
        
        assert result["type"] == "DisplayMedia"
        assert result["content_type"] == "image"
        assert result["uri"] == "file:///tmp/test.png"
        assert result["era"] == "current"
        assert result["biome"] == "temperate_forest"
    
    def test_create_display_media_message_video(self):
        """Test creating DisplayMedia message for video content."""
        result = create_display_media_message(
            content_type="video",
            uri="file:///tmp/test.mp4",
            duration=10.0,
            loop=True
        )
        
        assert result["type"] == "DisplayMedia"
        assert result["content_type"] == "video"
        assert result["uri"] == "file:///tmp/test.mp4"
        assert result["duration"] == 10.0
        assert result["loop"] is True
    
    def test_image_ready_to_display_media_invalid_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="Expected message to be an instance of ImageReady"):
            image_ready_to_display_media({"type": "ImageReady"})


class TestImageTransportIntegration:
    """Integration tests for image transport utilities."""
    
    @patch('experimance_common.zmq.zmq_utils.prepare_image_message')
    def test_create_display_media_with_image_data(self, mock_prepare):
        """Test DisplayMedia creation with actual image data."""
        mock_prepare.return_value = {
            "type": "DisplayMedia",
            "content_type": "image",
            "uri": "file:///tmp/test.png",
            "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        }
        
        result = create_display_media_message(
            content_type="image",
            image_data="/tmp/test.png",
            target_address="tcp://localhost:5555"
        )
        
        mock_prepare.assert_called_once()
        assert result["type"] == "DisplayMedia"
        assert result["content_type"] == "image"
