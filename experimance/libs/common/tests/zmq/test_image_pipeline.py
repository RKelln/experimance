"""
Tests for image generation to display media pipeline.

These tests validate the message conversion utilities used to transport
images from the image server through the core service to the display service.
"""

import pytest
from experimance_common.schemas import ImageReady
from experimance_common.zmq.zmq_utils import (
    image_ready_to_display_media,
    create_display_media_message
)


def test_image_ready_to_display_media():
    """Test conversion of ImageReady to DisplayMedia format."""
    # Create test ImageReady message
    image_ready = ImageReady(
        request_id="test-request-123",
        uri="file:///tmp/test_image.png"
    )
    
    # Convert to DisplayMedia
    display_media = image_ready_to_display_media(image_ready)
    
    # Verify structure
    assert display_media["type"] == "DisplayMedia"
    assert display_media["content_type"] == "image"
    assert display_media["request_id"] == "test-request-123"  # Should use request_id
    assert display_media["uri"] == "file:///tmp/test_image.png"


def test_create_display_media_message():
    """Test creating DisplayMedia messages with various parameters."""
    # Test with URI
    message = create_display_media_message(
        content_type="image",
        uri="file:///tmp/test.png",
        request_id="test-123",
        era="wilderness",
        biome="forest"
    )
    
    assert message["type"] == "DisplayMedia"
    assert message["content_type"] == "image"
    assert message["request_id"] == "test-123"
    assert message["uri"] == "file:///tmp/test.png"
    assert message["era"] == "wilderness"
    assert message["biome"] == "forest"


def test_message_base_conversion():
    """Test converting between message types."""
    from experimance_common.schemas import MessageBase, DisplayMedia
    
    # Create a raw dict message
    raw_message = {
        "type": "DisplayMedia",
        "content_type": "image",
        "request_id": "test-456",
        "uri": "file:///tmp/test.png"
    }
    
    # Convert to DisplayMedia object
    display_media = MessageBase.to_message_type(raw_message, DisplayMedia)
    assert isinstance(display_media, DisplayMedia)
    assert display_media is not None
    assert display_media.type == "DisplayMedia"
    assert display_media.content_type == "image"
    assert display_media.request_id == "test-456"
    assert display_media.uri == "file:///tmp/test.png"


def test_image_transport_mode_selection():
    """Test that image transport mode selection works correctly."""
    from experimance_common.zmq.zmq_utils import choose_image_transport_mode
    from experimance_common.constants import IMAGE_TRANSPORT_MODES
    
    # Test local vs remote address detection
    local_mode = choose_image_transport_mode(
        target_address="tcp://localhost:5555",
        transport_mode=IMAGE_TRANSPORT_MODES["AUTO"]
    )
    
    remote_mode = choose_image_transport_mode(
        target_address="tcp://192.168.1.100:5555", 
        transport_mode=IMAGE_TRANSPORT_MODES["AUTO"]
    )
    
    # Both should return valid transport modes
    assert local_mode in IMAGE_TRANSPORT_MODES.values()
    assert remote_mode in IMAGE_TRANSPORT_MODES.values()


def test_request_id_in_temp_files():
    """Test that request_id is incorporated into temporary file names."""
    import tempfile
    import os
    from experimance_common.image_utils import save_pil_image_as_tempfile
    from PIL import Image
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='red')
    request_id = "test-request-12345"
    
    # Save with request_id
    temp_path = save_pil_image_as_tempfile(test_image, request_id=request_id)
    
    try:
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Verify request_id is in the filename
        filename = os.path.basename(temp_path)
        assert request_id in filename
        # New pattern: experimance_img_YYYYMMDD_HHMMSS_request_id_
        assert filename.startswith("experimance_img_")
        assert f"_{request_id}_" in filename
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_request_id_in_zmq_message():
    """Test that request_id is properly extracted and used in prepare_image_message."""
    from experimance_common.zmq.zmq_utils import prepare_image_message
    from experimance_common.constants import IMAGE_TRANSPORT_MODES
    from PIL import Image
    import os
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='blue')
    request_id = "test-request-67890"
    
    # Prepare message with request_id
    message = prepare_image_message(
        image_data=test_image,
        transport_mode=IMAGE_TRANSPORT_MODES["FILE_URI"],
        request_id=request_id,
        type="DisplayMedia"
    )
    
    try:
        # Verify message structure
        assert message["type"] == "DisplayMedia"
        assert message["request_id"] == request_id
        assert "uri" in message
        assert "_temp_file" in message
        
        # Verify temp file contains request_id
        temp_file = message["_temp_file"]
        filename = os.path.basename(temp_file)
        assert request_id in filename
        
    finally:
        # Clean up temp file
        temp_file = message.get("_temp_file")
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
