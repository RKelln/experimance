"""
Integration tests for image message handling.

Tests the integration between prepare_image_message and load_image_from_message
to ensure round-trip functionality works correctly across all transport modes
and formats.
"""

import tempfile
import pytest
from pathlib import Path
from PIL import Image

from experimance_common.zmq.zmq_utils import prepare_image_message
from experimance_common.image_utils import load_image_from_message, ImageLoadFormat, cleanup_temp_file


class TestImageMessageIntegration:
    """Integration tests for image message round-trip functionality."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Create a test image
        self.test_image = Image.new('RGB', (200, 150), (255, 128, 0))
    
    def test_pil_image_round_trip(self):
        """Test PIL Image -> message -> PIL Image round trip."""
        # Send PIL image
        message = prepare_image_message(
            image_data=self.test_image,
            target_address="tcp://localhost:5555",
            mask_id="test_mask_pil"
        )
        
        assert "mask_id" in message
        assert "image_data" in message or "uri" in message
        
        # Receive back as PIL Image
        received_image = load_image_from_message(message, ImageLoadFormat.PIL)
        assert received_image is not None
        assert isinstance(received_image, Image.Image), f"Expected PIL Image, got {type(received_image)}"
        assert received_image.size == (200, 150)
    
    def test_filepath_requirement_with_temp_file_creation(self):
        """Test that systems requiring file paths get them (with temp file creation)."""
        message = prepare_image_message(
            image_data=self.test_image,
            target_address="tcp://localhost:5555",
            mask_id="test_mask_filepath"
        )
        
        # For systems that need file paths (like pyglet)
        result = load_image_from_message(message, ImageLoadFormat.FILEPATH)
        assert result is not None
        assert isinstance(result, tuple)
        
        file_path, is_temp = result
        assert Path(file_path).exists()
        
        # Clean up temp file if needed
        if is_temp:
            cleanup_temp_file(file_path)
            assert not Path(file_path).exists(), "Temp file should be cleaned up"
    
    def test_file_path_round_trip_with_uri_mode(self):
        """Test file path -> message -> file path with URI transport mode."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            self.test_image.save(tmp.name)
            temp_file_path = tmp.name
        
        try:
            message = prepare_image_message(
                image_data=temp_file_path,
                target_address="tcp://localhost:5555",
                transport_mode="file_uri",  # Force file URI transport mode
                mask_id="test_mask_file"
            )
            
            # Should prefer URI for existing files
            assert "uri" in message, f"Expected 'uri' key in message, got keys: {list(message.keys())}"
            
            # Receive for systems that need file paths should return same file path
            result = load_image_from_message(message, ImageLoadFormat.FILEPATH)
            assert result is not None
            assert isinstance(result, tuple)
            
            received_path, is_temp = result
            assert received_path == temp_file_path
            assert not is_temp  # Should not be temp since original file exists
            
        finally:
            Path(temp_file_path).unlink()
    
    def test_numpy_format_conversion(self):
        """Test PIL Image -> message -> numpy array conversion."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("NumPy not available")
        
        # Create a fresh message with base64 data for numpy testing
        numpy_test_message = prepare_image_message(
            image_data=self.test_image,
            target_address="tcp://localhost:5555",
            mask_id="test_mask_numpy"
        )
        
        numpy_array = load_image_from_message(numpy_test_message, ImageLoadFormat.NUMPY)
        assert numpy_array is not None, "NumPy conversion should not fail"
        assert isinstance(numpy_array, np.ndarray), f"Expected numpy array, got {type(numpy_array)}"
        assert numpy_array.shape == (150, 200, 3), f"Expected shape (150, 200, 3), got {numpy_array.shape}"
    
    def test_all_formats_with_same_message(self):
        """Test that all formats work with the same message."""
        message = prepare_image_message(
            image_data=self.test_image,
            target_address="tcp://localhost:5555",
            mask_id="test_all_formats"
        )
        
        # PIL format
        pil_result = load_image_from_message(message, ImageLoadFormat.PIL)
        assert pil_result is not None
        assert isinstance(pil_result, Image.Image)
        assert pil_result.size == (200, 150)
        
        # NumPy format (if available)
        try:
            import numpy as np
            numpy_result = load_image_from_message(message, ImageLoadFormat.NUMPY)
            assert numpy_result is not None
            assert isinstance(numpy_result, np.ndarray)
        except ImportError:
            pass  # Skip if NumPy not available
        
        # File path format
        filepath_result = load_image_from_message(message, ImageLoadFormat.FILEPATH)
        assert filepath_result is not None
        assert isinstance(filepath_result, tuple)
        file_path, is_temp = filepath_result
        assert Path(file_path).exists()
        
        # Clean up if temp
        if is_temp:
            cleanup_temp_file(file_path)
    
    def test_invalid_message_handling(self):
        """Test that invalid messages are handled gracefully."""
        invalid_message = {"invalid": "data"}
        
        # All formats should return None for invalid messages
        assert load_image_from_message(invalid_message, ImageLoadFormat.PIL) is None
        assert load_image_from_message(invalid_message, ImageLoadFormat.NUMPY) is None
        assert load_image_from_message(invalid_message, ImageLoadFormat.FILEPATH) is None
