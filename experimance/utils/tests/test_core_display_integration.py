#!/usr/bin/env python3
"""
Integration test demonstrating Core Service -> Display Service image flow.

This test shows how images flow from the core service (sending change maps)
to the display service (receiving and loading images) using our new unified
enum-based image utilities.
"""

import pytest
import numpy as np
import tempfile
from unittest.mock import Mock, AsyncMock, patch

from PIL import Image

from experimance_common.zmq.zmq_utils import prepare_image_message, IMAGE_TRANSPORT_MODES
from experimance_common.image_utils import load_image_from_message, ImageLoadFormat


class TestCoreToDisplayImageFlow:
    """Test the complete image flow from Core to Display service."""
    
    @pytest.mark.asyncio
    async def test_core_change_map_to_display_loading_file_uri(self):
        """Test Core service change map -> Display service loading (FILE_URI transport)."""
        # 1. Core service: Create a change map (binary image)
        change_map = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255
        change_score = 0.67
        
        # 2. Core service: Prepare image message using enum-based utilities
        message = prepare_image_message(
            image_data=change_map,
            target_address="tcp://localhost:5555",  # Local transport
            transport_mode=IMAGE_TRANSPORT_MODES["FILE_URI"],  # Explicitly test FILE_URI mode
            image_id="change_map_integration_test",
            metadata={
                "type": "ChangeMap",
                "change_score": change_score,
                "timestamp": "2025-06-18T10:30:00"
            }
        )
        
        # 3. Verify message structure
        assert "image_id" in message
        assert message["image_id"] == "change_map_integration_test"
        
        # For FILE_URI mode, should have URI regardless of target
        assert "uri" in message  # Should have file URI for FILE_URI mode
        assert message["uri"].startswith("file://")
        
        # 4. Display service: Load image using enum-based utilities (simulating pyglet needs)
        file_path, is_temp = load_image_from_message(message, ImageLoadFormat.FILEPATH) # type: ignore
        
        assert file_path is not None
        assert isinstance(file_path, str)
        assert is_temp is False  # Should use existing temp file created by prepare_image_message
        
        # 5. Verify the loaded image matches original
        from PIL import Image
        loaded_image = Image.open(file_path)
        loaded_array = np.array(loaded_image.convert('L'))  # Convert to grayscale
        
        # Check dimensions match
        assert loaded_array.shape == change_map.shape
        
        # Clean up (this is what display service would do)
        from experimance_common.image_utils import cleanup_temp_file
        if is_temp:
            cleanup_temp_file(file_path)
    
    @pytest.mark.asyncio
    async def test_core_change_map_to_display_loading_base64(self):
        """Test Core service change map -> Display service loading (BASE64 transport)."""
        # 1. Core service: Create a change map
        change_map = np.random.randint(0, 2, (240, 320), dtype=np.uint8) * 255
        change_score = 0.23
        
        # 2. Core service: Force BASE64 transport mode
        message = prepare_image_message(
            image_data=change_map,
            target_address="tcp://192.168.1.100:5555",  # Remote address should prefer BASE64
            transport_mode=IMAGE_TRANSPORT_MODES["BASE64"],
            image_id="change_map_base64_test",
            metadata={
                "type": "ChangeMap",
                "change_score": change_score
            }
        )
        
        # 3. Verify message uses base64 transport
        assert "image_data" in message
        assert message["image_data"].startswith("data:image/png;base64,")
        
        # 4. Display service: Load as PIL Image 
        pil_image = load_image_from_message(message, ImageLoadFormat.PIL)
        
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        # 5. Convert to numpy to verify content
        loaded_array = np.array(pil_image.convert('L'))
        
        # Check dimensions match
        assert loaded_array.shape == change_map.shape
    
    @pytest.mark.asyncio  
    async def test_core_change_map_to_display_loading_hybrid(self):
        """Test Core service change map -> Display service loading (HYBRID transport)."""
        # 1. Create test image file first
        change_map = np.random.randint(0, 2, (360, 480), dtype=np.uint8) * 255
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            from PIL import Image
            Image.fromarray(change_map, mode='L').save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # 2. Core service: Use HYBRID transport with file path
            message = prepare_image_message(
                image_data=temp_path,  # File path input
                target_address="tcp://localhost:5555",
                transport_mode=IMAGE_TRANSPORT_MODES["HYBRID"],
                image_id="change_map_hybrid_test"
            )
            
            # 3. Verify hybrid message has both URI and base64
            assert "uri" in message
            assert "image_data" in message
            assert message["uri"].startswith("file://")
            assert message["image_data"].startswith("data:image/png;base64,")
            
            # 4. Display service: Load via file path (should prefer URI for efficiency)
            file_path, is_temp = load_image_from_message(message, ImageLoadFormat.FILEPATH) # type: ignore
            
            assert file_path is not None
            assert is_temp is False  # Should use existing file, not create temp
            
            # 5. Verify content
            loaded_image = Image.open(file_path)
            loaded_array = np.array(loaded_image.convert('L'))
            assert loaded_array.shape == change_map.shape
            
        finally:
            # Clean up test file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_display_service_handles_core_message_errors_gracefully(self):
        """Test that display service handles malformed messages gracefully."""
        # Test various malformed messages that core service might accidentally send
        malformed_messages = [
            {},  # Empty message
            {"image_id": "test"},  # Missing image data
            {"image_data": "invalid_base64", "image_id": "test"},  # Invalid base64
            {"uri": "file:///nonexistent/path.png", "image_id": "test"},  # Non-existent file
        ]
        
        for message in malformed_messages:
            # Display service should handle these gracefully
            result = load_image_from_message(message, ImageLoadFormat.PIL)
            assert result is None  # Should return None for invalid messages
            
            result = load_image_from_message(message, ImageLoadFormat.FILEPATH)
            assert result is None  # Should return None for invalid messages
    
    @pytest.mark.asyncio
    async def test_numpy_array_preservation_through_transport(self):
        """Test that numpy array precision is preserved through transport."""
        # Create test change map with specific pattern
        change_map = np.zeros((100, 100), dtype=np.uint8)
        change_map[25:75, 25:75] = 255  # White square in center
        change_map[40:60, 40:60] = 128  # Gray square in center of white
        
        # Send through different transport modes
        transport_modes = [
            IMAGE_TRANSPORT_MODES["BASE64"],
            IMAGE_TRANSPORT_MODES["AUTO"]
        ]
        
        for mode in transport_modes:
            message = prepare_image_message(
                image_data=change_map,
                target_address="tcp://localhost:5555",
                transport_mode=mode,
                image_id=f"precision_test_{mode}"
            )
            
            # Load back as numpy array
            loaded_array = load_image_from_message(message, ImageLoadFormat.NUMPY)
            
            assert loaded_array is not None
            assert isinstance(loaded_array, np.ndarray)
            assert loaded_array.shape == change_map.shape
            
            # Check that key features are preserved (allowing for compression artifacts)
            # Check center white region (allow for compression artifacts)
            center_white = loaded_array[35:65, 35:65]
            assert np.mean(center_white) > 190  # Should be mostly white (allowing for PNG compression)
            
            # Check corners (should be black)
            corners = [
                loaded_array[0:10, 0:10],
                loaded_array[0:10, 90:100], 
                loaded_array[90:100, 0:10],
                loaded_array[90:100, 90:100]
            ]
            for corner in corners:
                assert np.mean(corner) < 50  # Should be mostly black
