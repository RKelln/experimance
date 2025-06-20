#!/usr/bin/env python3
"""
Test script for the image generation to display media pipeline.

This script tests the core functionality of:
1. ImageReady message conversion to DisplayMedia
2. MessageBase type conversion utilities
3. Image transport helpers in ZMQ utils
"""

import sys
import os
import uuid
from pathlib import Path

# Add the source directories to Python path
libs_common_src = Path(__file__).parent / "libs" / "common" / "src"
sys.path.insert(0, str(libs_common_src))

# Import modules directly without package init
sys.path.insert(0, str(libs_common_src / "experimance_common"))
import schemas
import constants
sys.path.insert(0, str(libs_common_src / "experimance_common" / "zmq"))
import zmq_utils

def test_message_conversion():
    """Test MessageBase type conversion utility."""
    print("Testing MessageBase.to_message_type()...")
    
    # Create test ImageReady message as dict
    image_ready_dict = {
        "type": "ImageReady",
        "request_id": str(uuid.uuid4()),
        "image_id": "test_image_001",
        "uri": "file:///tmp/test_image.png"
    }
    
    # Convert to ImageReady object
    image_ready = schemas.MessageBase.to_message_type(image_ready_dict, schemas.ImageReady)
    
    assert image_ready is not None, "Failed to convert dict to ImageReady"
    assert isinstance(image_ready, schemas.ImageReady), f"Expected ImageReady, got {type(image_ready)}"
    assert image_ready.request_id == image_ready_dict["request_id"], "request_id mismatch"
    assert image_ready.image_id == image_ready_dict["image_id"], "image_id mismatch"
    assert image_ready.uri == image_ready_dict["uri"], "uri mismatch"
    
    print("✓ MessageBase.to_message_type() works correctly")
    return image_ready

def test_image_ready_to_display_media(image_ready):
    """Test conversion from ImageReady to DisplayMedia."""
    print("Testing image_ready_to_display_media()...")
    
    # Convert ImageReady to DisplayMedia format
    display_media_dict = zmq_utils.image_ready_to_display_media(
        message=image_ready,
        target_address="tcp://localhost:5555",
        transport_mode=constants.DEFAULT_IMAGE_TRANSPORT_MODE
    )
    
    assert display_media_dict is not None, "Failed to convert ImageReady to DisplayMedia"
    assert display_media_dict["type"] == "DisplayMedia", f"Expected type 'DisplayMedia', got {display_media_dict.get('type')}"
    assert display_media_dict["content_type"] == "image", f"Expected content_type 'image', got {display_media_dict.get('content_type')}"
    assert display_media_dict["uri"] == image_ready.uri, "URI mismatch"
    assert display_media_dict["source_request_id"] == image_ready.request_id, "source_request_id mismatch"
    
    print("✓ image_ready_to_display_media() works correctly")
    return display_media_dict

def test_create_display_media_message():
    """Test the create_display_media_message helper."""
    print("Testing create_display_media_message()...")
    
    # Create DisplayMedia message
    message = zmq_utils.create_display_media_message(
        content_type="image",
        image_id="test_image_002",
        uri="file:///tmp/another_test.png",
        era=schemas.Era.CURRENT,
        biome=schemas.Biome.TEMPERATE_FOREST,
        source_request_id=str(uuid.uuid4()),
        target_address="tcp://localhost:5556",
        transport_mode=constants.DEFAULT_IMAGE_TRANSPORT_MODE
    )
    
    assert message is not None, "Failed to create DisplayMedia message"
    assert message["type"] == "DisplayMedia", f"Expected type 'DisplayMedia', got {message.get('type')}"
    assert message["content_type"] == "image", f"Expected content_type 'image', got {message.get('content_type')}"
    assert message["image_id"] == "test_image_002", "image_id mismatch"
    assert message["era"] == schemas.Era.CURRENT, "era mismatch"
    assert message["biome"] == schemas.Biome.TEMPERATE_FOREST, "biome mismatch"
    
    print("✓ create_display_media_message() works correctly")
    return message

def test_display_media_schema_validation():
    """Test DisplayMedia schema validation."""
    print("Testing DisplayMedia schema validation...")
    
    # Create DisplayMedia object
    display_media = schemas.DisplayMedia(
        content_type=schemas.ContentType.IMAGE,
        image_id="test_schema",
        uri="file:///tmp/schema_test.png",
        era=schemas.Era.FUTURE,
        biome=schemas.Biome.ARCTIC,
        source_request_id=str(uuid.uuid4())
    )
    
    assert display_media.type == "DisplayMedia", "type field incorrect"
    assert display_media.content_type == "image", "content_type incorrect"
    assert display_media.image_id == "test_schema", "image_id incorrect"
    assert display_media.era == schemas.Era.FUTURE, "era incorrect"
    assert display_media.biome == schemas.Biome.ARCTIC, "biome incorrect"
    
    # Test conversion to dict
    display_dict = display_media.model_dump() if hasattr(display_media, 'model_dump') else display_media.dict()
    assert isinstance(display_dict, dict), "Failed to convert to dict"
    assert display_dict["type"] == "DisplayMedia", "type lost in conversion"
    
    print("✓ DisplayMedia schema validation works correctly")
    return display_media

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Image Generation to Display Media Pipeline")
    print("=" * 60)
    
    try:
        # Test 1: Message conversion
        image_ready = test_message_conversion()
        
        # Test 2: ImageReady to DisplayMedia conversion  
        display_media_dict = test_image_ready_to_display_media(image_ready)
        
        # Test 3: Helper function
        helper_message = test_create_display_media_message()
        
        # Test 4: Schema validation
        display_media_obj = test_display_media_schema_validation()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! The image pipeline is working correctly.")
        print("=" * 60)
        
        # Show example usage
        print("\nExample DisplayMedia message:")
        print("-" * 30)
        import json
        print(json.dumps(display_media_dict, indent=2, default=str))
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
