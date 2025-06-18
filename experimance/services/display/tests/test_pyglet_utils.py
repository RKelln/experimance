#!/usr/bin/env python3
"""
Test script for pyglet utilities.

Tests the pyglet utilities for loading images from ZMQ messages.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from PIL import Image as PILImage

from experimance_common.zmq.zmq_utils import prepare_image_message, IMAGE_TRANSPORT_MODES

# Import the utilities we want to test
import sys
sys.path.append('/home/ryankelln/Documents/Projects/Art/experimance/installation/software/experimance/services/display/src')

from experimance_display.utils.pyglet_utils import (
    load_pyglet_image_from_message,
    cleanup_temp_file,
    create_positioned_sprite
)

logger = logging.getLogger(__name__)


def create_test_image() -> PILImage.Image:
    """Create a simple test image."""
    img = PILImage.new('RGBA', (256, 256), (255, 0, 0, 255))  # Red square
    return img


async def test_pyglet_utilities():
    """Test the pyglet utilities with ZMQ messages."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing pyglet utilities with ZMQ messages...")
    
    # Create test image
    test_image = create_test_image()
    
    # Test 1: Load from ZMQ message with base64 transport
    logger.info("Test 1: Loading from ZMQ message (base64 transport)")
    message_base64 = prepare_image_message(
        image_data=test_image,
        target_address="tcp://localhost:5555",
        transport_mode=IMAGE_TRANSPORT_MODES["BASE64"],
        image_id="test_base64"
    )
    
    pyglet_image, temp_file = load_pyglet_image_from_message(
        message=message_base64,
        image_id="test_base64",
        set_center_anchor=True
    )
    
    if pyglet_image:
        logger.info(f"✓ Base64 message loaded: {pyglet_image.width}x{pyglet_image.height}")
        logger.info(f"✓ Anchor: ({pyglet_image.anchor_x}, {pyglet_image.anchor_y})")
        
        # Test sprite creation
        sprite = create_positioned_sprite(pyglet_image, (1920, 1080))
        logger.info(f"✓ Sprite created at position: ({sprite.x}, {sprite.y})")
    else:
        logger.error("✗ Failed to load from base64 message")
    
    cleanup_temp_file(temp_file)
    
    # Test 2: Save to file and load from ZMQ message with file URI transport
    logger.info("\nTest 2: Loading from ZMQ message (file URI transport)")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
        test_image.save(temp.name, 'PNG')
        temp_path = temp.name
    
    try:
        message_file_uri = prepare_image_message(
            image_data=temp_path,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["FILE_URI"],
            image_id="test_file_uri"
        )
        
        pyglet_image, temp_file = load_pyglet_image_from_message(
            message=message_file_uri,
            image_id="test_file_uri",
            set_center_anchor=False
        )
        
        if pyglet_image:
            logger.info(f"✓ File URI message loaded: {pyglet_image.width}x{pyglet_image.height}")
            logger.info(f"✓ Anchor: ({pyglet_image.anchor_x}, {pyglet_image.anchor_y})")
        else:
            logger.error("✗ Failed to load from file URI message")
        
        cleanup_temp_file(temp_file)
        
    finally:
        Path(temp_path).unlink()
    
    # Test 3: Load from ZMQ message with hybrid transport
    logger.info("\nTest 3: Loading from ZMQ message (hybrid transport)")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
        test_image.save(temp.name, 'PNG')
        temp_path = temp.name
    
    try:
        message_hybrid = prepare_image_message(
            image_data=temp_path,
            target_address="tcp://localhost:5555",
            transport_mode=IMAGE_TRANSPORT_MODES["HYBRID"],
            image_id="test_hybrid"
        )
        
        pyglet_image, temp_file = load_pyglet_image_from_message(
            message=message_hybrid,
            image_id="test_hybrid",
            set_center_anchor=True
        )
        
        if pyglet_image:
            logger.info(f"✓ Hybrid message loaded: {pyglet_image.width}x{pyglet_image.height}")
            logger.info(f"✓ Anchor: ({pyglet_image.anchor_x}, {pyglet_image.anchor_y})")
        else:
            logger.error("✗ Failed to load from hybrid message")
        
        cleanup_temp_file(temp_file)
        
    finally:
        Path(temp_path).unlink()
    
    logger.info("\nPyglet utilities test completed!")


if __name__ == "__main__":
    asyncio.run(test_pyglet_utilities())
