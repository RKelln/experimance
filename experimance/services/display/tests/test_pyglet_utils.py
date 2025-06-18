#!/usr/bin/env python3
"""
Test script for pyglet utilities.

Tests the new shared utilities for loading images in the display service.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from PIL import Image as PILImage

from experimance_common.image_utils import png_to_base64url

# Import the utilities we want to test
import sys
sys.path.append('/home/ryankelln/Documents/Projects/Art/experimance/installation/software/experimance/services/display/src')

from experimance_display.utils.pyglet_utils import (
    load_pyglet_image_from_data,
    cleanup_temp_file,
    create_positioned_sprite
)

logger = logging.getLogger(__name__)


def create_test_image() -> PILImage.Image:
    """Create a simple test image."""
    img = PILImage.new('RGBA', (256, 256), (255, 0, 0, 255))  # Red square
    return img


async def test_pyglet_utilities():
    """Test the pyglet utilities."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing pyglet utilities...")
    
    # Create test image
    test_image = create_test_image()
    
    # Test 1: Load from PIL Image
    logger.info("Test 1: Loading from PIL Image")
    pyglet_image, temp_file = load_pyglet_image_from_data(
        image_data=test_image,
        image_id="test_pil",
        set_center_anchor=True
    )
    
    if pyglet_image:
        logger.info(f"✓ PIL Image loaded: {pyglet_image.width}x{pyglet_image.height}")
        logger.info(f"✓ Anchor: ({pyglet_image.anchor_x}, {pyglet_image.anchor_y})")
        
        # Test sprite creation
        sprite = create_positioned_sprite(pyglet_image, (1920, 1080))
        logger.info(f"✓ Sprite created at position: ({sprite.x}, {sprite.y})")
    else:
        logger.error("✗ Failed to load PIL Image")
    
    cleanup_temp_file(temp_file)
    
    # Test 2: Save to file and load from path
    logger.info("\nTest 2: Loading from file path")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
        test_image.save(temp.name, 'PNG')
        temp_path = temp.name
    
    try:
        pyglet_image, temp_file = load_pyglet_image_from_data(
            image_data=temp_path,
            image_id="test_file",
            set_center_anchor=False
        )
        
        if pyglet_image:
            logger.info(f"✓ File loaded: {pyglet_image.width}x{pyglet_image.height}")
            logger.info(f"✓ Anchor: ({pyglet_image.anchor_x}, {pyglet_image.anchor_y})")
        else:
            logger.error("✗ Failed to load from file")
        
        cleanup_temp_file(temp_file)
        
    finally:
        Path(temp_path).unlink()
    
    # Test 3: Load from base64 data
    logger.info("\nTest 3: Loading from base64 data")
    base64_data = png_to_base64url(test_image)
    
    pyglet_image, temp_file = load_pyglet_image_from_data(
        image_data=base64_data,
        image_id="test_base64",
        set_center_anchor=True
    )
    
    if pyglet_image:
        logger.info(f"✓ Base64 loaded: {pyglet_image.width}x{pyglet_image.height}")
        logger.info(f"✓ Anchor: ({pyglet_image.anchor_x}, {pyglet_image.anchor_y})")
    else:
        logger.error("✗ Failed to load from base64")
    
    cleanup_temp_file(temp_file)
    
    logger.info("\nPyglet utilities test completed!")


if __name__ == "__main__":
    asyncio.run(test_pyglet_utilities())
