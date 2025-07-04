#!/usr/bin/env python3
"""
Pyglet utilities for the Display Service.

Shared utilities for loading and handling images with pyglet.
"""

import logging
import os
import tempfile
from typing import Optional, Tuple, Union
from pathlib import Path

import pyglet
from PIL import Image as PILImage

from experimance_common.image_utils import base64url_to_png, load_image_from_message, ImageLoadFormat
from experimance_common.constants import BASE64_PNG_PREFIX

logger = logging.getLogger(__name__)

"""
Recommended usage pattern for any renderer handling ZMQ messages:

```python
    from experimance_display.utils.pyglet_utils import load_pyglet_image_from_message, cleanup_temp_file, create_positioned_sprite

    # Load image from ZMQ message (handles all transport modes automatically)
    pyglet_image, temp_file_path = load_pyglet_image_from_message(
        message=zmq_message,  # ZMQ message from prepare_image_message
        image_id="my_image",
        set_center_anchor=True
    )

    try:
        if pyglet_image:
            sprite = create_positioned_sprite(pyglet_image, window_size)
            # Use sprite...
    finally:
        cleanup_temp_file(temp_file_path)
```
"""

def _load_pyglet_from_base64(base64_data: str) -> Tuple[Optional[pyglet.image.AbstractImage], Optional[str]]:
    """Load pyglet image from base64 data.
    
    Args:
        base64_data: Base64 encoded image with data URL prefix
        
    Returns:
        Tuple of (pyglet_image, temp_file_path) or (None, None) if failed
    """
    try:
        # Convert base64 to PIL Image using our utility
        pil_image = base64url_to_png(base64_data)
        
        # Convert PIL to pyglet via temporary file
        return _load_pyglet_from_pil(pil_image)
        
    except Exception as e:
        logger.error(f"Error loading pyglet image from base64: {e}", exc_info=True)
        return None, None


def _load_pyglet_from_pil(pil_image: PILImage.Image) -> Tuple[Optional[pyglet.image.AbstractImage], Optional[str]]:
    """Load pyglet image from PIL Image.
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        Tuple of (pyglet_image, temp_file_path) or (None, None) if failed
        temp_file_path should be cleaned up by caller
    """
    try:
        # Convert to RGBA if needed for pyglet
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        # Save to temporary file for pyglet to load
        # (pyglet doesn't support loading directly from bytes easily)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            pil_image.save(temp_file.name, format='PNG')
            temp_path = temp_file.name
        
        # Load with pyglet
        pyglet_image = pyglet.image.load(temp_path)
        
        return pyglet_image, temp_path
        
    except Exception as e:
        logger.error(f"Error converting PIL image to pyglet: {e}", exc_info=True)
        return None, None


def cleanup_temp_file(temp_file_path: Optional[str]):
    """Clean up a temporary file created during image loading.
    
    Args:
        temp_file_path: Path to temporary file to delete
    """
    if temp_file_path:
        try:
            os.unlink(temp_file_path)
            logger.debug(f"Cleaned up temporary file: {temp_file_path}")
        except OSError as e:
            logger.warning(f"Could not delete temporary file {temp_file_path}: {e}")


def create_positioned_sprite(
    pyglet_image: pyglet.image.AbstractImage,
    window_size: Tuple[int, int],
    position: Optional[Tuple[int, int]] = None,
    batch: Optional[pyglet.graphics.Batch] = None,
    group: Optional[pyglet.graphics.Group] = None
) -> pyglet.sprite.Sprite:
    """Create a sprite from a pyglet image and position it.
    
    Args:
        pyglet_image: Pyglet image object
        window_size: (width, height) of the display window
        position: Optional (x, y) position. If None, centers in window
        
    Returns:
        Positioned pyglet sprite
    """
    sprite = pyglet.sprite.Sprite(pyglet_image, batch=batch, group=group)
    
    if position:
        sprite.x, sprite.y = position
    else:
        # Center in window
        sprite.x = window_size[0] // 2
        sprite.y = window_size[1] // 2
    
    return sprite


def load_pyglet_image_from_message(
    message: dict,
    image_id: Optional[str] = None,
    set_center_anchor: bool = True
) -> Tuple[Optional[pyglet.image.AbstractImage], Optional[str]]:
    """Load a pyglet image from a ZMQ message using generic utilities.
    
    This is the recommended way to load images from ZMQ messages in display service.
    It handles all transport modes (URI, base64, hybrid) automatically.
    
    Args:
        message: ZMQ message dict containing image data (from prepare_image_message)
        image_id: Optional identifier for logging
        set_center_anchor: Whether to set anchor to center of image
        
    Returns:
        Tuple of (pyglet_image, temp_file_path) or (None, None) if failed
        temp_file_path is returned for cleanup if a temporary file was created
        
    Example:
        # In a renderer's handle method:
        pyglet_image, temp_file = load_pyglet_image_from_message(message, "mask")
        try:
            if pyglet_image:
                self.mask_texture = pyglet_image.get_texture()
                # Use the image...
        finally:
            cleanup_temp_file(temp_file)
    """
    try:
        # Use the generic utility to get a file path for pyglet
        result = load_image_from_message(message, ImageLoadFormat.FILEPATH)
        
        if result is None:
            logger.error(f"Failed to extract image from message for: {image_id or 'unknown'}")
            return None, None
            
        file_path, is_temp = result  # type: ignore  # We know FILEPATH returns a tuple
        
        logger.debug(f"Loading pyglet image from {'temp ' if is_temp else ''}file: {file_path}")
        
        if not os.path.isfile(file_path):
            logger.error(f"Image file not found: {file_path}")
            return None, None
        
        # Load with pyglet
        pyglet_image = pyglet.image.load(file_path)
        
        if pyglet_image and set_center_anchor:
            # Set anchor point to center
            pyglet_image.anchor_x = pyglet_image.width // 2
            pyglet_image.anchor_y = pyglet_image.height // 2
        
        if pyglet_image:
            logger.debug(f"Successfully loaded pyglet image: {image_id or 'unknown'} size={pyglet_image.width}x{pyglet_image.height}")
        
        # Return temp file path only if it's actually a temp file that needs cleanup
        temp_file_path = file_path if is_temp else None
        return pyglet_image, temp_file_path
        
    except Exception as e:
        logger.error(f"Error loading pyglet image from message {image_id or 'unknown'}: {e}", exc_info=True)
        return None, None
