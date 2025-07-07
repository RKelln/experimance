#!/usr/bin/env python3
"""
Mask Renderer for the Display Service.

Renders a mask that sits between the video layer and text overlay,
masking the background image and video layers underneath.
Supports loading masks from files or creating a simple circular mask on the fly.
"""

import logging
import os
from typing import Tuple
from pathlib import Path

import pyglet
from PIL import Image, ImageDraw

from experimance_display.config import DisplayServiceConfig
from experimance_display.renderers.layer_manager import LayerRenderer
from experimance_display.utils.pyglet_utils import create_positioned_sprite

logger = logging.getLogger(__name__)


class MaskRenderer(LayerRenderer):
    """Renders a mask overlay.
    
    This renderer creates a mask that can be used to hide or reveal
    portions of the underlying layers (background image and video overlay).
    Supports either loading a mask from a file or creating a circular mask.
    """
    
    def __init__(self, config: DisplayServiceConfig, 
                 window: pyglet.window.BaseWindow, 
                 batch: pyglet.graphics.Batch,
                 order: int = 1):
        """Initialize the mask renderer.
        
        Args:
            config: Display service configuration
            window: Pyglet window instance
            batch: Graphics batch for efficient rendering
            order: Render order (higher numbers render on top)
        """
        super().__init__(config, window, batch, order)
        
        # Mask properties
        self._visible = self.config.display.mask is not None
        self._opacity = 1.0
        
        # Mask sprite
        self.mask_sprite = None
        
        # Create the mask
        self._create_mask()
        
        logger.info(f"MaskRenderer initialized")
    
    def _create_mask(self):
        """Create the mask from file or generate circular mask."""
        mask_config = getattr(self.config.display, 'mask', True)
        
        if isinstance(mask_config, str):
            if mask_config.lower() == "circle":
                # Create circular mask
                self._create_circular_mask()
            else:
                # Try to load mask from file
                self._load_mask_from_file(mask_config)
        else:
            # Default to circular mask
            self._create_circular_mask()
    
    def _create_circular_mask(self):
        """Create a circular mask texture."""
        # Use window dimensions for mask size
        width, height = self.window.width, self.window.height
        
        # Create a new image for the mask (black background with transparent circle)
        mask_image = Image.new('RGBA', (width, height), (0, 0, 0, 255))  # Black background (opaque)
        draw = ImageDraw.Draw(mask_image)
        
        # Calculate circle parameters - center circle that covers most of the screen
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 2
        
        # Draw transparent circle (this will be the "visible" area where content shows through)
        draw.ellipse([
            center_x - radius, 
            center_y - radius, 
            center_x + radius, 
            center_y + radius
        ], fill=(0, 0, 0, 0))  # Transparent (alpha=0) - reveals content underneath
        
        # Convert PIL image to pyglet texture
        self._create_sprite_from_pil_image(mask_image)
        
        logger.info(f"Created circular mask: center=({center_x}, {center_y}), radius={radius}")
        logger.info("Mask: black outside (hides content), transparent inside (reveals content)")
    
    def _load_mask_from_file(self, mask_path: str):
        """Load mask from a file.
        
        Args:
            mask_path: Path to the mask image file
        """
        try:
            # Check if file exists
            if not os.path.isfile(mask_path):
                logger.warning(f"Mask file not found: {mask_path}, falling back to circular mask")
                self._create_circular_mask()
                return
            
            # Load mask image with pyglet
            mask_image = pyglet.image.load(mask_path)
            mask_image.anchor_x = mask_image.width // 2
            mask_image.anchor_y = mask_image.height // 2
            
            # Create sprite
            self.mask_sprite = create_positioned_sprite(
                mask_image, 
                (self.window.width, self.window.height),
                batch=self.batch, 
                group=self
            )
            
            logger.info(f"Loaded mask from file: {mask_path}")
            
        except Exception as e:
            logger.error(f"Error loading mask from file {mask_path}: {e}")
            logger.info("Falling back to circular mask")
            self._create_circular_mask()
    
    def _create_sprite_from_pil_image(self, pil_image: Image.Image):
        """Create a pyglet sprite from a PIL image.
        
        Args:
            pil_image: PIL Image object
        """
        try:
            # Convert PIL image to pyglet format
            image_data = pil_image.tobytes()
            pyglet_image = pyglet.image.ImageData(
                pil_image.width, 
                pil_image.height, 
                'RGBA', 
                image_data,
                pitch=-pil_image.width * 4  # Negative pitch to flip image
            )
            
            # Set anchor to center
            pyglet_image.anchor_x = pyglet_image.width // 2
            pyglet_image.anchor_y = pyglet_image.height // 2
            
            # Create sprite
            self.mask_sprite = create_positioned_sprite(
                pyglet_image, 
                (self.window.width, self.window.height),
                batch=self.batch, 
                group=self
            )
            
        except Exception as e:
            logger.error(f"Error creating sprite from PIL image: {e}")
            # Create a fallback solid mask
            self._create_fallback_mask()
    
    def _create_fallback_mask(self):
        """Create a simple fallback mask."""
        try:
            # Create a simple solid color texture
            solid_image = pyglet.image.SolidColorImagePattern((255, 255, 255, 128)).create_image(
                self.window.width, self.window.height
            )
            solid_image.anchor_x = solid_image.width // 2
            solid_image.anchor_y = solid_image.height // 2
            
            self.mask_sprite = create_positioned_sprite(
                solid_image,
                (self.window.width, self.window.height),
                batch=self.batch,
                group=self
            )
            
            logger.info("Created fallback solid mask")
            
        except Exception as e:
            logger.error(f"Error creating fallback mask: {e}")
            # If even this fails, just set sprite to None
            self.mask_sprite = None
    
    def update(self, dt: float):
        """Update the mask state.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        # Update sprite properties for batch rendering
        if self.mask_sprite:
            self.mask_sprite.opacity = int(self._opacity * 255)
            self.mask_sprite.visible = self._visible
    
    async def cleanup(self):
        """Clean up mask resources."""
        if self.mask_sprite:
            # Sprite will be cleaned up by the batch
            self.mask_sprite = None
        logger.debug("MaskRenderer cleanup complete")
    
    @property
    def is_visible(self) -> bool:
        """Check if the mask should be rendered."""
        return self._visible
    
    def set_visible(self, visible: bool):
        """Set mask visibility.
        
        Args:
            visible: Whether the mask should be visible
        """
        self._visible = visible
        logger.debug(f"Mask visibility set to: {visible}")
    
    @property
    def opacity(self) -> float:
        """Get the mask opacity (0.0 to 1.0)."""
        return self._opacity
    
    def set_opacity(self, opacity: float):
        """Set mask opacity.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        opacity = max(0.0, min(1.0, opacity))  # Clamp to valid range
        if opacity != self._opacity:
            self._opacity = opacity
            logger.debug(f"Mask opacity set to: {opacity}")
    
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize events.
        
        Args:
            new_size: New (width, height) of the window
        """
        if new_size != (self.window.width, self.window.height):
            logger.debug(f"Mask resized to window {new_size}")
            
            # Recreate the mask for the new window size
            self._create_mask()
    
    def set_state(self):
        """Set OpenGL state for mask rendering."""
        # Enable blending for opacity support
        from pyglet.gl import glEnable, glBlendFunc, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def unset_state(self):
        """Restore OpenGL state after mask rendering."""
        from pyglet.gl import glDisable, GL_BLEND
        glDisable(GL_BLEND)
