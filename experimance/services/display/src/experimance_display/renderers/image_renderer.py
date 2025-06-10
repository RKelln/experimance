#!/usr/bin/env python3
"""
Image Renderer for the Display Service.

Handles satellite landscape image display with crossfade transitions.
Adapted from the ImageCycler class in pyglet_test.py with enhancements for:
- ZMQ message-driven updates
- Better resource management
- Error handling and fallbacks
- Future support for custom transitions and loops
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse

import pyglet
from pyglet.gl import GL_BLEND, glEnable, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA

from .layer_manager import LayerRenderer

logger = logging.getLogger(__name__)


class ImageRenderer(LayerRenderer):
    """Renders satellite landscape images with crossfade transitions."""
    
    def __init__(self, window_size: Tuple[int, int], config: Any, transitions_config: Any):
        """Initialize the image renderer.
        
        Args:
            window_size: (width, height) of the display window
            config: Rendering configuration
            transitions_config: Transition configuration
        """
        self.window_size = window_size
        self.config = config
        self.transitions_config = transitions_config
        
        # Current state
        self.current_image = None
        self.current_sprite = None
        self.current_image_id_value = None  # Store image_id separately
        self.next_image = None
        self.next_sprite = None
        self.next_image_id_value = None  # Store next image_id separately
        
        # Transition state
        self.transition_active = False
        self.transition_timer = 0.0
        self.transition_duration = transitions_config.default_crossfade_duration
        
        # Resource management
        self.image_cache = {}  # image_id -> (image, sprite)
        self.max_cache_size = 10  # Limit memory usage
        
        # Visibility and opacity
        self._visible = True
        self._opacity = 1.0
        
        logger.info(f"ImageRenderer initialized for {window_size[0]}x{window_size[1]}")
    
    @property
    def current_image_id(self) -> Optional[str]:
        """Compatibility property for tests - get current image ID."""
        return self.current_image_id_value
        
    @property
    def is_transitioning(self) -> bool:
        """Compatibility property for tests - check if transition is active."""
        return self.transition_active
        
    @property
    def is_visible(self) -> bool:
        """Check if the layer should be rendered."""
        return self._visible and self.current_sprite is not None
    
    @property
    def opacity(self) -> float:
        """Get the layer opacity (0.0 to 1.0)."""
        return self._opacity
    
    def update(self, dt: float):
        """Update transition state.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        if self.transition_active:
            self.transition_timer += dt
            
            # Check if transition is complete
            if self.transition_timer >= self.transition_duration:
                self._complete_transition()
    
    def render(self):
        """Render the current image(s)."""
        if not self.is_visible:
            return
        
        # Enable blending for smooth transitions
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        try:
            if self.transition_active and self.current_sprite and self.next_sprite:
                self._render_transition()
            elif self.current_sprite:
                self._render_current()
        except Exception as e:
            logger.error(f"Error rendering image: {e}", exc_info=True)
    
    def _render_current(self):
        """Render the current image only."""
        if self.current_sprite:
            self.current_sprite.opacity = int(self._opacity * 255)
            self.current_sprite.draw()
    
    def _render_transition(self):
        """Render crossfade transition between current and next image."""
        if not (self.current_sprite and self.next_sprite):
            return
        
        # Calculate transition progress (0.0 to 1.0)
        progress = min(self.transition_timer / self.transition_duration, 1.0)
        
        # Calculate opacities
        current_opacity = int((1.0 - progress) * self._opacity * 255)
        next_opacity = int(progress * self._opacity * 255)
        
        # Set sprite opacities
        self.current_sprite.opacity = current_opacity
        self.next_sprite.opacity = next_opacity
        
        # Draw both sprites (next first for proper alpha blending)
        self.next_sprite.draw()
        self.current_sprite.draw()
    
    def _complete_transition(self):
        """Complete the current transition."""
        logger.debug("Completing image transition")
        
        # Make next image the current image
        self.current_image = self.next_image
        self.current_sprite = self.next_sprite
        self.current_image_id_value = self.next_image_id_value
        
        # Clear next image
        self.next_image = None
        self.next_sprite = None
        self.next_image_id_value = None
        
        # Reset transition state
        self.transition_active = False
        self.transition_timer = 0.0
        
        logger.debug(f"Image transition completed to: {self.current_image_id_value}")
    
    async def handle_image_ready(self, message: Dict[str, Any]):
        """Handle ImageReady message.
        
        Args:
            message: ImageReady message with image_id and uri
        """
        try:
            image_id = message["image_id"]
            uri = message["uri"]
            
            logger.info(f"Loading image: {image_id} from {uri}")
            
            # Load image from URI
            image_path = self._uri_to_path(uri)
            if not image_path:
                logger.error(f"Could not resolve URI: {uri}")
                return
            
            # Load image
            image, sprite = await self._load_image(image_path, image_id)
            if not image or not sprite:
                logger.error(f"Failed to load image: {image_path}")
                return
            
            # Start transition to new image
            self._start_transition_to_image(image, sprite, image_id)
            
        except Exception as e:
            logger.error(f"Error handling ImageReady: {e}", exc_info=True)
    
    async def handle_transition_ready(self, message: Dict[str, Any]):
        """Handle TransitionReady message (future enhancement).
        
        Args:
            message: TransitionReady message with custom transition video
        """
        logger.info("Custom transitions not yet implemented")
        # Future: load and play custom transition video
    
    def _uri_to_path(self, uri: str) -> Optional[str]:
        """Convert URI to local file path.
        
        Args:
            uri: URI string (e.g., "file:///path/to/image.png")
            
        Returns:
            Local file path or None if invalid
        """
        try:
            parsed = urlparse(uri)
            if parsed.scheme == "file":
                return parsed.path
            elif parsed.scheme == "":
                # Assume it's already a file path
                return uri
            else:
                logger.warning(f"Unsupported URI scheme: {parsed.scheme}")
                return None
        except Exception as e:
            logger.error(f"Error parsing URI {uri}: {e}")
            return None
    
    async def _load_image(self, image_path: str, image_id: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Load image from file path.
        
        Args:
            image_path: Path to image file
            image_id: Unique identifier for the image
            
        Returns:
            Tuple of (image, sprite) or (None, None) if failed
        """
        try:
            # Check cache first
            if image_id in self.image_cache:
                logger.debug(f"Using cached image: {image_id}")
                return self.image_cache[image_id]
            
            # Check if file exists
            if not os.path.isfile(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None, None
            
            # Load image
            logger.debug(f"Loading image from: {image_path}")
            image = pyglet.image.load(image_path)
            
            # Set anchor point to center
            image.anchor_x = image.width // 2
            image.anchor_y = image.height // 2
            
            # Create sprite
            sprite = pyglet.sprite.Sprite(image)
            
            # Position sprite at center of window
            self._position_sprite(sprite)
            
            # Cache the image
            self._cache_image(image_id, image, sprite)
            
            logger.debug(f"Successfully loaded image: {image_id}")
            return image, sprite
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}", exc_info=True)
            return None, None
    
    def _position_sprite(self, sprite):
        """Position sprite at center of window.
        
        Args:
            sprite: Pyglet sprite to position
        """
        center_x = self.window_size[0] // 2
        center_y = self.window_size[1] // 2
        
        sprite.x = center_x
        sprite.y = center_y
    
    def _cache_image(self, image_id: str, image: Any, sprite: Any):
        """Cache an image and sprite.
        
        Args:
            image_id: Unique identifier for the image
            image: Pyglet image object
            sprite: Pyglet sprite object
        """
        # Remove oldest items if cache is full
        if len(self.image_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_id = next(iter(self.image_cache))
            del self.image_cache[oldest_id]
            logger.debug(f"Removed oldest cached image: {oldest_id}")
        
        self.image_cache[image_id] = (image, sprite)
        logger.debug(f"Cached image: {image_id}")
    
    def _start_transition_to_image(self, image: Any, sprite: Any, image_id: str):
        """Start transition to a new image.
        
        Args:
            image: Pyglet image object
            sprite: Pyglet sprite object
            image_id: Unique identifier for the image
        """
        # If no current image, just set it directly
        if not self.current_image:
            self.current_image = image
            self.current_sprite = sprite
            self.current_image_id_value = image_id
            logger.info(f"Set initial image: {image_id}")
            return
        
        # Start crossfade transition
        self.next_image = image
        self.next_sprite = sprite
        self.next_image_id_value = image_id
        self.transition_active = True
        self.transition_timer = 0.0
        
        logger.info(f"Started transition to image: {image_id}")
    
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize.
        
        Args:
            new_size: New (width, height) of the window
        """
        if new_size != self.window_size:
            logger.debug(f"ImageRenderer resize: {self.window_size} -> {new_size}")
            self.window_size = new_size
            
            # Reposition all sprites
            if self.current_sprite:
                self._position_sprite(self.current_sprite)
            if self.next_sprite:
                self._position_sprite(self.next_sprite)
            
            # Reposition cached sprites
            for image_id, (image, sprite) in self.image_cache.items():
                self._position_sprite(sprite)
    
    def set_visibility(self, visible: bool):
        """Set layer visibility.
        
        Args:
            visible: Whether the layer should be visible
        """
        self._visible = visible
        logger.debug(f"ImageRenderer visibility: {visible}")
    
    def set_opacity(self, opacity: float):
        """Set layer opacity.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        self._opacity = max(0.0, min(1.0, opacity))
        logger.debug(f"ImageRenderer opacity: {self._opacity}")
    
    async def cleanup(self):
        """Clean up image renderer resources."""
        logger.info("Cleaning up ImageRenderer...")
        
        # Clear current images
        self.current_image = None
        self.current_sprite = None
        self.current_image_id_value = None
        self.next_image = None
        self.next_sprite = None
        self.next_image_id_value = None
        
        # Clear cache
        self.image_cache.clear()
        
        logger.info("ImageRenderer cleanup complete")
