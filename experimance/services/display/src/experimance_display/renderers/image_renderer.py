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

from experimance_common.schemas import MessageBase, MessageType
from experimance_common.zmq.config import MessageDataType
from experimance_display.config import DisplayServiceConfig
import pyglet
from pyglet.gl import GL_BLEND, glEnable, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA

from .layer_manager import LayerRenderer
from ..utils.pyglet_utils import load_pyglet_image_from_message, cleanup_temp_file, create_positioned_sprite

logger = logging.getLogger(__name__)


class ImageRenderer(LayerRenderer):
    """Renders satellite landscape images with crossfade transitions."""
    
    def __init__(self, config: DisplayServiceConfig, 
                 window: pyglet.window.BaseWindow, 
                 batch: pyglet.graphics.Batch,
                 order: int = 0):
        """Initialize the image renderer."""
        super().__init__(config=config, window=window, batch=batch, order=order)

        # Current state
        self.current_image = None
        self.current_sprite = None
        self.current_image_id_value = None  # Store request_id separately
        self.next_image = None
        self.next_sprite = None
        self.next_image_id_value = None  # Store next request_id separately
        
        # Transition state
        self.transition_active = False
        self.transition_timer = 0.0
        self.transition_duration = config.transitions.default_crossfade_duration
        
        # Resource management
        self.image_cache = {}  # request_id -> (image, sprite)
        self.max_cache_size = 10  # Limit memory usage
        
        # Visibility and opacity
        self._visible = True
        self._opacity = 1.0
        
        logger.info(f"ImageRenderer initialized for {self.window}")
    
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
    
    async def handle_display_media(self, message: MessageDataType):
        """Handle DisplayMedia message.
        
        Args:
            message: DisplayMedia message with request_id and uri, and optionally image_data
        """
        try:
            if message is None:
                logger.error("Received None message in handle_display_media")
                return
            if message.get("type") != MessageType.DISPLAY_MEDIA:
                logger.error(f"Invalid display media message type: {message.get('type')}")
                return
            
            if isinstance(message, MessageBase):
                message = message.model_dump()
                
            request_id = message.get("request_id", None)
            if not request_id:
                logger.error("DisplayMedia message missing 'request_id'")
                return
            
            logger.info(f"Loading image: {request_id}")
            
            # Check cache first
            if request_id in self.image_cache:
                logger.debug(f"Using cached image: {request_id}")
                image, sprite = self.image_cache[request_id]
                self._start_transition_to_image(image, sprite, request_id)
                return
            
            # Use the robust image loading utility - it will handle fallbacks automatically
            pyglet_image, temp_file_path = load_pyglet_image_from_message(
                message=message,  # Pass the entire message
                image_id=request_id,  # Use request_id as image_id for the utility
                set_center_anchor=True
            )
            
            if not pyglet_image:
                logger.error(f"Failed to load image {request_id} - no valid data source")
                return
            
            try:
                # Create positioned sprite
                sprite = create_positioned_sprite(pyglet_image, self.window.get_size(), 
                                                  batch=self.batch, group=self)
                
                # Cache the image
                self._cache_image(request_id, pyglet_image, sprite)
                
                logger.info(f"Successfully loaded image: {request_id}")
                
                # Start transition to new image
                self._start_transition_to_image(pyglet_image, sprite, request_id)
                
            finally:
                # Clean up temporary file if one was created
                cleanup_temp_file(temp_file_path)
            
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
        if new_size != self.window.get_size():
            logger.debug(f"ImageRenderer resize: {new_size}")
        
            # Reposition all sprites
            if self.current_sprite:
                self.current_sprite.x = new_size[0] // 2
                self.current_sprite.y = new_size[1] // 2
            if self.next_sprite:
                self.next_sprite.x = new_size[0] // 2
                self.next_sprite.y = new_size[1] // 2
            
            # Reposition cached sprites
            for image_id, (image, sprite) in self.image_cache.items():
                sprite.x = new_size[0] // 2
                sprite.y = new_size[1] // 2
    
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
