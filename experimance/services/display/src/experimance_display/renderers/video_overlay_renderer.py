#!/usr/bin/env python3
"""
Video Overlay Renderer for the Display Service.

Renders masked video overlay that responds to sand interaction.
Adapted from the VideoOverlay class in pyglet_test.py with enhancements for:
- Dynamic mask updates from ZMQ messages
- Better error handling and fallbacks
- Performance optimizations
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional, Tuple

import pyglet
from pyglet.gl import GL_BLEND, glEnable, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA

from .layer_manager import LayerRenderer
from ..utils.pyglet_utils import load_pyglet_image_from_message, cleanup_temp_file

logger = logging.getLogger(__name__)


class VideoOverlayRenderer(LayerRenderer):
    """Renders masked video overlay responding to sand interaction."""
    
    def __init__(self, window_size: Tuple[int, int], config: Any, transitions_config: Any):
        """Initialize the video overlay renderer.
        
        Args:
            window_size: (width, height) of the display window
            config: Rendering configuration
            transitions_config: Transition configuration
        """
        self.window_size = window_size
        self.config = config
        self.transitions_config = transitions_config
        
        # Video state
        self.video_player = None
        self.video_texture = None
        self.video_loaded = False
        
        # Mask state
        self.mask_texture = None
        self.current_mask = None
        self.mask_loaded = False
        
        # Animation state
        self.fade_state = "hidden"  # "hidden", "fading_in", "visible", "fading_out"
        self.fade_timer = 0.0
        self.fade_in_duration = transitions_config.video_fade_in_duration
        self.fade_out_duration = transitions_config.video_fade_out_duration
        
        # Shader program for masking (future enhancement)
        self.shader_program = None
        
        # Visibility and opacity
        self._visible = True
        self._opacity = 1.0
        self._current_alpha = 0.0  # Actual alpha considering fade state
        
        # Default video (if configured)
        self._load_default_video()
        
        logger.info(f"VideoOverlayRenderer initialized for {window_size[0]}x{window_size[1]}")
    
    @property
    def is_fading(self) -> bool:
        """Compatibility property for tests - check if fading is active."""
        return self.fade_state in ("fading_in", "fading_out")
    
    @property
    def is_visible(self) -> bool:
        """Check if the layer should be rendered."""
        return (
            self._visible and 
            self.video_loaded and 
            self.mask_loaded and 
            self._current_alpha > 0.01
        )
    
    @property
    def opacity(self) -> float:
        """Get the layer opacity (0.0 to 1.0)."""
        return self._opacity * self._current_alpha
    
    def update(self, dt: float):
        """Update video and animation state.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        # Update fade animation
        self._update_fade_animation(dt)
        
        # Update video playback
        if self.video_player and self.video_loaded:
            try:
                # Keep video playing and looping
                if not self.video_player.playing:
                    self.video_player.play()
                    
                # Update video texture
                if hasattr(self.video_player, 'get_texture'):
                    self.video_texture = self.video_player.texture
                    
            except Exception as e:
                logger.error(f"Error updating video: {e}", exc_info=True)
    
    def render(self):
        """Render the masked video overlay."""
        if not self.is_visible:
            return
        
        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        try:
            # For now, render a simple placeholder
            # Future: implement shader-based masking
            self._render_simple_overlay()
            
        except Exception as e:
            logger.error(f"Error rendering video overlay: {e}", exc_info=True)
    
    def _update_fade_animation(self, dt: float):
        """Update fade animation state.
        
        Args:
            dt: Time elapsed since last update
        """
        if self.fade_state == "fading_in":
            self.fade_timer += dt
            progress = min(self.fade_timer / self.fade_in_duration, 1.0)
            self._current_alpha = progress
            
            if progress >= 1.0:
                self.fade_state = "visible"
                logger.debug("Video overlay fade in complete")
                
        elif self.fade_state == "fading_out":
            self.fade_timer += dt
            progress = min(self.fade_timer / self.fade_out_duration, 1.0)
            self._current_alpha = 1.0 - progress
            
            if progress >= 1.0:
                self.fade_state = "hidden"
                self._current_alpha = 0.0
                logger.debug("Video overlay fade out complete")
                
        elif self.fade_state == "visible":
            self._current_alpha = 1.0
        else:  # hidden
            self._current_alpha = 0.0
    
    def _render_simple_overlay(self):
        """Render a simple video overlay without advanced masking.
        
        This is a placeholder implementation until shader-based masking is added.
        """
        if not (self.video_texture and self.mask_texture):
            return
        
        # Calculate position and size
        center_x = self.window_size[0] // 2
        center_y = self.window_size[1] // 2
        
        # Simple rendering: apply global alpha to video
        # Future: implement proper shader-based masking
        
        # For now, just indicate that video overlay would be rendered
        if self._current_alpha > 0.01:
            logger.debug(f"Would render video overlay with alpha: {self._current_alpha:.2f}")
    
    async def handle_video_mask(self, message: Dict[str, Any]):
        """Handle VideoMask message.
        
        Args:
            message: VideoMask message with mask image URI or image data
        """
        try:
            mask_id = message["mask_id"]
            fade_in_duration = message.get("fade_in_duration", self.fade_in_duration)
            fade_out_duration = message.get("fade_out_duration", self.fade_out_duration)
            
            logger.info(f"Loading video mask: {mask_id}")
            
            # Use the generic message loader for pyglet compatibility
            mask_image, temp_file_path = load_pyglet_image_from_message(
                message, 
                image_id=mask_id,
                set_center_anchor=False  # We'll set this ourselves if needed
            )
            
            try:
                if not mask_image:
                    logger.error(f"Failed to load mask {mask_id} - no valid data source")
                    return
                
                # Set the mask texture
                self.mask_texture = mask_image.get_texture()
                self.current_mask = mask_image
                self.mask_loaded = True
                
                logger.info(f"Mask {mask_id} loaded successfully")
                
                # Update fade durations
                self.fade_in_duration = fade_in_duration
                self.fade_out_duration = fade_out_duration
                
                # Start fade in animation
                self._start_fade_in()
                
            finally:
                # Clean up any temporary file that was created
                cleanup_temp_file(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error handling VideoMask: {e}", exc_info=True)
    
    def _load_default_video(self):
        """Load default video if configured."""
        # This could be configured in the config
        default_video_path = "services/image_server/images/video_overlay.mp4"
        
        if os.path.isfile(default_video_path):
            logger.info(f"Loading default video: {default_video_path}")
            asyncio.create_task(self._load_video(default_video_path))
        else:
            logger.debug("No default video found")
    
    async def _load_video(self, video_path: str) -> bool:
        """Load video from file path.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.isfile(video_path):
                logger.error(f"Video file not found: {video_path}")
                return False
            
            logger.debug(f"Loading video: {video_path}")
            
            # Load video using pyglet media
            source = pyglet.media.load(video_path)
            self.video_player = pyglet.media.Player()
            self.video_player.queue(source)
            
            # Set to loop
            self.video_player.loop = True
            
            self.video_loaded = True
            logger.info(f"Video loaded successfully: {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}", exc_info=True)
            self.video_loaded = False
            return False
    
    def _start_fade_in(self):
        """Start fade in animation."""
        if self.fade_state != "fading_in":
            self.fade_state = "fading_in"
            self.fade_timer = 0.0
            logger.debug("Started video overlay fade in")
    
    def _start_fade_out(self):
        """Start fade out animation."""
        if self.fade_state in ["visible", "fading_in"]:
            self.fade_state = "fading_out"
            self.fade_timer = 0.0
            logger.debug("Started video overlay fade out")
    
    def hide_overlay(self):
        """Hide the video overlay (fade out)."""
        self._start_fade_out()
    
    def show_overlay(self):
        """Show the video overlay (fade in)."""
        if self.mask_loaded:
            self._start_fade_in()
        else:
            logger.warning("Cannot show overlay without mask")
    
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize.
        
        Args:
            new_size: New (width, height) of the window
        """
        if new_size != self.window_size:
            logger.debug(f"VideoOverlayRenderer resize: {self.window_size} -> {new_size}")
            self.window_size = new_size
            
            # Future: update shader uniforms for new window size
    
    def set_visibility(self, visible: bool):
        """Set layer visibility.
        
        Args:
            visible: Whether the layer should be visible
        """
        self._visible = visible
        logger.debug(f"VideoOverlayRenderer visibility: {visible}")
    
    def set_opacity(self, opacity: float):
        """Set layer opacity.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        self._opacity = max(0.0, min(1.0, opacity))
        logger.debug(f"VideoOverlayRenderer opacity: {self._opacity}")
    
    async def cleanup(self):
        """Clean up video overlay renderer resources."""
        logger.info("Cleaning up VideoOverlayRenderer...")
        
        # Stop and clean up video player
        if self.video_player:
            try:
                self.video_player.pause()
                self.video_player = None
            except Exception as e:
                logger.error(f"Error stopping video player: {e}")
        
        # Clear textures
        self.video_texture = None
        self.mask_texture = None
        self.current_mask = None
        
        # Reset state
        self.video_loaded = False
        self.mask_loaded = False
        self.fade_state = "hidden"
        
        logger.info("VideoOverlayRenderer cleanup complete")
