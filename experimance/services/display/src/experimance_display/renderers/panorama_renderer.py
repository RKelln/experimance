#!/usr/bin/env python3
"""
Panorama Renderer for the Display Service.

Handles panoramic image display with base image blur transitions and positioned tiles.
Designed for installations with wide-aspect displays or multi-projector setups.

Features:
- Base image with dynamic blur σ→0 transition
- Positioned tiles that fade in as they complete
- Horizontal mirroring support
- Configurable rescaling modes
- Support for squashed projector outputs (e.g., 1920x1080 stretched to 6 outputs)
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import math

from experimance_common.schemas import ContentType, MessageBase, MessageType
from experimance_common.zmq.config import MessageDataType
from experimance_display.config import DisplayServiceConfig
import pyglet
from pyglet.gl import GL_BLEND, glEnable, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
from pyglet.image.codecs import ImageDecodeException
from pyglet import gl

from .layer_manager import LayerRenderer
from ..utils.pyglet_utils import load_pyglet_image_from_message, cleanup_temp_file, create_positioned_sprite

logger = logging.getLogger(__name__)


class PanoramaTile:
    """Represents a single tile in the panorama with position and fade state."""
    
    def __init__(self, sprite: pyglet.sprite.Sprite, position: Tuple[int, int], 
                 tile_id: str, original_size: Tuple[int, int], fade_duration: float = 3.0):
        self.sprite = sprite
        self.position = position  # (x, y) position in panorama space
        self.tile_id = tile_id
        self.original_size = original_size  # (width, height) of original image
        self.fade_duration = fade_duration
        
        # Fade state
        self.fade_timer = 0.0
        self.is_fading = True
        self.sprite.opacity = 0  # Start transparent
        
    def update(self, dt: float) -> None:
        """Update tile fade animation."""
        if self.is_fading:
            self.fade_timer += dt
            progress = min(self.fade_timer / self.fade_duration, 1.0)
            
            # Smooth fade in
            self.sprite.opacity = int(255 * progress)
            
            if progress >= 1.0:
                self.is_fading = False
                logger.debug(f"Tile {self.tile_id} fade complete")


class PanoramaRenderer(LayerRenderer):
    """Renders panoramic images with base image + positioned tiles."""
    
    def __init__(self, config: DisplayServiceConfig, 
                 window: pyglet.window.BaseWindow, 
                 batch: pyglet.graphics.Batch,
                 order: int = 0):
        """Initialize the panorama renderer."""
        super().__init__(config=config, window=window, batch=batch, order=order)
        
        self.panorama_config = config.panorama
        
        # Base image state
        self.base_image = None
        self.base_sprite = None
        self.base_mirror_sprite = None  # For mirrored base image
        self.base_image_id = None
        
        # Base image blur transition
        self.blur_timer = 0.0
        self.blur_active = False
        self.current_blur = self.panorama_config.start_blur
        
        # Tiles
        self.tiles: Dict[str, PanoramaTile] = {}  # tile_id -> PanoramaTile
        self.tile_order: List[str] = []  # Track tile addition order
        
        # Panorama dimensions
        self.panorama_width = self.panorama_config.output_width
        self.panorama_height = self.panorama_config.output_height
        
        # Calculate scaling and projection
        self._calculate_panorama_transform()
        
        # Visibility
        self._visible = True
        self._opacity = 1.0
        
        logger.info(f"PanoramaRenderer initialized: {self.panorama_width}x{self.panorama_height}, "
                   f"mirror={self.panorama_config.mirror}")
        
        # Validate tile configuration
        tile_config = self.panorama_config.tiles
        if self.panorama_config.mirror:
            # In mirrored mode, tiles should fit within half the total panorama width
            # since output_width includes the mirrored portion
            max_tile_width = self.panorama_width / 2
            expected_tiles = (self.panorama_width / 2) / tile_config.width
            if tile_config.width > max_tile_width:
                logger.warning(f"Tile width ({tile_config.width}) is larger than half total panorama width "
                             f"({max_tile_width}) - tiles may overlap when mirrored!")
            logger.info(f"Mirrored panorama: expect ~{expected_tiles:.1f} tiles per half "
                       f"({tile_config.width}px each in {self.panorama_width}px total width)")
        else:
            # In non-mirrored mode, tiles should fit within full panorama
            expected_tiles = self.panorama_width / tile_config.width
            if tile_config.width > self.panorama_width:
                logger.warning(f"Tile width ({tile_config.width}) is larger than panorama width "
                             f"({self.panorama_width}) - tiles may extend beyond panorama!")
            logger.info(f"Non-mirrored panorama: expect ~{expected_tiles:.1f} tiles "
                       f"({tile_config.width}px each in {self.panorama_width}px total width)")
        
        if tile_config.height > self.panorama_height:
            logger.warning(f"Tile height ({tile_config.height}) is larger than panorama height "
                         f"({self.panorama_height}) - tiles may extend beyond panorama!")
        
        logger.debug(f"Tile config: {tile_config.width}x{tile_config.height}, "
                    f"rescale={tile_config.rescale or self.panorama_config.rescale}")
    
    def _calculate_panorama_transform(self) -> None:
        """Calculate scaling and positioning for panorama display.
        
        This ensures the panorama is properly scaled and centered regardless of
        window size - critical for dev mode where window != production size.
        """
        window_width = self.window.width
        window_height = self.window.height
        
        # Calculate scale to fit panorama in window
        scale_x = window_width / self.panorama_width
        scale_y = window_height / self.panorama_height
        
        # Apply rescaling strategy
        if self.panorama_config.rescale == "width":
            self.panorama_scale = scale_x
            # For very wide panoramas, this might make them tall and thin
            final_width = self.panorama_width * scale_x
            final_height = self.panorama_height * scale_x
        elif self.panorama_config.rescale == "height":
            self.panorama_scale = scale_y
            final_width = self.panorama_width * scale_y
            final_height = self.panorama_height * scale_y
        else:  # "shortest" - maintain aspect ratio, fit entirely in window
            self.panorama_scale = min(scale_x, scale_y)
            final_width = self.panorama_width * self.panorama_scale
            final_height = self.panorama_height * self.panorama_scale
        
        # Center the scaled panorama in the window
        self.panorama_offset_x = (window_width - final_width) / 2
        self.panorama_offset_y = (window_height - final_height) / 2
        
        # Store final dimensions for debugging
        self.final_panorama_width = final_width
        self.final_panorama_height = final_height
        
        logger.debug(f"Panorama transform: scale={self.panorama_scale:.3f}")
        logger.debug(f"  Window: {window_width}x{window_height}")
        logger.debug(f"  Panorama: {self.panorama_width}x{self.panorama_height}")
        logger.debug(f"  Final: {final_width:.1f}x{final_height:.1f}")
        logger.debug(f"  Offset: ({self.panorama_offset_x:.1f}, {self.panorama_offset_y:.1f})")
        
        # Warn if panorama is much larger than window (common in dev)
        if self.panorama_scale < 0.5:
            logger.info(f"Panorama is significantly larger than window (scale={self.panorama_scale:.2f}). "
                       f"This is normal in dev mode testing production-sized panoramas.")
        
        # Warn if panorama extends beyond window bounds
        if (final_width > window_width * 1.1 or final_height > window_height * 1.1):
            logger.warning(f"Panorama extends beyond window bounds. Consider using rescale='shortest' "
                          f"for dev mode.")
    
    def _panorama_to_screen(self, pano_x: int, pano_y: int) -> Tuple[int, int]:
        """Convert panorama coordinates to screen coordinates.
        
        Note: Uses top-left coordinate system to match image anchors.
        """
        screen_x = int(self.panorama_offset_x + (pano_x * self.panorama_scale))
        # For top-left coordinate system, y=0 should be at the top of the panorama area
        screen_y = int(self.panorama_offset_y + self.final_panorama_height - (pano_y * self.panorama_scale))
        return screen_x, screen_y
    
    def _scale_image_for_panorama(self, image: pyglet.image.AbstractImage) -> pyglet.image.AbstractImage:
        """Scale image according to panorama rescale mode.
        
        Note: For sprites, we'll handle scaling via sprite.scale properties
        rather than transforming the image texture directly.
        """
        # For now, return the image as-is and handle scaling via sprite properties
        # This is simpler and more reliable than texture transformations
        return image
    
    @property
    def is_visible(self) -> bool:
        """Check if the layer should be rendered."""
        return (self._visible and self.panorama_config.enabled and 
                (self.base_sprite is not None or len(self.tiles) > 0))
    
    @property
    def opacity(self) -> float:
        """Get current layer opacity."""
        return self._opacity
    
    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set layer opacity and update all sprites."""
        self._opacity = max(0.0, min(1.0, value))
        base_opacity = int(255 * self._opacity)
        
        if self.base_sprite:
            self.base_sprite.opacity = base_opacity
        if self.base_mirror_sprite:
            self.base_mirror_sprite.opacity = base_opacity
            
        for tile in self.tiles.values():
            # Combine layer opacity with tile fade opacity
            tile_opacity = int(tile.sprite.opacity * self._opacity / 255)
            tile.sprite.opacity = tile_opacity
    
    def handle_display_media(self, message: MessageDataType) -> None:
        """Handle DisplayMedia message for panorama content."""
        try:
            # Convert message to dict if needed (like in image_renderer)
            if isinstance(message, MessageBase):
                message = message.model_dump()
                
            content_type = message.get('content_type')
            request_id = message.get('request_id', 'unknown')
            position = message.get('position')
            
            if content_type != ContentType.IMAGE:
                logger.warning(f"PanoramaRenderer only supports IMAGE content, got {content_type}")
                return
            
            # Load image - handle the tuple return
            result = load_pyglet_image_from_message(message, image_id=request_id)
            if result is None or result[0] is None:
                logger.error(f"Failed to load image for panorama: {request_id}")
                return
                
            image, temp_file = result
            
            try:
                # Determine if this is a base image or tile based on position
                # Check image is not None before proceeding
                if image is not None:
                    if position is None:
                        self._handle_base_image(image, request_id)
                    else:
                        self._handle_tile(image, request_id, position)
                else:
                    logger.error(f"Image is None for panorama: {request_id}")
            finally:
                # Clean up temporary file if one was created
                cleanup_temp_file(temp_file)
                
        except Exception as e:
            logger.error(f"Error handling panorama display media: {e}", exc_info=True)
    
    def _handle_base_image(self, image: pyglet.image.AbstractImage, request_id: str) -> None:
        """Handle base image for panorama."""
        logger.info(f"Setting panorama base image: {request_id}")
        
        # Don't transform the image - use sprite scaling instead
        # This is more reliable and handles mirroring better
        
        # Set image anchor point to top-left to match our coordinate system
        # By default pyglet uses bottom-left anchor which causes positioning issues
        # Set this once and both sprites will use the same anchor
        image.anchor_x = 0
        image.anchor_y = image.height  # Top of image

        # Create sprite at panorama origin
        screen_x, screen_y = self._panorama_to_screen(0, 0)
        sprite = pyglet.sprite.Sprite(image, x=screen_x, y=screen_y, 
                                      batch=self.batch, group=self)
        
        # Calculate target dimensions for the base image
        # output_width is inclusive of mirroring - it represents total final width
        # If mirroring is enabled, the base image should fill half the total panorama width
        # If not mirroring, it should fill the full panorama width
        if self.panorama_config.mirror:
            target_width = self.panorama_width / 2  # Half of total width for the original
            target_height = self.panorama_height
            logger.info(f"Mirroring enabled: scaling base image to fit {target_width}x{target_height} "
                       f"(half of total panorama {self.panorama_width}x{self.panorama_height})")
        else:
            target_width = self.panorama_width  # Full width
            target_height = self.panorama_height
            logger.info(f"No mirroring: scaling base image to fit {target_width}x{target_height} (full panorama)")
        
        # Calculate scaling to fit target dimensions
        scale_x = target_width / image.width
        scale_y = target_height / image.height
        
        # Apply rescale mode to determine which scale to use
        if self.panorama_config.rescale == "width":
            base_scale = scale_x
        elif self.panorama_config.rescale == "height":
            base_scale = scale_y
        else:  # "shortest" - maintain aspect ratio
            base_scale = min(scale_x, scale_y)
        
        # Combine with screen scaling to get final scale
        combined_scale = base_scale * self.panorama_scale
        sprite.scale = combined_scale
        
        logger.info(f"Base image scaling: target={target_width:.0f}x{target_height:.0f}, "
                   f"base_scale={base_scale:.3f}, screen_scale={self.panorama_scale:.3f}, "
                   f"final_scale={combined_scale:.3f}")
        logger.info(f"Final base size: {image.width * combined_scale:.1f}x{image.height * combined_scale:.1f}")
        
        # Clean up old base images first
        if self.base_sprite:
            self.base_sprite.delete()
        if self.base_mirror_sprite:
            self.base_mirror_sprite.delete()
            self.base_mirror_sprite = None
        
        # Handle mirroring positioning
        if self.panorama_config.mirror:
            # Create mirrored copy of the base image
            mirror_screen_x = screen_x + (target_width * self.panorama_scale)  # Position at right half
            
            # Create second sprite using the same image (with shared anchor point)
            mirror_sprite = pyglet.sprite.Sprite(image, x=mirror_screen_x, y=screen_y,
                                                batch=self.batch, group=self)
            mirror_sprite.scale_x = -combined_scale  # Flip horizontally
            mirror_sprite.scale_y = combined_scale
            
            # Adjust position for right-edge anchoring behavior
            # When we flip horizontally, the sprite flips around its anchor point
            # Since the image has left-edge anchor, we need to adjust for the flip
            scaled_width = image.width * combined_scale
            mirror_sprite.x = mirror_screen_x + scaled_width
            
            logger.info(f"Created mirrored base image at ({mirror_sprite.x}, {mirror_sprite.y}) with scale ({-combined_scale:.3f}, {combined_scale:.3f})")
            
            # Store both sprites
            self.base_mirror_sprite = mirror_sprite
        
        self.base_image = image
        self.base_sprite = sprite
        self.base_image_id = request_id
        
        logger.info(f"Base image setup complete:")
        logger.info(f"  Original sprite at ({sprite.x}, {sprite.y}) with scale {sprite.scale}")
        if self.base_mirror_sprite:
            logger.info(f"  Mirror sprite at ({self.base_mirror_sprite.x}, {self.base_mirror_sprite.y}) with scale ({self.base_mirror_sprite.scale_x}, {self.base_mirror_sprite.scale_y})")
        else:
            logger.info(f"  No mirror sprite (mirroring disabled)")
        
        # Start blur transition
        self._start_blur_transition()
    
    def _handle_tile(self, image: pyglet.image.AbstractImage, request_id: str, 
                    position: Tuple[int, int] | str) -> None:
        """Handle tile image for panorama."""
        if isinstance(position, str):
            logger.error(f"String positions not yet supported for panorama tiles: {position}")
            return
            
        tile_x, tile_y = position
        logger.info(f"Adding panorama tile: {request_id} at ({tile_x}, {tile_y})")
        
        # Use tile-specific rescaling if configured, otherwise use panorama rescaling
        tile_config = self.panorama_config.tiles
        rescale_mode = tile_config.rescale or self.panorama_config.rescale
        
        # Tile target dimensions are specified in panorama space
        # These represent the final size the tile should have in the panorama
        target_tile_width = tile_config.width
        target_tile_height = tile_config.height
        
        logger.debug(f"Tile target size: {target_tile_width}x{target_tile_height} (in panorama space)")
        logger.debug(f"Input image size: {image.width}x{image.height}")
        
        # Calculate sprite scaling to reach target tile size
        if rescale_mode == "width":
            scale_factor = target_tile_width / image.width
            # For width scaling, center the image vertically if it's taller than target
            scaled_height = image.height * scale_factor
            if scaled_height > target_tile_height:
                # Image will be taller than target - center it vertically
                vertical_offset = (scaled_height - target_tile_height) / 2
                tile_y_adjusted = tile_y - int(vertical_offset)
                logger.debug(f"Width scaling: image will be {scaled_height:.1f}px tall (target: {target_tile_height}px), "
                           f"centering with offset {vertical_offset:.1f}px")
            else:
                tile_y_adjusted = tile_y
            tile_x_adjusted = tile_x  # No horizontal adjustment for width scaling
        elif rescale_mode == "height":
            scale_factor = target_tile_height / image.height
            # For height scaling, center the image horizontally if it's wider than target
            scaled_width = image.width * scale_factor
            if scaled_width > target_tile_width:
                # Image will be wider than target - center it horizontally
                horizontal_offset = (scaled_width - target_tile_width) / 2
                tile_x_adjusted = tile_x - int(horizontal_offset)
                logger.debug(f"Height scaling: image will be {scaled_width:.1f}px wide (target: {target_tile_width}px), "
                           f"centering with offset {horizontal_offset:.1f}px")
            else:
                tile_x_adjusted = tile_x
            tile_y_adjusted = tile_y
        else:  # "shortest" - maintain aspect ratio
            scale_factor = min(target_tile_width / image.width, target_tile_height / image.height)
            tile_x_adjusted = tile_x
            tile_y_adjusted = tile_y
        
        # Convert adjusted tile position to screen coordinates
        screen_x, screen_y = self._panorama_to_screen(tile_x_adjusted, tile_y_adjusted)
        
        # Set anchor point for consistent positioning - do this once for the image
        image.anchor_x = 0
        image.anchor_y = image.height  # Top of image
        
        # Create original tile sprite
        sprite = pyglet.sprite.Sprite(image, x=screen_x, y=screen_y, 
                                      batch=self.batch, group=self)
        
        # Apply combined scaling: tile rescaling * panorama screen scaling
        combined_scale = scale_factor * self.panorama_scale
        sprite.scale = combined_scale
        
        final_tile_width = image.width * combined_scale
        final_tile_height = image.height * combined_scale
        
        logger.info(f"Tile {request_id}: scale_factor={scale_factor:.3f}, screen_scale={self.panorama_scale:.3f}, "
                   f"combined_scale={combined_scale:.3f}")
        logger.info(f"Tile {request_id}: final size {final_tile_width:.1f}x{final_tile_height:.1f} at screen ({screen_x}, {screen_y})")
        
        # Remove old tiles if they exist
        if request_id in self.tiles:
            old_tile = self.tiles[request_id]
            old_tile.sprite.delete()
            if request_id in self.tile_order:
                self.tile_order.remove(request_id)
        
        # Also remove mirrored version if it exists
        mirror_id = f"{request_id}_mirror"
        if mirror_id in self.tiles:
            old_mirror = self.tiles[mirror_id]
            old_mirror.sprite.delete()
            if mirror_id in self.tile_order:
                self.tile_order.remove(mirror_id)
        
        # Create original tile
        tile = PanoramaTile(sprite, (tile_x, tile_y), request_id, 
                           original_size=(image.width, image.height),
                           fade_duration=tile_config.fade_duration)
        self.tiles[request_id] = tile
        self.tile_order.append(request_id)
        
        # Create mirrored tile if mirroring is enabled
        if self.panorama_config.mirror:
            # Calculate mirror position: reflect across the center of the panorama
            # Use the original tile position for mirror calculation, not the adjusted one
            panorama_center_x = self.panorama_width / 2
            # Use the target tile width (after scaling) for the mirror calculation
            target_scaled_width = target_tile_width  # This is the size in panorama space
            mirror_x = int(2 * panorama_center_x - tile_x - target_scaled_width)
            
            # Apply the same centering adjustments to the mirror
            if rescale_mode == "width":
                scaled_height = image.height * scale_factor
                if scaled_height > target_tile_height:
                    vertical_offset = (scaled_height - target_tile_height) / 2
                    mirror_y = tile_y - int(vertical_offset)
                else:
                    mirror_y = tile_y
            elif rescale_mode == "height":
                scaled_width = image.width * scale_factor
                if scaled_width > target_tile_width:
                    horizontal_offset = (scaled_width - target_tile_width) / 2
                    mirror_x_adjusted = mirror_x - int(horizontal_offset)
                else:
                    mirror_x_adjusted = mirror_x
                mirror_y = tile_y
            else:  # "shortest"
                mirror_x_adjusted = mirror_x
                mirror_y = tile_y
            
            # For width scaling, we already set mirror_y above
            if rescale_mode != "height":
                mirror_x_adjusted = mirror_x
            
            mirror_screen_x, mirror_screen_y = self._panorama_to_screen(mirror_x_adjusted, mirror_y)
            
            logger.debug(f"Mirror calculation: center={panorama_center_x}, tile_x={tile_x}, "
                        f"target_width={target_scaled_width}, mirror_x={mirror_x}, adjusted_mirror_x={mirror_x_adjusted}")
            
            # Create mirrored sprite using the same image (with shared anchor point)
            mirror_sprite = pyglet.sprite.Sprite(image, x=mirror_screen_x, y=mirror_screen_y,
                                                batch=self.batch, group=self)
            mirror_sprite.scale_x = -combined_scale  # Flip horizontally
            mirror_sprite.scale_y = combined_scale
            
            # Adjust x position for flipped sprite (pyglet flips around sprite anchor)
            scaled_width = image.width * combined_scale
            mirror_sprite.x = mirror_screen_x + scaled_width
            
            logger.info(f"Tile {request_id}: mirror at panorama ({mirror_x_adjusted}, {mirror_y}), screen ({mirror_sprite.x}, {mirror_sprite.y})")
            
            # Create mirrored tile
            mirror_tile = PanoramaTile(mirror_sprite, (mirror_x_adjusted, mirror_y), mirror_id,
                                     original_size=(image.width, image.height),
                                     fade_duration=tile_config.fade_duration)
            self.tiles[mirror_id] = mirror_tile
            self.tile_order.append(mirror_id)
    
    def _start_blur_transition(self) -> None:
        """Start the blur transition for the base image."""
        if not self.panorama_config.blur_duration:
            return
            
        self.blur_timer = 0.0
        self.blur_active = True
        self.current_blur = self.panorama_config.start_blur
        
        logger.debug(f"Starting blur transition: {self.panorama_config.start_blur} → "
                    f"{self.panorama_config.end_blur} over {self.panorama_config.blur_duration}s")
    
    def update(self, dt: float) -> None:
        """Update panorama animations."""
        # Update blur transition
        if self.blur_active and self.base_sprite:
            self.blur_timer += dt
            progress = min(self.blur_timer / self.panorama_config.blur_duration, 1.0)
            
            # Interpolate blur value
            blur_range = self.panorama_config.end_blur - self.panorama_config.start_blur
            self.current_blur = self.panorama_config.start_blur + (blur_range * progress)
            
            # TODO: Apply blur shader to base_sprite
            # This would require implementing a blur shader
            
            if progress >= 1.0:
                self.blur_active = False
                logger.debug("Blur transition complete")
        
        # Update tile fades
        for tile in self.tiles.values():
            tile.update(dt)
    
    def clear_panorama(self) -> None:
        """Clear all panorama content."""
        logger.info("Clearing panorama content")
        
        # Clear base image
        if self.base_sprite:
            self.base_sprite.delete()
            self.base_sprite = None
        if self.base_mirror_sprite:
            self.base_mirror_sprite.delete()
            self.base_mirror_sprite = None
        self.base_image = None
        self.base_image_id = None
        
        # Clear tiles
        for tile in self.tiles.values():
            tile.sprite.delete()
        self.tiles.clear()
        self.tile_order.clear()
        
        # Reset state
        self.blur_active = False
        self.blur_timer = 0.0
    
    def set_visibility(self, visible: bool) -> None:
        """Set renderer visibility."""
        self._visible = visible
        
        # Update sprite visibility
        if self.base_sprite:
            self.base_sprite.visible = visible
        if self.base_mirror_sprite:
            self.base_mirror_sprite.visible = visible
            
        for tile in self.tiles.values():
            tile.sprite.visible = visible
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the panorama state."""
        return {
            "panorama_enabled": self.panorama_config.enabled,
            "base_image_id": self.base_image_id,
            "tiles_count": len(self.tiles),
            "tile_ids": list(self.tiles.keys()),
            "blur_active": self.blur_active,
            "current_blur": self.current_blur,
            "panorama_size": f"{self.panorama_width}x{self.panorama_height}",
            "panorama_scale": self.panorama_scale,
            "mirror": self.panorama_config.mirror
        }
    
    async def cleanup(self) -> None:
        """Clean up panorama renderer resources."""
        logger.info("Cleaning up panorama renderer")
        
        # Clear all content
        self.clear_panorama()
        
        # Clean up any cached images
        # (Currently we don't have a cache, but this is where it would go)
        
    def resize(self, new_size: Tuple[int, int]) -> None:
        """Handle window resize by recalculating panorama transform."""
        logger.info(f"Panorama renderer handling resize to {new_size}")
        
        # Update window dimensions
        old_width, old_height = self.window.width, self.window.height
        self.window.width, self.window.height = new_size
        
        # Recalculate panorama transform
        self._calculate_panorama_transform()
        
        # Update existing sprite positions
        if self.base_sprite:
            screen_x, screen_y = self._panorama_to_screen(0, 0)
            self.base_sprite.x = screen_x
            self.base_sprite.y = screen_y
            self.base_sprite.scale = self.panorama_scale
            
            if self.panorama_config.mirror:
                self.base_sprite.scale_x = -self.panorama_scale
                self.base_sprite.x = screen_x + (self.panorama_width * self.panorama_scale)
        
        # Update tile positions
        for tile in self.tiles.values():
            screen_x, screen_y = self._panorama_to_screen(*tile.position)
            tile.sprite.x = screen_x
            tile.sprite.y = screen_y
            
            # Recalculate tile scaling using stored original size
            tile_config = self.panorama_config.tiles
            rescale_mode = tile_config.rescale or self.panorama_config.rescale
            
            image_width, image_height = tile.original_size
                
            if rescale_mode == "width":
                scale_factor = tile_config.width / image_width
            elif rescale_mode == "height": 
                scale_factor = tile_config.height / image_height
            else:  # "shortest"
                scale_factor = min(tile_config.width / image_width, tile_config.height / image_height)
            
            combined_scale = scale_factor * self.panorama_scale
            tile.sprite.scale = combined_scale
            
            if self.panorama_config.mirror:
                tile.sprite.scale_x = -combined_scale
                # Calculate the width of the scaled tile
                scaled_tile_width = image_width * combined_scale
                tile.sprite.x = screen_x + scaled_tile_width
        
        logger.debug(f"Panorama renderer resize complete: {old_width}x{old_height} → {new_size}")
