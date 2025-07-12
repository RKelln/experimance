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
from pyglet.gl import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture
from pyglet.gl import GL_COLOR_ATTACHMENT0, GL_NEAREST, GL_LINEAR
from pyglet.gl import GL_FRAMEBUFFER, glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D
from pyglet.image.codecs import ImageDecodeException
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.image import Framebuffer

from .layer_manager import LayerRenderer
from ..utils.pyglet_utils import load_pyglet_image_from_message, cleanup_temp_file, create_positioned_sprite

logger = logging.getLogger(__name__)


# GLSL Shaders for Post-Processing Blur
VERTEX_SHADER_SOURCE = """#version 150 core
in vec4 position;
in vec2 tex_coords;
out vec2 v_tex_coords;

uniform WindowBlock
{
    mat4 projection;
    mat4 view;
} window;

void main()
{
    gl_Position = window.projection * window.view * position;
    v_tex_coords = tex_coords;
}
"""

FRAGMENT_SHADER_SOURCE = """#version 150 core
in vec2 v_tex_coords;
out vec4 final_color;

uniform sampler2D scene_texture;
uniform float blur_sigma;

void main()
{
    if (blur_sigma <= 0.0) {
        final_color = texture(scene_texture, v_tex_coords);
        return;
    }
    
    vec2 tex_size = textureSize(scene_texture, 0);
    vec2 texel_size = 1.0 / tex_size;
    
    // Gaussian blur - smoother than box blur, eliminates "shake"
    vec4 color = vec4(0.0);
    float total_weight = 0.0;
    
    // Calculate radius (3-sigma rule covers ~99.7% of Gaussian distribution)
    float radius = blur_sigma * 3.0;
    int max_radius = int(ceil(radius));
    max_radius = min(max_radius, 20); // Clamp for performance
    
    // Two-pass separable Gaussian blur would be more efficient,
    // but this single-pass approach is simpler and works well for our needs
    for (int x = -max_radius; x <= max_radius; x++) {
        for (int y = -max_radius; y <= max_radius; y++) {
            vec2 offset = vec2(float(x), float(y)) * texel_size;
            
            // Gaussian weight calculation
            float distance_sq = float(x * x + y * y);
            float weight = exp(-distance_sq / (2.0 * blur_sigma * blur_sigma));
            
            color += texture(scene_texture, v_tex_coords + offset) * weight;
            total_weight += weight;
        }
    }
    
    // Normalize by total weight
    final_color = color / total_weight;
}
"""

# No-op fragment shader for debugging (bypasses blur)
FRAGMENT_SHADER_SOURCE_NOOP = """#version 150 core
in vec2 v_tex_coords;
out vec4 final_color;

uniform sampler2D scene_texture;

void main()
{
    final_color = texture(scene_texture, v_tex_coords);
}
"""


class _CaptureGroup(pyglet.graphics.Group):
    """Child group that captures sprites to framebuffer."""
    
    def __init__(self, panorama_renderer):
        super().__init__(order=0, parent=panorama_renderer)  # First child (order 0)
        self.panorama_renderer = panorama_renderer
    
    def set_state(self):
        """Bind framebuffer for off-screen rendering."""
        if self.panorama_renderer.framebuffer:
            self.panorama_renderer.framebuffer.bind()
            
            # Clear framebuffer
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            # Set viewport for framebuffer rendering
            fb_width = self.panorama_renderer.texture.width
            fb_height = self.panorama_renderer.texture.height
            gl.glViewport(0, 0, fb_width, fb_height)
            
            # Enable blending for sprite rendering
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        else:
            logger.error(f"_CaptureGroup: framebuffer is None!")
    
    def unset_state(self):
        """Unbind framebuffer."""
        if self.panorama_renderer.framebuffer:
            self.panorama_renderer.framebuffer.unbind()
            
            # Restore viewport to window size
            win_width = self.panorama_renderer.window.width
            win_height = self.panorama_renderer.window.height
            gl.glViewport(0, 0, win_width, win_height)
        else:
            logger.error(f"_CaptureGroup: framebuffer is None!")


class _DisplayGroup(pyglet.graphics.Group):
    """Child group that renders the blurred result."""
    
    def __init__(self, panorama_renderer):
        super().__init__(order=1, parent=panorama_renderer)  # Second child (order 1)
        self.panorama_renderer = panorama_renderer
    
    def set_state(self):
        """Setup shader and render fullscreen quad."""
        renderer = self.panorama_renderer
        
        try:
            # Ensure framebuffer is unbound (safety)
            renderer.framebuffer.unbind()
            
            if renderer.blur_enabled:
                # Use shader-based blur post-processing
                if (renderer.blur_shader_program and 
                    renderer.blur_quad and 
                    renderer.texture):
                    
                    # Activate shader
                    renderer.blur_shader_program.use()
                    
                    # Bind the captured texture
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(renderer.texture.target, renderer.texture.id)
                    
                    # Set shader uniforms
                    renderer.blur_shader_program['scene_texture'] = 0
                    
                    # Scale blur sigma for screen space
                    scaled_blur_sigma = renderer.current_blur_sigma * renderer.panorama_scale
                    renderer.blur_shader_program['blur_sigma'] = scaled_blur_sigma
                    
                    # Draw fullscreen quad
                    renderer.blur_quad.draw(gl.GL_TRIANGLE_STRIP)
                    
                    # Clean up shader
                    renderer.blur_shader_program.stop()
                else:
                    logger.error(f"_DisplayGroup: missing components for blur - shader_program={renderer.blur_shader_program is not None}, "
                                f"quad={renderer.blur_quad is not None}, texture={renderer.texture is not None}")
            else:
                # Use direct texture blitting (no blur)
                if renderer.texture:
                    renderer.texture.blit(0, 0)
                else:
                    logger.error(f"_DisplayGroup: no texture available for direct blitting")
                    
        except Exception as e:
            logger.error(f"_DisplayGroup: error during render: {e}", exc_info=True)
            # Try to clean up shader if it was activated
            try:
                if renderer.blur_enabled and renderer.blur_shader_program:
                    renderer.blur_shader_program.stop()
            except:
                pass
    
    def unset_state(self):
        """No cleanup needed - shader is stopped in set_state."""
        pass


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
        
        # Store opacity when clearing starts (for smooth transition from current state)
        self._clearing_start_opacity: Optional[int] = None
        
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
        
        # Blur post-processing setup - always use post-processing architecture
        self.blur_enabled = self.panorama_config.blur
        logger.info(f"Panorama blur {'enabled' if self.blur_enabled else 'DISABLED'} (config.panorama.blur={self.panorama_config.blur})")
        
        # PanoramaRenderer is the parent group for post-processing (blur or no-op)
        self.texture = pyglet.image.Texture.create(
            window.width, window.height, 
            min_filter=GL_LINEAR, mag_filter=GL_LINEAR
        )
        self.framebuffer = Framebuffer()
        self.framebuffer.attach_texture(self.texture, attachment=GL_COLOR_ATTACHMENT0)
        
        # Shader setup - choose blur or no-op based on config
        self.blur_shader_program = None
        self.blur_quad = None
        self._setup_blur_shader()
        
        # Child groups for capture and display
        self.capture = _CaptureGroup(self)  # order 0 (relative to parent)
        self.display = _DisplayGroup(self)  # order 1 (relative to parent)
        
        # Create a dummy sprite for the display group to ensure it gets rendered
        # This is needed because pyglet only calls set_state/unset_state for groups that have sprites
        dummy_image = pyglet.image.create(1, 1)
        self.dummy_sprite = pyglet.sprite.Sprite(dummy_image, x=-1000, y=-1000,  # Off-screen
                                               batch=self.batch, group=self.display)
        self.dummy_sprite.visible = False  # Make it invisible
        logger.debug(f"Created dummy sprite for _DisplayGroup to ensure rendering")
        
        # Create a test sprite for debugging - visible white square in center of screen
        test_image = pyglet.image.SolidColorImagePattern((255, 255, 255, 255)).create_image(100, 100)
        test_image.anchor_x = 50
        test_image.anchor_y = 50

        # Base image blur transition
        self.blur_timer = 0.0
        self.blur_active = False  # Only start when base image is received
        self.current_blur = self.panorama_config.start_blur
        self.current_blur_sigma = self.panorama_config.start_blur
        
        # Base image opacity fade-in
        self.base_fade_timer = 0.0
        self.base_fade_duration = 0.0
        self.base_fading = False
        
        # Tiles
        self.tiles: Dict[str, PanoramaTile] = {}  # tile_id -> PanoramaTile
        self.tile_order: List[str] = []  # Track tile addition order
        
        # Clearing/fading state
        self.is_clearing = False
        self.clear_timer = 0.0
        self.clear_duration = 3.0  # Default fade out duration
        self.clear_target_blur = self.config.panorama.start_blur
        
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
    
    def _setup_blur_shader(self):
        """Setup shader and fullscreen quad - choose blur or no-op based on config."""
        try:
            # Choose shader based on blur configuration
            vert_shader = Shader(VERTEX_SHADER_SOURCE, 'vertex')
            if self.blur_enabled:
                frag_shader = Shader(FRAGMENT_SHADER_SOURCE, 'fragment')
            else:
                frag_shader = Shader(FRAGMENT_SHADER_SOURCE_NOOP, 'fragment')
            
            self.blur_shader_program = ShaderProgram(vert_shader, frag_shader)
            
            # Create fullscreen quad using window coordinates (not NDC)
            positions = (
                0, 0, 0, 1,                          # bottom-left
                self.window.width, 0, 0, 1,          # bottom-right
                0, self.window.height, 0, 1,         # top-left
                self.window.width, self.window.height, 0, 1,  # top-right
            )
            tex_coords = (0, 0,  1, 0,  0, 1,  1, 1)
            
            self.blur_quad = self.blur_shader_program.vertex_list(
                4, gl.GL_TRIANGLE_STRIP,
                position=('f', positions),
                tex_coords=('f', tex_coords)
            )
            
            logger.debug(f"Blur shader setup complete: {self.window.width}x{self.window.height}")
            
        except Exception as e:
            logger.error(f"Failed to setup blur shader: {e}", exc_info=True)
            self.blur_shader_program = None
            self.blur_quad = None
    
    def _update_blur_effect(self) -> None:
        """Update the blur effect with current parameters."""
        # Blur parameters are stored directly on the PanoramaRenderer
        # and accessed by the child groups
        pass
    
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
        visible = (self._visible and self.panorama_config.enabled and 
                  (self.base_sprite is not None or len(self.tiles) > 0))
        logger.debug(f"PanoramaRenderer.is_visible: _visible={self._visible}, "
                    f"enabled={self.panorama_config.enabled}, "
                    f"has_base_sprite={self.base_sprite is not None}, "
                    f"tiles_count={len(self.tiles)}, result={visible}")
        return visible
    
    @property
    def opacity(self) -> float:
        """Get current layer opacity."""
        return self._opacity
    
    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set layer opacity for base images (tiles manage their own opacity)."""
        self._opacity = max(0.0, min(1.0, value))
        base_opacity = int(255 * self._opacity)
        
        # Only update base image sprites - tiles manage their own opacity
        if self.base_sprite:
            self.base_sprite.opacity = base_opacity
        if self.base_mirror_sprite:
            self.base_mirror_sprite.opacity = base_opacity
    
    def handle_display_media(self, message: MessageDataType) -> None:
        """Handle DisplayMedia message for panorama content."""
        try:
            # Convert message to dict if needed (like in image_renderer)
            if isinstance(message, MessageBase):
                message = message.model_dump()
                
            content_type = message.get('content_type')
            request_id = message.get('request_id', 'unknown')
            position = message.get('position')
            
            # Check for clear command (explicit clear flag or truly empty message)
            content = message.get('content')
            uri = message.get('uri')
            clear_flag = content_type == ContentType.CLEAR
            
            # Only treat as clear if explicitly flagged or if both content and uri are empty
            if clear_flag or (content is None and uri is None):
                logger.info(f"Clear command received: {request_id}")
                fade_duration = message.get('fade_in', 3.0)
                self.clear_panorama(fade_duration=fade_duration, blur_during_fade=True)
                return
            
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
                        self._handle_base_image(image, request_id, message)
                    else:
                        self._handle_tile(image, request_id, position, message)
                else:
                    logger.error(f"Image is None for panorama: {request_id}")
            finally:
                # Clean up temporary file if one was created
                cleanup_temp_file(temp_file)
                
        except Exception as e:
            logger.error(f"Error handling panorama display media: {e}", exc_info=True)
    
    def _handle_base_image(self, image: pyglet.image.AbstractImage, request_id: str, message: Dict[str, Any]) -> None:
        """Handle base image for panorama."""
        logger.info(f"Setting panorama base image: {request_id}")
        
        # Extract fade_in duration from message
        fade_in_duration = message.get('fade_in')
        
        # Clear any existing base image sprites
        if self.base_sprite:
            self.base_sprite.delete()
            self.base_sprite = None
        if self.base_mirror_sprite:
            self.base_mirror_sprite.delete()
            self.base_mirror_sprite = None
        
        # Don't transform the image - use sprite scaling instead
        # This is more reliable and handles mirroring better
        
        # Set image anchor point to top-left to match our coordinate system
        # By default pyglet uses bottom-left anchor which causes positioning issues
        # Set this once and both sprites will use the same anchor
        image.anchor_x = 0
        image.anchor_y = image.height  # Top of image

        # Create sprite at panorama origin
        screen_x, screen_y = self._panorama_to_screen(0, 0)
        
        # Create base image sprite (z=0 for bottom layer)
        sprite = pyglet.sprite.Sprite(image, x=screen_x, y=screen_y, 
                                      batch=self.batch, group=self.capture, z=0)
        
        # Set initial group opacity for fade-in effect (affects entire panorama)
        if fade_in_duration is not None and fade_in_duration > 0:
            self.opacity = 0.0  # Start transparent for fade-in
        else:
            self.opacity = 1.0  # Full opacity immediately
            
        logger.debug(f"Created base sprite: batch={self.batch}, group={self.capture}, z=0")
        logger.debug(f"Base sprite visible={sprite.visible}, opacity={sprite.opacity}")
        logger.debug(f"Base sprite position=({sprite.x}, {sprite.y}), size=({sprite.width}, {sprite.height})")
        logger.debug(f"Base sprite image size=({image.width}, {image.height}), anchor=({image.anchor_x}, {image.anchor_y})")
        
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
        
        logger.debug(f"After scaling: sprite size=({sprite.width}, {sprite.height}), scale={sprite.scale}")
        logger.debug(f"Final sprite bounds: left={sprite.x}, right={sprite.x + sprite.width}, "
                    f"top={sprite.y}, bottom={sprite.y - sprite.height}")
        logger.debug(f"Window bounds: width={self.window.width}, height={self.window.height}")
        
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
                                                batch=self.batch, group=self.capture, z=0)
            mirror_sprite.scale_x = -combined_scale  # Flip horizontally
            mirror_sprite.scale_y = combined_scale
            
            # Group opacity will handle both sprites together
            
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
        
        # Start opacity fade-in if specified
        if fade_in_duration is not None and fade_in_duration > 0:
            self._start_base_opacity_fade(fade_in_duration)
        
        logger.info(f"Base image setup complete:")
        logger.info(f"  Original sprite at ({sprite.x}, {sprite.y}) with scale {sprite.scale}")
        if self.base_mirror_sprite:
            logger.info(f"  Mirror sprite at ({self.base_mirror_sprite.x}, {self.base_mirror_sprite.y}) with scale ({self.base_mirror_sprite.scale_x}, {self.base_mirror_sprite.scale_y})")
        else:
            logger.info(f"  No mirror sprite (mirroring disabled)")
        
        # Start blur transition independently (uses config blur_duration, not fade_in_duration)
        self._start_blur_transition()
    
    def _handle_tile(self, image: pyglet.image.AbstractImage, request_id: str, 
                    position: Tuple[int, int] | str, message: Dict[str, Any]) -> None:
        """Handle tile image for panorama."""
        if isinstance(position, str):
            logger.error(f"String positions not yet supported for panorama tiles: {position}")
            return
            
        tile_x, tile_y = position
        logger.info(f"Adding panorama tile: {request_id} at ({tile_x}, {tile_y})")
        
        # Extract fade_in duration from message for tile fade duration
        fade_in_duration = message.get('fade_in')
        
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
        
        # Create original tile sprite (z=1 for top layer)
        sprite = pyglet.sprite.Sprite(image, x=screen_x, y=screen_y, 
                                      batch=self.batch, group=self.capture, z=1)
        
        # Set initial opacity for tile fade-in effect (individual tile control)
        if fade_in_duration is not None and fade_in_duration > 0:
            sprite.opacity = 0  # Start transparent for fade-in
        else:
            sprite.opacity = 255  # Full opacity immediately
        
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
        
        # Create original tile - use message fade_in for custom duration or config default
        if fade_in_duration is not None and fade_in_duration > 0:
            # Use custom fade duration from message
            tile = PanoramaTile(sprite, (tile_x, tile_y), request_id, 
                               original_size=(image.width, image.height),
                               fade_duration=fade_in_duration)
        else:
            # Use config fade duration for built-in animation
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
            
            # Create mirrored sprite using the same image (z=1 for top layer)
            mirror_sprite = pyglet.sprite.Sprite(image, x=mirror_screen_x, y=mirror_screen_y,
                                                batch=self.batch, group=self.capture, z=1)
            mirror_sprite.scale_x = -combined_scale  # Flip horizontally
            mirror_sprite.scale_y = combined_scale
            
            # Set same initial opacity as main tile sprite
            mirror_sprite.opacity = sprite.opacity
            
            # Adjust x position for flipped sprite (pyglet flips around sprite anchor)
            scaled_width = image.width * combined_scale
            mirror_sprite.x = mirror_screen_x + scaled_width
            
            logger.info(f"Tile {request_id}: mirror at panorama ({mirror_x_adjusted}, {mirror_y}), screen ({mirror_sprite.x}, {mirror_sprite.y})")
            
            # Create mirrored tile
            mirror_tile = PanoramaTile(mirror_sprite, (mirror_x_adjusted, mirror_y), mirror_id,
                                     original_size=(image.width, image.height),
                                     fade_duration=tile.fade_duration)  # Use same fade duration as original tile
                    
            self.tiles[mirror_id] = mirror_tile
            self.tile_order.append(mirror_id)
    
    def _start_base_opacity_fade(self, fade_duration: float) -> None:
        """Start opacity fade-in for entire panorama group."""
        self.base_fade_timer = 0.0
        self.base_fade_duration = fade_duration
        self.base_fading = True
    
    def _start_blur_transition(self) -> None:
        """Start the blur transition for the base image."""
        if not self.blur_enabled:
            return
        
        # Always use config blur_duration for blur transition (independent of fade_in)
        effective_blur_duration = self.panorama_config.blur_duration
        
        if not effective_blur_duration:
            return
            
        self.blur_timer = 0.0
        self.blur_active = True
        self.current_blur = self.panorama_config.start_blur
        self.current_blur_sigma = self.panorama_config.start_blur
        
        # Store the effective duration for use in update()
        self._effective_blur_duration = effective_blur_duration
        
        # Update blur effect with initial parameters
        self._update_blur_effect()
        
        logger.debug(f"Starting blur transition: {self.panorama_config.start_blur} → "
                    f"{self.panorama_config.end_blur} over {effective_blur_duration}s")
    
    def update(self, dt: float) -> None:
        """Update panorama animations."""
        # Handle clearing/fading
        if self.is_clearing:
            self.clear_timer += dt
            progress = min(self.clear_timer / self.clear_duration, 1.0)
            
            # Fade out entire panorama using group opacity
            group_opacity = 1.0 - progress
            self.opacity = group_opacity
            
            # Handle individual tiles that were already fading when clear started
            for tile in self.tiles.values():
                if tile._clearing_start_opacity is not None:
                    # Fade out from the opacity the tile had when clearing started
                    tile.sprite.opacity = int(tile._clearing_start_opacity * (1.0 - progress))
            
            # Increase blur during fade if enabled
            if self.blur_active and hasattr(self, 'original_blur'):
                self.current_blur_sigma = self.original_blur + (self.clear_target_blur * progress)
            
            # Complete clearing when fade is done
            if progress >= 1.0:
                self.is_clearing = False
                self._clear_all_sprites()
                # Reset blur to initial value if blur is enabled
                if self.blur_enabled:
                    self.current_blur_sigma = self.panorama_config.start_blur
                else:
                    self.current_blur_sigma = 0.0
                # Clean up original_blur if it was set
                if hasattr(self, 'original_blur'):
                    delattr(self, 'original_blur')
                logger.info("Panorama clear complete")
                return
        
        # Update base image opacity fade-in (affects entire panorama group)
        if self.base_fading and self.base_sprite:
            self.base_fade_timer += dt
            progress = min(self.base_fade_timer / self.base_fade_duration, 1.0)
            
            # Fade in entire panorama group
            self.opacity = progress
            
            if progress >= 1.0:
                self.base_fading = False
        
        # Update blur transition (only if blur enabled, not clearing and we have content)
        if (self.blur_enabled and not self.is_clearing and 
            self.blur_active and self.base_sprite):
            self.blur_timer += dt
            # Use the stored effective duration
            effective_duration = getattr(self, '_effective_blur_duration', self.panorama_config.blur_duration)
            progress = min(self.blur_timer / effective_duration, 1.0)
            
            # Interpolate blur value
            blur_range = self.panorama_config.end_blur - self.panorama_config.start_blur
            self.current_blur = self.panorama_config.start_blur + (blur_range * progress)
            
            # Update the current blur sigma for the shader
            self.current_blur_sigma = self.current_blur
            self._update_blur_effect()  # Update blur group with new parameters
            #logger.debug(f"Blur transition progress: {progress:.2f}, "
            #            f"current_blur_sigma={self.current_blur_sigma:.3f}")
            if progress >= 1.0:
                self.blur_active = False
                logger.debug("Blur transition complete")
        
        # Update tile fades (only if not clearing)
        if not self.is_clearing:
            for tile in self.tiles.values():
                tile.update(dt)
    
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
    
    def clear_panorama(self, fade_duration: float = 3.0, blur_during_fade: bool = True) -> None:
        """Start clearing/fading out the panorama.
        
        Args:
            fade_duration: Duration of fade out in seconds
            blur_during_fade: Whether to increase blur during fade
        """
        logger.info(f"Starting panorama clear with {fade_duration}s fade")
        self.is_clearing = True
        self.clear_timer = 0.0
        self.clear_duration = fade_duration
        
        # Stop all ongoing tile fade animations and store their current opacity
        for tile in self.tiles.values():
            tile.is_fading = False
            # Store the current opacity so we can fade from it
            tile._clearing_start_opacity = tile.sprite.opacity
            logger.debug(f"Stopped fade animation for tile {tile.tile_id}, current opacity: {tile.sprite.opacity}")
        
        # Stop base image fade-in if active
        if self.base_fading:
            self.base_fading = False
        
        if blur_during_fade:
            # Store original blur to restore later
            self.original_blur = self.current_blur_sigma
    
    def _clear_all_sprites(self) -> None:
        """Immediately remove all sprites from the panorama."""
        # Remove base image sprites
        if self.base_sprite:
            self.base_sprite.delete()
            self.base_sprite = None
        if self.base_mirror_sprite:
            self.base_mirror_sprite.delete()
            self.base_mirror_sprite = None
        
        self.base_image = None
        self.base_image_id = None
        
        # Remove all tile sprites
        for tile in self.tiles.values():
            tile.sprite.delete()
            # Clean up clearing state
            tile._clearing_start_opacity = None
        
        self.tiles.clear()
        self.tile_order.clear()
        
        logger.info("All panorama sprites cleared")

    async def cleanup(self) -> None:
        """Clean up panorama renderer resources."""
        logger.info("Cleaning up panorama renderer")
        
        # Clear all content
        self._clear_all_sprites()
        
        # Clean up dummy sprite
        if hasattr(self, 'dummy_sprite') and self.dummy_sprite:
            self.dummy_sprite.delete()
            self.dummy_sprite = None
        
        # Clean up blur resources
        if self.blur_quad:
            self.blur_quad.delete()
            self.blur_quad = None
        
        # Framebuffer and texture will be cleaned up by pyglet
        self.framebuffer = None
        self.texture = None
        self.blur_shader_program = None
        
    def resize(self, new_size: Tuple[int, int]) -> None:
        """Handle window resize by recalculating panorama transform and recreating resources."""
        logger.info(f"Panorama renderer handling resize to {new_size}")
        
        # Update window dimensions
        old_width, old_height = self.window.width, self.window.height
        self.window.width, self.window.height = new_size
        
        # Recreate resources for new size
        self.texture = pyglet.image.Texture.create(
            new_size[0], new_size[1], 
            min_filter=GL_LINEAR, mag_filter=GL_LINEAR
        )
        self.framebuffer = Framebuffer()
        self.framebuffer.attach_texture(self.texture, attachment=GL_COLOR_ATTACHMENT0)
        
        # Recreate child groups
        self.capture = _CaptureGroup(self)
        self.display = _DisplayGroup(self)
        
        # Recreate dummy sprite for display group
        if hasattr(self, 'dummy_sprite') and self.dummy_sprite:
            self.dummy_sprite.delete()
        dummy_image = pyglet.image.create(1, 1)
        self.dummy_sprite = pyglet.sprite.Sprite(dummy_image, x=-1000, y=-1000,
                                               batch=self.batch, group=self.display)
        self.dummy_sprite.visible = False
        
        # Recalculate panorama transform
        self._calculate_panorama_transform()
        
        # Update existing sprite positions and groups
        if self.base_sprite:
            screen_x, screen_y = self._panorama_to_screen(0, 0)
            
            # Update base sprite to use new capture group
            old_sprite = self.base_sprite
            self.base_sprite = pyglet.sprite.Sprite(
                old_sprite.image, x=screen_x, y=screen_y,
                batch=self.batch, group=self.capture, z=0
            )
            self.base_sprite.scale = self.panorama_scale
            old_sprite.delete()
            
            if self.base_mirror_sprite:
                old_mirror = self.base_mirror_sprite
                mirror_screen_x = screen_x + (self.panorama_width / 2 * self.panorama_scale)
                self.base_mirror_sprite = pyglet.sprite.Sprite(
                    old_mirror.image, x=mirror_screen_x, y=screen_y,
                    batch=self.batch, group=self.capture, z=0
                )
                self.base_mirror_sprite.scale_x = -self.panorama_scale
                self.base_mirror_sprite.scale_y = self.panorama_scale
                # Adjust for flipped positioning
                scaled_width = old_mirror.image.width * self.panorama_scale
                self.base_mirror_sprite.x = mirror_screen_x + scaled_width
                old_mirror.delete()
        
        # Update tile positions and groups
        for tile in self.tiles.values():
            screen_x, screen_y = self._panorama_to_screen(*tile.position)
            
            # Recreate tile sprite with new capture group
            old_sprite = tile.sprite
            tile.sprite = pyglet.sprite.Sprite(
                old_sprite.image, x=screen_x, y=screen_y,
                batch=self.batch, group=self.capture, z=1
            )
            
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
            old_sprite.delete()
            
            # Handle mirrored tiles
            if tile.tile_id.endswith('_mirror'):
                tile.sprite.scale_x = -combined_scale
                scaled_tile_width = image_width * combined_scale
                tile.sprite.x = screen_x + scaled_tile_width
        
        logger.debug(f"Panorama renderer resize complete: {old_width}x{old_height} → {new_size}")
