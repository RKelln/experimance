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
- Support             self.blur_shader_program = ShaderProgram(vert_shader, frag_shader)
            
            positions = (-1.0, -1.0, 0.0, 1.0,   # bl
             1.0, -1.0, 0.0, 1.0,   # br
            -1.0,  1.0, 0.0, 1.0,   # tlrojector outputs (e.g., 1920x1080 stretched to 6 outputs)
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
from pyglet.math import Mat4

from .layer_manager import LayerRenderer
from ..utils.pyglet_utils import load_pyglet_image_from_message, cleanup_temp_file, create_positioned_sprite

logger = logging.getLogger(__name__)


# GLSL Shaders for Post-Processing Blur
VERTEX_SHADER_SOURCE = """#version 150 core
in  vec4 position;
in  vec2 tex_coords;
out vec2 v_tex;

void main() {
    gl_Position = position;
    v_tex       = tex_coords;
}
"""

FRAGMENT_SHADER_SOURCE = """#version 150 core
in  vec2 v_tex;
out vec4 frag;

uniform sampler2D scene_texture;
uniform float     blur_sigma;     // in **framebuffer** pixels
uniform int       enable_mirror;  // use int (!) for pyglet

vec4 fast_blur(vec2 tc) {
    if (blur_sigma < 0.5)               // ~no-blur fast-path
        return texture(scene_texture, tc);

    vec2 texel = 1.0 / vec2(textureSize(scene_texture, 0));
    
    // Increased blur radius for more dramatic effects - up to 9x9 kernel
    int r = int(min(4.0, ceil(blur_sigma * 0.6)));
    float sigma2 = blur_sigma * blur_sigma;
    
    vec4 col = vec4(0.0);
    float wsum = 0.0;
    
    // Sample in a cross pattern for better performance
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            float dist2 = float(x*x + y*y);
            float w = exp(-dist2 / (2.0 * sigma2));
            col += texture(scene_texture, tc + vec2(x, y) * texel) * w;
            wsum += w;
        }
    }
    return col / wsum;
}

void main() {
    vec2 tc = v_tex;

    // Handle mirroring first
    if (enable_mirror == 1 && tc.x > 0.5) {
        tc.x = 1.0 - tc.x;
    }

    vec4 blurred = fast_blur(tc);
    frag = blurred;
}
"""

# Optimized fragment shader for better performance
FRAGMENT_SHADER_SOURCE_NOOP = """#version 150 core
in vec2 v_tex;
out vec4 frag;

uniform sampler2D scene_texture;
uniform float blur_sigma;
uniform int enable_mirror;

void main()
{
    vec2 tc = v_tex;

    // Handle mirroring if enabled
    if (enable_mirror == 1 && tc.x > 0.5) {
        tc.x = 1.0 - tc.x;
    }
    
    // Handle blur if enabled - use much more efficient implementation
    if (blur_sigma < 0.5) {
        // No blur - direct sampling
        frag = texture(scene_texture, tc);
    } else {
        // Enhanced box blur for more dramatic effects - up to 9x9 kernel
        vec2 texel = 1.0 / textureSize(scene_texture, 0);
        int r = int(min(4.0, ceil(blur_sigma * 0.5)));
        
        vec3 result = vec3(0.0);
        int samples = 0;
        
        for (int x = -r; x <= r; x++) {
            for (int y = -r; y <= r; y++) {
                vec2 offset = vec2(float(x), float(y)) * texel;
                result += texture(scene_texture, tc + offset).rgb;
                samples++;
            }
        }
        
        frag = vec4(result / float(samples), 1.0);
    }
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
            
            # CRITICAL: Set up projection matrix for framebuffer size using modern pyglet
            # Store original projection to restore later
            self._original_projection = self.panorama_renderer.window.projection
            
            # Create orthographic projection for framebuffer coordinates
            self.panorama_renderer.window.projection = Mat4.orthogonal_projection(
                0, fb_width,   # left, right
                0, fb_height,  # bottom, top
                -1, 1          # near, far
            )
            
            # Enable blending for sprite rendering
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        else:
            logger.error(f"_CaptureGroup: framebuffer is None!")
    
    def unset_state(self):
        """Unbind framebuffer and restore original state."""
        if self.panorama_renderer.framebuffer:
            self.panorama_renderer.framebuffer.unbind()
            
            # Restore viewport to window size
            win_width = self.panorama_renderer.window.width
            win_height = self.panorama_renderer.window.height
            gl.glViewport(0, 0, win_width, win_height)
            
            # Restore original projection matrix
            if hasattr(self, '_original_projection'):
                self.panorama_renderer.window.projection = self._original_projection
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
            
            # Always use the shader for consistent rendering
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
                
                # Set blur sigma (0 if blur disabled)
                if renderer.blur_enabled:
                    sigma = renderer.current_blur_sigma
                else:
                    sigma = 0.0
                renderer.blur_shader_program['blur_sigma'] = float(sigma)
                
                # Set mirroring uniforms
                renderer.blur_shader_program['enable_mirror'] = int(renderer.panorama_config.mirror)
                
                # Draw fullscreen quad - shader handles mirroring + scaling
                renderer.blur_quad.draw(gl.GL_TRIANGLE_STRIP)
                
                # Clean up shader
                renderer.blur_shader_program.stop()
            else:
                logger.error(f"_DisplayGroup: missing components for render - shader_program={renderer.blur_shader_program is not None}, "
                            f"quad={renderer.blur_quad is not None}, texture={renderer.texture is not None}")
                    
        except Exception as e:
            logger.error(f"_DisplayGroup: error during render: {e}", exc_info=True)
            # Try to clean up shader if it was activated
            try:
                if renderer.blur_shader_program:
                    renderer.blur_shader_program.stop()
            except:
                pass
    
    def unset_state(self):
        """No cleanup needed - shader is stopped in set_state."""
        pass


class PanoramaTile:
    """Represents a single tile in the panorama with position and fade state."""
    
    def __init__(self, sprite: pyglet.sprite.Sprite, position: Tuple[int, int], 
                 tile_id: str, original_size: Tuple[int, int], fade_duration: float = 3.0,
                 debug_rect=None):
        self.sprite = sprite
        self.position = position  # (x, y) position in panorama space
        self.tile_id = tile_id
        self.original_size = original_size  # (width, height) of original image
        self.fade_duration = fade_duration
        self.debug_rect = debug_rect  # Optional debug rectangle (injected)
        
        # Fade state
        self.fade_timer = 0.0
        self.is_fading = True
        self.sprite.opacity = 0  # Start transparent
        
        # Store opacity when clearing starts (for smooth transition from current state)
        self._clearing_start_opacity: Optional[int] = None

    def cleanup(self) -> None:
        """Clean up resources used by the tile."""
        self.sprite.delete()
        if self.debug_rect:
            self.debug_rect.delete()
    
    @staticmethod
    def create_debug_outline(sprite: pyglet.sprite.Sprite, batch: pyglet.graphics.Batch, 
                           group: pyglet.graphics.Group, color: Tuple[int, int, int] = (255, 0, 0)) -> Optional[pyglet.shapes.Rectangle]:
        """Create a debug outline rectangle for a sprite.
        
        Args:
            sprite: The sprite to create an outline for
            batch: Pyglet batch to add the rectangle to
            group: Pyglet group for rendering order
            color: RGB color for the outline (default: red)
            
        Returns:
            The created rectangle shape, or None if creation failed
        """
        try:
            from pyglet import shapes
            
            # Calculate outline dimensions based on sprite's scaled size
            scaled_width = sprite.width * sprite.scale
            scaled_height = sprite.height * sprite.scale
            
            debug_rect = shapes.Rectangle(
                x=sprite.x, y=sprite.y,
                width=scaled_width, height=scaled_height,
                color=color,
                batch=batch, group=group
            )
            debug_rect.opacity = 128  # Semi-transparent
            logger.debug(f"Created debug outline: {scaled_width}x{scaled_height} at ({sprite.x}, {sprite.y})")
            return debug_rect
        except Exception as e:
            logger.warning(f"Could not create debug outline: {e}")
            return None
        
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
        self.base_image_id = None
        
        # Panorama dimensions - set these first before creating framebuffer
        self.panorama_width = self.panorama_config.output_width
        self.panorama_height = self.panorama_config.output_height
        
        # Blur post-processing setup - always use post-processing architecture
        self.blur_enabled = self.panorama_config.blur
        logger.info(f"Panorama blur {'enabled' if self.blur_enabled else 'DISABLED'} (config.panorama.blur={self.panorama_config.blur})")
        
        # PanoramaRenderer is the parent group for post-processing (blur or no-op)
        # Create framebuffer at panorama size - mirroring happens in panorama space
        # then entire result gets scaled to screen by fullscreen quad
        fb_width = self.panorama_width
        fb_height = self.panorama_height
        self.texture = pyglet.image.Texture.create(
            fb_width, fb_height, 
            min_filter=GL_LINEAR, mag_filter=GL_LINEAR
        )
        self.framebuffer = Framebuffer()
        self.framebuffer.attach_texture(self.texture, attachment=GL_COLOR_ATTACHMENT0)
        
        logger.info(f"Created panorama framebuffer: {fb_width}x{fb_height} (will be scaled to screen {window.width}x{window.height})")
        logger.info(f"Panorama config: mirror={self.panorama_config.mirror}")
        
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
        
        # Calculate scaling and projection
        self._calculate_panorama_transform()
        
        # Visibility
        self._visible = True
        self._opacity = 1.0
        
        # Debug settings
        self.debug_tiles = False  # Flag to enable/disable tile debug outlines
        
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
            logger.debug(f"Setting up blur shader: blur_enabled={self.blur_enabled}")
            # Choose shader based on blur configuration
            vert_shader = Shader(VERTEX_SHADER_SOURCE, 'vertex')
            if self.blur_enabled:
                frag_shader = Shader(FRAGMENT_SHADER_SOURCE, 'fragment')
            else:
                frag_shader = Shader(FRAGMENT_SHADER_SOURCE_NOOP, 'fragment')
            
            self.blur_shader_program = ShaderProgram(vert_shader, frag_shader)

            # Create quad based on squeeze setting
            # if self.panorama_config.squeeze:
            #     # “squeezed”: quad spans the whole window in NDC
            #     positions = (-1,-1,  1,-1, -1,1,  1,1)
            # else:
            #     # frame-buffer-sized quad (no squeeze)
            #     w, h = self.panorama_width / self.window.width, self.panorama_height / self.window.height
            #     positions = (-w, -h,  w, -h, -w,  h,  w,  h)
            #positions = (-1,-1,  1,-1, -1,1,  1,1)
            positions = (-1.0, -1.0, 0.0, 1.0,   # bl
             1.0, -1.0, 0.0, 1.0,   # br
            -1.0,  1.0, 0.0, 1.0,   # tl
             1.0,  1.0, 0.0, 1.0)   # tr
            
            self.blur_quad = self.blur_shader_program.vertex_list(
                4, gl.GL_TRIANGLE_STRIP,
                position=('f', positions),
                tex_coords=('f', (0, 0,  1, 0,  0, 1,  1, 1))  # Fixed Y coordinates for proper text orientation
            )
            
            logger.debug("Blur shader setup complete")
            
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
        # New panorama-space architecture: No scaling needed
        # All sprites work in panorama coordinates, shader handles screen scaling
        self.panorama_scale = self._calculate_panorama_scale()
        
        # For debugging only - track what screen dimensions would be
        window_width = self.window.width
        window_height = self.window.height
        
        logger.info(f"Panorama space: {self.panorama_width}x{self.panorama_height} → screen: {window_width}x{window_height}")
        
        # Warn if panorama is much larger than window (common in dev)
        if self.panorama_scale < 0.5:
            logger.info(f"Panorama is significantly larger than window (scale={self.panorama_scale:.2f}). "
                       f"This is normal in dev mode testing production-sized panoramas.")
    
    def _calculate_panorama_scale(self) -> float:
        """Calculate scale factor for panorama space positioning."""
        return 1.0  # No scaling - work in panorama space directly
    
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
        """Set layer opacity for base images (tiles manage their own opacity)."""
        self._opacity = max(0.0, min(1.0, value))
        base_opacity = int(255 * self._opacity)
        
        # Only update base image sprite - no mirror sprites with shader approach
        if self.base_sprite:
            self.base_sprite.opacity = base_opacity
    
    async def handle_display_media(self, message: MessageDataType) -> None:
        """Handle DisplayMedia message for panorama content."""
        try:
            # Convert message to dict if needed (like in image_renderer)
            if isinstance(message, MessageBase):
                message = message.model_dump()
                
            content_type = message.get('content_type')
            request_id = message.get('request_id', 'unknown')
            position = message.get('position', None)
            
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
        
        # Don't transform the image - use sprite scaling instead
        # This is more reliable and handles mirroring better
        
        # Set image anchor point to bottom-left (pyglet default) for consistent positioning
        # This ensures (0,0) places the bottom-left corner of the image at (0,0)
        image.anchor_x = 0
        image.anchor_y = 0  # Bottom of image at coordinate y

        # Create sprite at panorama origin (no screen scaling here)
        # Sprites work in panorama space, shader handles all screen scaling
        sprite = pyglet.sprite.Sprite(image, x=0, y=0, 
                                      batch=self.batch, group=self.capture, z=0)
        
        # Set initial group opacity for fade-in effect (affects entire panorama)
        if fade_in_duration is not None and fade_in_duration > 0:
            self.opacity = 0.0  # Start transparent for fade-in
        else:
            self.opacity = 1.0  # Full opacity immediately
        
        # Calculate target dimensions for the base image
        if self.panorama_config.mirror:
            target_width = self.panorama_width / 2  # Half of framebuffer width
            target_height = self.panorama_height
            logger.info(f"Mirroring mode: scaling base image to fill left half ({target_width}x{target_height})")
        else:
            target_width = self.panorama_width  # Full framebuffer width
            target_height = self.panorama_height
            logger.info(f"No mirroring: scaling base image to full framebuffer ({target_width}x{target_height})")
        
        # Calculate scaling to target panorama dimensions
        scale_x = target_width / image.width
        scale_y = target_height / image.height
        
        # Apply rescale mode to determine which scale to use
        if self.panorama_config.rescale == "width":
            final_scale = scale_x
        elif self.panorama_config.rescale == "height":
            final_scale = scale_y
        else:  # "shortest" - maintain aspect ratio
            final_scale = min(scale_x, scale_y)
        
        # Apply the scaling (sprites work in panorama space)
        sprite.scale = final_scale
        
        logger.info(f"Base image: {image.width}x{image.height} → {target_width:.0f}x{target_height:.0f} "
                   f"(scale: {final_scale:.3f})")
        
        self.base_image = image
        self.base_sprite = sprite
        self.base_image_id = request_id
        
        # Start opacity fade-in if specified
        if fade_in_duration is not None and fade_in_duration > 0:
            self._start_base_opacity_fade(fade_in_duration)
        
        logger.info(f"Base image setup complete at ({sprite.x}, {sprite.y}) with scale {sprite.scale}")
        if self.panorama_config.mirror:
            logger.info("Shader-based mirroring enabled")
        
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
        
        # Tile target dimensions - use size from message if provided, otherwise use config
        message_size = message.get('size')
        if message_size and isinstance(message_size, (list, tuple)) and len(message_size) == 2:
            target_tile_width, target_tile_height = message_size
            logger.debug(f"Using message-provided tile size: {target_tile_width}x{target_tile_height}")
        else:
            # Fallback to config-based dimensions
            target_tile_width = tile_config.width
            target_tile_height = tile_config.height
            logger.debug(f"Using config tile size: {target_tile_width}x{target_tile_height}")
        
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
        
        # Set image anchor point to bottom-left for consistent positioning (same as base image)
        # This ensures (tile_x, tile_y) places the bottom-left corner of the image at that coordinate
        image.anchor_x = 0
        image.anchor_y = 0  # Bottom of image at coordinate y
        
        # Create tile sprite in panorama space (no screen coordinate conversion)
        # Position tiles directly in panorama coordinates - shader handles all scaling
        sprite = pyglet.sprite.Sprite(image, x=tile_x_adjusted, y=tile_y_adjusted, 
                                      batch=self.batch, group=self.capture, z=1)
        
        # Set initial opacity for tile fade-in effect (individual tile control)
        if fade_in_duration is not None and fade_in_duration > 0:
            sprite.opacity = 0  # Start transparent for fade-in
        else:
            sprite.opacity = 255  # Full opacity immediately
        
        # Apply tile scaling (tiles work in panorama space)
        sprite.scale = scale_factor
        
        logger.info(f"Tile {request_id}: {image.width}x{image.height} → {target_tile_width}x{target_tile_height} "
                   f"at ({tile_x_adjusted}, {tile_y_adjusted}) scale={scale_factor:.3f}"
                   f" scaled_size=({image.width * scale_factor:.1f}x{image.height * scale_factor:.1f})")
        
        # Remove old tiles if they exist
        # Generate unique ID if request_id is None to avoid collisions
        tile_id = request_id if request_id is not None else f"tile_{tile_x}_{tile_y}"
        
        if tile_id in self.tiles:
            old_tile = self.tiles[tile_id]
            old_tile.cleanup()
            if tile_id in self.tile_order:
                self.tile_order.remove(tile_id)
        
        # Create original tile - use message fade_in for custom duration or config default
        debug_rect = None
        if self.debug_tiles:
            debug_rect = PanoramaTile.create_debug_outline(sprite, self.batch, self.display)
            
        if fade_in_duration is not None and fade_in_duration > 0:
            # Use custom fade duration from message
            tile = PanoramaTile(sprite, (tile_x, tile_y), tile_id, 
                               original_size=(image.width, image.height),
                               fade_duration=fade_in_duration,
                               debug_rect=debug_rect)
        else:
            # Use config fade duration for built-in animation
            tile = PanoramaTile(sprite, (tile_x, tile_y), tile_id, 
                               original_size=(image.width, image.height),
                               fade_duration=tile_config.fade_duration,
                               debug_rect=debug_rect)
            
        self.tiles[tile_id] = tile
        self.tile_order.append(tile_id)
    
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
            if progress >= 1.0:
                self.blur_active = False
                logger.debug("Blur transition complete")
        
        # Update tile fades (only if not clearing)
        if not self.is_clearing:
            for tile in self.tiles.values():
                tile.update(dt)
    
    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug outlines for tiles."""
        self.debug_tiles = enabled
        logger.info(f"Tile debug mode {'enabled' if enabled else 'disabled'}")
        
        # Update existing tiles
        for tile in self.tiles.values():
            if enabled and not tile.debug_rect:
                # Create debug outline for existing tile
                tile.debug_rect = PanoramaTile.create_debug_outline(tile.sprite, self.batch, self.display)
            elif not enabled and tile.debug_rect:
                # Remove debug outline from existing tile
                tile.debug_rect.delete()
                tile.debug_rect = None
    
    def set_visibility(self, visible: bool) -> None:
        """Set renderer visibility."""
        self._visible = visible
        
        # Update sprite visibility
        if self.base_sprite:
            self.base_sprite.visible = visible
            
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
            "mirror": self.panorama_config.mirror,
            "debug_tiles": self.debug_tiles
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
        self.clear_duration = fade_duration or 1.0
        
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
        
        self.base_image = None
        self.base_image_id = None
        
        # Remove all tile sprites
        for tile in self.tiles.values():
            # Use the tile's cleanup method which handles both sprite and debug_rect
            tile.cleanup()
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
        
        # Recreate resources for new size - keep framebuffer at panorama size
        fb_width = self.panorama_width
        fb_height = self.panorama_height
        self.texture = pyglet.image.Texture.create(
            fb_width, fb_height, 
            min_filter=GL_LINEAR, mag_filter=GL_LINEAR
        )
        self.framebuffer = Framebuffer()
        self.framebuffer.attach_texture(self.texture, attachment=GL_COLOR_ATTACHMENT0)
        
        # Recreate shader with correct quad size for new window dimensions
        self._setup_blur_shader()
        
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
        
        # Update existing sprite groups (positions don't change - they're in panorama space)
        if self.base_sprite:
            # Update base sprite to use new capture group (position stays the same)
            old_sprite = self.base_sprite
            self.base_sprite = pyglet.sprite.Sprite(
                old_sprite.image, x=0, y=0,
                batch=self.batch, group=self.capture, z=0
            )
            self.base_sprite.scale = old_sprite.scale  # Keep panorama scale
            old_sprite.delete()
        
        # Update tile groups (positions don't change - they're in panorama space)
        for tile in self.tiles.values():
            # Recreate tile sprite with new capture group (position stays the same)
            old_sprite = tile.sprite
            tile.sprite = pyglet.sprite.Sprite(
                old_sprite.image, x=old_sprite.x, y=old_sprite.y,
                batch=self.batch, group=self.capture, z=1
            )
            tile.sprite.scale = old_sprite.scale  # Keep panorama scale
            old_sprite.delete()
        
        logger.debug(f"Panorama renderer resize complete: {old_width}x{old_height} → {new_size}")
