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
"""

import logging
import asyncio
import time
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
from pyglet.image.buffer import Framebuffer
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
uniform float     panorama_visibility;  // Global panorama visibility (1.0 = full color, 0.0 = black)

vec4 fast_blur(vec2 tc) {
    if (blur_sigma < 0.5)               // ~no-blur fast-path
        return texture(scene_texture, tc);

    vec2 texel = 1.0 / vec2(textureSize(scene_texture, 0));
    
    // Extreme blur radius for dramatic "clouds of color" effect - up to 49x49 kernel
    int r = int(min(24.0, ceil(blur_sigma * 0.8)));
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
    
    // Apply global panorama fade by darkening colors (instead of alpha)
    // panorama_visibility: 1.0 = full color, 0.0 = black
    frag = vec4(blurred.rgb * panorama_visibility, blurred.a);
}
"""

# Optimized fragment shader for better performance
FRAGMENT_SHADER_SOURCE_NOOP = """#version 150 core
in vec2 v_tex;
out vec4 frag;

uniform sampler2D scene_texture;
uniform float blur_sigma;
uniform int enable_mirror;
uniform float panorama_visibility;  // Global panorama visibility (1.0 = full color, 0.0 = black)

void main()
{
    vec2 tc = v_tex;

    // Handle mirroring if enabled
    if (enable_mirror == 1 && tc.x > 0.5) {
        tc.x = 1.0 - tc.x;
    }
    
    vec4 color;
    // Handle blur if enabled - use much more efficient implementation
    if (blur_sigma < 0.5) {
        // No blur - direct sampling
        color = texture(scene_texture, tc);
    } else {
        // Enhanced box blur for extreme "clouds of color" effect - up to 49x49 kernel
        vec2 texel = 1.0 / textureSize(scene_texture, 0);
        int r = int(min(24.0, ceil(blur_sigma * 0.8)));
        
        vec3 result = vec3(0.0);
        int samples = 0;
        
        for (int x = -r; x <= r; x++) {
            for (int y = -r; y <= r; y++) {
                vec2 offset = vec2(float(x), float(y)) * texel;
                result += texture(scene_texture, tc + offset).rgb;
                samples++;
            }
        }
        
        color = vec4(result / float(samples), 1.0);
    }
    
    // Apply global panorama visibility by darkening colors (instead of alpha)
    // panorama_visibility: 1.0 = full color, 0.0 = black
    frag = vec4(color.rgb * panorama_visibility, color.a);
}
"""

# the max blur for a current blur -> HUMP % of start blur -> 0 transition
HUMP_BLUR_PERCENTAGE = 0.5  # 0.5 = 50% of start_blur, can be adjusted

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
                
                # Set panorama-wide visibility for fade effect
                renderer.blur_shader_program['panorama_visibility'] = float(renderer.panorama_visibility)
                
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
        
        # Fade in state
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

    @staticmethod 
    def create_debug_outline_from_dimensions(x: float, y: float, width: float, height: float, 
                                           batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, 
                                           color: Tuple[int, int, int] = (255, 0, 0)) -> Optional[pyglet.shapes.Rectangle]:
        """Create a debug outline rectangle using explicit dimensions.
        
        Args:
            x, y: Position coordinates
            width, height: Exact dimensions to use
            batch: Pyglet batch to add the rectangle to
            group: Pyglet group for rendering order 
            color: RGB color for the outline (default: red)
            
        Returns:
            The created rectangle shape, or None if creation failed
        """
        try:
            from pyglet import shapes
            
            logger.debug(f"Creating debug outline from dimensions: input=({x}, {y}) {width}x{height}")
            
            # Create a sub-group with higher z-order to ensure debug rects render on top
            class DebugGroup(pyglet.graphics.Group):
                def __init__(self, parent_group):
                    super().__init__(order=10, parent=parent_group)  # High order = render on top
            
            debug_group = DebugGroup(group)
            
            debug_rect = shapes.Rectangle(
                x=x, y=y,
                width=width, height=height,
                color=color,
                batch=batch, group=debug_group
            )
            # Start with opacity 0 - will be set by calling code based on debug mode
            debug_rect.opacity = 0
            logger.debug(f"Created debug outline from dimensions: {width}x{height} at ({x}, {y}), initial_opacity={debug_rect.opacity}")
            return debug_rect
        except Exception as e:
            logger.warning(f"Could not create debug outline from dimensions: {e}")
            return None
        
    def update(self, dt: float) -> None:
        """Update tile fade in animation."""
        if self.is_fading:
            # Handle fade-in animation
            self.fade_timer += dt
            progress = min(self.fade_timer / self.fade_duration, 1.0)
            
            # Smooth fade in
            self.sprite.opacity = int(255 * progress)
            
            if progress >= 1.0:
                self.is_fading = False
                logger.debug(f"Tile {self.tile_id} fade-in complete")


class PanoramaRenderer(LayerRenderer):
    """Renders panoramic images with base image + positioned tiles."""
    
    def __init__(self, config: DisplayServiceConfig, 
                 window: pyglet.window.BaseWindow, 
                 batch: pyglet.graphics.Batch,
                 order: int = 0):
        """Initialize the panorama renderer."""
        super().__init__(config=config, window=window, batch=batch, order=order)
        
        self.panorama_config = config.panorama
        
        # Base image state - support multiple sprites for smooth crossfade
        self.base_image = None  # Keep reference to current/latest image
        self.base_sprites = {}  # Dict of image_id -> {'sprite': sprite, 'fade_timer': float, 'fade_duration': float, 'is_fading': bool}
        self.base_image_id = None  # ID of the current/target base image
        
        # Panorama dimensions - set these first before creating framebuffer
        self.panorama_width = self.panorama_config.output_width
        self.panorama_height = self.panorama_config.output_height
        
        # Blur post-processing setup - always use post-processing architecture
        self.blur_enabled = self.panorama_config.blur
        self.blur_velocity = 1.0 # allows for the blur to be sped up or slowed down due to state/messages
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
        self.blur_delay_timer = 0.0  # Timer for delaying blur start during crossfade
        self.blur_active = False  # Only start when base image is received
        self.blur_easing_mode = "decrease"  # "decrease" for black->image, "hump" for image->image
        self.blur_velocity = 1.0  # Current velocity multiplier
        self.blur_target_velocity = 1.0  # Target velocity for smooth acceleration
        self.blur_velocity_ramp_speed = 3.0  # How fast velocity changes (per second)
        self.previous_blur_sigma = 0.0  # For jump detection
        self.previous_blur_sigma = 0.0  # Track previous frame's blur for jump detection
        self.current_blur = self.panorama_config.start_blur
        self.current_blur_sigma = self.panorama_config.start_blur
        
        # Base image opacity fade-in - now handles multiple sprites
        self.base_fade_timer = 0.0
        self.base_fade_duration = 0.0
        self.base_fading = False
        
        # Tiles
        self.tiles: Dict[str, PanoramaTile] = {}  # tile_id -> PanoramaTile
        self.tile_order: List[str] = []  # Track tile addition order
        
        # Calculate scaling and projection
        self._calculate_panorama_transform()
        
        # Visibility
        self._visible = True
        self._opacity = 1.0
        
        # Panorama-wide fade system
        self.panorama_visibility = 1.0  # Global visibility for entire panorama (0.0 = all black, 1.0 = full brightness)
        self.panorama_disappear_timer = 0.0
        self.panorama_disappear_duration = self.panorama_config.disappear_duration
        self.panorama_is_disappearing = False
        
        # Panorama reappear system (for new images after panorama has faded)
        self.panorama_is_reappearing = False
        self.panorama_reappear_timer = 0.0
        self.panorama_reappear_duration = 0.0
        self.panorama_start_visibility = 1.0  # Starting visibility for reappear fade

        # Debug settings
        self.debug_tiles = False  # Flag to enable/disable tile debug outlines visibility
        self.hide_tiles_for_debug = False  # Flag to temporarily hide tiles (shows only base image)
        
        # Create singleton black image for clears - calculate appropriate size
        if self.panorama_config.mirror:
            black_width = self.panorama_width // 2  # Half width for mirroring
            black_height = self.panorama_height
        else:
            black_width = self.panorama_width
            black_height = self.panorama_height
        
        # Create a solid black image that we can reuse
        black_data = bytearray(black_width * black_height * 4)  # RGBA
        for i in range(0, len(black_data), 4):
            black_data[i:i+4] = [0, 0, 0, 255]  # Black with full alpha
        
        self._black_image: Optional[pyglet.image.ImageData] = pyglet.image.ImageData(
            black_width, black_height, 'RGBA', bytes(black_data)
        )
        self._black_image.anchor_x = 0
        self._black_image.anchor_y = 0
        
        logger.info(f"Created singleton black image for clears: {black_width}x{black_height}")
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
                (len(self.base_sprites) > 0 or len(self.tiles) > 0))
    
    @property
    def opacity(self) -> float:
        """Get current layer opacity."""
        return self._opacity
    
    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set layer opacity for base images (tiles manage their own opacity)."""
        self._opacity = max(0.0, min(1.0, value))
        base_opacity = int(255 * self._opacity)
        
        # Update all base image sprites with global layer opacity
        for base_data in self.base_sprites.values():
            if base_data['sprite']:
                base_data['sprite'].opacity = base_opacity
    
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
                # For clear messages, prefer fade_out (semantic), fallback to fade_in, then default
                fade_duration = message.get('fade_out') or message.get('fade_in', 3.0)
                logger.info(f"Clearing panorama with {fade_duration}s fade-out duration - using black base image")
                
                # Create a black base image and use the existing crossfade system
                self._create_black_base_image(fade_duration, request_id or "clear")
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
        """Handle base image for panorama with smooth crossfade support."""
        # Generate unique ID if request_id is None to prevent collisions
        if request_id is None:
            request_id = f"base_image_{int(time.time() * 1000000)}"  # microsecond timestamp
            logger.debug(f"Generated missing base image request_id: {request_id}")
        
        logger.info(f"Setting panorama base image: {request_id}")
        
        # Extract fade_in duration from message
        fade_in_duration = message.get('fade_in')
        
        # Set image anchor point to bottom-left (pyglet default) for consistent positioning
        image.anchor_x = 0
        image.anchor_y = 0

        # Create sprite at panorama origin (no screen scaling here)
        sprite = pyglet.sprite.Sprite(image, x=0, y=0, 
                                      batch=self.batch, group=self.capture, z=0)
        
        # Calculate target dimensions for the base image
        if self.panorama_config.mirror:
            target_width = self.panorama_width / 2
            target_height = self.panorama_height
            logger.info(f"Mirroring mode: scaling base image to fill left half ({target_width}x{target_height})")
        else:
            target_width = self.panorama_width
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
        
        # Apply the scaling
        sprite.scale = final_scale
        
        logger.info(f"Base image: {image.width}x{image.height} → {target_width:.0f}x{target_height:.0f} "
                   f"(scale: {final_scale:.3f})")
        
        # Set initial opacity for fade-in
        if fade_in_duration is not None and fade_in_duration > 0:
            sprite.opacity = 0  # Start transparent for individual sprite fade-in
        else:
            sprite.opacity = 255  # Full opacity immediately
        
        # Update current base image references
        self.base_image = image
        self.base_image_id = request_id
        
        # Add sprite to the collection with fade state
        fade_duration = fade_in_duration if fade_in_duration is not None and fade_in_duration > 0 else 0.0
        self.base_sprites[request_id] = {
            'sprite': sprite,
            'fade_timer': 0.0,
            'fade_duration': fade_duration,
            'is_fading': fade_duration > 0
        }
        
        logger.info(f"Base image setup complete at ({sprite.x}, {sprite.y}) with scale {sprite.scale}")
        if fade_duration > 0:
            logger.info(f"Starting fade-in over {fade_duration}s")
        if self.panorama_config.mirror:
            logger.info("Shader-based mirroring enabled")
        
        # Start blur transition for the new base image (easing will be determined automatically)
        # Check if this is a clear/black image operation
        is_clear_operation = bool(request_id and ("clear" in request_id.lower() or request_id.startswith("clear")))
        self._start_blur_transition(is_clear_operation=is_clear_operation)
        
        # If panorama has faded to black and this isn't a clear operation, start reappear fade
        if not is_clear_operation and self.panorama_visibility < 1.0 and fade_duration > 0:
            self._start_panorama_reappear(fade_duration)
    
    def _create_black_base_image(self, fade_duration: float, request_id: str) -> None:
        """Create a black base image using the existing crossfade system for clearing."""
        logger.info(f"Creating black base image for clear with {fade_duration}s fade-in: {request_id}")
        
        # Use the singleton black image (already sized correctly)
        black_image = self._black_image
        if black_image is None:
            logger.error("Singleton black image is None, cannot create clear sprite")
            return
        
        # Create sprite using the existing base image system
        sprite = pyglet.sprite.Sprite(black_image, x=0, y=0, 
                                      batch=self.batch, group=self.capture, z=0)
        sprite.scale = 1.0  # Already at target size
        sprite.opacity = 0  # Start transparent for fade-in
        
        # Add to the existing base_sprites system - it will handle crossfade automatically
        self.base_sprites[request_id] = {
            'sprite': sprite,
            'fade_timer': 0.0,
            'fade_duration': fade_duration,
            'is_fading': True
        }
        
        # Update current references
        self.base_image = black_image
        self.base_image_id = request_id
        
        logger.info(f"Black base image created: {black_image.width}x{black_image.height} with {fade_duration}s fade-in (using singleton)")
    
    def _clear_panorama(self) -> None:
        """Clear panorama after fade completes so next image gets 'decrease' blur pattern."""
        # Log what we're about to clean up
        logger.info(f"Clear panorama: {len(self.base_sprites)} base sprites, {len(self.tiles)} tiles")

        # Clean up all base sprites
        for image_id, base_data in list(self.base_sprites.items()):
            sprite = base_data['sprite']
            if sprite:
                sprite.delete()
            del self.base_sprites[image_id]
            logger.debug(f"Cleared base sprite: {image_id}")
        
        # Clear all tiles
        for tile_id, tile in list(self.tiles.items()):
            tile.cleanup()
            del self.tiles[tile_id]
            logger.debug(f"Cleared tile: {tile_id}")
        
        # Clear tile order tracking
        if hasattr(self, 'tile_order'):
            self.tile_order.clear()
        
        # Reset references
        self.base_image = None
        self.base_image_id = None
        
        # Reset panorama state so it's ready for next panorama
        self.panorama_visibility = 1.0
    
    def _handle_tile(self, image: pyglet.image.AbstractImage, request_id: str, 
                    position: Tuple[int, int] | str, message: Dict[str, Any]) -> None:
        """Handle tile image for panorama."""
        if isinstance(position, str):
            logger.error(f"String positions not yet supported for panorama tiles: {position}")
            return
            
        tile_x, tile_y = position
        
        # Generate unique ID if request_id is None to prevent collisions
        if request_id is None:
            request_id = f"tile_{tile_x}_{tile_y}_{int(time.time() * 1000000)}"  # microsecond timestamp
            logger.debug(f"Generated missing tile request_id: {request_id}")
            
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
        # Always create debug rectangle, but control its visibility
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        debug_rect = PanoramaTile.create_debug_outline_from_dimensions(
            tile_x_adjusted, tile_y_adjusted, 
            target_tile_width, target_tile_height,
            self.batch, self.capture,  # Use same group as tile but higher z-order
            color=colors[len(self.tile_order) % len(colors)]  # Cycle through colors
        )
        if debug_rect:
            # Set initial opacity based on debug_tiles flag (opacity 0 = invisible, >0 = visible)
            debug_rect.opacity = 128 if self.debug_tiles else 0
            logger.debug(f"Created debug rect for new tile {tile_id}: {debug_rect.width}x{debug_rect.height} at ({debug_rect.x}, {debug_rect.y}), target_size={target_tile_width}x{target_tile_height}, opacity={debug_rect.opacity}")
        else:
            logger.warning(f"Failed to create debug rect for new tile {tile_id}")
            
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
        
        # Apply debug visibility state to new tile
        if self.hide_tiles_for_debug:
            sprite.visible = False
            if debug_rect:
                debug_rect.opacity = 0  # Hide debug rect when tiles are hidden for debug
            
        self.tiles[tile_id] = tile
        self.tile_order.append(tile_id)
        
        # Check if this is the final tile and accelerate blur if needed
        is_final_tile = message.get('final_tile', False)
        logger.debug(f"Tile {tile_id}: final_tile={is_final_tile}")
        if is_final_tile:
            logger.info(f"Final tile received: {tile_id}")
            if self.blur_active:
                self._accelerate_blur_transition()
                logger.info(f"Accelerating blur transition to complete within 1 second")
            else:
                logger.info(f"Blur transition not active, no acceleration needed")
    
    def _start_blur_transition(self, is_clear_operation: bool = False) -> None:
        """Start the blur transition for the base image.
        
        Args:
            is_clear_operation: True if this is a clear/black image operation that should ramp to max blur
        """
        if not self.blur_enabled:
            return
        
        # Always use config blur_duration for blur transition (independent of fade_in)
        effective_blur_duration = self.panorama_config.blur_duration
        
        if not effective_blur_duration:
            return
        
        # CRITICAL FIX: Always start from current visual state to prevent jumps
        current_visual_blur = getattr(self, 'current_blur_sigma', 0.0)
        
        self.blur_timer = 0.0
        self.blur_active = True
        self.blur_velocity = 1.0  # Reset to normal speed
        self.blur_target_velocity = 1.0  # Reset target speed
        
        # Determine blur pattern based on transition context
        # Special case: Clear operations should always ramp TO maximum blur (fade to black through blur)
        if is_clear_operation:
            self.blur_easing_mode = "increase"  # current → max for fade-to-black effect
            self.current_blur = current_visual_blur
            self.current_blur_sigma = current_visual_blur
            pattern_desc = f"increasing ({current_visual_blur:.1f} → {self.panorama_config.start_blur}) for clear"
        # If we have no existing content (base_sprites empty), this is black->image
        # If we have existing content, this is image->image crossfade
        elif len(self.base_sprites) <= 1:  # No existing content or just the new one
            self.blur_easing_mode = "decrease"  # max → 0 for black->image
            # Start from max blur and go to end blur (usually 0)
            # BUT: If we already have some blur active, start from current to avoid jump
            if current_visual_blur > 0:
                self.current_blur = current_visual_blur
                self.current_blur_sigma = current_visual_blur
                logger.debug(f"Continuing from existing blur {current_visual_blur:.1f} instead of jumping to {self.panorama_config.start_blur}")
            else:
                self.current_blur = self.panorama_config.start_blur  # Should be max (e.g., 20)
                self.current_blur_sigma = self.panorama_config.start_blur
            pattern_desc = f"decreasing ({self.current_blur:.1f} → {self.panorama_config.end_blur})"
        else:
            self.blur_easing_mode = "hump"  # current → max → 0 for image->image crossfade
            # ALWAYS start from current visual blur value to prevent jumps
            self.current_blur = current_visual_blur
            self.current_blur_sigma = current_visual_blur
            
            # Store the starting blur for the hump pattern
            self._hump_start_blur = current_visual_blur
            
            # Use 50% of configured start_blur for more subtle crossfade hump (adjustable)
            max_blur = self.panorama_config.start_blur * HUMP_BLUR_PERCENTAGE
            pattern_desc = f"hump ({current_visual_blur:.1f} → {max_blur:.1f} → 0)"
        
        # Store the effective duration and blur values for hump pattern
        self._effective_blur_duration = effective_blur_duration
        if self.blur_easing_mode == "hump":
            self._max_blur_for_hump = self.panorama_config.start_blur * HUMP_BLUR_PERCENTAGE
        
        # Now check if we're creating a jump (should be minimal with new approach)
        blur_change = abs(self.current_blur_sigma - self.previous_blur_sigma)
        if blur_change > 1.0:  # Reduced threshold since we should have minimal jumps now
            logger.warning(f"Minor blur discontinuity at transition start: {self.previous_blur_sigma:.2f} → {self.current_blur_sigma:.2f} "
                         f"(change: {blur_change:.2f}) mode={self.blur_easing_mode}")
        
        # Update previous for next frame
        self.previous_blur_sigma = self.current_blur_sigma
        
        # Update blur effect with initial parameters
        self._update_blur_effect()
        
        logger.debug(f"Starting blur transition over {effective_blur_duration}s using {pattern_desc}")
    
    def _accelerate_blur_transition(self) -> None:
        """Accelerate the blur transition by smoothly ramping up velocity."""
        if not self.blur_active:
            return

        self.blur_target_velocity = 3.0  # Set target, will ramp up smoothly
        logger.debug(f"Blur acceleration triggered: ramping velocity to {self.blur_target_velocity}x")
    
    def start_panorama_fade(self) -> None:
        """Start the panorama-wide fade effect if enabled."""
        if self.panorama_disappear_duration <= 0:
            return
            
        self.panorama_disappear_timer = 0.0
        self.panorama_is_disappearing = True
        self.panorama_visibility = 1.0  # Start fully opaque
        logger.debug(f"✅ PANORAMA FADE STARTED: {self.panorama_disappear_duration}s duration, "
                   f"opacity={self.panorama_visibility}, is_fading={self.panorama_is_disappearing}")

    def _start_panorama_reappear(self, fade_duration: float) -> None:
        """Start panorama reappear fade from current opacity to full brightness."""
        self.panorama_reappear_timer = 0.0
        self.panorama_is_reappearing = True
        self.panorama_reappear_duration = fade_duration
        self.panorama_start_visibility = self.panorama_visibility  # Remember where we started
        
        # Stop any existing disappear fade
        self.panorama_is_disappearing = False
        
        logger.debug(f"✅ PANORAMA REAPPEAR STARTED: {fade_duration}s duration, "
                   f"from opacity={self.panorama_start_visibility:.2f} to 1.0")

    def update(self, dt: float) -> None:
        """Update panorama animations with simple crossfade support.
        
        Strategy: Let sprites accumulate during rapid crossfades, then clean up ALL old sprites
        the moment the newest one finishes fading. Simple and bulletproof.
        """
        sprites_to_remove = []
        
        # Update fade animations and clean up immediately when new sprite finishes
        for image_id, base_data in list(self.base_sprites.items()):
            sprite = base_data['sprite']
            if not sprite:
                continue
                
            # Update fade-in animation for this sprite
            if base_data['is_fading']:
                base_data['fade_timer'] += dt
                progress = min(base_data['fade_timer'] / base_data['fade_duration'], 1.0)
                
                # Fade in this specific sprite
                sprite.opacity = int(255 * progress)
                
                if progress >= 1.0:
                    base_data['is_fading'] = False
                    logger.debug(f"Base image {image_id} fade complete")
                    
                    # If this is the current sprite, clean up all old sprites immediately
                    if image_id == self.base_image_id:
                        logger.debug(f"Current sprite {image_id} fully faded - cleaning up old sprites and tiles")
                        # Mark all other sprites for removal and clean up their tiles
                        for other_id in self.base_sprites.keys():
                            if other_id != image_id:
                                sprites_to_remove.append(other_id)
                                
                                # Clean up tiles belonging to this old base sprite
                                old_tiles = self.get_tiles_for_base_sprite(other_id)
                                if old_tiles:
                                    logger.debug(f"Clearing {len(old_tiles)} tiles for old base sprite {other_id}")
                                    for tile_id, tile in old_tiles.items():
                                        tile.cleanup()
                                        del self.tiles[tile_id]
                                        # Remove from tile order tracking
                                        if hasattr(self, 'tile_order') and tile_id in self.tile_order:
                                            self.tile_order.remove(tile_id)
                                        logger.debug(f"Cleared tile {tile_id} for old base sprite {other_id}")        # Clean up old sprites
        
        for image_id in sprites_to_remove:
            if image_id in self.base_sprites:
                sprite = self.base_sprites[image_id]['sprite']
                if sprite:
                    sprite.delete()
                del self.base_sprites[image_id]
                logger.debug(f"Cleaned up old base sprite: {image_id}")
        
        # Update blur transition (only if blur enabled and we have content)
        if (self.blur_enabled and self.blur_active and len(self.base_sprites) > 0):
            # Smoothly ramp blur velocity towards target
            if self.blur_velocity != self.blur_target_velocity:
                velocity_diff = self.blur_target_velocity - self.blur_velocity
                max_change = self.blur_velocity_ramp_speed * dt
                if abs(velocity_diff) <= max_change:
                    self.blur_velocity = self.blur_target_velocity
                else:
                    self.blur_velocity += max_change if velocity_diff > 0 else -max_change
                    
            self.blur_timer += dt * self.blur_velocity

            linear_progress = min(self.blur_timer / self.panorama_config.blur_duration, 1.0)
            
            # Apply appropriate blur pattern based on transition context
            if self.blur_easing_mode == "decrease":
                # Decreasing pattern: max → 0 (black → image)
                # Use ease-out for smooth deceleration
                eased_progress = 1 - (1 - linear_progress) ** 2  # Gentler ease-out
                self.current_blur = self.panorama_config.start_blur * (1 - eased_progress)
                pattern_desc = f"decrease({eased_progress:.3f})"
            elif self.blur_easing_mode == "increase":
                # Increasing pattern: current → max (fade to black through blur)
                # Use ease-in for smooth acceleration to maximum blur
                eased_progress = linear_progress ** 2  # Ease-in
                start_blur = getattr(self, 'current_blur', 0.0)
                self.current_blur = start_blur + (self.panorama_config.start_blur - start_blur) * eased_progress
                pattern_desc = f"increase({eased_progress:.3f})"
            else:  # hump
                # Hump pattern: start_blur → max → 0 (image → image crossfade)
                # Use modified sine wave that starts at current blur, peaks at max, ends at 0
                start_blur = getattr(self, '_hump_start_blur', 0.0)
                max_blur_for_hump = getattr(self, '_max_blur_for_hump', 10.0)
                
                # Create a curve that goes: start_blur → max_blur → 0
                # Use sine wave but offset and scaled to hit our target points
                sine_progress = math.sin(linear_progress * math.pi)  # 0 to 1 to 0
                
                # Interpolate between start_blur and 0, with peak at max_blur
                if linear_progress <= 0.5:
                    # First half: start_blur → max_blur
                    self.current_blur = start_blur + (max_blur_for_hump - start_blur) * sine_progress
                else:
                    # Second half: max_blur → 0
                    self.current_blur = max_blur_for_hump * sine_progress
                
                pattern_desc = f"hump({sine_progress:.3f}, {start_blur:.1f}→{max_blur_for_hump:.1f}→0)"
            
            # Update the current blur sigma for the shader
            self.current_blur_sigma = self.current_blur
            
            # BLUR JUMP DETECTION - Log sudden changes
            blur_change = abs(self.current_blur_sigma - self.previous_blur_sigma)
            if blur_change > 2.0 and self.previous_blur_sigma > 0:  # Threshold for "sudden jump"
                logger.warning(f"🚨 BLUR JUMP DETECTED! Previous: {self.previous_blur_sigma:.2f} → Current: {self.current_blur_sigma:.2f} "
                             f"(change: {blur_change:.2f}) at t={self.blur_timer:.2f}s vel={self.blur_velocity:.2f}→{self.blur_target_velocity:.2f} "
                             f"mode={self.blur_easing_mode} progress={linear_progress:.3f}")
            
            # Store current blur for next frame's jump detection
            self.previous_blur_sigma = self.current_blur_sigma
            
            # Debug logging every 0.5 seconds to show smooth progression
            # if int(self.blur_timer * 2) % 1 == 0 and self.blur_timer > 0:
            #     velocity_info = f"vel={self.blur_velocity:.2f}" + (f"→{self.blur_target_velocity:.1f}" if self.blur_velocity != self.blur_target_velocity else "")
            #     logger.debug(f"Blur transition: t={self.blur_timer:.1f}s {velocity_info} "
            #                f"linear={linear_progress:.3f} {pattern_desc} sigma={self.current_blur:.1f}")
            
            # Update the current blur sigma for the shader
            self.current_blur_sigma = self.current_blur
            self._update_blur_effect()
            if linear_progress >= 1.0:
                self.blur_active = False
                self.blur_velocity = 1.0
                self.blur_target_velocity = 1.0  # Reset target velocity too
                # final image now slowly darkens until being removed
                self.start_panorama_fade()
        
        # Update tile fades
        for tile in self.tiles.values():
            tile.update(dt)
        
        # TODO: Add smarter tile cleanup that doesn't interfere with crossfades
        # For now, only clean up tiles during the full panorama clear
        
        # Update panorama-wide fade (if enabled and active)
        if self.panorama_is_disappearing and self.panorama_disappear_duration > 0:
            self.panorama_disappear_timer += dt
            progress = min(self.panorama_disappear_timer / self.panorama_disappear_duration, 1.0)
            
            # Fade from 1.0 (opaque) to 0.0 (transparent)
            self.panorama_visibility = 1.0 - progress

            if progress >= 1.0:
                self.panorama_is_disappearing = False
                self.panorama_visibility = 0.0
                # Clear the panorama so next image gets "decrease" pattern instead of "hump"
                self._clear_panorama()
        
        # Update panorama reappear fade (bringing opacity back to 1.0 with new content)
        if self.panorama_is_reappearing and self.panorama_reappear_duration > 0:
            self.panorama_reappear_timer += dt
            progress = min(self.panorama_reappear_timer / self.panorama_reappear_duration, 1.0)
            
            # Fade from start_opacity to 1.0 (full brightness)
            self.panorama_visibility = self.panorama_start_visibility + (1.0 - self.panorama_start_visibility) * progress

            if progress >= 1.0:
                self.panorama_is_reappearing = False
                self.panorama_visibility = 1.0
                logger.debug(f"✅ PANORAMA REAPPEAR COMPLETE: opacity now {self.panorama_visibility}")
    
    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug outlines for tiles."""
        self.debug_tiles = enabled
        logger.info(f"Tile debug mode {'enabled' if enabled else 'disabled'}")
        
        # Update opacity of existing debug rectangles instead of visibility
        target_opacity = 128 if enabled else 0  # Semi-transparent when visible, invisible when disabled
        for tile_id, tile in self.tiles.items():
            if tile.debug_rect:
                tile.debug_rect.opacity = target_opacity
                logger.debug(f"Set debug rect opacity for tile {tile_id}: {target_opacity} (size: {tile.debug_rect.width}x{tile.debug_rect.height})")
            else:
                logger.warning(f"Tile {tile_id} has no debug rect to toggle")
    
    def set_tiles_hidden_for_debug(self, hidden: bool) -> None:
        """Hide or show tiles for debug purposes (allows seeing just the base image)."""
        self.hide_tiles_for_debug = hidden
        logger.info(f"Tiles {'hidden' if hidden else 'shown'} for debug mode")
        
        # Update tile visibility immediately
        for tile in self.tiles.values():
            tile.sprite.visible = not hidden and self._visible
            # Also hide debug rectangles when hiding tiles
            if tile.debug_rect:
                # Use opacity to control debug rect visibility: 0 if hidden, 128 if visible and debug enabled
                should_show = (not hidden and self._visible and self.debug_tiles)
                tile.debug_rect.opacity = 128 if should_show else 0
    
    def is_tiles_hidden_for_debug(self) -> bool:
        """Check if tiles are currently hidden for debug purposes."""
        return self.hide_tiles_for_debug
    
    def get_tiles_for_base_sprite(self, base_sprite_id: str) -> Dict[str, 'PanoramaTile']:
        """Get all tiles that belong to a specific base sprite/panorama.
        
        Args:
            base_sprite_id: The base request ID or image ID to find tiles for
            
        Returns:
            Dictionary of tile_id -> PanoramaTile for tiles belonging to this base sprite
        """
        matching_tiles = {}
        
        # Look for tiles whose IDs start with the base_sprite_id
        for tile_id, tile in self.tiles.items():
            # Handle both direct matches and tile IDs that start with base_sprite_id + "_tile"
            if (tile_id == base_sprite_id or 
                tile_id.startswith(f"{base_sprite_id}_tile")):
                matching_tiles[tile_id] = tile
        
        return matching_tiles
    
    def set_visibility(self, visible: bool) -> None:
        """Set renderer visibility."""
        self._visible = visible
        
        # Update all base sprite visibility
        for base_data in self.base_sprites.values():
            if base_data['sprite']:
                base_data['sprite'].visible = visible
            
        for tile in self.tiles.values():
            # Respect both overall visibility and debug hide state
            tile.sprite.visible = visible and not self.hide_tiles_for_debug
            # Also update debug rectangles if they exist
            if tile.debug_rect:
                # Use opacity to control debug rect visibility: consider overall visibility, debug hide state, and debug enabled
                should_show = (visible and not self.hide_tiles_for_debug and self.debug_tiles)
                tile.debug_rect.opacity = 128 if should_show else 0
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the panorama state."""
        # Get info about base sprites
        base_sprite_info = {}
        for image_id, base_data in self.base_sprites.items():
            base_sprite_info[image_id] = {
                'is_fading': base_data['is_fading'],
                'fade_timer': base_data['fade_timer'],
                'fade_duration': base_data['fade_duration'],
                'opacity': base_data['sprite'].opacity if base_data['sprite'] else 0
            }
        
        # Get info about tile fade states
        tile_info = {}
        for tile_id, tile in self.tiles.items():
            tile_info[tile_id] = {
                'is_fading_in': tile.is_fading,
                'fade_in_progress': tile.fade_timer / tile.fade_duration if tile.is_fading else 1.0,
                'opacity': tile.sprite.opacity,
                'position': tile.position
            }
        
        return {
            "panorama_enabled": self.panorama_config.enabled,
            "base_image_id": self.base_image_id,
            "base_sprites_count": len(self.base_sprites),
            "base_sprites": base_sprite_info,
            "tiles_count": len(self.tiles),
            "tile_ids": list(self.tiles.keys()),
            "tile_info": tile_info,
            "panorama_visibility": self.panorama_visibility,
            "panorama_is_disappearing": self.panorama_is_disappearing,
            "panorama_disappear_progress": self.panorama_disappear_timer / self.panorama_disappear_duration if self.panorama_is_disappearing and self.panorama_disappear_duration > 0 else 0.0,
            "panorama_disappear_enabled": self.panorama_disappear_duration > 0,
            "blur_active": self.blur_active,
            "current_blur": self.current_blur,
            "blur_progress": (self.blur_timer / getattr(self, '_effective_blur_duration', self.panorama_config.blur_duration)) if self.blur_active else 0.0,
            "blur_accelerated": hasattr(self, '_effective_blur_duration') and self._effective_blur_duration != self.panorama_config.blur_duration,
            "panorama_size": f"{self.panorama_width}x{self.panorama_height}",
            "panorama_scale": self.panorama_scale,
            "mirror": self.panorama_config.mirror,
            "debug_tiles": self.debug_tiles,
            "hide_tiles_for_debug": self.hide_tiles_for_debug
        }
    
    def _clear_all_sprites(self) -> None:
        """Immediately remove all sprites from the panorama."""
        # Remove all base image sprites
        for base_data in self.base_sprites.values():
            if base_data['sprite']:
                base_data['sprite'].delete()
                base_data['sprite'] = None
        
        self.base_sprites.clear()
        self.base_image = None
        self.base_image_id = None
        
        # Remove all tile sprites
        for tile in self.tiles.values():
            tile.cleanup()
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
        
        # Clean up singleton black image
        if hasattr(self, '_black_image'):
            self._black_image = None
        
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
        for image_id, base_data in list(self.base_sprites.items()):
            if base_data['sprite']:
                # Update base sprite to use new capture group (position stays the same)
                old_sprite = base_data['sprite']
                new_sprite = pyglet.sprite.Sprite(
                    old_sprite.image, x=0, y=0,
                    batch=self.batch, group=self.capture, z=0
                )
                new_sprite.scale = old_sprite.scale  # Keep panorama scale
                new_sprite.opacity = old_sprite.opacity  # Keep current opacity
                
                # Update the sprite reference and clean up old sprite
                base_data['sprite'] = new_sprite
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
