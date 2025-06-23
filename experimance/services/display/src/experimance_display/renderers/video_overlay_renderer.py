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

from experimance_common.config import BaseConfig
from experimance_common.schemas import MessageBase
from experimance_common.zmq.config import MessageDataType
from experimance_common.constants import VIDEOS_DIR_ABS
from experimance_display.config import DisplayServiceConfig
import pyglet
import pyglet.sprite
from pyglet.gl import (
    GL_BLEND, glEnable, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_TEXTURE_2D, GL_TRIANGLE_STRIP, glBindTexture, glActiveTexture,
    GL_TEXTURE0, GL_TEXTURE1
)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics.vertexdomain import VertexList
from pyglet.image import AbstractImage

from .layer_manager import LayerRenderer
from ..utils.pyglet_utils import load_pyglet_image_from_message, cleanup_temp_file

logger = logging.getLogger(__name__)


class VideoOverlayRenderer(LayerRenderer):
    """Renders masked video overlay responding to sand interaction."""
    quad: VertexList

    # Vertex shader: transforms vertices and passes texture coordinates
    vertex_shader_source = """#version 120
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
    """

    vertex_shader_source = """
    #version 330 core
    in vec2 position;
    in vec2 uv;
    out vec2 v_uv;
    uniform WindowBlock { mat4 proj; mat4 view; } window;
    void main()
    {
        gl_Position = window.proj * window.view * vec4(position, 0.0, 1.0);
        v_uv = uv;
    }
    """
    
    # Fragment shader: samples video and mask textures and blends them
    fragment_shader_source = """#version 120
    varying vec2 v_texcoord;
    uniform sampler2D video_tex;
    uniform sampler2D mask_tex;
    uniform float global_alpha;
    
    void main() {
        // Get colors from both textures
        vec4 video_col = texture2D(video_tex, v_texcoord);
        float mask_alpha = texture2D(mask_tex, v_texcoord).r;  // Use red channel for grayscale mask
        
        // Apply mask alpha and global alpha to video alpha
        float final_alpha = video_col.a * mask_alpha * global_alpha;
        
        // Output final color with calculated alpha
        gl_FragColor = vec4(video_col.rgb, final_alpha);
    }
    """

    fragment_shader_source = """
        #version 330 core
        in  vec2 v_uv;
        out vec4 frag;
        uniform sampler2D video_tex;
        uniform sampler2D mask_tex;
        uniform float global_alpha;

        void main()
        {
            vec4 vid  = texture(video_tex, v_uv);
            float m   = texture(mask_tex, v_uv).r;   // assume grayscale mask
            frag = vec4(vid.rgb * m, global_alpha);
        }
    """
    
    def __init__(self, config: DisplayServiceConfig, 
                 window: pyglet.window.BaseWindow, 
                 batch: pyglet.graphics.Batch,
                 order: int = 1):
        """Initialize the video overlay renderer."""
        super().__init__(config=config, window=window, batch=batch, order=order)
        
        # Extract video overlay specific config
        self.video_config = config.video_overlay
        self.transitions_config = config.transitions
        
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
        self.fade_in_duration = self.transitions_config.video_fade_in_duration
        self.fade_out_duration = self.transitions_config.video_fade_out_duration
        
        # Shader program for masking
        self.shader_program = None
        
        # Geometry and rendering state
        self.video_sprite = None
        self.mask_sprite = None
        
        # Set up OpenGL state for blending (one-time setup)
        self._setup_opengl_state()
        
        # Visibility and opacity
        self._visible = True
        self._opacity = 1.0
        self._current_alpha = 0.0  # Actual alpha considering fade state
        
        # Default video (if configured)
        self._load_default_video()
        
        # Load all white mask (i.e. none) or start-up mask
        self._load_mask()
        self._load_startup_mask()

        # Initialize shader program and geometry
        self._setup_shader()
        if not self.shader_program:
            logger.error("Failed to initialize shader program for video overlay")
            return


        # Disable fallback mask loading entirely for debugging
        # if (not self.mask_loaded and 
        #     hasattr(self.video_config, 'fallback_mask_enabled') and 
        #     self.video_config.fallback_mask_enabled):
        #     self._load_fallback_mask()
        
        logger.info(f"VideoOverlayRenderer initialized for {self.window}")
    
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
            self._current_alpha > 0.01
        )
    
    @property
    def opacity(self) -> float:
        """Get the layer opacity (0.0 to 1.0)."""
        return self._opacity * self._current_alpha

    def set_state(self):
        """Set OpenGL state for rendering the video overlay."""
        if not self.shader_program:
            logger.error("Shader program not initialized, cannot set state")
            return
        self._update_quad_vertices()

        self.shader_program.use()

        # -- video frame ---------------------------------------------------------
        if self.video_player is not None:
            self.video_player.update_texture()              # make sure ‘texture’ is fresh
            tex = self.video_player.texture
            if tex:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(tex.target, tex.id)
                self.shader_program["video_tex"] = 0
            else:
                return  # no frame yet – skip binding

        # -- mask ----------------------------------------------------------------
        if self.mask_texture is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(self.mask_texture.target, self.mask_texture.id)
            self.shader_program["mask_tex"] = 1


    def unset_state(self):
        """Unset OpenGL state after rendering the video overlay."""
        if not self.shader_program: return
        self.shader_program.stop()


    def update(self, dt: float):
        """Update video and animation state.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        # Disable fade animation for debugging
        # self._update_fade_animation(dt)
        
        # Update video playback
        if self.video_player and self.video_loaded:
            try:
                # Keep video playing and looping
                if not self.video_player.playing:
                    self.video_player.play()
                    
                # Update video texture from player
                if hasattr(self.video_player, 'texture') and self.video_player.texture:
                    if self.video_texture != self.video_player.texture:
                        self.video_texture = self.video_player.texture
                else:
                    if not hasattr(self.video_player, 'texture'):
                        logger.debug("Video player has no texture attribute")
                    elif not self.video_player.texture:
                        logger.debug("Video player texture is None")
                    
            except Exception as e:
                logger.error(f"Error updating video: {e}", exc_info=True)
    
    # def render(self):
    #     """Render the masked video overlay."""
    #     if not self.is_visible:
    #         logger.debug(f"Video not visible: _visible={self._visible}, video_loaded={self.video_loaded}, alpha={self._current_alpha}")
    #         return
        
    #     logger.debug(f"RENDERING VIDEO: alpha={self._current_alpha}, mask_loaded={self.mask_loaded}")
        
    #     try:
    #         # Render with shader-based masking
    #         self._render_simple_overlay()
            
    #     except Exception as e:
    #         logger.error(f"Error rendering video overlay: {e}", exc_info=True)
    
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
    
    # def _render_simple_overlay(self):
    #     """Render a video overlay with shader-based masking."""
    #     if not self.video_texture:
    #         logger.debug(f"No video texture available for rendering (video_loaded={self.video_loaded})")
    #         return
        
    #     # Debug: Check what we have available
    #     logger.debug(f"Render check: video_texture={self.video_texture is not None}, "
    #                 f"mask_texture={self.mask_texture is not None}, "
    #                 f"shader_program={self.shader_program is not None}, "
    #                 f"mask_loaded={self.mask_loaded}")
        
    #     # Only render if we have both video and mask textures and a shader program
    #     if self.mask_texture and self.shader_program and self.mask_loaded:
    #         logger.debug("Using shader-based rendering with mask")
    #         self._render_with_shader()
    #     else:
    #         # For debugging: Don't render anything if no proper mask is loaded
    #         logger.debug("No mask loaded - skipping video rendering entirely (for debugging)")
    #         return
    
    # def _render_with_shader(self):
    #     """Render video with mask using shaders."""
    #     try:
    #         logger.debug("SHADER RENDERING: Starting shader-based masked video rendering")
            
    #         # Update quad vertices to maintain aspect ratio with current texture
    #         self._update_quad_vertices()
            
    #         # Use the shader program
    #         logger.debug(f"Using shader program, current alpha: {self._current_alpha}")
    #         self.shader_program.use()
            
    #         # Bind video texture to unit 0
    #         glActiveTexture(GL_TEXTURE0)
    #         glBindTexture(GL_TEXTURE_2D, self.video_texture.id)
    #         logger.debug(f"Bound video texture ID: {self.video_texture.id} to unit 0")
            
    #         # Bind mask texture to unit 1
    #         glActiveTexture(GL_TEXTURE1)
    #         glBindTexture(GL_TEXTURE_2D, self.mask_texture.id)
    #         logger.debug(f"Bound mask texture ID: {self.mask_texture.id} to unit 1")
            
    #         # Set global alpha
    #         self.shader_program['global_alpha'] = self._current_alpha
    #         logger.debug(f"SHADER DEBUG: Set global_alpha uniform to: {self._current_alpha}")
    #         logger.debug(f"SHADER DEBUG: fade_state={self.fade_state}, mask_loaded={self.mask_loaded}")
    #         logger.debug(f"SHADER DEBUG: video_texture_id={self.video_texture.id}, mask_texture_id={self.mask_texture.id}")
            
    #         # Check if we have a valid quad
    #         if hasattr(self, 'quad') and self.quad:
    #             logger.debug(f"Drawing quad with {self.quad.count} vertices")
    #             self.quad.draw(GL_TRIANGLE_FAN)
    #             logger.debug("SHADER RENDERING: Quad draw completed")
    #         else:
    #             logger.warning("No quad vertex list available for shader rendering")
            
    #     except Exception as e:
    #         logger.error(f"Error rendering with shader: {e}", exc_info=True)
    #     finally:
    #         # Clean up shader state
    #         if self.shader_program:
    #             self.shader_program.stop()
    #         # Reset to texture unit 0
    #         glActiveTexture(GL_TEXTURE0)
    #         logger.debug("Shader rendering cleanup completed")
    
    # def _render_with_sprite(self):
    #     """Fallback sprite rendering without masking."""
    #     try:
    #         logger.debug("SPRITE RENDERING: Creating sprite-based video rendering (no masking)")
            
    #         # Create/update video sprite
    #         if not self.video_sprite and self.video_texture:
    #             self.video_sprite = pyglet.sprite.Sprite(self.video_texture)
    #             self._position_sprite()
    #             logger.debug(f"Created video sprite: {self.video_sprite.width}x{self.video_sprite.height}")
            
    #         # Update video texture if needed
    #         if self.video_sprite and self.video_texture:
    #             self.video_sprite.image = self.video_texture
                
    #         # Render the video sprite with current alpha
    #         if self.video_sprite:
    #             self.video_sprite.opacity = int(self._current_alpha * 255)
    #             self.video_sprite.draw()
    #             logger.debug(f"SPRITE RENDERING: Drew sprite with opacity {self.video_sprite.opacity}")
                
    #     except Exception as e:
    #         logger.error(f"Error rendering video overlay: {e}", exc_info=True)
    
    # def _position_sprite(self):
    #     """Position the video sprite to center and maintain aspect ratio."""
    #     if not self.video_sprite:
    #         return
            
    #     # Get sprite dimensions
    #     sprite_width = self.video_sprite.width
    #     sprite_height = self.video_sprite.height
        
    #     w, h =  self.window.get_size()

    #     # Scale to fit window while maintaining aspect ratio
    #     window_aspect = w / h
    #     sprite_aspect = sprite_width / sprite_height
        
    #     if sprite_aspect > window_aspect:
    #         # Video is wider - scale to fit width
    #         scale = w / sprite_width
    #     else:
    #         # Video is taller - scale to fit height  
    #         scale = h / sprite_height
            
    #     self.video_sprite.scale = scale
        
    #     # Center the sprite (position is bottom-left corner by default)
    #     scaled_width = sprite_width * scale
    #     scaled_height = sprite_height * scale
        
    #     self.video_sprite.x = (w - scaled_width) // 2
    #     self.video_sprite.y = (h - scaled_height) // 2
    
    def _setup_shader(self):
        """Set up shader program and vertex data."""
        try:
            # Create shader program using the shaders from the prototype
            vertex_shader = Shader(self.vertex_shader_source, 'vertex')
            fragment_shader = Shader(self.fragment_shader_source, 'fragment')
            self.shader_program = ShaderProgram(vertex_shader, fragment_shader)
            
            # Set texture uniforms (like in prototype)
            self.shader_program['video_tex'] = 0  # Will use texture unit 0
            self.shader_program['mask_tex'] = 1   # Will use texture unit 1
            
            w, h = self.window.get_size()
            # FIXME:
            # Accordingto docs: 
            # When using GL_LINE_STRIP and GL_TRIANGLE_STRIP, care must be taken to insert degenerate vertices at the beginning and end of each vertex list
            # This doesn't actually seem to be the case!
            positions = (
                0, 0, # bottom-left
                w, 0, # bottom-right
                0, h, # top left
                w, h  # top right
            )
            uvs        = ( 0,  0,   1,  0,    0, 1,   1, 1)

            self.quad = self.shader_program.vertex_list(4,
                    pyglet.gl.GL_TRIANGLE_STRIP,
                    batch=self.batch, group=self,
                    position=("f", positions),
                    uv=("f", uvs))
            
            # Track dimensions for optimization
            self.last_texture_width = None
            self.last_texture_height = None
            self.last_window_width = None
            self.last_window_height = None
            
            logger.info("Video overlay shader program and vertex data created successfully")
            
        except Exception as e:
            logger.error(f"Error setting up shader program: {e}", exc_info=True)
            self.shader_program = None
            #self.quad = None
    
    def _setup_opengl_state(self):
        """Set up OpenGL state for video overlay rendering (one-time setup)."""
        try:
            # Enable blending for transparency - this only needs to be done once
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            logger.debug("OpenGL blending state configured for video overlay")
        except Exception as e:
            logger.error(f"Error setting up OpenGL state: {e}", exc_info=True)
    
    def _update_quad_vertices(self):
        """Update the quad vertices based on the video's aspect ratio."""
        # Only update vertex positions if we have the shader setup
        if not self.shader_program or not self.quad or not self.video_texture:
            return
        
        # Get current window dimensions
        current_window_width, current_window_height = self.window.get_size()
        
        # Check if update is needed based on texture or window dimensions
        needs_update = (
            self.last_texture_width != self.video_texture.width or
            self.last_texture_height != self.video_texture.height or
            self.last_window_width != current_window_width or
            self.last_window_height != current_window_height
        )
        
        if needs_update:
            try:
                # Calculate new vertex positions based on video texture aspect ratio
                new_positions = self._calculate_quad_vertices(self.video_texture)
                
                # Update the existing vertex list
                self.quad.position[:] = new_positions # type: ignore
                
                # Store current dimensions
                self.last_texture_width = self.video_texture.width
                self.last_texture_height = self.video_texture.height
                self.last_window_width = current_window_width
                self.last_window_height = current_window_height
                
                logger.debug(f"Updated quad vertices for texture: {self.video_texture.width}x{self.video_texture.height} in window: {current_window_width}x{current_window_height}")
            except Exception as e:
                logger.error(f"Error updating quad vertices: {e}", exc_info=True)
    
    def _calculate_quad_vertices(self, texture=None) -> tuple:
        """Calculate quad vertices that preserve the aspect ratio of the video.
        
        Args:
            texture: The video texture (optional). If provided, uses its dimensions for aspect ratio.
        
        Returns:
            tuple: Quad vertex positions adjusted for aspect ratio
        """
        logger.debug("_calculate_quad_vertices")

        # Default to filling the screen while preserving aspect ratio
        # If no texture is provided yet, use a 1:1 aspect ratio
        # FIXME: get aspect from video player?
        video_aspect = 1.0
        
        if texture:
            # Calculate video aspect ratio from texture dimensions
            video_aspect = texture.width / texture.height
            logger.debug(f"Video dimensions: {texture.width}x{texture.height}, aspect ratio: {video_aspect:.2f}")
        
        # Get window dimensions and calculate aspect ratio
        window_width, window_height = self.window.get_size()
        window_aspect = window_width / window_height
        logger.debug(f"Window dimensions: {window_width}x{window_height}, aspect ratio: {window_aspect:.2f}")
        
        # Standard aspect ratio calculation
        if video_aspect > window_aspect:
            # Video is wider than window, fit to width
            scale_x = 1.0
            scale_y = (window_aspect / video_aspect)
            logger.debug(f"Video is wider than window, scaling height by: {scale_y:.2f}")
        else:
            # Video is taller than window, fit to height
            scale_x = (video_aspect / window_aspect)
            scale_y = 1.0
            logger.debug(f"Video is taller than window, scaling width by: {scale_x:.2f}")
        
        scaled_width = window_width * scale_x
        scaled_height = window_height * scale_y
        # Calculate offsets to center the quad
        x_offset = (window_width - scaled_width) / 2
        y_offset = (window_height - scaled_height) / 2

        # Define quad vertices (bottom-left, bottom-right, top-left, top-right)
        positions = (
            x_offset, y_offset,  # bottom-left
            x_offset + scaled_width, y_offset,  # bottom-right
            x_offset, y_offset + scaled_height,  # top-left
            x_offset + scaled_width, y_offset + scaled_height,  # top-right
        )

        return positions
    
    def _create_fallback_mask(self):
        """Create a plain white mask texture as fallback."""
        # Create a solid white image (fully opaque)
        return pyglet.image.SolidColorImagePattern((255, 255, 255, 255)).create_image(1, 1).get_texture()
    
    async def handle_video_mask(self, message: MessageDataType):
        """Handle VideoMask message.
        
        Args:
            message: VideoMask message with mask image URI or image data
        """
        try:
            if isinstance(message, MessageBase):
                message = message.model_dump()
                
            mask_id = message.get("mask_id")
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
                self._load_mask(mask_image)
                
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
    
    def _load_mask(self, mask_image:Optional[AbstractImage] = None):
        """Load a mask texture from an image.

        Args:
            mask_image: Pyglet image to use as mask
        """
        logger.debug(f"Loading mask texture from image: {mask_image}")
        if mask_image:
            try:
                self.mask_texture = mask_image.get_texture()
            except Exception as e:
                logger.error(f"Error loading mask texture: {e}", exc_info=True)
        else:
            # create an all white mask (i.e. no mask)
            self.mask_texture = self._create_fallback_mask()

        self.current_mask = mask_image
        self.mask_loaded = True
        logger.info("Mask loaded successfully")

    def _load_default_video(self):
        """Load default video if configured."""
        # Check config for default video path first
        default_video_path = None
        if hasattr(self.video_config, 'default_video_path') and self.video_config.default_video_path:
            default_video_path = self.video_config.default_video_path
            
            # Handle relative paths (relative to VIDEOS_DIR)
            if not os.path.isabs(default_video_path):
                default_video_path = str(VIDEOS_DIR_ABS / default_video_path)
        else:
            # Fallback to hardcoded path
            default_video_path = str(VIDEOS_DIR_ABS / "video_overlay.mp4")
        
        if os.path.isfile(default_video_path):
            logger.info(f"Loading default video: {default_video_path}")
            asyncio.create_task(self._load_video(default_video_path))
        else:
            logger.debug(f"No default video found at: {default_video_path}")
            
        # Note: Fallback mask loading is handled after startup mask loading in __init__
    
    def _load_fallback_mask(self):
        """Load a fallback mask texture."""
        try:
            self.mask_texture = self._create_fallback_mask()
            self.mask_loaded = True
            logger.debug("Fallback mask loaded (solid white)")
        except Exception as e:
            logger.error(f"Error creating fallback mask: {e}", exc_info=True)



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
            
            # Set to loop based on config
            should_loop = True
            if hasattr(self.video_config, 'loop_video'):
                should_loop = self.video_config.loop_video
                
            self.video_player.loop = should_loop
            
            # Start playing the video immediately
            self.video_player.play()
            
            self.video_loaded = True
            logger.info(f"Video loaded successfully: {video_path}")
            logger.debug(f"Video player state: playing={self.video_player.playing}")
            
            # For debugging: Set video to immediately visible instead of fading in
            self.fade_state = "visible"
            self._current_alpha = 1.0
            logger.debug("Set video overlay to immediately visible (no fade) for debugging")

            self._update_quad_vertices()  # Update vertices based on video aspect ratio

            self.show_overlay()

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
        """Show the video overlay (immediately visible for debugging)."""
        if self.video_loaded:
            # For debugging: Set immediately visible instead of fading
            self.fade_state = "visible"
            self._current_alpha = 1.0
            logger.debug("DEBUG: Video overlay set to immediately visible (no fade)")
        else:
            logger.warning("Cannot show overlay without video")
    
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize.
        
        Args:
            new_size: New (width, height) of the window
        """
        if new_size != self.window.get_size():
            logger.debug(f"VideoOverlayRenderer resize: {new_size}")
            # self.window_size = new_size
            
            # w, h = new_size
            # self.quad.position[:] = (0, 0, w, 0, 0, h, w, h)

            self._update_quad_vertices()

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
        
        try:
            # Clean up vertex list (shader-based rendering)
            if hasattr(self, 'quad') and self.quad:
                self.quad.delete()
        
            # Stop and clean up video player
            if self.video_player:
                try:
                    self.video_player.pause()
                    if hasattr(self.video_player, 'delete'):
                        self.video_player.delete()
                    self.video_player = None
                except Exception as e:
                    logger.error(f"Error stopping video player: {e}")
            
            # Clean up sprites
            if self.video_sprite:
                try:
                    if hasattr(self.video_sprite, 'delete'):
                        self.video_sprite.delete()
                    self.video_sprite = None
                except Exception as e:
                    logger.error(f"Error deleting video sprite: {e}")
                    
            if self.mask_sprite:
                try:
                    if hasattr(self.mask_sprite, 'delete'):
                        self.mask_sprite.delete()
                    self.mask_sprite = None
                except Exception as e:
                    logger.error(f"Error deleting mask sprite: {e}")
            
            # Clear textures and state
            self.video_texture = None
            self.mask_texture = None
            self.current_mask = None
            self.video_loaded = False
            self.mask_loaded = False
            self.fade_state = "hidden"
            
            logger.info("VideoOverlayRenderer cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during VideoOverlayRenderer cleanup: {e}", exc_info=True)
    
    def _load_startup_mask(self):
        """Load startup mask if configured."""

        if not hasattr(self.video_config, 'start_mask_path') or not self.video_config.start_mask_path:
            logger.debug("No startup mask path configured")
            return
            
        mask_path = self.video_config.start_mask_path
        logger.info(f"Loading startup mask from: {mask_path}")
        
        try:
            # Handle relative paths (relative to display service directory or media directory)
            if not os.path.isabs(mask_path):
                # Try multiple possible paths
                possible_paths = [
                    mask_path,  # try as-is first
                    os.path.join(os.getcwd(), mask_path),  # relative to current directory
                ]
                
                # Try relative to display service directory
                try:
                    from experimance_common.constants import DISPLAY_SERVICE_DIR
                    possible_paths.append(os.path.join(DISPLAY_SERVICE_DIR, mask_path))
                except ImportError:
                    pass
                
                # Try relative to media directory (handle case where path already starts with 'media/')
                try:
                    from experimance_common.constants import MEDIA_DIR_ABS
                    if mask_path.startswith('media/'):
                        # Remove 'media/' prefix to avoid duplication
                        clean_path = mask_path[6:]  # Remove 'media/' prefix
                        possible_paths.append(os.path.join(MEDIA_DIR_ABS, clean_path))
                    else:
                        possible_paths.append(os.path.join(MEDIA_DIR_ABS, mask_path))
                except ImportError:
                    pass
                
                # Find the first existing path
                mask_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        mask_path = path
                        logger.debug(f"Found startup mask at: {path}")
                        break
                
            if not mask_path:
                logger.warning(f"Startup mask file not found in any of the attempted paths for: {self.video_config.start_mask_path}")
                return
                
            # Load the mask image
            self._load_mask(pyglet.image.load(mask_path))
            
            logger.info(f"Startup mask loaded successfully: {mask_path}")
            if self.current_mask:
                logger.debug(f"Mask dimensions: {self.current_mask.width}x{self.current_mask.height}")
                
        except Exception as e:
            logger.error(f"Error loading startup mask from {mask_path}: {e}", exc_info=True)
