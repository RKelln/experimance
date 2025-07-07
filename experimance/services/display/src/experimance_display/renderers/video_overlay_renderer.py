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

    fragment_shader_source = """
        #version 330 core
        in  vec2 v_uv;
        out vec4 frag;
        uniform sampler2D video_tex;
        uniform sampler2D mask_tex;
        uniform float global_alpha;
        uniform vec2 video_size;        // Original video dimensions
        uniform vec2 target_size;       // Target size from config

        void main()
        {
            // Calculate scaling to fit the smallest side of video to smallest side of target
            float video_min_side = min(video_size.x, video_size.y);
            float target_min_side = min(target_size.x, target_size.y);
            float scale = target_min_side / video_min_side;
            
            // Calculate scaled video dimensions
            vec2 scaled_video = video_size * scale;
            
            // Calculate the UV scaling and offset to center the scaled video within target
            vec2 uv_scale = target_size / scaled_video;
            vec2 uv_offset = (1.0 - uv_scale) * 0.5;
            
            // Transform UV coordinates
            vec2 adjusted_uv = v_uv * uv_scale + uv_offset;
            
            // Sample video texture with adjusted UVs (clamped to avoid artifacts)
            vec4 vid = texture(video_tex, clamp(adjusted_uv, 0.0, 1.0));
            
            // If UVs are outside [0,1] range, make video transparent (crop effect)
            float in_bounds = step(0.0, adjusted_uv.x) * step(adjusted_uv.x, 1.0) * 
                             step(0.0, adjusted_uv.y) * step(adjusted_uv.y, 1.0);
            vid.a *= in_bounds;
            
            // Sample mask at original UV coordinates (mask covers the target area)
            float m = texture(mask_tex, v_uv).r;   // assume grayscale mask
            
            frag = vec4(vid.rgb, vid.a * m * global_alpha);
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

        if (not self.mask_loaded and
            self.video_config.fallback_mask_enabled):
            self._load_fallback_mask()
        
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
        
        if self.video_player is None:
            return
        
        # required (set every frame) for alpha channels on masks to work
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._update_quad_vertices() # TODO: before or after shader use?

        self.shader_program.use()

        # -- video frame ---------------------------------------------------------
        #self.video_player.update_texture()  # done automatically in pyglet
        tex = self.video_player.texture
        if tex:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(tex.target, tex.id)
            self.shader_program["video_tex"] = 0
            self.shader_program["video_size"] = (float(tex.width), float(tex.height))
            
        else:
            return  # no frame yet – skip binding

        # -- mask ----------------------------------------------------------------
        if self.mask_texture is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(self.mask_texture.target, self.mask_texture.id)
            self.shader_program["mask_tex"] = 1

        # global alpha
        self.shader_program["global_alpha"] = self._current_alpha
        
        # rescaling
        target_width, target_height = self.video_config.size
        self.shader_program["target_size"] = (float(target_width), float(target_height))

    def unset_state(self):
        """Unset OpenGL state after rendering the video overlay."""
        if self.shader_program:
            self.shader_program.stop()
        glActiveTexture(GL_TEXTURE0)


    def update(self, dt: float):
        """Update video and animation state.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        self._update_fade_animation(dt)
        
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
            
            # Get target size from config and calculate centered positions
            target_width, target_height = self.video_config.size
            window_width, window_height = self.window.get_size()
            
            # Center the target-sized quad in the window
            x_offset = (window_width - target_width) / 2
            y_offset = (window_height - target_height) / 2
            
            # FIXME:
            # According to docs: 
            # When using GL_LINE_STRIP and GL_TRIANGLE_STRIP, care must be taken to insert degenerate vertices at the beginning and end of each vertex list
            # This doesn't actually seem to be the case!
            positions = (
                x_offset, y_offset,  # bottom-left
                x_offset + target_width, y_offset,  # bottom-right
                x_offset, y_offset + target_height,  # top-left
                x_offset + target_width, y_offset + target_height  # top-right
            )
            uvs = (0, 0,   1, 0,    0, 1,   1, 1)

            self.quad = self.shader_program.vertex_list(4,
                    pyglet.gl.GL_TRIANGLE_STRIP,
                    batch=self.batch, 
                    group=self,
                    position=("f", positions),
                    uv=("f", uvs))
            
            # Track dimensions for optimization
            self.last_texture_width = None
            self.last_texture_height = None
            self.last_window_width = window_width
            self.last_window_height = window_height
            
            logger.info(f"Video overlay shader program created successfully for target size: {target_width}x{target_height}")
            
        except Exception as e:
            logger.error(f"Error setting up shader program: {e}", exc_info=True)
            self.shader_program = None
            #self.quad = None


    def _update_quad_vertices(self):
        """Update the quad vertices when window size changes."""
        # Only update vertex positions if we have the shader setup
        if not self.shader_program or not self.quad:
            return
        
        # Get current window dimensions
        current_window_width, current_window_height = self.window.get_size()
        
        # Check if update is needed based on window dimensions
        needs_update = (
            self.last_window_width != current_window_width or
            self.last_window_height != current_window_height
        )
        
        if needs_update:
            try:
                # Calculate new vertex positions for target size, centered in window
                new_positions = self._calculate_quad_vertices(self.video_texture)
                
                # Update the existing vertex list
                self.quad.position = new_positions  # type: ignore
                
                # Store current dimensions
                self.last_window_width = current_window_width
                self.last_window_height = current_window_height
                
                # Update texture dimensions for shader uniform (even if they don't affect quad positioning)
                if self.video_texture:
                    self.last_texture_width = self.video_texture.width
                    self.last_texture_height = self.video_texture.height
                
                logger.debug(f"Updated quad vertices for window: {current_window_width}x{current_window_height}")
            except Exception as e:
                logger.error(f"Error updating quad vertices: {e}", exc_info=True)


    def _calculate_quad_vertices(self, texture=None) -> tuple:
        """Calculate quad vertices for the target size, centered in window.
        
        Args:
            texture: The video texture (optional, used for logging only)
        
        Returns:
            tuple: Quad vertex positions for target size, centered in window
        """
        logger.debug("_calculate_quad_vertices")

        # Get target size from config
        target_width, target_height = self.video_config.size
        
        # Get window dimensions
        window_width, window_height = self.window.get_size()
        
        if texture:
            logger.debug(f"Video dimensions: {texture.width}x{texture.height}")
        logger.debug(f"Target size: {target_width}x{target_height}")
        logger.debug(f"Window dimensions: {window_width}x{window_height}")
        
        # Calculate position to center the target-sized quad in the window
        x_offset = (window_width - target_width) / 2
        y_offset = (window_height - target_height) / 2
        
        # Define quad vertices for the target size, centered in window
        # (bottom-left, bottom-right, top-left, top-right)
        positions = (
            x_offset, y_offset,  # bottom-left
            x_offset + target_width, y_offset,  # bottom-right
            x_offset, y_offset + target_height,  # top-left
            x_offset + target_width, y_offset + target_height,  # top-right
        )
        
        logger.debug(f"Video quad positioned at: offset=({x_offset}, {y_offset}), size=({target_width}, {target_height})")

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
        """Load a mask texture from an image, rescaling to target size if needed.

        Args:
            mask_image: Pyglet image to use as mask
        """
        logger.debug(f"Loading mask texture from image: {mask_image}")
        if mask_image:
            try:
                # Get target size from config
                target_width, target_height = self.video_config.size
                
                # Check if mask needs rescaling
                if (mask_image.width != target_width or mask_image.height != target_height):
                    logger.info(f"Rescaling mask from {mask_image.width}x{mask_image.height} to {target_width}x{target_height}")
                    
                    # Convert to PIL Image for rescaling
                    # Get image data as RGBA
                    image_data = mask_image.get_image_data()
                    raw_data = image_data.get_data('RGBA', image_data.width * 4)
                    
                    # Create PIL Image
                    from PIL import Image
                    pil_image = Image.frombytes('RGBA', (image_data.width, image_data.height), raw_data)
                    
                    # Resize using high-quality resampling
                    resized_pil = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    
                    # Convert back to pyglet image
                    resized_data = resized_pil.tobytes()
                    mask_image = pyglet.image.ImageData(target_width, target_height, 'RGBA', resized_data)
                    
                    logger.debug(f"Mask rescaled successfully to {target_width}x{target_height}")
                else:
                    logger.debug(f"Mask already correct size: {mask_image.width}x{mask_image.height}")
                
                self.mask_texture = mask_image.get_texture()
            except Exception as e:
                logger.error(f"Error loading mask texture: {e}", exc_info=True)
                # Fall back to white mask on error
                self.mask_texture = self._create_fallback_mask()
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
