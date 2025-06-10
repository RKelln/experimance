#!/usr/bin/env python3
"""
Fix for the pyglet_test.py script.

This script modifies the original pyglet_test.py script to fix issues with:
1. Image loading using direct file paths instead of the resource system
2. Modern OpenGL handling for alpha blending
3. Error handling for missing files
"""

import os
import math
import argparse
import time
import statistics
import atexit
from pathlib import Path
import pyglet
from pyglet import clock
from pyglet.window import key
from pyglet.gl import (
    GL_TEXTURE_2D, GL_TRIANGLE_FAN,
    glBindTexture, glActiveTexture,
    GL_TEXTURE0, GL_TEXTURE1,
    GL_BLEND, glEnable, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
)
from pyglet.graphics.shader import Shader, ShaderProgram

# ── Command-line Arguments ───────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Pyglet Display Application")
parser.add_argument(
    '-f', '--fullscreen',
    action='store_true',
    help='Enable fullscreen mode'
)
parser.add_argument(
    '-M', '--mask',
    required=False,
    help='Path to grayscale mask image (defines the overlay shape)'
)
parser.add_argument(
    '-v', '--video',
    required=False,
    help='Path to video file for overlay'
)
args = parser.parse_args()

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_FOLDER    = "services/image_server/images/generated"      # directory for background images
IMAGE_DURATION  = 5.0            # seconds per image (including fades)
IMAGE_FADE_TIME = 1.0            # seconds to fade in/out each image

VIDEO_PATH      = "services/image_server/images/video_overlay.mp4" # video file path
VIDEO_FADE_TIME = 1.0            # seconds fade in/out video overlay
VIDEO_HOLD_TIME = 3.0            # seconds fully visible

WINDOW_WIDTH    = 1920
WINDOW_HEIGHT   = 1080

# ── Utilities ────────────────────────────────────────────────────────────────
def compute_alpha(timer, fade, hold):
    """Cycle alpha: fade in, hold, fade out."""
    cycle = fade * 2 + hold
    t = timer % cycle  # Fixed typo: 'cycl' -> 'cycle'
    if t < fade:
        return t / fade
    if t < fade + hold:
        return 1.0
    return 1.0 - (t - fade - hold) / fade

# ── Core Classes ─────────────────────────────────────────────────────────────
class ImageCycler:
    def __init__(self, folder, duration, fade):
        self.images = []
        self.sprites = []  # We'll use sprites for better opacity control
        print(f"Loading images from: {folder} from {os.getcwd()}")
        folder_path = Path(folder)
        
        # Check if folder exists
        if not os.path.isdir(folder_path):
            print(f"Warning: Image folder not found at {folder_path}")
            folder_path = Path(os.getcwd()) / folder_path
            print(f"Trying absolute path: {folder_path}")
            if not os.path.isdir(folder_path):
                print(f"Error: Could not find image folder at {folder_path}")
                return

        # Load images directly without using resource system
        for fn in sorted(os.listdir(folder_path)):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filepath = os.path.join(folder_path, fn)
                if not os.path.isfile(filepath):
                    continue
                
                print(f"Loading image: {filepath}")
                img = pyglet.image.load(filepath)
                img.anchor_x = img.width // 2
                img.anchor_y = img.height // 2
                self.images.append(img)
                # Create sprite for each image (will position later)
                sprite = pyglet.sprite.Sprite(img)
                sprite.opacity = 0  # Start invisible
                self.sprites.append(sprite)
        
        print(f"Loaded {len(self.images)} images")
        
        self.duration = duration
        self.fade = fade
        self.timer = 0.0
        self.index = 0
        self.next_index = 0  # For cross-fading to next image
        self.last_size = None

    def update(self, dt):
        self.timer += dt
        if self.timer >= self.duration:
            self.timer -= self.duration
            # When crossfade completes, the next image becomes the current one
            self.index = (self.index + 1) % len(self.images)
            self.next_index = (self.index + 1) % len(self.images)

    def update_sprite_positions(self, window_size):
        """Update sprite positions to be centered on screen."""
        if not self.sprites:
            return
        print(f"Updating sprite positions for window size: {window_size}")
        self.last_size = window_size
        center_x = window_size[0] // 2
        center_y = window_size[1] // 2
        
        for sprite in self.sprites:
            # Center each sprite on the screen
            sprite.x = center_x
            sprite.y = center_y
            sprite.anchor_x = sprite.width // 2
            sprite.anchor_y = sprite.height // 2
            
            # Debug output
            print(f"Positioned sprite at ({sprite.x}, {sprite.y}), "
                  f"screen center at ({center_x}, {center_y}), "
                  f"sprite size: {sprite.width}x{sprite.height}")

    def draw(self, window_size):
        if not self.images or not self.sprites:
            return
            
        # Update sprite positions if center has changed
        if self.last_size != window_size:
            self.update_sprite_positions(window_size)
        
        # No fading needed if only one image
        if len(self.images) <= 1:
            self.sprites[0].opacity = 255
            self.sprites[0].draw()
            return

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Determine if we're in the fade-out period
        in_fade_out = self.timer >= (self.duration - self.fade)
        
        try:
            # Current image
            self.index = self.index % len(self.sprites)
            current_sprite = self.sprites[self.index]
            
            # Next image (for crossfade)
            self.next_index = (self.index + 1) % len(self.sprites)
            next_sprite = self.sprites[self.next_index]
            
            if in_fade_out:
                # When fading out current image, start fading in next image
                fade_progress = (self.timer - (self.duration - self.fade)) / self.fade
                fade_progress = max(0, min(fade_progress, 1))  # Clamp to [0, 1]
                
                # Calculate opacities for smoother transition
                current_opacity = int((1 - fade_progress) * 255)
                
                # Update opacities for both sprites
                current_sprite.opacity = current_opacity
                next_sprite.opacity = 255
                
                # Draw both images for crossfade effect
                # Draw next image first (behind current for alpha blending)
                next_sprite.draw()
                current_sprite.draw()
            else:
                # Normal display of current image
                # Make sure current image is fully visible during hold period
                current_sprite.opacity = 255
                current_sprite.draw()
        
        except Exception as e:
            print(f"Error drawing image: {e}")
            # Fallback to simple drawing of current image
            try:
                img = self.images[self.index]
                # Use window size to calculate center
                if self.last_size:
                    center_x = self.last_size[0] // 2
                    center_y = self.last_size[1] // 2
                    img.blit(center_x - img.width//2, center_y - img.height//2)
            except Exception:
                pass  # If even this fails, just skip drawing

class VideoOverlay:
    """
    A class that displays video with a grayscale mask overlay.
    
    The mask is applied to the video using shaders, where white areas of the mask
    show the video at full opacity, black areas are completely transparent, and
    gray areas are partially transparent.
    """
    
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

    def __init__(self, video_path, mask_path, fade_time=1.0, hold_time=3.0):
        """
        Initialize a new VideoOverlayFixed instance.
        
        Args:
            video_path (str): Path to the video file
            mask_path (str): Path to the grayscale mask image
            fade_time (float): Time in seconds for fade in/out animations
            hold_time (float): Time in seconds to hold at full opacity
        """
        print("Initializing VideoOverlayFixed...")
        print(f"Video path: {video_path}")
        print(f"Mask path: {mask_path}")

        self.video_loaded = False
        self.mask_loaded = False
        self.timer = 0.0
        self.fade_time = fade_time
        self.hold_time = hold_time
        
        # Keep track of texture dimensions for optimization
        self.last_texture_width = None
        self.last_texture_height = None
        self.last_window_width = None
        self.last_window_height = None
        
        # Load video
        self._load_video(video_path)
        
        # Load mask texture
        self._load_mask(mask_path)
        
        # Initialize shader program and geometry
        self._setup_shader()

    def _load_video(self, video_path):
        """Load video from the given path."""
        # Create media player
        self.player = pyglet.media.Player()
        
        # Handle relative paths
        if not os.path.isfile(video_path):
            print(f"Warning: Video file not found at {video_path}")
            video_path = os.path.join(os.getcwd(), video_path)
            print(f"Trying absolute path: {video_path}")
        
        # Load video if file exists
        if os.path.isfile(video_path):
            try:
                src = pyglet.media.load(video_path)
                self.player.queue(src)
                self.player.loop = True
                self.player.play()
                self.video_loaded = True
                print(f"Successfully loaded video: {video_path}")
            except Exception as e:
                print(f"Error loading video: {e}")
        else:
            print(f"Error: Could not find video file at {video_path}")

    def _load_mask(self, mask_path):
        """Load mask texture from the given path."""
        # Handle mask loading
        if mask_path and os.path.isfile(mask_path):
            try:
                self.mask_img = pyglet.image.load(mask_path)
                self.mask_tex = self.mask_img.get_texture()
                self.mask_loaded = True
                print(f"Successfully loaded mask: {mask_path}")
            except Exception as e:
                print(f"Error loading mask: {e}")
                # Fall back to a white texture (no masking)
                self.mask_tex = self._create_fallback_mask()
        else:
            print(f"No mask provided or mask not found at {mask_path}")
            # Fall back to a white texture (no masking)
            self.mask_tex = self._create_fallback_mask()

    def _create_fallback_mask(self):
        """Create a plain white mask texture as fallback."""
        # Create a solid white image (fully opaque)
        return pyglet.image.SolidColorImagePattern((255, 255, 255, 255)).create_image(1, 1).get_texture()

    def _calculate_quad_vertices(self, texture=None):
        """
        Calculate quad vertices that preserve the aspect ratio of the video.
        
        Args:
            texture: The video texture (optional). If provided, uses its dimensions for aspect ratio.
        
        Returns:
            tuple: Quad vertex positions adjusted for aspect ratio
        """
        # Default to filling the screen while preserving aspect ratio
        # If no texture is provided yet, use a 1:1 aspect ratio
        video_aspect = 1.0
        
        if texture:
            # Calculate video aspect ratio from texture dimensions
            video_aspect = texture.width / texture.height
            print(f"Video dimensions: {texture.width}x{texture.height}, aspect ratio: {video_aspect:.2f}")
        
        # Get window dimensions and calculate aspect ratio
        # First try to get windows directly from pyglet.app.windows
        window_width = WINDOW_WIDTH  # Default fallback
        window_height = WINDOW_HEIGHT  # Default fallback
        window_aspect = window_width / window_height  # Default fallback
        
        # Try to get windows from app
        try:
            # pyglet.app.windows might be a WeakSet in some versions
            windows = list(pyglet.app.windows)
            if windows:
                # Use the first window (there's usually only one)
                window = windows[0]
                window_width = window.width
                window_height = window.height
                window_aspect = window_width / window_height
                print(f"Window dimensions: {window_width}x{window_height}, aspect ratio: {window_aspect:.2f}")
            else:
                print(f"No windows found, using default window aspect ratio: {window_aspect:.2f}")
        except Exception as e:
            # Fallback to default window size from globals
            print(f"Error getting window dimensions: {e}")
            print(f"Using default window aspect ratio: {window_aspect:.2f}")
        
        # Standard aspect ratio calculation
        if video_aspect > window_aspect:
            # Video is wider than window, fit to width
            scale_x = 1.0
            scale_y = (window_aspect / video_aspect)
            print(f"Video is wider than window, scaling height by: {scale_y:.2f}")
        else:
            # Video is taller than window, fit to height
            scale_x = (video_aspect / window_aspect)
            scale_y = 1.0
            print(f"Video is taller than window, scaling width by: {scale_x:.2f}")
            
        # Generate quad positions that preserve aspect ratio
        positions = (
            -scale_x, -scale_y,  # bottom-left
             scale_x, -scale_y,  # bottom-right
             scale_x,  scale_y,  # top-right
            -scale_x,  scale_y   # top-left
        )
        
        return positions

    def _setup_shader(self):
        """Set up shader program and vertex data."""
        try:
            # Compile shaders and create shader program
            vert_shader = Shader(self.vertex_shader_source, 'vertex')
            frag_shader = Shader(self.fragment_shader_source, 'fragment')
            self.shader_program = ShaderProgram(vert_shader, frag_shader)
            
            # Set texture uniforms
            self.shader_program['video_tex'] = 0  # Will use texture unit 0
            self.shader_program['mask_tex'] = 1   # Will use texture unit 1
            
            # Define quad vertices that preserve aspect ratio
            positions = self._calculate_quad_vertices()
            
            # Define texture coordinates
            texcoords = (
                0.0, 0.0,  # bottom-left
                1.0, 0.0,  # bottom-right
                1.0, 1.0,  # top-right
                0.0, 1.0   # top-left
            )
            
            # Create vertex list - properly pass attributes to avoid "too many values to unpack" error
            self.quad = self.shader_program.vertex_list(
                4,                                    # 4 vertices for a quad
                GL_TRIANGLE_FAN,                      # Drawing mode
                position=('f', positions),           # Format: 'f' = float, followed by the data
                texcoord=('f', texcoords)            # Format: 'f' = float, followed by the data 
            )
            
            print("Shader program and vertex data set up successfully")
        except Exception as e:
            import traceback
            print(f"Error setting up shader program: {e}")
            print("Shader sources for debugging:")
            print("Vertex shader source:")
            print(self.vertex_shader_source)
            print("Fragment shader source:")
            print(self.fragment_shader_source)
            traceback.print_exc()
            self.shader_program = None
            self.quad = None

    def update(self, dt):
        """
        Update the overlay state.
        
        Args:
            dt (float): Time elapsed since last update
        """
        self.timer += dt

    def compute_alpha(self):
        """Calculate the current alpha value based on timer."""
        cycle_duration = self.fade_time * 2 + self.hold_time
        t = self.timer % cycle_duration
        
        if t < self.fade_time:
            # Fade in
            return t / self.fade_time
        elif t < self.fade_time + self.hold_time:
            # Hold at full opacity
            return 1.0
        else:
            # Fade out
            return 1.0 - (t - self.fade_time - self.hold_time) / self.fade_time

    def update_quad_vertices(self, texture):
        """
        Update the quad vertices based on the texture's aspect ratio.
        Only updates if the texture size or window size has changed.
        
        Args:
            texture: The video texture to use for aspect ratio calculation
        """
        if not self.shader_program or not self.quad:
            return
            
        # Get current window dimensions - safely get window dimensions
        current_window_width = WINDOW_WIDTH  # Default fallback
        current_window_height = WINDOW_HEIGHT  # Default fallback
        
        try:
            # pyglet.app.windows might be a WeakSet in some versions
            windows = list(pyglet.app.windows)
            if windows:
                window = windows[0]
                current_window_width = window.width
                current_window_height = window.height
        except Exception as e:
            print(f"Error getting window dimensions in update_quad_vertices: {e}")
        
        # Check if update is needed based on texture or window dimensions
        needs_update = (
            self.last_texture_width != texture.width or
            self.last_texture_height != texture.height or
            self.last_window_width != current_window_width or
            self.last_window_height != current_window_height
        )
        
        if needs_update:
            try:
                # Calculate new vertex positions
                new_positions = self._calculate_quad_vertices(texture)
                
                # Update the existing vertex list
                self.quad.position[:] = new_positions
                
                # Store current dimensions
                self.last_texture_width = texture.width
                self.last_texture_height = texture.height
                self.last_window_width = current_window_width
                self.last_window_height = current_window_height
                
                print(f"Updated quad vertices for texture: {texture.width}x{texture.height} in window: {current_window_width}x{current_window_height}")
            except Exception as e:
                print(f"Error updating quad vertices: {e}")
    
    def draw(self):
        """Draw the video overlay with mask."""
        # Check if video is loaded and ready
        if not self.video_loaded:
            return
            
        # Check if shader program is initialized
        if not self.shader_program or not self.quad:
            return
            
        try:
            # Get video texture (using the texture property instead of get_texture method)
            tex = self.player.texture
            if tex is None:
                return
                
            # Update quad vertices to maintain aspect ratio with current texture
            self.update_quad_vertices(tex)

            # Calculate current alpha value for fading
            alpha = self.compute_alpha()

            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Bind video texture to unit 0
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tex.id)
            
            # Bind mask texture to unit 1
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.mask_tex.id)

            # Use shader and update alpha uniform
            self.shader_program.use()
            self.shader_program['global_alpha'] = alpha
            
            # Draw the quad
            self.quad.draw(GL_TRIANGLE_FAN)
            
            # Stop using the shader program
            self.shader_program.stop()

            # Reset to texture unit 0
            glActiveTexture(GL_TEXTURE0)
        except Exception as e:
            print(f"Error drawing video overlay: {e}")

    def delete(self):
        """Clean up resources."""
        if hasattr(self, 'quad') and self.quad:
            self.quad.delete()
        
        if hasattr(self, 'player') and self.player:
            self.player.pause()
            self.player.delete()


# ── Main Window ──────────────────────────────────────────────────────────────
class MainWindow(pyglet.window.Window):
    def __init__(self, fullscreen=False):
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, caption="Pyglet Display", fullscreen=fullscreen)

        print("Initializing Pyglet Display Application...")
        print("Fullscreen mode:", fullscreen)
        print("Window size:", self.width, "x", self.height, self.get_size())

        # instantiate image cycler
        self.images = ImageCycler(IMAGE_FOLDER, IMAGE_DURATION, IMAGE_FADE_TIME)
        
        # Initialize video overlay only if both video and mask are available
        self.video = None
        video_path = args.video or VIDEO_PATH
        
        if video_path and args.mask:
            try:
                self.video = VideoOverlay(video_path, args.mask, VIDEO_FADE_TIME, VIDEO_HOLD_TIME)
                # schedule video update loop
                clock.schedule_interval(self.video.update, 1/60.0)
            except Exception as e:
                print(f"Error initializing video overlay: {e}")
                self.video = None
        else:
            print(f"Video overlay disabled. Video path: {video_path}, Mask: {args.mask}")

        # schedule image update loop
        clock.schedule_interval(self.images.update, 1/60.0)
        
        # FPS tracking
        self.fps_display = pyglet.window.FPSDisplay(self)
        self.fps_display.label.font_size = 18
        self.fps_display.label.color = (255, 255, 0, 255)  # Yellow text
        
        # Stats tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.show_fps = True
        
        # Register exit function to print stats
        atexit.register(self.print_stats)
        
        # Schedule FPS update (lower frequency to avoid overhead)
        clock.schedule_interval(self.update_fps, 0.5)
        
    def on_resize(self, width, height):
        """Handle window resize events."""
        super().on_resize(width, height)
        print(f"Window resized to {width}x{height}")
        
        # Force aspect ratio recalculation on video overlay
        if self.video and hasattr(self.video, "last_window_width"):
            # Reset window size tracking to force update
            self.video.last_window_width = None
            self.video.last_window_height = None
            
        # Force image position update when window size changes
        if hasattr(self.images, 'last_size'):
            self.images.last_size = None  # This will force an update
            
        return True

    def update_fps(self, dt):
        """Update FPS tracking data."""
        if self.frame_count > 0:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed > 0:  # Avoid division by zero
                fps = self.frame_count / elapsed
                self.frame_times.append(fps)
                self.frame_count = 0
                self.last_frame_time = current_time
    
    def print_stats(self):
        """Print framerate statistics on exit."""
        if not self.frame_times:
            print("No frame data collected")
            return
            
        avg_fps = statistics.mean(self.frame_times) if self.frame_times else 0
        min_fps = min(self.frame_times) if self.frame_times else 0
        max_fps = max(self.frame_times) if self.frame_times else 0
        
        print("\n=== Frame Rate Statistics ===")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Minimum FPS: {min_fps:.2f}")
        print(f"Maximum FPS: {max_fps:.2f}")
        print(f"Samples collected: {len(self.frame_times)}")

    def on_draw(self):
        self.clear()
        
        # Draw background images first
        if hasattr(self.images, 'draw') and self.images.images:
            self.images.draw(self.get_size())
            
        # Draw video overlay on top of images
        if self.video and hasattr(self.video, 'draw'):
            self.video.draw()
            
        # Always increment frame count
        self.frame_count += 1
        
        # Draw FPS display if enabled (on top of everything)
        if self.show_fps:
            self.fps_display.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()
        elif symbol == key.F11:
            self.set_fullscreen(not self.fullscreen)
            # Force image position update when fullscreen changes
            if hasattr(self.images, 'last_size'):
                self.images.last_size = None  # This will force an update
        elif symbol == key.F:
            # Toggle FPS display
            self.show_fps = not self.show_fps
        elif symbol == key.P:
            # Print current stats
            self.print_stats()

# ── Entry Point ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Pyglet Display Application...")
    
    # Setup initial OpenGL configuration
    # Enable alpha blending for proper transparency
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    try:
        print("Creating main window...")
        window = MainWindow(fullscreen=args.fullscreen)
        print("Window created successfully. Running application...")
        print("Press 'F' to toggle FPS display")
        print("Press 'P' to print framerate statistics")
        print("Press 'F11' to toggle fullscreen")
        print("Press 'ESC' to exit")
        
        # Actually run the application
        pyglet.app.run()
        
    except Exception as e:
        import traceback
        print(f"ERROR: Application crashed: {e}")
        traceback.print_exc()
        import time
        print("Keeping terminal open for 5 seconds to read error...")
        time.sleep(5)