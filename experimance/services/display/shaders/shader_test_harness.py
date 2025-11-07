#!/usr/bin/env python3
"""
Shader Test Harness for Display Service

This test harness automatically loads all fragment shaders from the current directory
and provides an interactive environment for testing and debugging shader effects.

Usage:
    cd services/display/shaders
    uv run shader_test_harness.py
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pyglet
from pyglet.gl import *
from pyglet.graphics.shader import Shader, ShaderProgram

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ShaderTestHarness:
    """Interactive test harness for shader development and debugging."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.start_time = time.time()
        
        # Shader directory (current directory when run from shaders/)
        self.shader_dir = Path(__file__).parent
        
        # Create window
        self.window = pyglet.window.Window(
            width=width,
            height=height,
            caption="Shader Test Harness",
            resizable=False
        )
        
        # Set background color
        pyglet.gl.glClearColor(0.1, 0.1, 0.2, 1.0)  # Dark blue background
        
        # Bind events
        self.window.on_draw = self.on_draw
        self.window.on_key_press = self.on_key_press
        
        # OpenGL setup
        self._init_opengl()
        
        # Background content
        self.background_batch = pyglet.graphics.Batch()
        self.create_test_background()
        
        # Shader programs
        self.shaders = {}
        self.current_shader_name = None
        self.shader_names = []
        self.current_shader_index = 0
        
        # Control flags
        self.show_shaders = True
        self.auto_reload = False
        self.last_reload_time = 0
        
        # Uniforms (adjustable parameters)
        self.uniforms = {
            'vignette_strength': 0.7,
            'turbulence_amount': 0.25,
            'spark_intensity': 0.5,
            'full_mask_zone': 0.08,
            'gradient_zone': 0.25,
            'horizontal_compression': 1.0,
            'color_intensity': 0.6
        }
        
        # Batch system
        self.batch = pyglet.graphics.Batch()
        
        # Load and setup shaders
        self.load_all_shaders()
        self.create_vertex_lists()
        
        self._print_controls()
    
    def _init_opengl(self):
        """Initialize OpenGL settings."""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        logger.debug("OpenGL initialized")
    
    def create_test_background(self):
        """Create a test background with bright spots for shader testing."""
        try:
            # Create a procedural background with interesting features
            width, height = 512, 512
            data = []
            
            for y in range(height):
                for x in range(width):
                    # Base gradient
                    r = int(30 + (x / width) * 60)
                    g = int(20 + (y / height) * 80)
                    b = int(60 + ((x + y) / (width + height)) * 100)
                    
                    # Add some bright spots for spark testing
                    center_x, center_y = width // 2, height // 4
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if dist < 20:
                        brightness = max(0, 1.0 - dist / 20)
                        r = min(255, int(r + brightness * 200))
                        g = min(255, int(g + brightness * 150))
                        b = min(255, int(b + brightness * 50))
                    
                    # Add more bright spots
                    for spot_x, spot_y in [(100, 100), (400, 150), (300, 350), (150, 400)]:
                        dist = ((x - spot_x) ** 2 + (y - spot_y) ** 2) ** 0.5
                        if dist < 15:
                            brightness = max(0, 1.0 - dist / 15)
                            r = min(255, int(r + brightness * 180))
                            g = min(255, int(g + brightness * 140))
                            b = min(255, int(b + brightness * 60))
                    
                    data.extend([r, g, b, 255])
            
            # Create texture
            image_data = pyglet.image.ImageData(width, height, 'RGBA', bytes(data))
            
            # Create sprite that fills the screen
            sprite = pyglet.sprite.Sprite(
                img=image_data,
                x=0, y=0,
                batch=self.background_batch
            )
            sprite.scale_x = self.width / width
            sprite.scale_y = self.height / height
            
            logger.info("Created test background with bright spots")
            
        except Exception as e:
            logger.error(f"Failed to create test background: {e}")
    
    def load_all_shaders(self):
        """Automatically load all fragment shaders from the current directory."""
        self.shaders.clear()
        
        # Standard vertex shader
        vertex_source = """#version 330 core
in vec2 position;
in vec2 tex_coords;
out vec2 v_tex_coords;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_tex_coords = tex_coords;
}
"""
        
        try:
            vertex_shader = Shader(vertex_source, 'vertex')
            
            # Find all .frag files
            frag_files = list(self.shader_dir.glob("*.frag"))
            
            # Filter out passthrough (it's boring)
            frag_files = [f for f in frag_files if f.name != "passthrough.frag"]
            
            for frag_path in sorted(frag_files):
                name = frag_path.stem
                
                try:
                    with open(frag_path, 'r') as f:
                        frag_source = f.read()
                    
                    # Skip if shader doesn't use our expected interface
                    if "in vec2 tex_coords;" not in frag_source:
                        logger.warning(f"Skipping {name}: incompatible interface")
                        continue
                    
                    # Adapt shader to use our vertex shader output
                    # Only replace the input declaration and the main usage
                    frag_source = frag_source.replace("in vec2 tex_coords;", "in vec2 v_tex_coords;")
                    frag_source = frag_source.replace("vec2 uv = tex_coords;", "vec2 uv = v_tex_coords;")
                    
                    frag_shader = Shader(frag_source, 'fragment')
                    self.shaders[name] = ShaderProgram(vertex_shader, frag_shader)
                    logger.info(f"Loaded shader: {name}")
                    
                except Exception as e:
                    logger.error(f"Failed to compile {name}: {e}")
            
            self.shader_names = sorted(self.shaders.keys())
            
            if self.shader_names:
                self.current_shader_name = self.shader_names[0]
                logger.info(f"Active shader: {self.current_shader_name}")
                logger.info(f"Total shaders loaded: {len(self.shader_names)}")
            else:
                logger.warning("No compatible shaders found")
            
        except Exception as e:
            logger.error(f"Failed to initialize shaders: {e}")
    
    def create_vertex_lists(self):
        """Create vertex lists for all loaded shaders."""
        self.vertex_lists = {}
        
        # Full-screen quad
        position_data = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
        texcoord_data = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        indices = [0, 1, 2, 2, 3, 0]
        
        for name, shader_program in self.shaders.items():
            try:
                vertex_list = shader_program.vertex_list_indexed(
                    4, GL_TRIANGLES, indices,
                    position=('f', position_data),
                    tex_coords=('f', texcoord_data)
                )
                self.vertex_lists[name] = vertex_list
                logger.debug(f"Created vertex list for: {name}")
            except Exception as e:
                logger.error(f"Failed to create vertex list for {name}: {e}")
    
    def reload_shaders_if_needed(self):
        """Auto-reload shaders if files have changed."""
        if not self.auto_reload:
            return
        
        current_time = time.time()
        if current_time - self.last_reload_time < 1.0:  # Check at most once per second
            return
        
        self.last_reload_time = current_time
        
        # Check if any shader file has been modified
        should_reload = False
        for frag_path in self.shader_dir.glob("*.frag"):
            if frag_path.stat().st_mtime > self.start_time:
                should_reload = True
                break
        
        if should_reload:
            logger.info("Shader files changed, reloading...")
            self.load_all_shaders()
            self.create_vertex_lists()
            self.start_time = current_time
    
    def on_draw(self):
        """Render the current shader effect."""
        self.reload_shaders_if_needed()
        
        self.window.clear()
        
        # Draw background
        self.background_batch.draw()
        
        # Draw shader effect
        if (not self.show_shaders or 
            not self.current_shader_name or 
            self.current_shader_name not in self.shaders):
            return
        
        shader_program = self.shaders[self.current_shader_name]
        vertex_list = self.vertex_lists.get(self.current_shader_name)
        
        if not vertex_list:
            return
        
        # Set appropriate blend mode
        glEnable(GL_BLEND)
        if 'sparks' in self.current_shader_name or 'additive' in self.current_shader_name:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive
        else:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Alpha
        
        shader_program.use()
        
        try:
            # Set common uniforms
            current_time = time.time() - self.start_time
            
            if 'time' in shader_program.uniforms:
                shader_program['time'] = current_time
            
            if 'resolution' in shader_program.uniforms:
                shader_program['resolution'] = (float(self.width), float(self.height))
            
            # Set adjustable uniforms
            for uniform_name, value in self.uniforms.items():
                if uniform_name in shader_program.uniforms:
                    shader_program[uniform_name] = value
            
            # Draw
            vertex_list.draw(GL_TRIANGLES)
            
        except Exception as e:
            logger.error(f"Error rendering {self.current_shader_name}: {e}")
        finally:
            shader_program.stop()
            glDisable(GL_BLEND)
    
    def switch_shader(self, direction=1):
        """Switch to next/previous shader."""
        if not self.shader_names:
            return
        
        self.current_shader_index = (self.current_shader_index + direction) % len(self.shader_names)
        self.current_shader_name = self.shader_names[self.current_shader_index]
        logger.info(f"Switched to shader: {self.current_shader_name} ({self.current_shader_index + 1}/{len(self.shader_names)})")
    
    def adjust_uniform(self, name, delta):
        """Adjust a uniform parameter."""
        if name in self.uniforms:
            self.uniforms[name] = max(0.0, min(2.0, self.uniforms[name] + delta))
            logger.info(f"{name}: {self.uniforms[name]:.2f}")
    
    def on_key_press(self, symbol, modifiers):
        """Handle keyboard input."""
        if symbol == pyglet.window.key.ESCAPE:
            self.window.close()
        
        elif symbol == pyglet.window.key.SPACE:
            self.switch_shader()
        
        elif symbol == pyglet.window.key.BACKSPACE:
            self.switch_shader(-1)
        
        elif symbol == pyglet.window.key.S:
            self.show_shaders = not self.show_shaders
            status = "ON" if self.show_shaders else "OFF"
            logger.info(f"Shaders {status}")
        
        elif symbol == pyglet.window.key.R:
            logger.info("Manually reloading shaders...")
            self.load_all_shaders()
            self.create_vertex_lists()
        
        elif symbol == pyglet.window.key.A:
            self.auto_reload = not self.auto_reload
            status = "ON" if self.auto_reload else "OFF"
            logger.info(f"Auto-reload {status}")
        
        elif symbol == pyglet.window.key.H:
            self._print_controls()
        
        # Uniform adjustments
        elif symbol == pyglet.window.key.Q:
            self.adjust_uniform('vignette_strength', 0.1 if not (modifiers & pyglet.window.key.MOD_SHIFT) else -0.1)
        elif symbol == pyglet.window.key.W:
            self.adjust_uniform('turbulence_amount', 0.05 if not (modifiers & pyglet.window.key.MOD_SHIFT) else -0.05)
        elif symbol == pyglet.window.key.E:
            self.adjust_uniform('spark_intensity', 0.1 if not (modifiers & pyglet.window.key.MOD_SHIFT) else -0.1)
        elif symbol == pyglet.window.key.T:
            self.adjust_uniform('full_mask_zone', 0.01 if not (modifiers & pyglet.window.key.MOD_SHIFT) else -0.01)
        elif symbol == pyglet.window.key.Y:
            self.adjust_uniform('gradient_zone', 0.05 if not (modifiers & pyglet.window.key.MOD_SHIFT) else -0.05)
        elif symbol == pyglet.window.key.U:
            self.adjust_uniform('horizontal_compression', 0.5 if not (modifiers & pyglet.window.key.MOD_SHIFT) else -0.5)
        
        # Quick shader selection (1-9)
        elif pyglet.window.key._1 <= symbol <= pyglet.window.key._9:
            index = symbol - pyglet.window.key._1
            if index < len(self.shader_names):
                self.current_shader_index = index
                self.current_shader_name = self.shader_names[index]
                logger.info(f"Selected shader: {self.current_shader_name}")
    
    def _print_controls(self):
        """Print control instructions."""
        print("\n" + "="*60)
        print("SHADER TEST HARNESS CONTROLS")
        print("="*60)
        print("Navigation:")
        print("  Space/Backspace : Next/Previous shader")
        print("  1-9             : Quick select shader")
        print("  S               : Toggle shaders on/off")
        print("  R               : Reload shaders")
        print("  A               : Toggle auto-reload")
        print("  H               : Show this help")
        print("  Escape          : Exit")
        print()
        print("Uniform Adjustments (Shift = decrease):")
        print("  Q/Shift+Q       : Vignette strength")
        print("  W/Shift+W       : Turbulence amount")
        print("  E/Shift+E       : Spark intensity")
        print("  T/Shift+T       : Full mask zone")
        print("  Y/Shift+Y       : Gradient zone")
        print("  U/Shift+U       : Horizontal compression")
        print()
        print(f"Available shaders ({len(self.shader_names)}):")
        for i, name in enumerate(self.shader_names, 1):
            marker = " <-- ACTIVE" if name == self.current_shader_name else ""
            print(f"  {i}: {name}{marker}")
        print("="*60 + "\n")
    
    def run(self):
        """Run the test harness."""
        logger.info("Starting shader test harness...")
        if self.shader_names:
            logger.info(f"Loaded {len(self.shader_names)} shaders. Press H for help.")
        else:
            logger.error("No shaders loaded. Check shader directory and files.")
        pyglet.app.run()


def main():
    """Main entry point."""
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if current_dir.name != "shaders":
        print("ERROR: This script should be run from the shaders directory.")
        print("Usage: cd services/display/shaders && uv run shader_test_harness.py")
        return 1
    
    try:
        harness = ShaderTestHarness()
        harness.run()
    except Exception as e:
        logger.error(f"Failed to start test harness: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
