#!/usr/bin/env python3
"""
Single Shader Renderer for the Display Service.

A clean, single-responsibility renderer that loads and executes a single
fragment shader from an external file. Designed to be used with the layer
manager for compositing multiple shader effects.

This renderer follows the single responsibility principle - each instance
handles exactly one shader effect. Multiple effects are achieved by using
multiple instances coordinated through the layer manager.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pyglet
from pyglet.gl import *
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4

from experimance_display.config import DisplayServiceConfig
from experimance_display.renderers.layer_manager import LayerRenderer

logger = logging.getLogger(__name__)


class ShaderRenderer(LayerRenderer):
    """A renderer that loads and executes a single fragment shader.
    
    This renderer is designed for the single responsibility principle - each
    instance handles exactly one shader effect. Multiple effects are achieved
    by using multiple instances in the layer manager.
    """
    
    def __init__(self, 
                 config: DisplayServiceConfig,
                 window: pyglet.window.BaseWindow,
                 batch: pyglet.graphics.Batch,
                 shader_path: str,
                 order: int = 10,
                 uniforms: Optional[Dict[str, Any]] = None):
        """Initialize the single shader renderer.
        
        Args:
            config: Display service configuration
            window: Pyglet window instance
            batch: Graphics batch for efficient rendering
            shader_path: Path to the fragment shader file (.frag)
            order: Render order (higher numbers render on top)
            uniforms: Dictionary of uniform values to pass to shader
        """
        super().__init__(config, window, batch, order)
        
        self.shader_path = Path(shader_path)
        self.uniforms = uniforms or {}
        
        # Shader state
        self._visible = True
        self._opacity = 1.0
        self.shader_program: Optional[ShaderProgram] = None
        
        # Scene texture for shaders that need it
        self.scene_texture = None
        
        # Timing
        self.start_time = time.time()
        
        # Full-screen quad for shader rendering
        self.quad_vertex_list = None
        
        # Initialize OpenGL and load shader
        self._init_opengl()
        self._load_shader()
        self._create_screen_quad()
        self._load_scene_texture()
        
        logger.info(f"ShaderRenderer initialized with shader: {self.shader_path.name}")
    
    def _init_opengl(self):
        """Initialize OpenGL settings for shader rendering."""
        # Enable blending for transparency effects
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Check shader support with compatibility for different context types
        try:
            if hasattr(self.window.context, 'check_gl_extension'):
                # Our mock context
                shader_support = self.window.context.check_gl_extension('GL_ARB_shading_language_100')
            elif hasattr(self.window.context, 'get_info'):
                # Real pyglet context - check OpenGL version instead
                gl_info = self.window.context.get_info()
                # Assume shaders are supported if we can get GL info
                shader_support = True
                logger.debug(f"OpenGL context info available: {type(gl_info)}")
            else:
                # Fallback - try to import shader classes to test support
                try:
                    from pyglet.graphics.shader import Shader
                    shader_support = True
                    logger.debug("Shader support detected via Shader class import")
                except ImportError:
                    shader_support = False
                    logger.warning("Shader support not detected - Shader class import failed")
        except Exception as e:
            logger.warning(f"Could not determine shader support: {e}")
            # Assume shaders are supported and let shader compilation fail if they're not
            shader_support = True
        
        if not shader_support:
            logger.error("OpenGL shaders not supported")
            self._visible = False
            return
        
        logger.debug("OpenGL shader support confirmed")
    
    def _load_shader(self):
        """Load the fragment shader from file."""
        if not self.shader_path.exists():
            logger.error(f"Shader file not found: {self.shader_path}")
            self._visible = False
            return
        
        try:
            # Load fragment shader from file
            with open(self.shader_path, 'r') as f:
                fragment_source = f.read()
            
            # Standard vertex shader for full-screen quad
            vertex_source = """#version 150 core
in vec2 position;
in vec2 tex_coords;
out vec2 v_tex;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_tex = tex_coords;
}
"""
            
            # Create shader program
            vertex_shader = Shader(vertex_source, 'vertex')
            fragment_shader = Shader(fragment_source, 'fragment')
            self.shader_program = ShaderProgram(vertex_shader, fragment_shader)
            
            logger.info(f"Successfully loaded shader: {self.shader_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load shader {self.shader_path}: {e}")
            self._visible = False
    
    def _create_screen_quad(self):
        """Create a full-screen quad for shader rendering using Pyglet's vertex list."""
        if not self.shader_program:
            return
        
        # Vertex data for a full-screen quad in NDC coordinates
        position_data = [
            -1.0, -1.0,  # Bottom-left
             1.0, -1.0,  # Bottom-right  
             1.0,  1.0,  # Top-right
            -1.0,  1.0   # Top-left
        ]
        
        texcoord_data = [
            0.0, 0.0,  # Bottom-left
            1.0, 0.0,  # Bottom-right
            1.0, 1.0,  # Top-right
            0.0, 1.0   # Top-left
        ]
        
        # Indices for two triangles forming a quad
        indices = [0, 1, 2, 2, 3, 0]
        
        # Create vertex list using the shader program (this is the correct Pyglet API)
        try:
            self.quad_vertex_list = self.shader_program.vertex_list_indexed(
                4,  # 4 vertices
                pyglet.gl.GL_TRIANGLES,
                indices,
                batch=self.batch,
                group=self,
                position=('f', position_data),
                tex_coords=('f', texcoord_data)
            )
            logger.debug(f"Created vertex list for shader: {self.shader_path.name}")
        except Exception as e:
            logger.error(f"Failed to create vertex list for shader {self.shader_path.name}: {e}")
            self.quad_vertex_list = None
        
        logger.debug("Created full-screen quad vertex list for shader rendering")
    
    def update(self, dt: float):
        """Update shader state.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        if not self._visible or not self.shader_program:
            return
        
        # Update timing-based uniforms automatically
        current_time = time.time() - self.start_time
        self.uniforms['time'] = current_time
    
    def render(self):
        """Render the shader effect using Pyglet's batch system.
        
        Since we're using the batch system with a Group, this method is primarily 
        used for updating uniforms. The actual drawing is handled by the batch.
        """
        if not self._visible or not self.shader_program or not self.quad_vertex_list:
            return

        # Update uniforms before batch rendering
        try:
            # Set common uniforms only if they exist in the shader
            current_time = time.time() - self.start_time
            if 'time' in self.shader_program.uniforms:
                self.shader_program['time'] = current_time
            
            # Set resolution uniform only if declared in shader
            if 'resolution' in self.shader_program.uniforms:
                self.shader_program['resolution'] = (float(self.window.width), float(self.window.height))
            
            # Set custom uniforms
            for uniform_name, value in self.uniforms.items():
                if uniform_name in self.shader_program.uniforms:
                    self.shader_program[uniform_name] = value
            
        except Exception as e:
            logger.error(f"Error setting uniforms for shader {self.shader_path.name}: {e}")
    
    def set_state(self):
        """Set OpenGL state for rendering (called by batch system)."""
        if not self._visible or not self.shader_program:
            return
            
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Use shader program
        self.shader_program.use()
        
        # Handle scene_texture uniform if shader needs it
        if 'scene_texture' in self.shader_program.uniforms:
            # For now, use our test image as scene_texture
            # TODO: In full implementation, this would capture the current framebuffer
            self._bind_scene_texture()
        
        # Update uniforms
        try:
            current_time = time.time() - self.start_time
            if 'time' in self.shader_program.uniforms:
                self.shader_program['time'] = current_time
            
            if 'resolution' in self.shader_program.uniforms:
                self.shader_program['resolution'] = (float(self.window.width), float(self.window.height))
            
            for uniform_name, value in self.uniforms.items():
                if uniform_name in self.shader_program.uniforms:
                    self.shader_program[uniform_name] = value
                    
        except Exception as e:
            logger.error(f"Error setting uniforms for shader {self.shader_path.name}: {e}")
    
    def unset_state(self):
        """Unset OpenGL state for rendering (called by batch system)."""
        try:
            ShaderProgram.unbind()
            glDisable(GL_BLEND)
        except Exception as e:
            logger.error(f"Error unsetting state for shader {self.shader_path.name}: {e}")
    
    def set_uniform(self, name: str, value: Any):
        """Set a uniform value for the shader.
        
        Args:
            name: Name of the uniform
            value: Value to set
        """
        self.uniforms[name] = value
        logger.debug(f"Set uniform {name} = {value} for shader {self.shader_path.name}")
    
    def get_uniform(self, name: str) -> Any:
        """Get a uniform value from the shader.
        
        Args:
            name: Name of the uniform
            
        Returns:
            The uniform value or None if not found
        """
        return self.uniforms.get(name)
    
    @property
    def is_visible(self) -> bool:
        """Check if the shader layer should be rendered."""
        return self._visible and self.shader_program is not None
    
    @property
    def opacity(self) -> float:
        """Get the layer opacity."""
        return self._opacity
    
    @opacity.setter
    def opacity(self, value: float):
        """Set the layer opacity.
        
        Args:
            value: Opacity value (0.0 to 1.0)
        """
        self._opacity = max(0.0, min(1.0, value))
    
    @property
    def visible(self) -> bool:
        """Get the layer visibility."""
        return self._visible
    
    @visible.setter
    def visible(self, value: bool):
        """Set the layer visibility.
        
        Args:
            value: Whether the layer should be visible
        """
        self._visible = value
    
    def reload_shader(self):
        """Reload the shader from file.
        
        Useful for development and hot-reloading of shader effects.
        """
        logger.info(f"Reloading shader: {self.shader_path.name}")
        
        # Clean up existing shader
        if self.shader_program:
            if hasattr(self.shader_program, 'delete'):
                self.shader_program.delete()
            self.shader_program = None
        
        # Reload
        self._load_shader()
        
        if self.shader_program:
            logger.info(f"Successfully reloaded shader: {self.shader_path.name}")
        else:
            logger.error(f"Failed to reload shader: {self.shader_path.name}")
    
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize.
        
        Args:
            new_size: New (width, height) of the window
        """
        # Update resolution uniform
        self.uniforms['resolution'] = (float(new_size[0]), float(new_size[1]))
        logger.debug(f"Shader {self.shader_path.name} resized to: {new_size}")
    
    async def cleanup(self):
        """Clean up shader resources."""
        if self.shader_program:
            if hasattr(self.shader_program, 'delete'):
                self.shader_program.delete()
            self.shader_program = None
        
        if self.quad_vertex_list:
            if hasattr(self.quad_vertex_list, 'delete'):
                self.quad_vertex_list.delete()
            self.quad_vertex_list = None
        
        logger.info(f"Cleaned up shader renderer: {self.shader_path.name}")
    
    def _load_scene_texture(self):
        """Load a scene texture for shaders that need it."""
        # TODO: In a proper implementation, this would capture the current framebuffer
        # or load from a configured scene source. For now, no scene texture.
        self.scene_texture = None
        logger.debug("No scene texture configured - shaders will need to generate their own content")
    
    def _bind_scene_texture(self):
        """Bind scene texture for shader if available."""
        if (self.scene_texture and 
            self.shader_program and 
            'scene_texture' in self.shader_program.uniforms):
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.scene_texture.id)
            self.shader_program['scene_texture'] = 0
            logger.debug("Bound scene texture to shader")


class MultiShaderRenderer:
    """Manages multiple ShaderRenderer instances for complex effects.
    
    This class coordinates multiple shader renderers to create layered effects
    while maintaining the single responsibility principle for individual shaders.
    """
    
    def __init__(self, 
                 config: DisplayServiceConfig,
                 window: pyglet.window.BaseWindow,
                 batch: pyglet.graphics.Batch):
        """Initialize the multi-shader renderer.
        
        Args:
            config: Display service configuration
            window: Pyglet window instance
            batch: Graphics batch for efficient rendering
        """
        self.config = config
        self.window = window
        self.batch = batch
        
        self.shader_renderers: Dict[str, ShaderRenderer] = {}
        
        logger.info("MultiShaderRenderer initialized")
    
    def add_shader(self, 
                   name: str,
                   shader_path: str,
                   order: int = 10,
                   uniforms: Optional[Dict[str, Any]] = None) -> ShaderRenderer:
        """Add a shader renderer.
        
        Args:
            name: Unique name for the shader
            shader_path: Path to the fragment shader file
            order: Render order (higher numbers render on top)
            uniforms: Initial uniform values
            
        Returns:
            The created ShaderRenderer instance
        """
        if name in self.shader_renderers:
            logger.warning(f"Shader '{name}' already exists, replacing it")
            # Clean up existing shader
            old_shader = self.shader_renderers[name]
            # Note: cleanup is async, but we'll call it sync here for simplicity
            # In production, this should be handled properly with async/await
        
        shader_renderer = ShaderRenderer(
            config=self.config,
            window=self.window,
            batch=self.batch,
            shader_path=shader_path,
            order=order,
            uniforms=uniforms
        )
        
        self.shader_renderers[name] = shader_renderer
        logger.info(f"Added shader '{name}' with order {order}")
        
        return shader_renderer
    
    def remove_shader(self, name: str):
        """Remove a shader renderer.
        
        Args:
            name: Name of the shader to remove
        """
        if name in self.shader_renderers:
            shader = self.shader_renderers[name]
            # Note: cleanup should be async
            del self.shader_renderers[name]
            logger.info(f"Removed shader '{name}'")
        else:
            logger.warning(f"Shader '{name}' not found for removal")
    
    def get_shader(self, name: str) -> Optional[ShaderRenderer]:
        """Get a shader renderer by name.
        
        Args:
            name: Name of the shader
            
        Returns:
            The ShaderRenderer instance or None if not found
        """
        return self.shader_renderers.get(name)
    
    def update(self, dt: float):
        """Update all shader renderers.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        for shader in self.shader_renderers.values():
            shader.update(dt)
    
    def render_all(self):
        """Render all shader effects in order."""
        # Sort by render order
        sorted_shaders = sorted(
            self.shader_renderers.values(),
            key=lambda s: s.order
        )
        
        for shader in sorted_shaders:
            if shader.is_visible:
                shader.render()
    
    def set_uniform_all(self, name: str, value: Any):
        """Set a uniform value for all shaders.
        
        Args:
            name: Name of the uniform
            value: Value to set
        """
        for shader in self.shader_renderers.values():
            shader.set_uniform(name, value)
    
    def reload_all_shaders(self):
        """Reload all shaders from files."""
        logger.info("Reloading all shaders")
        for shader in self.shader_renderers.values():
            shader.reload_shader()
    
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize for all shaders.
        
        Args:
            new_size: New (width, height) of the window
        """
        for shader in self.shader_renderers.values():
            shader.resize(new_size)
    
    async def cleanup(self):
        """Clean up all shader resources."""
        logger.info("Cleaning up MultiShaderRenderer")
        
        for name, shader in self.shader_renderers.items():
            await shader.cleanup()
        
        self.shader_renderers.clear()
        logger.info("MultiShaderRenderer cleanup complete")
