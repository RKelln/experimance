#!/usr/bin/env python3
"""
Layer Manager for the Display Service.

Coordinates the rendering of multiple visual layers in the correct z-order:
1. Background layer (satellite landscape images)
2. Video overlay layer (masked video responding to sand interaction)
3. Text overlay layer (agent/system text)
4. Debug overlay layer (performance metrics, if enabled)

The layer manager handles opacity, visibility, and compositing of all layers.
"""

import logging
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod

from experimance_common.config import Config

logger = logging.getLogger(__name__)


class LayerRenderer(ABC):
    """Abstract base class for layer renderers."""
    
    @abstractmethod
    def update(self, dt: float):
        """Update the layer state."""
        pass
    
    @abstractmethod
    def render(self):
        """Render the layer."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up layer resources."""
        pass
    
    @property
    @abstractmethod
    def is_visible(self) -> bool:
        """Check if the layer should be rendered."""
        pass
    
    @property
    @abstractmethod
    def opacity(self) -> float:
        """Get the layer opacity (0.0 to 1.0)."""
        pass

    @abstractmethod
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize events.
        
        Args:
            new_size: New (width, height) of the window
        """
        # Default implementation does nothing, can be overridden by subclasses
        pass


class LayerManager:
    """Manages multiple rendering layers with proper z-order and compositing."""
    
    def __init__(self, window_size: Tuple[int, int], config: Config):
        """Initialize the layer manager.
        
        Args:
            window_size: (width, height) of the display window
            config: Display service configuration
        """
        self.window_size = window_size
        self.config = config
        
        # Layer registry in render order (bottom to top)
        self.layers: Dict[str, LayerRenderer] = {}
        self.layer_order = ["background", "video_overlay", "text_overlay", "debug_overlay"]
        
        # Performance tracking
        self.frame_count = 0
        self.total_render_time = 0.0
        
        logger.info(f"LayerManager initialized for {window_size[0]}x{window_size[1]}")
    
    def register_renderer(self, layer_name: str, renderer: LayerRenderer):
        """Register a renderer for a specific layer.
        
        Args:
            layer_name: Name of the layer (must be in layer_order)
            renderer: LayerRenderer instance
        """
        if layer_name not in self.layer_order:
            logger.warning(f"Unknown layer name: {layer_name}")
            return
        
        self.layers[layer_name] = renderer
        logger.info(f"Registered renderer for layer: {layer_name}")
    
    def unregister_renderer(self, layer_name: str):
        """Unregister a renderer for a specific layer.
        
        Args:
            layer_name: Name of the layer to unregister
        """
        if layer_name in self.layers:
            del self.layers[layer_name]
            logger.info(f"Unregistered renderer for layer: {layer_name}")
    
    def get_renderer(self, layer_name: str) -> Optional[LayerRenderer]:
        """Get the renderer for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            LayerRenderer instance or None if not found
        """
        return self.layers.get(layer_name)
    
    def update(self, dt: float):
        """Update all layer states.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        # Update all registered layers
        for layer_name in self.layer_order:
            if layer_name in self.layers:
                try:
                    self.layers[layer_name].update(dt)
                except Exception as e:
                    logger.error(f"Error updating layer {layer_name}: {e}", exc_info=True)
    
    def render(self):
        """Render all visible layers in the correct z-order."""
        import time
        start_time = time.time()
        
        # Render layers from bottom to top
        layers_rendered = 0
        
        for layer_name in self.layer_order:
            if layer_name in self.layers:
                renderer = self.layers[layer_name]
                
                # Skip invisible layers for performance
                if not renderer.is_visible:
                    continue
                
                try:
                    renderer.render()
                    layers_rendered += 1
                except Exception as e:
                    logger.error(f"Error rendering layer {layer_name}: {e}", exc_info=True)
        
        # Track performance
        render_time = time.time() - start_time
        self.total_render_time += render_time
        self.frame_count += 1
        
        # Log performance every 300 frames (10 seconds at 30fps)
        if self.frame_count % 300 == 0:
            avg_render_time = self.total_render_time / self.frame_count
            logger.debug(f"Average render time: {avg_render_time*1000:.2f}ms, Layers rendered: {layers_rendered}")
    
    def set_layer_visibility(self, layer_name: str, visible: bool):
        """Set the visibility of a specific layer.
        
        Args:
            layer_name: Name of the layer
            visible: Whether the layer should be visible
        """
        if layer_name in self.layers:
            # This would require adding visibility property to LayerRenderer
            # For now, just log the request
            logger.info(f"Layer visibility change requested: {layer_name} = {visible}")
        else:
            logger.warning(f"Cannot set visibility for unknown layer: {layer_name}")
    
    def set_layer_opacity(self, layer_name: str, opacity: float):
        """Set the opacity of a specific layer.
        
        Args:
            layer_name: Name of the layer
            opacity: Opacity value (0.0 to 1.0)
        """
        if layer_name in self.layers:
            # This would require adding opacity setter to LayerRenderer
            # For now, just log the request
            opacity = max(0.0, min(1.0, opacity))  # Clamp to valid range
            logger.info(f"Layer opacity change requested: {layer_name} = {opacity}")
        else:
            logger.warning(f"Cannot set opacity for unknown layer: {layer_name}")
    
    def get_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered layers.
        
        Returns:
            Dictionary with layer information
        """
        info = {}
        
        for layer_name in self.layer_order:
            if layer_name in self.layers:
                renderer = self.layers[layer_name]
                info[layer_name] = {
                    "registered": True,
                    "visible": renderer.is_visible,
                    "opacity": renderer.opacity,
                    "type": type(renderer).__name__
                }
            else:
                info[layer_name] = {
                    "registered": False,
                    "visible": False,
                    "opacity": 0.0,
                    "type": None
                }
        
        return info
    
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize events.
        
        Args:
            new_size: New (width, height) of the window
        """
        if new_size != self.window_size:
            logger.info(f"Window resized from {self.window_size} to {new_size}")
            self.window_size = new_size
            
            # Notify all layers of the resize
            for layer_name, renderer in self.layers.items():
                if hasattr(renderer, 'resize'):
                    try:
                        renderer.resize(new_size)
                    except Exception as e:
                        logger.error(f"Error resizing layer {layer_name}: {e}", exc_info=True)
    
    async def cleanup(self):
        """Clean up all layer resources."""
        logger.info("Cleaning up LayerManager...")
        
        # Clean up all registered layers
        for layer_name, renderer in self.layers.items():
            try:
                await renderer.cleanup()
                logger.debug(f"Cleaned up layer: {layer_name}")
            except Exception as e:
                logger.error(f"Error cleaning up layer {layer_name}: {e}", exc_info=True)
        
        # Clear the registry
        self.layers.clear()
        
        logger.info("LayerManager cleanup complete")
