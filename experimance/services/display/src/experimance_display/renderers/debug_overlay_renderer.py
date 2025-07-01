#!/usr/bin/env python3
"""
Debug Overlay Renderer for the Display Service.

Renders debug information including:
- Window center crosshair
- FPS counter
- Layer information
- Performance metrics
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

from experimance_display.config import DisplayServiceConfig
import pyglet
from pyglet.gl import GL_LINES
from pyglet.graphics import Batch
from pyglet.shapes import Line
from pyglet.text import Label

from .layer_manager import LayerRenderer

logger = logging.getLogger(__name__)


class DebugOverlayRenderer(LayerRenderer):
    """Renders debug overlay with crosshair, FPS, and performance info."""
    
    def __init__(self, config: DisplayServiceConfig, 
                 window: pyglet.window.BaseWindow, 
                 batch: pyglet.graphics.Batch,
                 layer_manager: Any = None,
                 order: int = 3):
        """Initialize the debug overlay renderer."""
        super().__init__(config=config, window=window, batch=batch, order=order)
        
        self.layer_manager = layer_manager
        
        # Visibility and opacity
        self._visible = True
        self._opacity = 1.0
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_timer = 0.0
        self.current_fps = 0.0
        
        # Crosshair settings
        self.crosshair_enabled = True
        self.crosshair_size = 20  # Length of crosshair arms in pixels
        self.crosshair_color = (255, 0, 0, 200)  # Red with some transparency (RGBA 0-255)
        
        # Crosshair lines
        self.crosshair_horizontal = None
        self.crosshair_vertical = None
        self._create_crosshair()
        
        # FPS label
        self.fps_label = None
        self._create_fps_label()
        
        logger.info(f"DebugOverlayRenderer initialized for {window}")
    
    def _create_fps_label(self):
        """Create the FPS display label."""
        # Position in top-left corner
        x = 10
        y = self.window.get_size()[1] - 30
        
        self.fps_label = Label(
            text="FPS: --",
            font_name="Arial",
            font_size=16,
            color=(255, 255, 255, 200),  # White with slight transparency
            x=x,
            y=y,
            anchor_x="left",
            anchor_y="top",
            batch=self.batch,
            group=self,
        )
    
    def _create_crosshair(self):
        """Create the crosshair lines."""
        window_size = self.window.get_size()
        center_x = window_size[0] // 2
        center_y = window_size[1] // 2
        
        # Create horizontal line
        self.crosshair_horizontal = Line(
            center_x - self.crosshair_size, center_y,
            center_x + self.crosshair_size, center_y,
            color=self.crosshair_color,
            batch=self.batch,
            group=self,
        )
        
        # Create vertical line
        self.crosshair_vertical = Line(
            center_x, center_y - self.crosshair_size,
            center_x, center_y + self.crosshair_size,
            color=self.crosshair_color,
            batch=self.batch,
            group=self,
        )
    
    @property
    def is_visible(self) -> bool:
        """Check if the layer should be rendered."""
        return self._visible and self.config.display.debug_overlay
    
    @property
    def opacity(self) -> float:
        """Get the layer opacity (0.0 to 1.0)."""
        return self._opacity
    
    def update(self, dt: float):
        """Update debug overlay state and ensure all elements are in the batch and group.
        Args:
            dt: Time elapsed since last update in seconds
        """
        if not self.is_visible:
            # Hide all debug overlay elements
            if self.fps_label:
                self.fps_label.visible = False
            if self.crosshair_horizontal:
                self.crosshair_horizontal.visible = False
            if self.crosshair_vertical:
                self.crosshair_vertical.visible = False
            return

        # Show all debug overlay elements
        if self.fps_label:
            self.fps_label.visible = True
        if self.crosshair_horizontal:
            self.crosshair_horizontal.visible = self.crosshair_enabled
        if self.crosshair_vertical:
            self.crosshair_vertical.visible = self.crosshair_enabled

        # Update FPS calculation
        self.fps_counter += 1
        self.fps_timer += dt

        # Update FPS display every second
        if self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / self.fps_timer
            self.fps_counter = 0
            self.fps_timer = 0.0

            # Update FPS label text
            if self.fps_label:
                layer_count = 0
                if self.layer_manager:
                    layer_info = self.layer_manager.get_layer_info()
                    layer_count = sum(1 for info in layer_info.values() if info.get("visible", False))
                self.fps_label.text = f"FPS: {self.current_fps:.1f} | Layers: {layer_count}"
    
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize.
        
        Args:
            new_size: New (width, height) of the window
        """
        if new_size != self.window.get_size():
            logger.debug(f"DebugOverlayRenderer resize: {new_size} -> {new_size}")
            
            # Recreate FPS label with new position
            self._create_fps_label()
            
            # Recreate crosshair with new center position
            self._create_crosshair()
    
    def set_visibility(self, visible: bool):
        """Set layer visibility.
        
        Args:
            visible: Whether the layer should be visible
        """
        self._visible = visible
        logger.debug(f"DebugOverlayRenderer visibility: {visible}")
    
    def set_opacity(self, opacity: float):
        """Set layer opacity.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        self._opacity = max(0.0, min(1.0, opacity))
        logger.debug(f"DebugOverlayRenderer opacity: {self._opacity}")
    
    def set_crosshair_enabled(self, enabled: bool):
        """Enable or disable crosshair rendering.
        
        Args:
            enabled: Whether to render the crosshair
        """
        self.crosshair_enabled = enabled
        logger.debug(f"Crosshair enabled: {enabled}")
    
    def set_crosshair_size(self, size: int):
        """Set the size of the crosshair arms.
        
        Args:
            size: Length of crosshair arms in pixels
        """
        self.crosshair_size = max(1, size)
        logger.debug(f"Crosshair size: {self.crosshair_size}")
    
    def set_crosshair_color(self, color: Tuple[int, int, int, int]):
        """Set the color of the crosshair.
        
        Args:
            color: RGBA color tuple (values 0 to 255)
        """
        self.crosshair_color = color
        logger.debug(f"Crosshair color: {color}")
        
        # Recreate crosshair with new color
        self._create_crosshair()

    async def cleanup(self):
        """Clean up debug overlay renderer resources."""
        logger.info("Cleaning up DebugOverlayRenderer...")
        
        # Clear label and crosshair
        self.fps_label = None
        self.crosshair_horizontal = None
        self.crosshair_vertical = None
        
        logger.info("DebugOverlayRenderer cleanup complete")
