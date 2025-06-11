#!/usr/bin/env python3
"""
Text Overlay Manager for the Display Service.

Manages multiple concurrent text overlays with different styles and positions.
Supports:
- Multiple text items with unique IDs
- Speaker-specific styling (agent, system, debug)
- Text replacement for streaming updates
- Automatic expiration based on duration
- Different positioning options
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

from pyglet.text import Label

from .layer_manager import LayerRenderer

from pyglet.customtypes import AnchorX, AnchorY

logger = logging.getLogger(__name__)


class TextItem:
    """Represents a single text overlay item."""
    
    def __init__(
        self,
        text_id: str,
        content: str,
        label: Label,
        duration: Optional[float] = None,
        creation_time: Optional[float] = None
    ):
        """Initialize a text item.
        
        Args:
            text_id: Unique identifier for this text
            content: Text content to display
            label: Pyglet label for rendering
            duration: Duration in seconds (None for infinite)
            creation_time: Time when text was created
        """
        self.text_id = text_id
        self.content = content
        self.label = label
        self.duration = duration
        self.creation_time = creation_time or time.time()
        
        # Animation state
        self.opacity = 1.0
        self.fade_in_progress = 0.0
        self.fade_out_progress = 0.0
        self.is_fading_in = True
        self.is_fading_out = False
    
    def is_expired(self) -> bool:
        """Check if this text item has expired."""
        if self.duration is None:
            return False
        return (time.time() - self.creation_time) >= self.duration
    
    def update(self, dt: float, fade_duration: float):
        """Update text animation state.
        
        Args:
            dt: Time elapsed since last update
            fade_duration: Duration of fade in/out animations
        """
        # Handle fade in
        if self.is_fading_in:
            self.fade_in_progress += dt / fade_duration
            if self.fade_in_progress >= 1.0:
                self.fade_in_progress = 1.0
                self.is_fading_in = False
            self.opacity = self.fade_in_progress
        
        # Handle fade out
        elif self.is_fading_out:
            self.fade_out_progress += dt / fade_duration
            if self.fade_out_progress >= 1.0:
                self.fade_out_progress = 1.0
                self.opacity = 0.0
            else:
                self.opacity = 1.0 - self.fade_out_progress
        
        # Update label opacity
        self.label.color = (*self.label.color[:3], int(self.opacity * 255))
    
    def start_fade_out(self):
        """Start fading out this text."""
        if not self.is_fading_out:
            self.is_fading_out = True
            self.is_fading_in = False
            self.fade_out_progress = 0.0


class TextOverlayManager(LayerRenderer):
    """Manages multiple text overlays with different styles and positions."""
    
    def __init__(self, window_size: Tuple[int, int], config: Any, transitions_config: Any):
        """Initialize the text overlay manager.
        
        Args:
            window_size: (width, height) of the display window
            config: Text styles configuration
            transitions_config: Transition configuration
        """
        self.window_size = window_size
        self.config = config
        self.transitions_config = transitions_config
        
        # Active text items
        self.text_items: Dict[str, TextItem] = {}
        
        # Positioning calculations
        self.position_map = self._create_position_map()
        
        # Visibility and opacity
        self._visible = True
        self._opacity = 1.0
        
        logger.info(f"TextOverlayManager initialized for {window_size[0]}x{window_size[1]}")
    
    @property
    def active_texts(self) -> Dict[str, TextItem]:
        """Compatibility property for tests to access text_items."""
        return self.text_items
    
    @property
    def is_visible(self) -> bool:
        """Check if the layer should be rendered."""
        return self._visible and len(self.text_items) > 0
    
    @property
    def opacity(self) -> float:
        """Get the layer opacity (0.0 to 1.0)."""
        return self._opacity
    
    def _create_position_map(self) -> Dict[str, Tuple[int, int, str]]:
        """Create mapping from position names to (x, y, anchor) coordinates.
        
        Returns:
            Dictionary mapping position names to (x, y, anchor_position)
        """
        width, height = self.window_size
        margin = 20  # Margin from edges
        
        return {
            "top_left": (margin, height - margin, "top_left"),
            "top_center": (width // 2, height - margin, "top_center"),
            "top_right": (width - margin, height - margin, "top_right"),
            "center_left": (margin, height // 2, "center_left"),
            "center": (width // 2, height // 2, "center"),
            "center_right": (width - margin, height // 2, "center_right"),
            "bottom_left": (margin, margin, "bottom_left"),
            "bottom_center": (width // 2, margin, "bottom_center"),
            "bottom_right": (width - margin, margin, "bottom_right"),
        }
    
    def update(self, dt: float):
        """Update all text items.
        
        Args:
            dt: Time elapsed since last update in seconds
        """
        # Update animation states
        items_to_remove = []
        fade_duration = self.transitions_config.text_fade_duration
        
        for text_id, item in self.text_items.items():
            # Check for expiration
            if item.is_expired() and not item.is_fading_out:
                item.start_fade_out()
            
            # Update animation
            item.update(dt, fade_duration)
            
            # Mark for removal if fully faded out
            if item.is_fading_out and item.fade_out_progress >= 1.0:
                items_to_remove.append(text_id)
        
        # Remove fully faded items
        for text_id in items_to_remove:
            del self.text_items[text_id]
            logger.debug(f"Removed expired text: {text_id}")
    
    def render(self):
        """Render all visible text items."""
        if not self.is_visible:
            return
        
        try:
            for item in self.text_items.values():
                if item.opacity > 0:
                    item.label.draw()
        except Exception as e:
            logger.error(f"Error rendering text overlays: {e}", exc_info=True)
    
    async def handle_text_overlay(self, message: Dict[str, Any]):
        """Handle TextOverlay message.
        
        Args:
            message: TextOverlay message
        """
        try:
            text_id = message["text_id"]
            content = message["content"]
            speaker = message.get("speaker", "system")
            duration = message.get("duration")
            style_overrides = message.get("style", {})
            
            logger.info(f"Adding text overlay: {text_id} ({speaker})")
            
            # Get base style for speaker
            style = self._get_style_for_speaker(speaker)
            
            # Apply style overrides
            style.update(style_overrides)
            
            # Create label
            label = self._create_label(content, style)
            
            # Create text item
            text_item = TextItem(
                text_id=text_id,
                content=content,
                label=label,
                duration=duration
            )
            
            # If text with same ID exists, replace it (for streaming text)
            if text_id in self.text_items:
                logger.debug(f"Replacing existing text: {text_id}")
            
            self.text_items[text_id] = text_item
            
        except Exception as e:
            logger.error(f"Error handling TextOverlay: {e}", exc_info=True)
    
    async def handle_remove_text(self, message: Dict[str, Any]):
        """Handle RemoveText message.
        
        Args:
            message: RemoveText message with text_id
        """
        try:
            text_id = message["text_id"]
            
            if text_id in self.text_items:
                logger.info(f"Removing text overlay: {text_id}")
                self.text_items[text_id].start_fade_out()
            else:
                logger.warning(f"Cannot remove unknown text: {text_id}")
                
        except Exception as e:
            logger.error(f"Error handling RemoveText: {e}", exc_info=True)
    
    def _get_style_for_speaker(self, speaker: str) -> Dict[str, Any]:
        """Get style configuration for a speaker.
        
        Args:
            speaker: Speaker name (e.g., "agent", "system", "debug")
            
        Returns:
            Style configuration dictionary
        """
        # Map speaker to config attribute
        if hasattr(self.config, speaker):
            style_config = getattr(self.config, speaker)
            return {
                "font_size": style_config.font_size,
                "color": style_config.color,
                "position": style_config.position,
                "background": style_config.background,
                "background_color": style_config.background_color,
                "padding": style_config.padding,
                "max_width": style_config.max_width,
            }
        else:
            # Fallback to system style
            logger.warning(f"Unknown speaker '{speaker}', using system style")
            style_config = self.config.system
            return {
                "font_size": style_config.font_size,
                "color": style_config.color,
                "position": style_config.position,
                "background": style_config.background,
                "background_color": style_config.background_color,
                "padding": style_config.padding,
                "max_width": style_config.max_width,
            }
    
    def _create_label(self, content: str, style: Dict[str, Any]) -> Label:
        """Create a pyglet label with the specified style.
        
        Args:
            content: Text content
            style: Style configuration
            
        Returns:
            Configured pyglet Label
        """
        # Get position
        position_name = style.get("position", "bottom_center")
        if position_name in self.position_map:
            x, y, anchor = self.position_map[position_name]
        else:
            logger.warning(f"Unknown position '{position_name}', using bottom_center")
            x, y, anchor = self.position_map["bottom_center"]
        
        # Handle text wrapping
        max_width = style.get("max_width")
        if max_width:
            # Simple word wrapping (could be enhanced)
            content = self._wrap_text(content, max_width, style["font_size"])
        
        # Create label
        color = style.get("color", (255, 255, 255, 255))
        font_size = style.get("font_size", 24)
        
        # Set default width for multiline text if not provided
        if max_width is None:
            # Use a reasonable default based on window width (80% of window width)
            max_width = int(self.window_size[0] * 0.8)
        
        label = Label(
            text=content,
            font_name="Arial",  # Could be configurable
            font_size=font_size,
            color=color,
            x=x,
            y=y,
            anchor_x=self._get_anchor_x(anchor),
            anchor_y=self._get_anchor_y(anchor),
            multiline=True,
            width=max_width
        )
        
        return label
    
    def _wrap_text(self, text: str, max_width: int, font_size: int) -> str:
        """Simple text wrapping based on character count.
        
        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            font_size: Font size
            
        Returns:
            Wrapped text with newlines
        """
        # Rough character count based on font size
        chars_per_line = max_width // (font_size // 2)
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= chars_per_line:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)
    
    def _get_anchor_x(self, anchor: str) -> AnchorX:
        """Get pyglet anchor_x value from position anchor.
        
        Args:
            anchor: Position anchor string
            
        Returns:
            Pyglet anchor_x value
        """
        if "left" in anchor:
            return "left"
        elif "right" in anchor:
            return "right"
        else:
            return "center"
    
    def _get_anchor_y(self, anchor: str) -> AnchorY:
        """Get pyglet anchor_y value from position anchor.
        
        Args:
            anchor: Position anchor string
            
        Returns:
            Pyglet anchor_y value
        """
        # AnchorY = "top", "bottom", "center", "baseline"
        if "top" in anchor:
            return "top"
        elif "bottom" in anchor:
            return "bottom"
        elif "baseline" in anchor:
            return "baseline"
        else:
            return "center"
    
    def resize(self, new_size: Tuple[int, int]):
        """Handle window resize.
        
        Args:
            new_size: New (width, height) of the window
        """
        if new_size != self.window_size:
            logger.debug(f"TextOverlayManager resize: {self.window_size} -> {new_size}")
            self.window_size = new_size
            
            # Recreate position map
            self.position_map = self._create_position_map()
            
            # Update all existing text positions
            for item in self.text_items.values():
                style = {"position": "bottom_center"}  # Default, could be stored
                label = self._create_label(item.content, style)
                item.label = label
    
    def set_visibility(self, visible: bool):
        """Set layer visibility.
        
        Args:
            visible: Whether the layer should be visible
        """
        self._visible = visible
        logger.debug(f"TextOverlayManager visibility: {visible}")
    
    def set_opacity(self, opacity: float):
        """Set layer opacity.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        self._opacity = max(0.0, min(1.0, opacity))
        logger.debug(f"TextOverlayManager opacity: {self._opacity}")
    
    def get_active_text_count(self) -> int:
        """Get the number of active text items.
        
        Returns:
            Number of active text overlays
        """
        return len(self.text_items)
    
    async def cleanup(self):
        """Clean up text overlay manager resources."""
        logger.info("Cleaning up TextOverlayManager...")
        
        # Clear all text items
        self.text_items.clear()
        
        logger.info("TextOverlayManager cleanup complete")
