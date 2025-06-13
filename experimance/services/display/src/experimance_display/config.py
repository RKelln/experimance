#!/usr/bin/env python3
"""
Configuration schema for the Experimance Display Service.

This module defines Pydantic models for validating and accessing
display service configuration in a type-safe way.
"""

from typing import Dict, List, Literal, Optional, Tuple, TypeAlias
from pathlib import Path

from pydantic import BaseModel, Field

from experimance_common.config import Config
from experimance_common.constants import DEFAULT_PORTS


class ZmqConfig(BaseModel):
    """ZeroMQ configuration for the Display Service."""
    
    core_sub_address: str = Field(
        default=f"tcp://localhost:{DEFAULT_PORTS['core']}",
        description="Address for subscribing to messages"
    )



class DisplayConfig(BaseModel):
    """Main display configuration."""
    
    fullscreen: bool = Field(
        default=True,
        description="Whether to run in fullscreen mode"
    )
    
    monitor: int = Field(
        default=0,
        description="Monitor to display on (0 = primary)"
    )
    
    resolution: Tuple[int, int] = Field(
        default=(1920, 1080),
        description="Resolution to use if not fullscreen"
    )
    
    fps_limit: int = Field(
        default=60,
        description="Frame rate limit"
    )
    
    vsync: bool = Field(
        default=True,
        description="Whether to use vertical sync"
    )
    
    debug_overlay: bool = Field(
        default=False,
        description="Whether to show debug overlay"
    )
    
    debug_text: bool = Field(
        default=False,
        description="Whether to show debug text in all positions"
    )

    profile: bool = Field(
        default=False,
        description="Whether to enable profiling for performance analysis"
    )
    
    headless: bool = Field(
        default=False,
        description="Whether to run in headless mode (no window creation)"
    )


class RenderingConfig(BaseModel):
    """Rendering system configuration."""
    
    max_texture_cache_mb: int = Field(
        default=512,
        description="Maximum texture cache size in MB"
    )
    
    shader_path: Path = Field(
        default=Path("shaders/"),
        description="Path to shader files"
    )
    
    font_path: Path = Field(
        default=Path("fonts/"),
        description="Path to font files"
    )
    
    preload_common_resources: bool = Field(
        default=True,
        description="Whether to preload commonly used resources"
    )
    
    backend: Literal["opengl"] = Field(
        default="opengl",
        description="Rendering backend to use"
    )


class TransitionsConfig(BaseModel):
    """Configuration for image and video transitions."""
    
    default_crossfade_duration: float = Field(
        default=1.0,
        description="Duration of default crossfade in seconds"
    )
    
    video_fade_in_duration: float = Field(
        default=0.2,
        description="Duration of video overlay fade in"
    )
    
    video_fade_out_duration: float = Field(
        default=1.0,
        description="Duration of video overlay fade out"
    )
    
    text_fade_duration: float = Field(
        default=0.3,
        description="Duration of text fade in/out"
    )
    
    preload_frames: bool = Field(
        default=True,
        description="Whether to preload transition frames"
    )
    
    max_preload_mb: int = Field(
        default=500,
        description="Maximum memory to use for preloading in MB"
    )


class TextStyleConfig(BaseModel):
    """Configuration for text rendering styles."""
    
    font_size: int = Field(
        default=28,
        description="Font size in pixels"
    )
    
    color: Tuple[int, int, int, int] = Field(
        default=(255, 255, 255, 255),
        description="Text color as RGBA tuple"
    )
    
    anchor: Literal["top_left", "top_center", "top_right", 
                     "center_left", "center", "center_right",
                     "bottom_left", "bottom_center", "bottom_right",
                     "baseline_left", "baseline_center", "baseline_right"] = Field(
        default="baseline_center",
        description="Text anchor position"
    )

    position: Literal["top_left", "top_center", "top_right", 
                     "center_left", "center", "center_right",
                     "bottom_left", "bottom_center", "bottom_right"] = Field(
        default="bottom_center",
        description="Text position on screen"
    )
    
    background: bool = Field(
        default=True,
        description="Whether to show background behind text"
    )
    
    background_color: Tuple[int, int, int, int] = Field(
        default=(0, 0, 0, 128),
        description="Background color as RGBA tuple"
    )
    
    padding: int = Field(
        default=10,
        description="Padding around text in pixels"
    )
    
    max_width: Optional[int] = Field(
        default=None,
        description="Maximum text width before wrapping (None = no limit)"
    )
    
    align: Literal["left", "center", "right"] = Field(
        default="left",
        description="Text alignment within the label ('left', 'center', 'right')"
    )


class TextStylesConfig(BaseModel):
    """Text styles for different speakers."""
    
    agent: TextStyleConfig = Field(
        default_factory=lambda: TextStyleConfig(
            font_size=28,
            color=(255, 255, 255, 255),
            anchor="baseline_center",
            position="bottom_center",
            background=True,
            background_color=(0, 0, 0, 128)
        ),
        description="Style for agent text"
    )
    
    system: TextStyleConfig = Field(
        default_factory=lambda: TextStyleConfig(
            font_size=24,
            color=(200, 200, 200, 255),
            anchor="baseline_center",
            position="top_right",
            background=False
        ),
        description="Style for system text"
    )
    
    debug: TextStyleConfig = Field(
        default_factory=lambda: TextStyleConfig(
            font_size=16,
            color=(255, 255, 0, 255),
            anchor="baseline_center",
            position="top_center",
            background=False
        ),
        description="Style for debug text"
    )
    
    title: TextStyleConfig = Field(
        default_factory=lambda: TextStyleConfig(
            font_size=72,
            color=(255, 255, 255, 255),
            anchor="baseline_center",
            position="center",
            background=False
        ),
        description="Style for title screen text"
    )

FadeDurationType: TypeAlias = float | tuple[float, float]

class TitleScreenConfig(BaseModel):
    """Configuration for the startup title screen."""
    
    enabled: bool = Field(
        default=True,
        description="Whether to show the title screen on startup"
    )
    
    text: str = Field(
        default="Experimance",
        description="Text to display on the title screen"
    )
    
    duration: float = Field(
        default=3.0,
        description="Duration to show title screen in seconds"
    )

    fade_duration: FadeDurationType = Field(
        default=(2.0, 5.0),
        description="Duration of fade in animation in seconds"
    )


class DisplayServiceConfig(Config):
    """Complete configuration schema for the Display Service."""
    
    service_name: str = "display-service"
    
    # Main configuration sections
    zmq: ZmqConfig = Field(default_factory=ZmqConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    transitions: TransitionsConfig = Field(default_factory=TransitionsConfig)
    text_styles: TextStylesConfig = Field(default_factory=TextStylesConfig)
    title_screen: TitleScreenConfig = Field(default_factory=TitleScreenConfig)
