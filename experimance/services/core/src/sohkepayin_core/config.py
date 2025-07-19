#!/usr/bin/env python3
"""
Configuration schema for the Sohkepayin Core Service.

This module defines Pydantic models for validating and accessing
core service configuration in a type-safe way.
"""

from dataclasses import dataclass
from typing import Optional, List
from pydantic import BaseModel, Field

from experimance_common.config import BaseServiceConfig
from experimance_common.constants import (
    DEFAULT_PORTS, CORE_SERVICE_DIR, ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX,
    get_project_config_path
)
from experimance_common.zmq.config import (
    ControllerPullConfig, ControllerPushConfig, ControllerServiceConfig, PublisherConfig, SubscriberConfig, 
    WorkerConfig, WorkerPushConfig, WorkerPullConfig
)
from experimance_common.schemas import MessageType

# Define the default configuration path with project-aware fallback
DEFAULT_CONFIG_PATH = get_project_config_path("core", CORE_SERVICE_DIR)

# dataclasses
@dataclass
class ImagePrompt:
    """Data class for image generation prompts."""
    
    prompt: str
    negative_prompt: Optional[str] = ""

# config classess

class PanoramaConfig(BaseModel):
    """Configuration for panorama generation."""
    
    display_width: int = Field(
        default=5760,  # 3 tiles * 1920 before mirroring
        description="Base panorama width in pixels",
        ge=512
    )
    
    display_height: int = Field(
        default=1080,
        description="Base panorama height in pixels", 
        ge=120
    )

    generated_width: int = Field(
        default=2240,  # 3 tiles
        description="Base panorama width in pixels",
        ge=512
    )
    
    generated_height: int = Field(
        default=420,
        description="Base panorama height in pixels", 
        ge=120
    )

    generator: str = Field(
        default="mock",
        description="Image generator to use for panorama generation"
    )


class TileConfig(BaseModel):
    """Configuration for tile generation."""
    
    display_width: int = Field(
        default=1920,
        description="Base panorama width in pixels",
        ge=512
    )
    
    display_height: int = Field(
        default=1080,
        description="Base panorama height in pixels", 
        ge=120
    )

    generated_width: int = Field(
        default=1344,
        description="Base panorama width in pixels",
        ge=512
    )
    
    generated_height: int = Field(
        default=758,
        description="Base panorama height in pixels", 
        ge=120
    )
    
    min_overlap_percent: float = Field(
        default=10.0,
        description="Minimum overlap between tiles as percentage",
        ge=0.0,
        le=50.0
    )
    
    max_megapixels: float = Field(
        default=1.0,
        description="Maximum megapixels per tile (1MP = 1,000,000 pixels)",
        ge=0.5,
        le=5.0
    )

    generator: str = Field(
        default="mock",
        description="Image generator to use for panorama generation"
    )


class LLMConfig(BaseModel):
    """Configuration for LLM integration."""
    
    provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, local)"
    )
    
    model: str = Field(
        default="gpt-4o",
        description="Model name to use"
    )
    
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the LLM provider"
    )
    
    max_tokens: int = Field(
        default=500,
        description="Maximum tokens for LLM responses",
        ge=100,
        le=2000
    )
    
    temperature: float = Field(
        default=0.7,
        description="Temperature for LLM responses",
        ge=0.0,
        le=2.0
    )
    
    timeout: float = Field(
        default=30.0,
        description="Timeout for LLM requests in seconds",
        ge=10.0,
        le=120.0
    )

    system_prompt_or_file: Optional[str] = Field(
        default=None,
        description="System prompt as a string or path to a file containing the prompt"
    )


class RenderingConfig(BaseModel):
    """Configuration for image rendering."""
    
    timeout: float = Field(
        default=120.0,
        description="Timeout for image generation in seconds",
        ge=30.0,
        le=300.0
    )
    
    base_image_timeout: float = Field(
        default=60.0,
        description="Timeout for base image generation in seconds",
        ge=20.0,
        le=180.0
    )
    
    tile_timeout: float = Field(
        default=90.0,
        description="Timeout for tile generation in seconds", 
        ge=30.0,
        le=180.0
    )


class SohkepayinCoreConfig(BaseServiceConfig):
    """Sohkepayin core service configuration."""
    
    service_name: str = Field(
        default="sohkepayin_core",
        description="Service instance name"
    )
    
    panorama: PanoramaConfig = Field(
        default_factory=PanoramaConfig,
        description="Panorama generation configuration"
    )
    
    tiles: TileConfig = Field(
        default_factory=TileConfig,
        description="Tile generation configuration"
    )
    
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM integration configuration"
    )
    
    rendering: RenderingConfig = Field(
        default_factory=RenderingConfig,
        description="Image rendering configuration"
    )
    
    zmq: ControllerServiceConfig = Field(
        default_factory=lambda: ControllerServiceConfig(
            publisher=PublisherConfig(
                address=ZMQ_TCP_BIND_PREFIX,
                port=DEFAULT_PORTS["events"],  # Publish to events channel
                default_topic=MessageType.DISPLAY_MEDIA
            ),
            subscriber=SubscriberConfig(
                address=ZMQ_TCP_CONNECT_PREFIX,
                port=DEFAULT_PORTS["agent"],  # Subscribe to agent messages
                topics=[MessageType.STORY_HEARD, MessageType.UPDATE_LOCATION]
            ),
            workers={
                "image_server": WorkerConfig(
                    name="image_server",
                    push_config=ControllerPushConfig(
                        port=DEFAULT_PORTS["image_requests"]
                    ),
                    pull_config=ControllerPullConfig(
                        port=DEFAULT_PORTS["image_results"]
                    )
                ),
            }
        ),
        description="ZMQ communication configuration"
    )
