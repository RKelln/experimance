#!/usr/bin/env python3
"""
Configuration schema for the Experimance Image Server service.

This module defines Pydantic models for validating and accessing
image server configuration in a type-safe way using the new ZMQ architecture.
"""

from typing import Annotated, Dict, Literal, Optional
from pathlib import Path

from pydantic import BaseModel, Field, StringConstraints

from experimance_common.config import BaseConfig, BaseServiceConfig
from experimance_common.constants import (
    DEFAULT_PORTS, IMAGE_SERVER_SERVICE_DIR, ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX,
    get_project_config_path, GENERATED_IMAGES_DIR
)
from experimance_common.zmq.config import (
    WorkerServiceConfig, PublisherConfig, SubscriberConfig, 
    WorkerPushConfig, WorkerPullConfig, MessageType
)
from image_server.generators.config import GENERATOR_NAMES

# Import all generator config types
from image_server.generators.mock.mock_generator_config import MockGeneratorConfig
from image_server.generators.fal.fal_comfy_config import FalComfyGeneratorConfig
from image_server.generators.vastai.vastai_config import VastAIGeneratorConfig
from image_server.generators.local.sdxl_generator import LocalSDXLConfig

# Import audio generator configs
from image_server.generators.audio.audio_config import Prompt2AudioConfig


DEFAULT_CONFIG_PATH = get_project_config_path("image_server", IMAGE_SERVER_SERVICE_DIR)

class GeneratorConfig(BaseModel):
    """Configuration for image generator selection and common settings."""
    strategy: Annotated[GENERATOR_NAMES,
                        StringConstraints(to_lower=True)] = Field(
        default="vastai",
        description="Image generation strategy to use."
    )
    timeout: int = Field(
        default=60,
        description="Default timeout for image generation in seconds"
    )


class AudioGeneratorConfig(BaseModel):
    """Configuration for audio generator selection and common settings."""
    strategy: Literal["prompt2audio"] = Field(
        default="prompt2audio",
        description="Audio generation strategy to use."
    )
    timeout: int = Field(
        default=120,
        description="Default timeout for audio generation in seconds"
    )
    enabled: bool = Field(
        default=True,
        description="Whether audio generation is enabled"
    )


class ImageServerConfig(BaseServiceConfig):
    """Complete configuration schema for the Image Server service."""
    
    # Override service name with default for image server
    service_name: str ="image-server"
    
    # Cache configuration
    cache_dir: Path = Field(
        default=GENERATED_IMAGES_DIR,
        description="Directory to store generated images"
    )
    
    max_cache_size_gb: float = Field(
        default=2.0,
        description="Maximum cache size in gigabytes",
        gt=0
    )
    
    # ZeroMQ configuration using new WorkerServiceConfig pattern
    zmq: WorkerServiceConfig = Field(
        default_factory=lambda: WorkerServiceConfig(
            name="image-server",
            publisher=None,
            # publisher=PublisherConfig(
            #     address=ZMQ_TCP_BIND_PREFIX,
            #     port=DEFAULT_PORTS['events'],
            #     default_topic=str(MessageType.IMAGE_READY)
            # ),
            subscriber=SubscriberConfig(
                address=ZMQ_TCP_CONNECT_PREFIX,
                port=DEFAULT_PORTS['events'],
                topics=[str(MessageType.RENDER_REQUEST)]
            ),
            push=WorkerPushConfig(
                port=DEFAULT_PORTS['image_results']
            ),
            pull=WorkerPullConfig(
                port=DEFAULT_PORTS['image_requests']
            )
        ),
        description="ZeroMQ communication settings using worker pattern"
    )
    
    # Generator configuration
    generator: GeneratorConfig = Field(
        default_factory=GeneratorConfig,
        description="Image generation settings"
    )
    
    # Audio generator configuration
    audio_generator: AudioGeneratorConfig = Field(
        default_factory=AudioGeneratorConfig,
        description="Audio generation settings"
    )
    
    # Strategy-specific configurations
    mock: MockGeneratorConfig = Field(
        default_factory=MockGeneratorConfig,
        description="Configuration for mock generator"
    )
    
    local_sdxl: LocalSDXLConfig = Field(
        default_factory=LocalSDXLConfig,
        description="Configuration for local SDXL generator"
    )
    
    sdxl: Dict = Field(
        default_factory=dict,
        description="Configuration for SDXL generator (deprecated, use local_sdxl)"
    )
    
    fal_comfy: FalComfyGeneratorConfig = Field(
        default_factory=FalComfyGeneratorConfig,
        description="Configuration for FAL.AI generator"
    )
    
    openai: Dict = Field(
        default_factory=dict,
        description="Configuration for OpenAI DALL-E generator"
    )
    
    vastai: VastAIGeneratorConfig = Field(
        default_factory=VastAIGeneratorConfig,
        description="Configuration for VastAI generator"
    )
    
    # Audio generator configurations
    prompt2audio: Prompt2AudioConfig = Field(
        default_factory=Prompt2AudioConfig,
        description="Configuration for prompt-to-audio generator"
    )
    
    def get_generator_config(self, strategy: Optional[str] = None) -> Dict:
        """Get configuration for a specific generator strategy.
        
        Args:
            strategy: Generator strategy name (defaults to default_strategy)
            
        Returns:
            Configuration dictionary for the generator
        """
        if strategy is None:
            strategy = self.generator.strategy
            
        # Base configuration
        config = {
            "output_dir": str(self.cache_dir),
            "timeout": self.generator.timeout,
        }
        
        # Add strategy-specific configuration
        if hasattr(self, strategy):
            strategy_config = getattr(self, strategy)
            if strategy_config:
                config.update(strategy_config.model_dump())
            
        return config

# =============================================================================
# HELPER FUNCTIONS FOR QUICK SETUP
# =============================================================================

def create_image_server_config(
    service_name: str = "image-server",
    cache_dir: str = "images",
    default_strategy: GENERATOR_NAMES = "mock"
) -> ImageServerConfig:
    """
    Create an ImageServerConfig for quick setup and convenience.
    
    ⚠️  USAGE GUIDANCE:
    - **Production services**: Use ImageServerConfig.from_overrides() with proper config files
    - **Unit/integration tests**: Use mocks or minimal inline configs
    - **Quick setup/examples/prototypes**: Use this factory function for convenience
    
    Args:
        service_name: Name of the image server service
        cache_dir: Directory to store generated images
        default_strategy: Default image generation strategy
        
    Returns:
        ImageServerConfig configured for worker pattern communication
        
    Example:
        # Quick setup for examples
        config = create_image_server_config(
            service_name="my-image-server",
            default_strategy="mock"
        )
    """
    return ImageServerConfig(
        service_name=service_name,
        cache_dir=Path(cache_dir),
        generator=GeneratorConfig(strategy=default_strategy),
    )