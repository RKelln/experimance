#!/usr/bin/env python3
"""
Configuration schema for the Experimance Image Server service.

This module defines Pydantic models for validating and accessing
image server configuration in a type-safe way.
"""

from typing import Dict, Literal, Optional, Union, Type, Annotated
from pathlib import Path

from pydantic import BaseModel, Field

from experimance_common.config import Config
from experimance_common.constants import DEFAULT_PORTS
from image_server.generators.config import BaseGeneratorConfig

# Import all generator config types
from image_server.generators.fal.fal_comfy_config import FalComfyGeneratorConfig

# For future use when other generators are implemented:
#from image_server.generators.mock.mock_generator import MockGeneratorConfig
#from image_server.generators.sdxl.sdxl_generator import SDXLGeneratorConfig
#from image_server.generators.openai.openai_generator import OpenAIGeneratorConfig

class ZmqConfig(BaseModel):
    """ZeroMQ configuration for the Image Server service."""
    
    events_sub_address: str = Field(
        default=f"tcp://localhost:{DEFAULT_PORTS['image_request_pub']}",
        description="Address for subscribing to event messages"
    )
    
    images_pub_address: str = Field(
        default=f"tcp://*:{DEFAULT_PORTS['image_server_pub']}",
        description="Address for publishing image messages"
    )

GeneratorConfigUnion = Union[
#    MockGeneratorConfig,
#    SDXLGeneratorConfig,
    FalComfyGeneratorConfig,
#    OpenAIGeneratorConfig,
]

class GeneratorConfig(BaseModel):
    """Configuration for image generator selection and common settings."""
    default_strategy: Literal["mock", "sdxl", "falai", "openai"] = "falai"
    config: GeneratorConfigUnion = Field(..., discriminator="strategy")

class ImageServerConfig(Config):
    """Complete configuration schema for the Image Server service."""
    
    # General service configuration
    service_name: str = Field(
        default="image-server",
        description="Name of this service instance"
    )
    
    # Cache configuration
    cache_dir: Path = Field(
        default=Path("images"),
        description="Directory to store generated images"
    )
    
    max_cache_size_gb: float = Field(
        default=2.0,
        description="Maximum cache size in gigabytes",
        gt=0
    )
    
    # ZeroMQ configuration
    zmq: ZmqConfig = Field(
        default_factory=ZmqConfig,
        description="ZeroMQ communication settings"
    )
    
    # Generator configuration
    generator: GeneratorConfig = Field(
        default_factory=GeneratorConfig,
        description="Image generation settings"
    )
    
    # Strategy-specific configurations
    mock: Dict = Field(
        default_factory=dict,
        description="Configuration for mock generator"
    )
    
    sdxl: Dict = Field(
        default_factory=dict,
        description="Configuration for SDXL generator"
    )
    
    falai: Dict = Field(
        default_factory=dict,
        description="Configuration for FAL.AI generator"
    )
    
    openai: Dict = Field(
        default_factory=dict,
        description="Configuration for OpenAI DALL-E generator"
    )
    
    def get_generator_config(self, strategy: Optional[str] = None) -> Dict:
        """Get configuration for a specific generator strategy.
        
        Args:
            strategy: Generator strategy name (defaults to default_strategy)
            
        Returns:
            Configuration dictionary for the generator
        """
        if strategy is None:
            strategy = self.generator.default_strategy
            
        # Base configuration
        config = {
            "output_dir": str(self.cache_dir),
            "timeout": self.generator.timeout_seconds
        }
        
        # Add strategy-specific configuration
        if hasattr(self, strategy):
            strategy_config = getattr(self, strategy)
            config.update(strategy_config)
            
        return config
