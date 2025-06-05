#!/usr/bin/env python3
"""
Configuration schema for the Experimance Image Server service.

This module defines Pydantic models for validating and accessing
image server configuration in a type-safe way.
"""

from typing import Dict, Literal, Optional
from pathlib import Path

from pydantic import BaseModel, Field

from experimance_common.config import Config
from experimance_common.constants import DEFAULT_PORTS


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


class GeneratorConfig(BaseModel):
    """Configuration for image generation strategies."""
    
    default_strategy: Literal["mock", "sdxl", "falai", "openai"] = Field(
        default="mock",
        description="Default image generation strategy"
    )
    
    timeout_seconds: int = Field(
        default=30,
        description="Timeout for image generation in seconds",
        gt=0
    )
    
    # Strategy-specific configuration could go here
    # For example:
    # sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # openai_api_key: Optional[str] = None


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
