"""
Feed the Fires Core Service Package.

This package provides the core orchestration service for the Feed the Fires
interactive art installation. It handles story analysis, image generation
coordination, and panoramic visualization pipeline.
"""

__version__ = "0.1.0"

from .config import FireCoreConfig, ImagePrompt, DEFAULT_CONFIG_PATH
from .fire_core import FireCoreService, run_fire_core_service
from .llm_prompt_builder import LLMPromptBuilder
from .llm import LLMProvider
from .tiler import PanoramaTiler, TileSpec

__all__ = [
    "LLMPromptBuilder",
    "LLMProvider",
    "FireCoreConfig",
    "DEFAULT_CONFIG_PATH", 
    "FireCoreService",
    "run_fire_core_service",
    "ImagePrompt",
    "PanoramaTiler",
    "TileSpec"
]
