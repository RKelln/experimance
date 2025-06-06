"""
Experimance Image Server Service: Image generation and management component.

This module handles the creation, storage, and delivery of AI-generated
satellite imagery for the installation.
"""

from .image_service import ImageServerService
from .generators.generator import (
    ImageGenerator,
    MockImageGenerator,
)
from .generators.fal.fal_comfy_generator import FalComfyGenerator
from .generators.openai.openai_generator import OpenAIGenerator
from .generators.local.sdxl_generator import LocalSDXLGenerator
from .config import ImageServerConfig
from .generators.factory import create_generator

__version__ = '0.1.0'
__all__ = [
    'ImageServerService',
    'ImageGenerator',
    'MockImageGenerator',
    'FalComfyGenerator',
    'OpenAIGenerator',
    'LocalSDXLGenerator',
    'ImageServerConfig',
    'create_generator',
]

