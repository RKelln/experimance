"""
Experimance Image Server Service: Image generation and management component.

This module handles the creation, storage, and delivery of AI-generated
satellite imagery for the installation.
"""

from .image_service import ImageServerService
from .generators.generator import (
    ImageGenerator,
)
from .generators.mock.mock_generator import MockImageGenerator
from .generators.fal.fal_comfy_generator import FalComfyGenerator
from .generators.fal.fal_lightning_i2i_generator import FalLightningI2IGenerator
from .generators.openai.openai_generator import OpenAIGenerator
from .generators.local.sdxl_generator import LocalSDXLGenerator
from .config import ImageServerConfig
from .generators.factory import create_generator, GeneratorManager

__version__ = '0.1.0'
__all__ = [
    'ImageServerService',
    'ImageGenerator',
    'MockImageGenerator',
    'FalComfyGenerator',
    'FalLightningI2IGenerator',
    'OpenAIGenerator',
    'LocalSDXLGenerator',
    'ImageServerConfig',
    'create_generator',
    'GeneratorManager',
]

