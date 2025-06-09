from typing import Literal
from pydantic import BaseModel, Field

DEFAULT_GENERATOR_TIMEOUT = 30  # Default timeout for image generation in seconds


class BaseGeneratorConfig(BaseModel):
    """Base configuration for all image generators.
    
    This class defines common fields and provides a base for all generator configs.
    Subclasses should override the strategy field with their specific literal value.
    """
    # This is a placeholder that subclasses will override with a specific literal value
    strategy: str

    # Common configuration options for all generators
    timeout: int = DEFAULT_GENERATOR_TIMEOUT

class SDXLConfig(BaseGeneratorConfig):
    lora_strength: float = 0.8
    dimensions: list[int] = [1024, 1024]
    num_inference_steps: int = 4
    negative_prompt: str = "distorted, warped, blurry, text, cartoon"
    ksampler_seed: int = 1
    prompt: str = "A beautiful landscape"