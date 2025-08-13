from typing import Literal, Union ,TYPE_CHECKING
from experimance_common.schemas import MessageSchema
from pydantic import BaseModel, Field

from experimance_common.config import BaseConfig

DEFAULT_GENERATOR_TIMEOUT = 30  # Default timeout for image generation in seconds

GENERATOR_NAMES = Literal[
    "mock",
    "vastai",
    "fal_comfy",
    "fal_lightning_i2i",
    "local_sdxl",
]

class BaseGeneratorConfig(BaseConfig, MessageSchema):
    """Base configuration for all image generators.
    
    This class defines common fields and provides a base for all generator configs.
    Subclasses should override the strategy field with their specific literal value.
    """
    # This is a placeholder that subclasses will override with a specific literal value
    strategy: str

    # Common configuration options for all generators
    timeout: int = DEFAULT_GENERATOR_TIMEOUT

    pre_warm: bool = Field(
        default=False,
        description="Whether to pre-warm the generator. If True, the generator will be sent a test image at startup."
    )

class SDXLConfig(BaseGeneratorConfig):
    lora_strength: float = 0.8
    dimensions: list[int] = [1024, 1024]
    num_inference_steps: int = 4
    negative_prompt: str = "distorted, warped, blurry, text, cartoon"
    seed: int = 1
    prompt: str = "A beautiful landscape"


# Only import and create the union type when type checking
if TYPE_CHECKING:
    from image_server.generators.mock.mock_generator_config import MockGeneratorConfig
    from image_server.generators.fal.fal_comfy_config import FalComfyGeneratorConfig
    from image_server.generators.fal.fal_lightning_i2i_config import FalLightningI2IConfig
    from image_server.generators.vastai.vastai_config import VastAIGeneratorConfig
    
    # Type for all possible generator configs
    GeneratorConfigType = Union[
        MockGeneratorConfig,
        FalComfyGeneratorConfig,
        FalLightningI2IConfig,
        VastAIGeneratorConfig
    ]