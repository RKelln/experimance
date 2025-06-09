import os
from typing import Optional, Literal
from dotenv import load_dotenv

from image_server.generators.config import SDXLConfig, BaseGeneratorConfig

load_dotenv(dotenv_path="../../.env", override=True)


FALCOMFY_ENDPOINT = os.getenv("FALCOMFY_ENDPOINT", "fal-ai/fast-lightning-sdxl")
LORA_URL= os.getenv(
    "FALCOMFY_LORA_URL", 
    "https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor")
MODEL_URL = os.getenv(
    "FALCOMFY_MODEL_URL",
    "https://civitai.com/api/download/models/471120?type=Model&format=SafeTensor&size=full&fp=fp16"
)

class FalComfyGeneratorConfig(SDXLConfig):
    """Configuration schema for FAL.AI image generator.
    Uses custom Comfy workflow that has specific parameters.
    """
    
    # Override the strategy field with a literal value for this specific generator
    strategy: Literal["falai"] = "falai"
    
    endpoint: str = FALCOMFY_ENDPOINT
    model_url: Optional[str] = MODEL_URL
    lora_url: Optional[str] = LORA_URL
    lora_strength: float = 0.8
    depth_map: Optional[str] = None
    
    def to_args(self) -> dict:
        """Convert configuration to args to pass to fal endpoint call."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model_url": self.model_url,
            "lora_url": self.lora_url,
            "lora_strength": self.lora_strength,
            "depth_map": self.depth_map,
            "ksampler_seed": self.seed,
        }
    
    def __repr__(self) -> str:
        # don't ouput the depth_map if given as it is a large string
        if self.depth_map:
            return super().__repr__().replace(f", depth_map='{self.depth_map}'", "depth_map=True")
        return super().__repr__()

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"FalComfyGeneratorConfig("
            f"endpoint={self.endpoint!r}, model_url={self.model_url!r}, "
            f"lora_url={self.lora_url!r}, lora_strength={self.lora_strength}, "
            f"depth_map={'Provided' if self.depth_map else 'None'}, "
            f"prompt={self.prompt[:50]!r}, "
            f"negative_prompt={self.negative_prompt[:50]!r}, seed={self.seed})"
        )

class FalGeneratorConfig(SDXLConfig):
    """Configuration schema for FAL.AI image generator.
    Useful for testing with various FAL.AI endpoints.
    """
    
    # Override the strategy field with a literal value for this specific generator
    strategy: Literal["falai"] = "falai"
    
    endpoint: str = FALCOMFY_ENDPOINT
    model_url: Optional[str] = MODEL_URL
    lora_url: Optional[str] = LORA_URL
    lora_strength: float = 0.8
    depth_map: Optional[str] = None
    format: Literal["png", "jpeg"] = "png"
    
    def to_args(self) -> dict:
        """Convert configuration to args to pass to fal endpoint call."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image_size": {
                "width": self.dimensions[0],
                "height": self.dimensions[1]
            },
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "enable_safety_checker": False,
        }
    
    def __repr__(self) -> str:
        # don't ouput the depth_map if given as it is a large string
        if self.depth_map:
            return super().__repr__().replace(f", depth_map='{self.depth_map}'", "depth_map=True")
        return super().__repr__()
