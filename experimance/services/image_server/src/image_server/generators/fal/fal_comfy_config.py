import os
from typing import Optional, Literal
from dotenv import load_dotenv

from image_server.generators.config import SDXLConfig

load_dotenv(dotenv_path="../../.env", override=True)


FALCOMFY_ENDPOINT = os.getenv("FALCOMFY_ENDPOINT", "fal-ai/sdxl-lightning")
LORA_URL= os.getenv(
    "FALCOMFY_LORA_URL", 
    "https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor")
MODEL_URL = os.getenv(
    "FALCOMFY_MODEL_URL",
    "https://civitai.com/api/download/models/471120?type=Model&format=SafeTensor&size=full&fp=fp16"
)

class FalComfyGeneratorConfig(SDXLConfig):
    """Configuration schema for FAL.AI image generator."""
    
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
            "model_url": self.model_url,
            "lora_url": self.lora_url,
            "lora_strength": self.lora_strength,
            "depth_map": self.depth_map,
            "image_size": {
                "width": self.dimensions[0],
                "height": self.dimensions[1]
            },
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "ksampler_seed": self.ksampler_seed,
        }