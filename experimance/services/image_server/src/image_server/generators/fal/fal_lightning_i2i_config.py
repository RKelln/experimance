import os
from typing import Optional, Literal
from dotenv import load_dotenv

from image_server.generators.config import SDXLConfig

load_dotenv(dotenv_path="../../.env", override=True)


FALLIGHTNINGI2I_ENDPOINT = os.getenv("FALLIGHTNINGI2I_ENDPOINT", "fal-ai/fast-lightning-sdxl/image-to-image")


class FalLightningI2IConfig(SDXLConfig):
    """Configuration schema for FAL.AI Lightning image-to-image generator.
    
    This configuration is for the fast-lightning-sdxl/image-to-image endpoint
    which provides high-speed image-to-image transformation.
    """
    
    # Override the strategy field with a literal value for this specific generator
    strategy: Literal["falai_lightning_i2i"] = "falai_lightning_i2i"
    
    endpoint: str = FALLIGHTNINGI2I_ENDPOINT
    
    # Image-to-image specific parameters
    image_url: Optional[str] = None  # Required for I2I generation
    strength: float = 0.95  # How much the generated image resembles the initial image
    
    # Standard Lightning SDXL parameters
    image_size: str = "square_hd"  # Can be: square_hd, square, portrait_4_3, portrait_16_9, landscape_4_3, landscape_16_9
    num_images: int = 1
    format: Literal["jpeg", "png"] = "jpeg"
    enable_safety_checker: bool = True
    safety_checker_version: Literal["v1", "v2"] = "v1"
    expand_prompt: bool = False
    guidance_rescale: Optional[float] = None
    preserve_aspect_ratio: bool = False
    crop_output: bool = False
    
    def to_args(self) -> dict:
        """Convert configuration to args to pass to fal Lightning I2I endpoint call."""
        args = {
            "image_url": self.image_url,
            "prompt": self.prompt,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "num_images": self.num_images,
            "format": self.format,
            "enable_safety_checker": self.enable_safety_checker,
            "safety_checker_version": self.safety_checker_version,
            "expand_prompt": self.expand_prompt,
            "preserve_aspect_ratio": self.preserve_aspect_ratio,
            "crop_output": self.crop_output,
        }
        
        # Add negative prompt if provided
        if self.negative_prompt:
            # Note: The API docs don't explicitly mention negative_prompt for Lightning I2I,
            # but it's common in SD models, so we'll include it if available
            pass  # Lightning I2I doesn't seem to support negative prompts based on the schema
        
        # Handle image size - can be a preset string or custom dimensions
        if hasattr(self, 'dimensions') and self.dimensions:
            args["image_size"] = {
                "width": self.dimensions[0],
                "height": self.dimensions[1]
            }
        else:
            args["image_size"] = self.image_size
            
        # Add guidance rescale if specified
        if self.guidance_rescale is not None:
            args["guidance_rescale"] = self.guidance_rescale
        
        # Remove None values to keep request clean
        return {k: v for k, v in args.items() if v is not None}
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"FalLightningI2IConfig("
            f"endpoint={self.endpoint!r}, "
            f"image_url={'Provided' if self.image_url else 'None'}, "
            f"strength={self.strength}, "
            f"image_size={self.image_size if isinstance(self.image_size, str) else self.dimensions}, "
            f"prompt={self.prompt[:50]!r}, seed={self.seed})"
        )
