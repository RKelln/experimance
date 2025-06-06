import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from image_server.generators.generator import ImageGenerator


class FalComfyGenerator(ImageGenerator):
    """FAL.AI image generator implementation.
    
    Uses the FAL.AI API for remote image generation.
    """
    
    def _configure(self, **kwargs):
        """Configure FAL.AI generator settings."""
        self.endpoint = kwargs.get("endpoint", "fal-ai/sdxl-lightning")
        self.model_url = kwargs.get("model_url")
        self.lora_url = kwargs.get("lora_url")
        self.lora_strength = kwargs.get("lora_strength", 0.8)
        self.dimensions = kwargs.get("dimensions", [1024, 1024])
        self.num_inference_steps = kwargs.get("num_inference_steps", 4)
        self.negative_prompt = kwargs.get("negative_prompt", "distorted, warped, blurry, text, cartoon")
        self.timeout = kwargs.get("timeout", 30)
    
    async def generate_image(self, prompt: str, depth_map_b64: Optional[str] = None, **kwargs) -> str:
        """Generate an image using FAL.AI API."""
        self._validate_prompt(prompt)
        
        try:
            import fal_client
        except ImportError:
            raise RuntimeError("fal_client library not installed. Install with: pip install fal-client")
        
        logger.info(f"FalAIGenerator: Generating image with FAL.AI for prompt: {prompt[:50]}...")
        
        # Prepare arguments for FAL.AI
        arguments = {
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "image_size": {
                "width": self.dimensions[0],
                "height": self.dimensions[1]
            },
            "num_inference_steps": self.num_inference_steps,
            "ksampler_seed": kwargs.get("seed", 1),
        }
        
        # Add depth map if provided
        if depth_map_b64:
            arguments["depth_map"] = depth_map_b64
        
        # Add model and LoRA URLs if configured
        if self.model_url:
            arguments["model_url"] = self.model_url
        if self.lora_url:
            arguments["lora_url"] = self.lora_url
            arguments["lora_strength"] = self.lora_strength
        
        try:
            # Submit the request to FAL.AI
            result = await asyncio.to_thread(
                fal_client.submit,
                self.endpoint,
                arguments=arguments
            )
            
            # Wait for the result
            response = await asyncio.to_thread(result.get)
            
            # Download the generated image
            if "images" in response and len(response["images"]) > 0:
                image_url = response["images"][0]["url"]
                return await self._download_image(image_url)
            else:
                raise RuntimeError("No images returned from FAL.AI")
                
        except Exception as e:
            logger.error(f"FalAIGenerator: Error generating image: {e}")
            raise RuntimeError(f"FAL.AI generation failed: {e}")
    
    

