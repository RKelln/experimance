import asyncio
import logging
import requests
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env", override=True)

logger = logging.getLogger(__name__)

from image_server.generators.generator import ImageGenerator
from image_server.generators.config import DEFAULT_GENERATOR_TIMEOUT
from .fal_comfy_config import FalComfyGeneratorConfig


class FalComfyGenerator(ImageGenerator):
    """FAL.AI image generator implementation.
    
    Uses the FAL.AI API for remote image generation.
    """
    
    # def __init__(self, output_dir: str = "/tmp", **kwargs):
    #     """Initialize the FAL.AI image generator.
        
    #     Args:
    #         output_dir: Directory to save generated images
    #         **kwargs: Additional configuration options or can be a FalComfyGeneratorConfig instance
    #     """
    #     # Check if a config object was passed directly
    #     config_obj = kwargs.pop("config", None)
    #     if isinstance(config_obj, FalComfyGeneratorConfig):
    #         self.config = config_obj
    #         # Use the provided config object
    #         super().__init__(output_dir=output_dir, **kwargs)
    #     else:
    #         # Initialize normally with kwargs
    #         super().__init__(output_dir=output_dir, **kwargs)
    
    def _configure(self, **kwargs):
        """Configure FAL.AI generator settings from kwargs or create default config."""
        config_obj = kwargs.pop("config", None)
        if isinstance(config_obj, FalComfyGeneratorConfig):
            self.config = config_obj
        
        if not hasattr(self, 'config'):
            # Create a new config with kwargs as overrides
            self.config = FalComfyGeneratorConfig(**kwargs)
        
        logger.info(f"FalComfyGenerator initialized with endpoint: {self.config.endpoint}")
    
    async def generate_image(self, prompt: str, depth_map_b64: Optional[str] = None, 
                             config_overrides: Optional[FalComfyGeneratorConfig|dict] = None) -> str:
        """Generate an image using FAL.AI API.
        
        Args:
            prompt: Text description of the image to generate
            depth_map_b64: Optional base64-encoded depth map PNG
            config_overrides: Optional configuration overrides for this specific generation
            
        Returns:
            Path to the generated image file
            
        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails or dependencies missing
        """
        self._validate_prompt(prompt)
        
        # Handle config overrides for this specific request
        current_config = self.config
        if config_overrides:
            if isinstance(config_overrides, dict):
                # Create a new config with overrides applied
                current_config = FalComfyGeneratorConfig(**{
                    **self.config.model_dump(),
                    **config_overrides
                })
            else:
                # Use the provided config object directly
                current_config = config_overrides
        
        try:
            import fal_client
        except ImportError:
            raise RuntimeError("fal_client library not installed. Install with: pip install fal-client")
        
        logger.info(f"FalAIGenerator: Generating image with FAL.AI for prompt: {prompt[:50]}...")
        
        # Use the config's to_args method to get arguments
        arguments = current_config.to_args()
        
        # Add the prompt to the arguments
        arguments["prompt"] = prompt
        
        # Add depth map if provided
        if depth_map_b64:
            arguments["depth_map"] = depth_map_b64

        try:
            # Submit the request to FAL.AI
            handler = await fal_client.submit_async(
                current_config.endpoint,
                arguments=arguments
            )
            
            start_time = time.monotonic()
            timeout = getattr(current_config, "timeout", DEFAULT_GENERATOR_TIMEOUT) 
            async for event in handler.iter_events(with_logs=logger.isEnabledFor(logging.DEBUG)):
                elapsed_time = time.monotonic() - start_time
                if elapsed_time > timeout:
                    logger.error(f"FalAIGenerator: Generation timed out after {elapsed_time:.2f} seconds")
                    raise RuntimeError(f"FAL.AI generation timed out after {elapsed_time:.2f} seconds")
                logger.debug(event)

            response = await handler.get()

            # Download the generated image
            if "images" in response and len(response["images"]) > 0:
                image_url = response["images"][0]["url"]
                return await self._download_image(image_url)
            else:
                raise RuntimeError("No images returned from FAL.AI")
                
        except asyncio.TimeoutError:
            logger.error(f"FalAIGenerator: Generation timed out after {timeout} seconds")
            raise RuntimeError(f"FAL.AI generation timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"FalAIGenerator: Error generating image: {e}")
            raise RuntimeError(f"FAL.AI generation failed: {e}")
    
    # Using _download_image from the base ImageGenerator class
    

if __name__ == "__main__":
    # Example usage
    # $ uv run -m image_server.generators.fal.fal_comfy_generator
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting FAL.AI image generation example...")

    # Example 1: Create a config instance and pass it to the generator
    config = FalComfyGeneratorConfig(
        strategy="falai",
        endpoint="fal-ai/fast-lightning-sdxl",
        dimensions=[1024, 1024],
        num_inference_steps=4,
        negative_prompt="distorted, warped, blurry, text, cartoon"
    )
    generator = FalComfyGenerator(config=config)
    
    # Example 2: Pass configuration directly to constructor
    # generator = FalComfyGenerator(
    #     endpoint="fal-ai/sdxl-lightning",
    #     model_url="https://example.com/model",
    #     lora_url="https://example.com/lora"
    # )
    
    # Example 3: Override configuration for a specific request
    config_overrides = {
        #"dimensions": [512, 512],  # Generate a smaller image
        "num_inference_steps": 8,  # More steps for better quality
    }
    
    # Run an example generation (this would normally be done in an async context)
    try:
        # Example with config overrides
        image_path = asyncio.run(
            generator.generate_image(
                "A beautiful landscape with mountains and lakes",
                config_overrides=config_overrides
            )
        )
        logger.info(f"Generated image saved to: {image_path}")
    except RuntimeError as e:
        logger.error(f"Generation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    else:
        logger.info("Image generation completed successfully.")
