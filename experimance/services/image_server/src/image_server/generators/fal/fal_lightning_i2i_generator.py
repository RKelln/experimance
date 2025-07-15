"""
FAL.AI Lightning Image-to-Image Generation

$ uv run -m image_server.generators.fal.fal_lightning_i2i_generator
"""
import asyncio
import logging
import time
from typing import Generator, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env", override=True)

# Configure logging
logger = logging.getLogger(__name__)

from image_server.generators.generator import ImageGenerator, configure_external_loggers
from image_server.generators.config import BaseGeneratorConfig, DEFAULT_GENERATOR_TIMEOUT
from .fal_lightning_i2i_config import FalLightningI2IConfig
from experimance_common.image_utils import png_to_base64url


class FalLightningI2IGenerator(ImageGenerator):
    """FAL.AI Lightning image-to-image generator implementation.
    
    Uses the FAL.AI Lightning SDXL image-to-image API for fast image generation from input images.
    """
    
    def __init__(self, config: BaseGeneratorConfig, output_dir: str = "/tmp", **kwargs):
        """Initialize the FAL.AI Lightning image-to-image generator.
        
        Args:
            config: Configuration object for the generator
            output_dir: Directory to save generated images
            **kwargs: Additional configuration options
        """
        super().__init__(config, output_dir, **kwargs)
        
        self._stop_event = asyncio.Event()

    def _configure(self, config: BaseGeneratorConfig, **kwargs):
        """Configure FAL.AI Lightning I2I generator settings from kwargs or create default config."""
        self.config = FalLightningI2IConfig(**{
            **config.model_dump(),
            **kwargs
        })
        logger.info(f"FalLightningI2IGenerator initialized with endpoint: {self.config.endpoint}")

    async def generate_image(self, prompt: str, image_b64: Optional[str] = None, 
                             **kwargs) -> str:
        """Generate an image using FAL.AI Lightning image-to-image API.
        
        Args:
            prompt: Text description of the image to generate
            image_url: URL or base64 data URI of the source image for image-to-image transformation
            depth_map_b64: Optional base64-encoded depth map PNG (not used in Lightning I2I)
            **kwargs: Additional parameters including request_id for filename tracking
            
        Returns:
            Path to the generated image file
            
        Raises:
            ValueError: If prompt is empty or invalid, or if image_url is not provided
            RuntimeError: If generation fails or dependencies missing
        """
        self._validate_prompt(prompt)
        
        if not image_b64:
            raise ValueError("image_url is required for image-to-image generation")
        
        # Handle config overrides for this specific request
        current_config = self.config
        
        try:
            import fal_client
        except ImportError:
            raise RuntimeError("fal_client library not installed. Install with: pip install fal-client")
        
        logger.info(f"FalLightningI2IGenerator: Generating image with FAL.AI Lightning I2I for prompt: {prompt[:50]}...")
        
        current_config.prompt = prompt
        current_config.image_url = image_b64

        logger.info(current_config)

        try:
            start_time = time.monotonic()

            # Submit the request to FAL.AI
            handler = await fal_client.submit_async(
                current_config.endpoint,
                arguments=current_config.to_args()
            )
            
            timeout = getattr(current_config, "timeout", DEFAULT_GENERATOR_TIMEOUT) 
            async for event in handler.iter_events(with_logs=logger.isEnabledFor(logging.DEBUG)):
                if self._stop_event.is_set():
                    raise RuntimeError("FAL.AI Lightning I2I generation stopped by user")
                elapsed_time = time.monotonic() - start_time
                if elapsed_time > timeout:
                    logger.error(f"FalLightningI2IGenerator: Generation timed out after {elapsed_time:.2f} seconds")
                    raise RuntimeError(f"FAL.AI Lightning I2I generation timed out after {elapsed_time:.2f} seconds")
                logger.debug(event)

            if self._stop_event.is_set():
                raise RuntimeError("FAL.AI Lightning I2I generation stopped by user")
            
            response = await handler.get()

            if self._stop_event.is_set():
                raise RuntimeError("FAL.AI Lightning I2I generation stopped by user")

            # Download the generated image
            for image_url in self.falai_image_url_generator(response):
                # we just need one image URL, so we can break after the first
                request_id = kwargs.get('request_id')
                return await self._download_image(image_url, request_id=request_id)

            # If no image URL was found, raise an error to ensure a str is always returned or an exception is raised
            raise RuntimeError("No image URL found in FAL.AI Lightning I2I response.")
                
        except asyncio.TimeoutError:
            logger.error(f"FalLightningI2IGenerator: Generation timed out after {timeout} seconds")
            raise RuntimeError(f"FAL.AI Lightning I2I generation timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"FalLightningI2IGenerator: Error generating image: {e}")
            raise RuntimeError(f"FAL.AI Lightning I2I generation failed: {e}")
        finally:
            logger.info(f"FalLightningI2IGenerator: Generation completed in {time.monotonic() - start_time:.2f} seconds")

    
    async def stop(self):
        """Stop the FAL.AI Lightning I2I generator if running."""
        logger.info("Stopping FAL.AI Lightning I2I generator...")
        self._stop_event.set()

    
    def falai_image_url_generator(self, response: dict) -> Generator[str, None, None]:
        """Generator to extract image URLs from FAL.AI Lightning I2I response.
        Args:
            response: Response dictionary from FAL.AI API
        Yields:
            str: URLs of generated images
        Raises:
            ValueError: If response format is unknown or unsupported
        """

        # Lightning I2I endpoint response format:
        # {
        #   "images": [
        #     {
        #       "url": "https://fal.media/files/panda/xxx.jpeg",
        #       "width": 1024,
        #       "height": 1024,
        #       "content_type": "image/jpeg"
        #     }
        #   ],
        #   "timings": {"inference": 0.xx},
        #   "seed": 123456,
        #   "has_nsfw_concepts": [false],
        #   "prompt": "..."
        # }
        if 'images' in response:
            for image_result in response['images']:
                yield image_result['url']
            return
        
        raise ValueError(f"Unknown response type: {response}")


if __name__ == "__main__":
    import argparse

    # Example usage
    # $ uv run -m image_server.generators.fal.fal_lightning_i2i_generator
    parser = argparse.ArgumentParser(description="FAL.AI Lightning Image-to-Image Generation Example")
    parser.add_argument("--config", type=str, help="Config toml")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output_dir", type=str, default="/tmp", help="Directory to save generated images")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for image generation in seconds")
    parser.add_argument("--image_url", type=str, help="Source image URL for image-to-image transformation")
    parser.add_argument("--strength", type=float, default=0.95, help="Strength for image-to-image transformation")
    parser.add_argument("--seed", type=int, default=1, help="Seed for image generation")
    args = parser.parse_args()

    # Configure root logger
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    # Configure external library loggers
    configure_external_loggers(logging.WARNING)
    
    logger.info("Starting FAL.AI Lightning Image-to-Image generation example...")

    # Create configuration
    i2i_config = FalLightningI2IConfig(
        strategy="falai_lightning_i2i",
        endpoint="fal-ai/fast-lightning-sdxl/image-to-image",
        dimensions=[1024, 1024],
        num_inference_steps=4,
        strength=args.strength,
        seed=args.seed,
        timeout=args.timeout,
        negative_prompt="distorted, warped, blurry, text, cartoon, illustration, low quality, lowres"
    )
    
    i2i_generator = FalLightningI2IGenerator(config=i2i_config, output_dir=args.output_dir)
    
    async def run_example():
        """Run the image-to-image generation example."""
        test_prompt = "a beautiful landscape painting in the style of Bob Ross, happy little trees, peaceful lake, mountains in background, oil painting style"
        
        # Use a sample image URL or the provided one
        source_image_url = args.image_url or "https://fal-cdn.batuhan-941.workers.dev/files/tiger/IExuP-WICqaIesLZAZPur.jpeg"
        
        try:
            logger.info("Running FAL.AI Lightning Image-to-Image example...")
            image_path = await i2i_generator.generate_image(
                test_prompt,
                image_url=source_image_url
            )
            logger.info(f"Example complete - Generated image saved to: {image_path}")
            return True
        except Exception as e:
            logger.error(f"Example failed: {e}")
            return False
    
    # Run the example
    try:
        success = asyncio.run(run_example())
        if success:
            logger.info("Example completed successfully!")
        else:
            logger.info("Example failed. See logs above for details.")
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}")
