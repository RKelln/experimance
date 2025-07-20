"""
FAL.AI Image Generation using ComfyUI Workflow

$ uv run -m image_server.generators.fal.fal_comfy_generator
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
from .fal_comfy_config import FalGeneratorConfig, FalComfyGeneratorConfig
from experimance_common.image_utils import png_to_base64url


class FalComfyGenerator(ImageGenerator):
    """FAL.AI image generator implementation.
    
    Uses the FAL.AI API for remote image generation.
    """
    
    def __init__(self, config: BaseGeneratorConfig, output_dir: str = "/tmp",  **kwargs):
        """Initialize the FAL.AI image generator.
        
        Args:
            config: Configuration object for the generator
            output_dir: Directory to save generated images
            **kwargs: Additional configuration options
        """
        super().__init__(config, output_dir, **kwargs)
        
        self._stop_event = asyncio.Event()
        
        # Pre-warm the generator if enabled
        if self.config.pre_warm:
            asyncio.create_task(self._pre_warm())

    def _configure(self, config:BaseGeneratorConfig, **kwargs):
        """Configure FAL.AI generator settings from kwargs or create default config."""
        self.config = FalComfyGeneratorConfig(**{
            **config.model_dump(),
            **kwargs
        })
        logger.info(f"FalComfyGenerator initialized with endpoint: {self.config.endpoint}")

    async def _pre_warm(self):
        """Pre-warm the generator by sending a test generation request.
        
        This helps reduce cold start latency for the first real generation.
        The result is discarded and not saved.
        """
        try:
            logger.info("Pre-warming FAL.AI generator...")
            from image_server.generators.generator import mock_depth_map
            from experimance_common.image_utils import png_to_base64url
            
            # Generate the test image (this will be discarded)
            await self.generate_image(
                prompt="test warm-up image generation",
                depth_map_b64=png_to_base64url(mock_depth_map()),
                request_id="prewarm_discard"
            )
            
            logger.info("FAL.AI generator pre-warming completed successfully")
            
        except Exception as e:
            logger.warning(f"FAL.AI generator pre-warming failed (continuing anyway): {e}")
            # Don't raise the exception - pre-warming failure shouldn't stop initialization

    async def generate_image(self, prompt: str, depth_map_b64: Optional[str] = None, 
                             **kwargs) -> str:
        """Generate an image using FAL.AI API.
        
        Args:
            prompt: Text description of the image to generate
            depth_map_b64: Optional base64-encoded depth map PNG
            **kwargs: Additional parameters including request_id for filename tracking
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
        if isinstance(kwargs, dict):
            # modify lora strength based on era
            era = kwargs.get('era', None)
            if era:
                from experimance_common.schemas import Era
                if era == Era.WILDERNESS:
                    kwargs['lora_url'] = "https://civitai.com/api/download/models/179152?type=Model&format=SafeTensor"
                    kwargs['lora_strength'] = self.config.lora_strength * 0.8
                elif era == Era.PRE_INDUSTRIAL:
                    kwargs['lora_url'] = "https://civitai.com/api/download/models/179152?type=Model&format=SafeTensor"
                    kwargs['lora_strength'] = self.config.lora_strength * 0.8
                elif era == 'future':
                    kwargs['lora_strength'] = self.config.lora_strength * 1.2

            # Create a new config with overrides applied
            # remove kwargs that are not in the config schema
            kwargs = {k: v for k, v in kwargs.items() if k in self.config.model_fields}

            # Create a new config instance with the current config and overrides
            current_config = FalComfyGeneratorConfig(**{
                **self.config.model_dump(),
                **kwargs
            })
            print(f"FalComfyGenerator: Using config overrides: {current_config}")
        else:
            current_config = self.config
        
        try:
            import fal_client
        except ImportError:
            raise RuntimeError("fal_client library not installed. Install with: pip install fal-client")
        
        logger.info(f"FalAIGenerator: Generating image with FAL.AI for prompt: {prompt[:50]}...")
        
        current_config.prompt = prompt
        # Add depth map if provided
        if depth_map_b64:
            current_config.depth_map = depth_map_b64

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
                    raise RuntimeError("FAL.AI generation stopped by user")
                elapsed_time = time.monotonic() - start_time
                if elapsed_time > timeout:
                    logger.error(f"FalAIGenerator: Generation timed out after {elapsed_time:.2f} seconds")
                    raise RuntimeError(f"FAL.AI generation timed out after {elapsed_time:.2f} seconds")
                logger.debug(event)

            if self._stop_event.is_set():
                raise RuntimeError("FAL.AI generation stopped by user")
            
            response = await handler.get()

            if self._stop_event.is_set():
                raise RuntimeError("FAL.AI generation stopped by user")

            # Download the generated image
            for image_url in self.falai_image_url_generator(response):
                # we just need one image URL, so we can break after the first
                request_id = kwargs.get('request_id')
                return await self._download_image(image_url, request_id=request_id)

            # If no image URL was found, raise an error to ensure a str is always returned or an exception is raised
            raise RuntimeError("No image URL found in FAL.AI response.")
                
        except asyncio.TimeoutError:
            logger.error(f"FalAIGenerator: Generation timed out after {timeout} seconds")
            raise RuntimeError(f"FAL.AI generation timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"FalAIGenerator: Error generating image: {e}")
            raise RuntimeError(f"FAL.AI generation failed: {e}")
        finally:
            logger.info(f"FalAIGenerator: Generation completed in {time.monotonic() - start_time:.2f} seconds")

    
    async def stop(self):
        """Stop the FAL.AI generator if running."""
        logger.info("Stopping FAL.AI generator...")
        self._stop_event.set()

    
    def falai_image_url_generator(self, response:dict) -> Generator[str, None, None]:
        """Generator to extract image URLs from FAL.AI response.
        Args:
            response: Response dictionary from FAL.AI API
        Yields:
            str: URLs of generated images
        Raises:
            ValueError: If response format is unknown or unsupported
        """

        # model endpoint
        # {'images': [{'url': 'https://fal.media/files/panda/uQsLdbOPox-ntaMxeDzsy.png', 'width': 1024, 'height': 768, 'content_type': 'image/jpeg'}], 'timings': {'inference': 0.3568768650002312}, 'seed': 852971348, 'has_nsfw_concepts': [False], 'prompt': 'test'}
        if 'images' in response:
            for image_result in response['images']:
                yield image_result['url']
            return
        
        # comfy workflow endpoint:
        # {
        #     "outputs": {
        #         "14": {
        #           "images": [
        #               {
        #               "url": "https://fal.media/files/panda/YB89Jz_d95I7_ZeBFhslU_ComfyUI_00003_.png",
        #               "type": "output",
        #               "filename": "ComfyUI_00003_.png",
        #               "subfolder": ""
        #               }
        #           ]
        #         }
        #     }, ...
        if 'outputs' in response:
            # the next key could change based on workflow, but it will always be first index
            first_output_key = list(response['outputs'].keys())[0]
            for image_result in response['outputs'][first_output_key]['images']:
                if 'image_output' not in image_result:
                    yield image_result['url']
                else:
                    yield image_result['image_output']
            return
        
        raise ValueError(f"Unknown response type: {response}")

if __name__ == "__main__":
    import argparse

    # Example usage
    # $ uv run -m image_server.generators.fal.fal_comfy_generator
    parser = argparse.ArgumentParser(description="FAL.AI Image Generation Example")
    parser.add_argument("--config", type=str, help="Config toml")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output_dir", type=str, default="/tmp", help="Directory to save generated images")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for image generation in seconds")
    parser.add_argument("--endpoint", type=str, default="fal-ai/fast-lightning-sdxl", help="FAL.AI endpoint to use")
    parser.add_argument("--seed", type=int, default=1, help="Seed for image generation")
    args = parser.parse_args()

    # Configure root logger
    logging.basicConfig(level=logging.INFO)
    # Configure external library loggers
    configure_external_loggers(logging.WARNING)
    
    logger.info("Starting FAL.AI image generation example...")

    # Example 1: Create a config instance and pass it to the generator
    sdxl_config = FalGeneratorConfig(
        strategy="falai",
        endpoint="fal-ai/fast-lightning-sdxl",
        dimensions=[1024, 1024],
        num_inference_steps=4,
        negative_prompt="distorted, warped, blurry, text, cartoon"
    )
    fal_generator = FalComfyGenerator(config=sdxl_config)
    
    # Example 2: Pass configuration directly to constructor
    comfy_config = FalComfyGeneratorConfig(
        strategy="falai",
        #endpoint="comfy/RKelln/experimance_hyper_depth_v5",
        endpoint="comfy/RKelln/experimancexilightningdepth",
        model_url="https://civitai.com/api/download/models/471120?type=Model&format=SafeTensor&size=full&fp=fp16",
        lora_url="https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor",
        seed=args.seed,
        timeout=args.timeout,
        negative_prompt="distorted, warped, blurry, text, cartoon, illustration, low quality, lowres"
    )
    experimance_generator = FalComfyGenerator(config=comfy_config)
    
    from image_server.generators.generator import mock_depth_map
    
    async def run_examples():
        """Run all examples in sequence within a single event loop."""
        results = []
        
        test_prompt = "colorful RAW photo modern masterpiece, overhead top down aerial shot, in the style of (Edward Burtynsky:1.2) and (Gerhard Richter:1.2), (dense urban:1.2) dramatic landscape, buildings, farmland, (industrial:1.1), (rivers, lakes:1.1), busy highways, hills, vibrant hyper detailed photorealistic maximum detail, 32k, high resolution ultra HD"
        
        # Example 1: Run the fast SDXL model
        # try:
        #     logger.info("Running Example 1: Fast SDXL model...")
        #     image_path = await fal_generator.generate_image(
        #         test_prompt
        #     )
        #     logger.info(f"Example 1 complete - Generated image saved to: {image_path}")
        #     results.append(True)
        # except Exception as e:
        #     logger.error(f"Example 1 failed: {e}")
        #     results.append(False)
        

        # Example 2: Run the ComfyUI workflow with depth map
        try:
            logger.info("Running Example 2: ComfyUI workflow with depth map...")
            image_path = await experimance_generator.generate_image(
                test_prompt,
                depth_map_b64=png_to_base64url(mock_depth_map())
            )
            logger.info(f"Example 2 complete - Generated image saved to: {image_path}")
            results.append(True)
        except Exception as e:
            logger.error(f"Example 2 failed: {e}")
            results.append(False)

        return results
    
    # Run all examples in a single event loop
    try:
        results = asyncio.run(run_examples())
        successful = sum(results)
        logger.info(f"Test run completed. {successful}/{len(results)} examples completed successfully.")
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}")
    else:
        if all(results):
            logger.info("All examples completed successfully!")
        else:
            logger.info("Some examples failed. See logs above for details.")
