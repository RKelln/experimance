#!/usr/bin/env python3
"""
Image generation strategy implementations for the Experimance image server.

This module provides an abstract base class and concrete implementations
for different image generation backends (mock, local, remote APIs).
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import tempfile

logger = logging.getLogger(__name__)


class ImageGenerator(ABC):
    """Abstract base class for image generation strategies."""
    
    def __init__(self, output_dir: str = "/tmp", **kwargs):
        """Initialize the image generator.
        
        Args:
            output_dir: Directory to save generated images
            **kwargs: Additional configuration options
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._configure(**kwargs)
    
    def _configure(self, **kwargs):
        """Configure generator-specific settings.
        
        Subclasses can override this to handle their specific configuration.
        """
        pass
    
    @abstractmethod
    async def generate_image(self, prompt: str, depth_map_b64: Optional[str] = None, **kwargs) -> str:
        """Generate an image based on the given prompt and optional depth map.
        
        Args:
            prompt: Text description of the image to generate
            depth_map_b64: Optional base64-encoded depth map PNG
            **kwargs: Additional generation parameters
            
        Returns:
            Path to the generated image file
            
        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails
        """
        pass
    
    def _validate_prompt(self, prompt: str):
        """Validate the input prompt."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
    
    def _get_output_path(self, extension: str = "png") -> str:
        """Generate a unique output path for an image."""
        image_id = str(uuid.uuid4())
        return str(self.output_dir / f"generated_{image_id}.{extension}")


class MockImageGenerator(ImageGenerator):
    """Mock image generator for testing purposes.
    
    Generates simple placeholder images with the prompt text.
    """
    
    def _configure(self, **kwargs):
        """Configure mock generator settings."""
        self.image_size = kwargs.get("image_size", (1024, 1024))
        self.background_color = kwargs.get("background_color", (100, 150, 200))
        self.text_color = kwargs.get("text_color", (255, 255, 255))
    
    async def generate_image(self, prompt: str, depth_map_b64: Optional[str] = None, **kwargs) -> str:
        """Generate a mock image with the prompt text."""
        self._validate_prompt(prompt)
        
        logger.info(f"MockImageGenerator: Generating image for prompt: {prompt[:50]}...")
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Create a simple image with the prompt text
        image = Image.new("RGB", self.image_size, self.background_color)
        draw = ImageDraw.Draw(image)
        
        # Try to use a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Add prompt text (truncated to fit)
        text = prompt[:100] + "..." if len(prompt) > 100 else prompt
        
        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.image_size[0] - text_width) // 2
        y = (self.image_size[1] - text_height) // 2
        
        draw.text((x, y), text, fill=self.text_color, font=font)
        
        # Add era/biome info if provided
        era = kwargs.get("era", "unknown")
        biome = kwargs.get("biome", "unknown")
        info_text = f"Era: {era}, Biome: {biome}"
        
        # Add info text at the bottom
        info_bbox = draw.textbbox((0, 0), info_text, font=font)
        info_width = info_bbox[2] - info_bbox[0]
        info_x = (self.image_size[0] - info_width) // 2
        info_y = self.image_size[1] - 50
        
        draw.text((info_x, info_y), info_text, fill=self.text_color, font=font)
        
        # Save the image
        output_path = self._get_output_path("png")
        image.save(output_path)
        
        logger.info(f"MockImageGenerator: Saved image to {output_path}")
        return output_path


class FalAIGenerator(ImageGenerator):
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
    
    async def _download_image(self, image_url: str) -> str:
        """Download an image from a URL."""
        import aiohttp
        
        output_path = self._get_output_path("png")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    logger.info(f"FalAIGenerator: Downloaded image to {output_path}")
                    return output_path
                else:
                    raise RuntimeError(f"Failed to download image: HTTP {response.status}")


class OpenAIGenerator(ImageGenerator):
    """OpenAI DALL-E image generator implementation."""
    
    def _configure(self, **kwargs):
        """Configure OpenAI generator settings."""
        self.model = kwargs.get("model", "dall-e-3")
        self.quality = kwargs.get("quality", "standard")
        self.dimensions = kwargs.get("dimensions", [1024, 1024])
        self.timeout = kwargs.get("timeout", 60)
    
    async def generate_image(self, prompt: str, depth_map_b64: Optional[str] = None, **kwargs) -> str:
        """Generate an image using OpenAI DALL-E API."""
        self._validate_prompt(prompt)
        
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai library not installed. Install with: pip install openai")
        
        logger.info(f"OpenAIGenerator: Generating image with DALL-E for prompt: {prompt[:50]}...")
        
        # Note: DALL-E doesn't support depth maps directly
        if depth_map_b64:
            logger.warning("OpenAIGenerator: Depth maps not supported by DALL-E, ignoring")
        
        try:
            client = openai.Client()  # Uses OPENAI_API_KEY environment variable
            
            # Generate image with DALL-E
            response = await asyncio.to_thread(
                client.images.generate,
                model=self.model,
                prompt=prompt,
                size=f"{self.dimensions[0]}x{self.dimensions[1]}",
                quality=self.quality,
                n=1
            )
            
            # Download the generated image
            if response.data and len(response.data) > 0:
                image_url = response.data[0].url
                return await self._download_image(image_url)
            else:
                raise RuntimeError("No images returned from OpenAI")
                
        except Exception as e:
            logger.error(f"OpenAIGenerator: Error generating image: {e}")
            raise RuntimeError(f"OpenAI generation failed: {e}")
    
    async def _download_image(self, image_url: str) -> str:
        """Download an image from a URL."""
        import aiohttp
        
        output_path = self._get_output_path("png")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    logger.info(f"OpenAIGenerator: Downloaded image to {output_path}")
                    return output_path
                else:
                    raise RuntimeError(f"Failed to download image: HTTP {response.status}")


class LocalSDXLGenerator(ImageGenerator):
    """Local SDXL image generator implementation.
    
    Uses a local SDXL model for image generation.
    """
    
    def _configure(self, **kwargs):
        """Configure local SDXL generator settings."""
        self.model_path = kwargs.get("model_path", "stabilityai/stable-diffusion-xl-base-1.0")
        self.dimensions = kwargs.get("dimensions", [1024, 1024])
        self.num_inference_steps = kwargs.get("num_inference_steps", 20)
        self.guidance_scale = kwargs.get("guidance_scale", 7.5)
        self.device = kwargs.get("device", "cuda" if self._cuda_available() else "cpu")
        self._pipeline = None
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def generate_image(self, prompt: str, depth_map_b64: Optional[str] = None, **kwargs) -> str:
        """Generate an image using local SDXL model."""
        self._validate_prompt(prompt)
        
        # Initialize pipeline if not already done
        if self._pipeline is None:
            await self._initialize_pipeline()
        
        logger.info(f"LocalSDXLGenerator: Generating image locally for prompt: {prompt[:50]}...")
        
        try:
            # Prepare generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "width": self.dimensions[0],
                "height": self.dimensions[1]
            }
            
            # Add depth map support if available and provided
            if depth_map_b64 and hasattr(self._pipeline, 'depth_map'):
                # Convert base64 to PIL Image
                depth_image = self._decode_depth_map(depth_map_b64)
                generation_kwargs["image"] = depth_image
            
            # Generate image (run in thread to avoid blocking)
            result = await asyncio.to_thread(
                self._pipeline,
                **generation_kwargs
            )
            
            # Save the generated image
            if hasattr(result, 'images') and len(result.images) > 0:
                output_path = self._get_output_path("png")
                result.images[0].save(output_path)
                logger.info(f"LocalSDXLGenerator: Saved image to {output_path}")
                return output_path
            else:
                raise RuntimeError("No images generated by local model")
                
        except Exception as e:
            logger.error(f"LocalSDXLGenerator: Error generating image: {e}")
            raise RuntimeError(f"Local SDXL generation failed: {e}")
    
    async def _initialize_pipeline(self):
        """Initialize the SDXL pipeline."""
        try:
            from diffusers import StableDiffusionXLPipeline
            import torch
        except ImportError:
            raise RuntimeError("diffusers library not installed. Install with: pip install diffusers torch")
        
        logger.info(f"LocalSDXLGenerator: Initializing SDXL pipeline with model: {self.model_path}")
        
        # Load the pipeline (run in thread to avoid blocking)
        self._pipeline = await asyncio.to_thread(
            StableDiffusionXLPipeline.from_pretrained,
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        )
        
        # Move to device
        self._pipeline = self._pipeline.to(self.device)
        
        # Enable memory efficient attention if available
        if hasattr(self._pipeline, 'enable_attention_slicing'):
            self._pipeline.enable_attention_slicing()
        
        logger.info(f"LocalSDXLGenerator: Pipeline initialized on {self.device}")
    
    def _decode_depth_map(self, depth_map_b64: str) -> Image.Image:
        """Decode base64 depth map to PIL Image."""
        import base64
        from io import BytesIO
        
        # Remove data URL prefix if present
        if depth_map_b64.startswith('data:'):
            depth_map_b64 = depth_map_b64.split(',', 1)[1]
        
        # Decode base64
        image_data = base64.b64decode(depth_map_b64)
        
        # Load as PIL Image
        return Image.open(BytesIO(image_data))


# Factory function to create generators
def create_generator(strategy: str, **kwargs) -> ImageGenerator:
    """Factory function to create image generators.
    
    Args:
        strategy: Generator strategy ("mock", "fal", "openai", "local")
        **kwargs: Configuration options for the generator
        
    Returns:
        Configured ImageGenerator instance
        
    Raises:
        ValueError: If strategy is not supported
    """
    generators = {
        "mock": MockImageGenerator,
        "fal": FalAIGenerator,
        "openai": OpenAIGenerator,
        "local": LocalSDXLGenerator
    }
    
    if strategy not in generators:
        raise ValueError(f"Unsupported generator strategy: {strategy}. "
                        f"Available strategies: {list(generators.keys())}")
    
    return generators[strategy](**kwargs)
