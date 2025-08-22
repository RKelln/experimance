import asyncio
import logging
import random
import shutil
from pathlib import Path
from typing import Optional, Literal, List

from image_server.generators.generator import ImageGenerator, GeneratorCapabilities
from image_server.generators.config import BaseGeneratorConfig
from .mock_generator_config import MockGeneratorConfig

from PIL import Image, ImageDraw, ImageFont

# Configure logging
logger = logging.getLogger(__name__)


class MockImageGenerator(ImageGenerator):
    """Mock image generator for testing purposes.
    
    Can either generate simple placeholder images with prompt text or 
    use existing images from a specified directory for more realistic testing.
    """
    
    # Mock generator supports most capabilities for testing purposes
    supported_capabilities = {
        GeneratorCapabilities.IMAGE_TO_IMAGE,
        GeneratorCapabilities.CONTROLNET,
        GeneratorCapabilities.LORAS,
        GeneratorCapabilities.NEGATIVE_PROMPTS,
        GeneratorCapabilities.SEEDS,
        GeneratorCapabilities.INPAINTING,
        GeneratorCapabilities.UPSCALING,
        GeneratorCapabilities.BATCH_GENERATION
    }
    
    def _configure(self, config:BaseGeneratorConfig, **kwargs):
        """Configure mock generator settings."""
        self.config = MockGeneratorConfig(**{
            **config.model_dump(),
            **kwargs
        })
        
        # If using existing images, validate the directory and collect image files
        self._existing_images: List[Path] = []
        if self.config.use_existing_images and self.config.existing_images_dir:
            self._load_existing_images()
    
    def _load_existing_images(self) -> None:
        """Load list of existing images from the configured directory."""
        if not self.config.existing_images_dir or not self.config.existing_images_dir.exists():
            logger.warning(f"Existing images directory not found: {self.config.existing_images_dir}")
            return
            
        # Find all image files (common formats)
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp'}
        self._existing_images = [
            f.relative_to(self.config.existing_images_dir) for f in self.config.existing_images_dir.rglob("*")
            if f.suffix.lower() in image_extensions and f.is_file()
        ]
        
        logger.info(f"MockImageGenerator: Found {len(self._existing_images)} existing images in {self.config.existing_images_dir}")
        if not self._existing_images:
            logger.warning("No existing images found - will fall back to generated placeholders")
    
    async def _generate_image_impl(self, prompt: str, **kwargs) -> str:
        """Generate a mock image with the prompt text or copy an existing image."""
        self._validate_prompt(prompt)
        
        logger.info(f"MockImageGenerator: Generating image for prompt: {prompt[:50]}...")
        
        immediate = kwargs.get('immediate', False)
        if not immediate:
            # Simulate some processing time
            if self.config.delay > 0:
                await asyncio.sleep(self.config.delay)
            else:
                await asyncio.sleep(random.uniform(0.5, 2.5))
        
        # If we have existing images and are configured to use them, pick one randomly
        if self.config.use_existing_images and self._existing_images:
            return await self._copy_existing_image(prompt, **kwargs)
        else:
            return await self._generate_placeholder_image(prompt, **kwargs)
    
    async def _copy_existing_image(self, prompt: str, **kwargs) -> str:
        """Copy a random existing image to the output location."""
        match_paths = []
        
        if era := kwargs.get('era', None):
            match_paths.append(era.lower())

        if biome := kwargs.get('biome', None):
            match_paths.append(biome.lower())

        if len(match_paths) == 0:
            # Pick a random existing image
            source_image = random.choice(self._existing_images)
        else:
            # Pick a random existing image that matches paths 
            # images stored as era/biome/filename
            matching_images = [
                img for img in self._existing_images if all(part in img.parts for part in match_paths)
            ]
            logger.debug(f"Matching images for {match_paths}: {len(matching_images)} found")
            source_image = random.choice(matching_images or self._existing_images)

        # Determine output format based on source image
        output_ext = source_image.suffix.lower()
        if output_ext == '.jpeg':
            output_ext = '.jpg'

        source_image = (self.config.existing_images_dir or Path(".")) / source_image

        return str(source_image)  # Return the path to the existing image

        # Create output path with request_id if provided
        request_id = kwargs.get('request_id')
        #output_path = self._get_output_path(output_ext.lstrip('.'), request_id=request_id)
        
        # Copy the image
        #shutil.copy2(source_image, output_path)
        
        #logger.info(f"MockImageGenerator: Copied existing image {source_image.name} to {output_path}")
        #return str(output_path)
    
    async def _generate_placeholder_image(self, prompt: str, **kwargs) -> str:
        """Generate a simple placeholder image with the prompt text."""
        # Create a simple image with the prompt text
        default_size = {
            "width": self.config.image_size[0],
            "height": self.config.image_size[1]
        }
        width = kwargs.get('width', kwargs.get("image_size", default_size)["width"])
        height = kwargs.get('height', kwargs.get("image_size", default_size)["height"])

        background_color = self.config.background_color or (random.randint(0,128), random.randint(0,128), random.randint(0,128))

        logger.debug(f"Creating placeholder image of size {width}x{height} with background {background_color}")

        image = Image.new("RGB", (width, height), background_color)
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
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill=self.config.text_color, font=font)
        
        # draw a box outline along all outside edges
        draw.rectangle([0, 0, width - 1, height - 1], outline=self.config.text_color, width=5)

        # Save the image
        request_id = kwargs.get('request_id')
        output_path = self._get_output_path("png", request_id=request_id)
        image.save(output_path)
        
        logger.info(f"MockImageGenerator: Saved placeholder image to {output_path}")
        return str(output_path)
        
    async def stop(self):
        """Stop any ongoing generation processes.
        
        For the mock generator, this is a no-op since there are no background 
        processes to clean up.
        """
        logger.info("MockImageGenerator: Stopping (no-op)")
        return