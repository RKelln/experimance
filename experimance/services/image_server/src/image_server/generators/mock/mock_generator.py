import asyncio
import logging
from typing import Optional, Literal

from image_server.generators.generator import ImageGenerator
from image_server.generators.config import BaseGeneratorConfig
from .mock_generator_config import MockGeneratorConfig

from PIL import Image, ImageDraw, ImageFont

# Configure logging
logger = logging.getLogger(__name__)


class MockImageGenerator(ImageGenerator):
    """Mock image generator for testing purposes.
    
    Generates simple placeholder images with the prompt text.
    """
    
    def _configure(self, config:BaseGeneratorConfig, **kwargs):
        """Configure mock generator settings."""
        self.config = MockGeneratorConfig(**{
            **config.model_dump(),
            **kwargs
        })
    
    async def generate_image(self, prompt: str, depth_map_b64: Optional[str] = None, **kwargs) -> str:
        """Generate a mock image with the prompt text."""
        self._validate_prompt(prompt)
        
        logger.info(f"MockImageGenerator: Generating image for prompt: {prompt[:50]}...")
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Create a simple image with the prompt text
        image = Image.new("RGB", self.config.image_size, self.config.background_color)
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
        x = (self.config.image_size[0] - text_width) // 2
        y = (self.config.image_size[1] - text_height) // 2
        
        draw.text((x, y), text, fill=self.config.text_color, font=font)
        
        # Add era/biome info if provided
        era = kwargs.get("era", "unknown")
        biome = kwargs.get("biome", "unknown")
        info_text = f"Era: {era}, Biome: {biome}"
        
        # Add info text at the bottom
        info_bbox = draw.textbbox((0, 0), info_text, font=font)
        info_width = info_bbox[2] - info_bbox[0]
        info_x = (self.config.image_size[0] - info_width) // 2
        info_y = self.config.image_size[1] - 50
        
        draw.text((info_x, info_y), info_text, fill=self.config.text_color, font=font)
        
        # Save the image
        output_path = self._get_output_path("png")
        image.save(output_path)
        
        logger.info(f"MockImageGenerator: Saved image to {output_path}")
        return output_path
        
    async def stop(self):
        """Stop any ongoing generation processes.
        
        For the mock generator, this is a no-op since there are no background 
        processes to clean up.
        """
        logger.info("MockImageGenerator: Stopping (no-op)")
        return