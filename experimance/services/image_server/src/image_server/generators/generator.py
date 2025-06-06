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
