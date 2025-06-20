#!/usr/bin/env python3
"""
Image generation strategy implementations for the Experimance image server.

This module provides an abstract base class and concrete implementations
for different image generation backends (mock, local, remote APIs).
"""

import asyncio
from datetime import datetime
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont

from image_server.generators.config import BaseGeneratorConfig
from experimance_common.logger import configure_external_loggers

# Configure logging
logger = logging.getLogger(__name__)

# Configure external loggers when module is imported
configure_external_loggers()

VALID_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp']

class ImageGenerator(ABC):
    """Abstract base class for image generation strategies."""
    
    def __init__(self, config: BaseGeneratorConfig, output_dir: str = "/tmp",  **kwargs):
        """Initialize the image generator.
        
        Args:
            output_dir: Directory to save generated images
            **kwargs: Additional configuration options
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._configure(config, **kwargs)
    
    def _configure(self, config, **kwargs):
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
    
    @abstractmethod
    async def stop(self):
        """Stop any ongoing generation processes.
        
        This method should be implemented by subclasses to handle cleanup.
        """
        pass

    def _validate_prompt(self, prompt: str):
        """Validate the input prompt."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
    
    def _get_output_path(self, file_or_extension: str = "png", request_id: Optional[str] = None) -> str:
        """Generate a unique output path for an image.
        
        Args:
            file_or_extension: File extension or full filename
            request_id: Optional request ID to include in filename for traceability
        """
        name = None
        if file_or_extension in VALID_EXTENSIONS:
            # If a file extension is provided, use it directly
            extension = f".{file_or_extension}"
        elif isinstance(file_or_extension, str):
            # If a string is provided, assume it's a filename with extension
            path = Path(file_or_extension)
            name, extension = path.stem, path.suffix

        if extension[1:] not in VALID_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {extension}. Must be one of png, jpg, jpeg, webp")
            
        # Create ID using request_id if provided, otherwise fall back to timestamp
        image_id = f"{self.__class__.__name__.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if request_id:
            image_id += f"_{request_id}"
        if name:
            image_id += f"_{name}"
        return str(self.output_dir / f"{image_id}{extension}")
        
    async def _download_image(self, image_url: str, request_id: Optional[str] = None) -> str:
        """Download the generated image from the provided URL.
        
        Args:
            image_url: URL of the generated image
            request_id: Optional request ID to include in filename for traceability
        Returns:
            Path to the downloaded image file   
        Raises:
            RuntimeError: If download fails
        """
        try:
            import aiohttp
            
            if not image_url:
                raise ValueError("Image URL cannot be empty")
            
            # get extension from URL
            
            output_path = self._get_output_path(image_url, request_id=request_id)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        with open(output_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        logger.info(f"{self.__class__.__name__}: Image downloaded and saved to {output_path}")
                        return output_path
                    else:
                        error_message = f"Failed to download image: HTTP {response.status}"
                        logger.error(f"{self.__class__.__name__}: {error_message}")
                        raise RuntimeError(error_message)
        
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Error downloading image: {e}")
            raise RuntimeError(f"Failed to download image: {e}")


def mock_depth_map(size: tuple = (1024, 1024)) -> Image.Image:
    """Generate a mock depth map image.
    
    Args:
        size: Size of the depth map image
        color: Color to fill the depth map (default gray)
        
    Returns:
        PIL Image object representing the depth map
    """
    # check for depthmap in mock images
    mock = Path("services/image_server/images/mocks/depth_map.png")
    if size == (1024,1024) and mock.exists():
        return Image.open(mock.resolve()).convert("L")
    else:
        depth_map = Image.new("L", size, color=128)  # Create a gray depth map

    return depth_map