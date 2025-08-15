import asyncio
import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)

from image_server.generators.generator import ImageGenerator
from image_server.generators.config import BaseGeneratorConfig

class OpenAIGeneratorConfig:
    model: str = "dall-e-3"
    quality: Literal['standard', 'hd', 'low', 'medium', 'high', 'auto']
    size: Literal['auto', '1024x1024', '1536x1024', '1024x1536', '256x256', '512x512', '1792x1024', '1024x1792'] = '1024x1024'

class OpenAIGenerator(ImageGenerator):
    """OpenAI DALL-E image generator implementation."""
    

    def _configure(self, config:BaseGeneratorConfig, **kwargs):
        """Configure OpenAI generator settings from kwargs or create default config."""
        self.config = OpenAIGeneratorConfig(**{
            **config.model_dump(),
            **kwargs
        })
        logger.info(f"OpenAIGenerator initialized: {self.config}")
    
    async def _generate_image_impl(self, prompt: str, depth_map_b64: Optional[str] = None, **kwargs) -> str:
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
                model=self.config.model,
                prompt=prompt,
                size=self.config.size,
                quality=self.config.quality,
                n=1
            )
            
            # Download the generated image
            if response.data and len(response.data) > 0:
                image_url = response.data[0].url
                if not image_url:
                    raise RuntimeError("No image URL returned from OpenAI")
                return await self._download_image(image_url)
            else:
                raise RuntimeError("No images returned from OpenAI")
                
        except Exception as e:
            logger.error(f"OpenAIGenerator: Error generating image: {e}")
            raise RuntimeError(f"OpenAI generation failed: {e}")
    
    # Using _download_image from the base ImageGenerator class
