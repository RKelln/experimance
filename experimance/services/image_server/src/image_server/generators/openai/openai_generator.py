import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from image_server.generators.generator import ImageGenerator

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
