#!/usr/bin/env python3
"""
VastAI image generator implementation.

This generator uses VastAI instances to generate images remotely using the 
experimance ControlNet model server. It manages a single instance lifecycle
and handles image generation requests serially.

The generator supports:
- Multiple ControlNet models (sdxl_small, llite, etc.)
- Era-specific LoRA selection (experimance, drone)
- Depth map conditioning via base64 or PIL Image
- Configurable generation parameters
"""

import asyncio
import logging
import time
import base64
import io
from typing import Dict, Any, Optional
from PIL import Image
from experimance_common.image_utils import base64url_to_png, png_to_base64url
import aiohttp
import requests  # Keep requests for quick health checks

from image_server.generators.generator import ImageGenerator, mock_depth_map
from image_server.generators.vastai.vastai_config import VastAIGeneratorConfig
from image_server.generators.vastai.vastai_manager import VastAIManager, InstanceEndpoint
from image_server.generators.vastai.server.data_types import ControlNetGenerateData

from experimance_common.schemas import Era

logger = logging.getLogger(__name__)

class VastAIGenerator(ImageGenerator):
    """VastAI-based image generator using remote ControlNet model server."""
    config : VastAIGeneratorConfig

    def __init__(self, config: VastAIGeneratorConfig, output_dir: str = "/tmp", **kwargs):
        """Initialize VastAI generator with configuration."""
        super().__init__(config, output_dir, **kwargs)
        self.manager = VastAIManager()
        self.current_endpoint = None
        self._instance_ready = False
        self._initialized = False
        
        # Pre-warm the generator if enabled
        if self.config.pre_warm:
            asyncio.create_task(self._pre_warm())
        
    def _configure(self, config: VastAIGeneratorConfig, **kwargs):
        """Configure the generator with VastAI-specific settings."""
        logger.info(f"Configuring VastAI generator with model: {config.model_name}")
        logger.info(f"Steps: {config.steps}, CFG: {config.cfg}")
        logger.info(f"Scheduler: {config.scheduler}, ControlNet strength: {config.controlnet_strength}")
    
    async def _pre_warm(self):
        """Pre-warm the generator by sending a test generation request.
        
        This helps reduce cold start latency for the first real generation.
        The result is discarded and not saved.
        """
        try:
            logger.info("Pre-warming VastAI generator...")

            # Send a simple test prompt with mock depth map
            test_prompt = "test image for warming up"
            
            # Use a shorter timeout for pre-warming to avoid hanging
            original_timeout = self.config.instance_timeout
            self.config.instance_timeout = self.config.pre_warm_timeout
            
            try:
                await self.generate_image(
                    prompt=test_prompt,
                    mock_depth=True  # Use server's built-in mock depth generation
                )
                logger.info("VastAI generator pre-warming completed successfully")
            finally:
                # Restore original timeout
                self.config.instance_timeout = original_timeout
            
        except Exception as e:
            logger.warning(f"VastAI generator pre-warming failed (continuing anyway): {e}")
            # Don't raise the exception - pre-warming failure shouldn't stop initialization
        
    def _ensure_instance_ready(self) -> InstanceEndpoint:
        """Ensure we have a ready VastAI instance and return its endpoint."""
        logger.info("Ensuring VastAI instance is ready...")
        
        # Check if we have a current endpoint and it's still healthy
        if self.current_endpoint:
            if self._health_check(self.current_endpoint):
                logger.info(f"Using existing healthy instance: {self.current_endpoint.instance_id}")
                return self.current_endpoint
            else:
                logger.warning(f"Instance {self.current_endpoint.instance_id} is unhealthy, will find/create new one")
                self.current_endpoint = None
        
        # Find or create a ready instance
        endpoint = self.manager.find_or_create_instance(
            create_if_none=self.config.create_if_none,
            wait_for_ready=self.config.wait_for_ready
        )
        
        if not endpoint:
            raise RuntimeError("Failed to get a ready VastAI instance")
            
        self.current_endpoint = endpoint
        logger.info(f"VastAI instance ready: {endpoint.instance_id} at {endpoint.url}")
        return endpoint
        
    async def _initialize_instance(self):
        """Initialize the VastAI instance on first use (lazy initialization)."""
        if self._initialized:
            return
            
        logger.info("Initializing VastAI instance for first use...")
        try:
            # Ensure we have a ready instance
            self.current_endpoint = self._ensure_instance_ready()
            self._instance_ready = True
            self._initialized = True
            logger.info("VastAI instance initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VastAI instance: {e}")
            self._instance_ready = False
            raise
        
    def _health_check(self, endpoint: InstanceEndpoint) -> bool:
        """Check if the instance endpoint is healthy."""
        try:
            response = requests.get(
                f"{endpoint.url}/healthcheck",
                timeout=5  # Short timeout for health checks
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
            return False
        except Exception as e:
            logger.warning(f"Health check failed for instance {endpoint.instance_id}: {e}")
            return False
    
    def _encode_image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Encode PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _decode_base64_to_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate an image using VastAI remote model server.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (optional)
            depth_image: Depth map for ControlNet conditioning (optional)
            seed: Random seed for reproducible results (optional)
            **kwargs: Additional generation parameters
                depth_map_b64: Base64 encoded depth map (alternative to depth_image)
                controlnet: ControlNet model to use (default: "sdxl_small")
                era: Era for LoRA selection (influences era-specific styling)
            
        Returns:
            Path to the saved generated image
        """
        self._validate_prompt(prompt)
        logger.info(f"Generating image with VastAI: {prompt[:50]}...")
        
        # Lazy initialization - ensure instance is ready on first call
        if not self._initialized:
            await self._initialize_instance()
        
        # Fast path: use pre-initialized instance
        if not self._instance_ready or not self.current_endpoint:
            raise RuntimeError("VastAI generator not ready. Instance is unavailable.")
        
        start_time = time.time()
    
        # Get depth map if provided
        depth_map_b64 = kwargs.get("depth_map_b64", None)
        mock_depth = kwargs.get("mock_depth", depth_map_b64 is None)  # Allow explicit mock_depth override

        # Log depth map information for debugging
        if depth_map_b64:
            logger.info(f"📷 VastAI: Using provided depth map (length: {len(depth_map_b64)} chars)")
            # Validate the depth map format on client side
            if not depth_map_b64.startswith(("data:image/", "iVBORw0KGgo")):  # Common base64 image prefixes
                logger.warning(f"⚠️  VastAI: Depth map doesn't look like valid base64 image data. Prefix: {depth_map_b64[:50]}...")
        elif mock_depth:
            logger.info("🎭 VastAI: Using mock depth map")
        else:
            logger.warning("⚠️  VastAI: No depth map and mock_depth=False - server may fail")

        era = "experimance"
        if kwargs.get("era"):
            if kwargs["era"] in [Era.WILDERNESS, Era.PRE_INDUSTRIAL]:
                era = "drone"

        try:
            endpoint = self.current_endpoint  # Use pre-initialized endpoint
            
            # Create the generation request using ControlNetGenerateData
            data = ControlNetGenerateData(
                prompt=prompt,
                negative_prompt=negative_prompt,
                depth_map_b64=depth_map_b64,
                mock_depth=mock_depth,
                model=self.config.model_name,
                controlnet=kwargs.get("controlnet", "sdxl_small"),
                era=era,
                steps=self.config.steps or 6,
                cfg=self.config.cfg or 2.0,
                seed=seed,
                scheduler=self.config.scheduler,
                use_karras_sigmas=self.config.use_karras_sigmas,
                lora_strength=self.config.lora_strength,
                controlnet_strength=self.config.controlnet_strength,
                width=self.config.width,
                height=self.config.height,
            )
            
            # Validate the request
            validation_errors = data.validate()
            if validation_errors:
                raise RuntimeError(f"Generation request validation failed: {validation_errors}")
            
            # Convert to JSON payload
            payload = data.generate_payload_json()
            
            logger.info(f"Payload keys: {list(payload.keys())}, mock_depth: {payload.get('mock_depth')}")
            
            # Send generation request (no health check for speed)
            logger.debug(f"Sending generation request to {endpoint.url}/generate")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{endpoint.url}/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.instance_timeout)
                ) as response:
                    if response.status != 200:
                        # Instance might be down, mark as not ready for next request
                        self._instance_ready = False
                        error_text = await response.text()
                        raise RuntimeError(f"Generation request failed: {response.status} - {error_text}")
                    
                    result = await response.json()
            
            if not result.get("success", True):
                error_msg = result.get("error_message", "Unknown error")
                raise RuntimeError(f"Generation failed on remote server: {error_msg}")
            
            # Decode the generated image
            image_b64 = result.get("image_b64")
            if not image_b64:
                raise RuntimeError("No image data received from remote server")
            
            generated_image = base64url_to_png(image_b64)
            if generated_image is None:
                logger.error("🚨 Failed to decode generated image from server!")
                logger.error(f"Image data prefix: {image_b64[:50]}..." if len(image_b64) > 50 else f"Full image data: {image_b64}")
                raise RuntimeError("Failed to decode generated image from remote server")
            
            # Save the image and return the path
            request_id = kwargs.get('request_id')
            output_path = self._get_output_path("png", request_id=request_id)
            generated_image.save(output_path)
            
            # Calculate total time (including network overhead)
            total_time = time.time() - start_time
            model_time = result.get("generation_time", 0)
            
            logger.info(f"Image generated successfully in {total_time:.2f}s (model: {model_time:.2f}s)")
            logger.debug(f"Image saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"VastAI generation failed: {e}")
            # Mark instance as not ready if there was a connection issue
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                self._instance_ready = False
            raise RuntimeError(f"VastAI generation failed: {e}")
    
    async def stop(self):
        """Stop the generator and clean up resources."""
        logger.info("Stopping VastAI generator...")
        self.cleanup()
        logger.info("VastAI generator stopped")
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to VastAI and model server.
        
        Returns:
            Dictionary with test results
        """
        try:
            # Test VastAI API connection
            instances = self.manager.show_instances()
            
            # Try to get a ready instance
            endpoint = self._ensure_instance_ready()
            
            # Test model server endpoints
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint.url}/healthcheck", timeout=aiohttp.ClientTimeout(total=10)) as health_response:
                    health_ok = health_response.status == 200
                
                async with session.get(f"{endpoint.url}/models", timeout=aiohttp.ClientTimeout(total=10)) as models_response:
                    models_ok = models_response.status == 200
            
            return {
                "vastai_connection": True,
                "instances_found": len(instances),
                "endpoint_ready": endpoint is not None,
                "health_check": health_ok,
                "models_endpoint": models_ok,
                "endpoint_url": endpoint.url if endpoint else None,
                "instance_id": endpoint.instance_id if endpoint else None
            }
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return {
                "vastai_connection": False,
                "error": str(e)
            }

    def cleanup(self):
        """Clean up resources and optionally destroy instances."""
        logger.info("Cleaning up VastAI generator...")
        # Note: We don't automatically destroy instances here since they're expensive to recreate
        # Users can manually destroy instances through the VastAI manager if needed
        self.current_endpoint = None
        logger.info("VastAI generator cleanup complete")
