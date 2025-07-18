"""
VastAI Image Generator

This generator manages remote image generation using VastAI instances.
It automatically finds or creates VastAI instances and routes image generation
requests to them, providing automatic scaling and fallback capabilities.
"""

import asyncio
import logging
import time
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path
import aiohttp
import json

from experimance_common.schemas import GeneratedImage
from ..base_generator import BaseImageGenerator
from .vastai.vastai_manager import VastAIManager, InstanceEndpoint

logger = logging.getLogger(__name__)


class VastAIImageGenerator(BaseImageGenerator):
    """
    Image generator that uses VastAI instances for remote generation.
    
    Features:
    - Automatic instance discovery and creation
    - Load balancing across multiple instances
    - Health monitoring and failover
    - Automatic scaling based on demand
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VastAI generator.
        
        Args:
            config: Configuration dictionary with VastAI settings
        """
        super().__init__(config)
        
        # VastAI configuration
        self.vastai_api_key = config.get("vastai_api_key")
        self.max_instances = config.get("max_instances", 3)
        self.min_instances = config.get("min_instances", 1)
        self.instance_timeout = config.get("instance_timeout", 300)  # 5 minutes
        self.health_check_interval = config.get("health_check_interval", 60)  # 1 minute
        self.auto_scale = config.get("auto_scale", True)
        
        # Instance management
        self.manager = VastAIManager(api_key=self.vastai_api_key)
        self.active_instances: List[InstanceEndpoint] = []
        self.unhealthy_instances: List[InstanceEndpoint] = []
        self.last_health_check = 0
        self.generation_queue_size = 0
        
        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"VastAI generator initialized with max_instances={self.max_instances}")
    
    async def initialize(self) -> bool:
        """Initialize the generator and discover existing instances."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.instance_timeout)
            )
            
            # Discover existing instances
            await self._discover_instances()
            
            # Ensure we have at least min_instances running
            if len(self.active_instances) < self.min_instances:
                logger.info(f"Need {self.min_instances - len(self.active_instances)} more instances")
                await self._scale_up(self.min_instances - len(self.active_instances))
            
            logger.info(f"VastAI generator initialized with {len(self.active_instances)} instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize VastAI generator: {e}")
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("VastAI generator cleaned up")
    
    async def _discover_instances(self):
        """Discover existing VastAI instances."""
        try:
            # Use sync call in executor to avoid blocking
            loop = asyncio.get_event_loop()
            instances = await loop.run_in_executor(
                None, 
                self.manager.find_experimance_instances
            )
            
            # Check health of discovered instances
            self.active_instances = []
            for instance in instances:
                endpoint = self.manager.get_model_server_endpoint(instance['id'])
                if endpoint and await self._check_instance_health(endpoint):
                    self.active_instances.append(endpoint)
                    logger.info(f"Discovered healthy instance: {endpoint.url}")
                else:
                    logger.warning(f"Instance {instance['id']} is not healthy")
            
        except Exception as e:
            logger.error(f"Failed to discover instances: {e}")
    
    async def _check_instance_health(self, endpoint: InstanceEndpoint) -> bool:
        """Check if an instance is healthy and responsive."""
        try:
            if not self.session:
                return False
                
            async with self.session.get(f"{endpoint.url}/healthcheck") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy"
                return False
                
        except Exception as e:
            logger.debug(f"Health check failed for {endpoint.url}: {e}")
            return False
    
    async def _periodic_health_check(self):
        """Periodically check instance health."""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = current_time
        
        # Check all active instances
        healthy_instances = []
        for instance in self.active_instances:
            if await self._check_instance_health(instance):
                healthy_instances.append(instance)
            else:
                logger.warning(f"Instance {instance.url} became unhealthy")
                self.unhealthy_instances.append(instance)
        
        self.active_instances = healthy_instances
        
        # Auto-scale if needed
        if self.auto_scale:
            await self._auto_scale()
    
    async def _auto_scale(self):
        """Automatically scale instances based on demand and health."""
        active_count = len(self.active_instances)
        
        # Scale up if below minimum or queue is getting full
        should_scale_up = (
            active_count < self.min_instances or
            (self.generation_queue_size > active_count * 2 and active_count < self.max_instances)
        )
        
        if should_scale_up:
            scale_count = min(
                self.max_instances - active_count,
                max(1, self.min_instances - active_count)
            )
            await self._scale_up(scale_count)
    
    async def _scale_up(self, count: int):
        """Create new instances."""
        logger.info(f"Scaling up by {count} instances")
        
        for i in range(count):
            try:
                # Use executor to avoid blocking on instance creation
                loop = asyncio.get_event_loop()
                endpoint = await loop.run_in_executor(
                    None,
                    lambda: self.manager.find_or_create_instance(
                        create_if_none=True,
                        wait_for_ready=True
                    )
                )
                
                if endpoint:
                    self.active_instances.append(endpoint)
                    logger.info(f"Successfully created instance: {endpoint.url}")
                else:
                    logger.error(f"Failed to create instance {i+1}/{count}")
                    
            except Exception as e:
                logger.error(f"Error creating instance {i+1}/{count}: {e}")
    
    async def _select_instance(self) -> Optional[InstanceEndpoint]:
        """Select the best available instance for generation."""
        await self._periodic_health_check()
        
        if not self.active_instances:
            logger.error("No healthy instances available")
            return None
        
        # Simple round-robin selection for now
        # Could be enhanced with load balancing metrics
        return self.active_instances[0]
    
    async def generate_image(self, 
                           prompt: str,
                           negative_prompt: Optional[str] = None,
                           depth_map_data: Optional[bytes] = None,
                           **kwargs) -> Optional[GeneratedImage]:
        """
        Generate an image using a VastAI instance.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (optional)
            depth_map_data: Depth map as bytes (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            GeneratedImage if successful, None otherwise
        """
        start_time = time.time()
        
        # Track queue size for auto-scaling
        self.generation_queue_size += 1
        
        try:
            # Select an instance
            instance = await self._select_instance()
            if not instance:
                logger.error("No available instances for generation")
                return None
            
            # Prepare the request payload
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model": kwargs.get("model", "hyper"),
                "era": kwargs.get("era"),
                "steps": kwargs.get("steps", 6),
                "cfg": kwargs.get("cfg", 2.0),
                "seed": kwargs.get("seed"),
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024),
                "lora_strength": kwargs.get("lora_strength", 1.0),
                "controlnet_strength": kwargs.get("controlnet_strength", 0.8),
                "scheduler": kwargs.get("scheduler", "auto"),
                "use_karras_sigmas": kwargs.get("use_karras_sigmas"),
                "mock_depth": depth_map_data is None
            }
            
            # Add depth map if provided
            if depth_map_data:
                depth_map_b64 = base64.b64encode(depth_map_data).decode()
                payload["depth_map_b64"] = depth_map_b64
            
            logger.info(f"Generating image on {instance.url} with prompt: {prompt[:50]}...")
            
            # Make the generation request
            if not self.session:
                raise Exception("HTTP session not initialized")
                
            async with self.session.post(
                f"{instance.url}/generate",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Generation failed with status {response.status}: {error_text}")
                    return None
                
                result = await response.json()
                
                # Extract image data
                if "image_b64" not in result:
                    logger.error("No image data in response")
                    return None
                
                image_data = base64.b64decode(result["image_b64"])
                generation_time = time.time() - start_time
                
                # Create GeneratedImage
                generated_image = GeneratedImage(
                    image_data=image_data,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=result.get("seed_used"),
                    model=payload["model"],
                    generation_time=result.get("generation_time", generation_time),
                    metadata={
                        "instance_url": instance.url,
                        "instance_id": instance.instance_id,
                        "total_time": generation_time,
                        **result.get("metadata", {})
                    }
                )
                
                logger.info(f"Image generated successfully in {generation_time:.2f}s on {instance.url}")
                return generated_image
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
            
        finally:
            self.generation_queue_size = max(0, self.generation_queue_size - 1)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the VastAI generator."""
        await self._periodic_health_check()
        
        return {
            "generator_type": "vastai",
            "active_instances": len(self.active_instances),
            "unhealthy_instances": len(self.unhealthy_instances),
            "max_instances": self.max_instances,
            "min_instances": self.min_instances,
            "queue_size": self.generation_queue_size,
            "auto_scale": self.auto_scale,
            "instances": [
                {
                    "url": inst.url,
                    "instance_id": inst.instance_id,
                    "status": inst.status
                }
                for inst in self.active_instances
            ]
        }
    
    def supports_controlnet(self) -> bool:
        """Check if this generator supports ControlNet."""
        return True
    
    def supports_lora(self) -> bool:
        """Check if this generator supports LoRA."""
        return True
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return ["lightning", "hyper", "base"]
    
    def get_supported_eras(self) -> List[str]:
        """Get list of supported eras."""
        return ["drone", "experimance"]


# Configuration helper
def create_vastai_generator(config: Dict[str, Any]) -> VastAIImageGenerator:
    """
    Create a VastAI generator with default configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured VastAI generator
    """
    default_config = {
        "max_instances": 3,
        "min_instances": 1,
        "instance_timeout": 300,
        "health_check_interval": 60,
        "auto_scale": True
    }
    
    # Merge with provided config
    merged_config = {**default_config, **config}
    
    return VastAIImageGenerator(merged_config)
