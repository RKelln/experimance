#!/usr/bin/env python3
"""
Image Server Service for the Experimance project.

This service subscribes to RenderRequest messages from the events channel
and publishes ImageReady messages to the images channel using ZeroMQ.
It supports multiple image generation strategies including mock, local SDXL,
FAL.AI, and OpenAI DALL-E.
"""

import argparse
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from experimance_common.zmq.worker import ZmqWorkerService
from experimance_common.zmq.zmq_utils import MessageType
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.config import load_config_with_overrides
from experimance_common.logger import configure_external_loggers

from .config import ImageServerConfig
from .generators.factory import create_generator_from_config
from .generators.generator import ImageGenerator

from image_server.generators.mock.mock_generator import MockImageGenerator
from image_server.generators.fal.fal_comfy_generator import FalComfyGenerator
from image_server.generators.fal.fal_comfy_config import FalComfyGeneratorConfig, FalGeneratorConfig

# Future generator implementations:
# from image_server.generators.mock.mock_generator import MockGenerator, MockGeneratorConfig
# from image_server.generators.openai.openai_generator import OpenAIGenerator, OpenAIGeneratorConfig
# from image_server.generators.sdxl.sdxl_generator import SDXLGenerator, SDXLGeneratorConfig


logger = logging.getLogger(__name__)

class ImageServerService(ZmqWorkerService):
    """Main image server service that handles render requests and publishes generated images.
    
    This service uses the worker pattern:
    - Pulls RenderRequest messages from Core service 
    - Pushes ImageReady messages back to Core service
    - Supports multiple image generation strategies
    """
    image_generator: ImageGenerator
    config: ImageServerConfig
    
    def __init__(
        self,
        config: ImageServerConfig
    ):
        """Initialize the Image Server Service.
        
        Args:
            config: Service configuration object
            service_name: Name of this service instance
            
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            self.config = config

            # Create cache directory
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            
            configure_external_loggers(logging.WARNING)

            # Initialize the base service using worker pattern
            super().__init__(
                service_name=self.config.service_name,
                sub_address=self.config.zmq.events_sub_address,  # For control messages
                pull_address=f"tcp://localhost:{DEFAULT_PORTS['images']}",  # Pull RENDER_REQUEST from Core
                push_address=f"tcp://*:{DEFAULT_PORTS['image_results']}",  # Push IMAGE_READY back to Core
                topics=[],  # No specific subscription topics for now
                service_type="image-server"
            )
            
            # For compatibility with tests
            self._default_strategy = self.config.generator.default_strategy
            
            # Create generator directly from the configuration
            strategy = config.generator.default_strategy
            strategy_config = {}
            
            # Get strategy-specific config if available
            if hasattr(config, strategy):
                strategy_config_obj = getattr(config, strategy)
                if strategy_config_obj:
                    if hasattr(strategy_config_obj, "model_dump"):
                        strategy_config = strategy_config_obj.model_dump()
                    else:
                        strategy_config = dict(strategy_config_obj)
            
            # Create the generator
            self.generator = create_generator_from_config(
                strategy=strategy,
                config_data=strategy_config,
                cache_dir=config.cache_dir,
                timeout=getattr(config.generator, "timeout", 60)
            )
            
            logger.info(f"ImageServerService initialized with strategy: {self.config.generator.default_strategy}")
        except Exception as e:
            error_msg = f"Failed to initialize ImageServerService: {e}"
            logger.error(error_msg, exc_info=True)
            # We can't record an error here yet because super() hasn't been initialized
            raise RuntimeError(error_msg) from e
    
    def _load_config(self, service_name: str, config: Optional[Dict[str, Any]], 
                   config_file: Optional[str], args: Optional[Any]) -> ImageServerConfig:
        """Load and validate configuration using Pydantic.
        
        Args:
            service_name: Service name to use if not overridden
            config: Configuration dictionary (highest priority)
            config_file: Path to TOML configuration file
            args: Command line arguments as an argparse.Namespace
            
        Returns:
            Validated ImageServerConfig instance
        """
        # Create default config with service_name
        default_config = {"service_name": service_name}
        
        # Load and validate with Pydantic
        return ImageServerConfig.from_overrides(
            override_config=config,
            config_file=config_file,
            default_config=default_config,
            args=args
        )

    async def start(self):
        """Start the image server service."""
        try:
            # Register message handlers
            self.register_handler(MessageType.RENDER_REQUEST, self._handle_render_request)
            
            # Register periodic cache cleanup task (run every 10 minutes)
            self.add_task(self._periodic_cache_cleanup(interval=600))
            
            # Start the base service
            await super().start()
            
            logger.info(f"ImageServerService started, listening for {MessageType.RENDER_REQUEST} messages")
        except Exception as e:
            self.record_error(e, is_fatal=True, custom_message=f"Failed to start ImageServerService: {e}")
            raise
    
    async def _handle_render_request(self, message: Dict[str, Any]):
        """Handle incoming RenderRequest messages.
        
        Args:
            message: The RenderRequest message containing generation parameters
        """
        try:
            logger.debug(f"Received RenderRequest message: {message}")

            # Validate required fields
            if not self._validate_render_request(message):
                # Record validation error but continue service operation
                self.record_error(
                    ValueError(f"Invalid RenderRequest message: {message}"), 
                    is_fatal=False,
                    custom_message=f"Invalid RenderRequest message: {message}"
                )
                return
            
            request_id = message["request_id"]
            prompt = message["prompt"]
            depth_map_b64 = message.get("depth_map_png")
            
            # Extract era and biome for context-aware generation
            era = message.get("era")
            biome = message.get("biome")
            seed = message.get("seed")
            negative_prompt = message.get("negative_prompt")
            style = message.get("style")
            
            logger.info(f"Processing RenderRequest {request_id} for {era}/{biome}")
            
            # Create and properly register image generation task with full context
            task = self._process_render_request(
                request_id=request_id, 
                prompt=prompt, 
                depth_map_b64=depth_map_b64,
                era=era,
                biome=biome,
                seed=seed,
                negative_prompt=negative_prompt,
                style=style
            )
            self.add_task(task)
            
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message=f"Error handling RenderRequest: {e}")
    
    def _validate_render_request(self, message: Dict[str, Any]) -> bool:
        """Validate a RenderRequest message.
        
        Args:
            message: The message to validate
            
        Returns:
            True if message is valid, False otherwise
        """
        required_fields = ["request_id", "prompt"]
        
        for field in required_fields:
            if field not in message or not message[field]:
                # Note: We don't log here since the caller will use record_error
                return False
        
        # Validate message type
        if message.get("type") != MessageType.RENDER_REQUEST:
            # Note: We don't log here since the caller will use record_error
            return False
        
        return True
    
    async def _process_render_request(
        self,
        request_id: str,
        prompt: str,
        depth_map_b64: Optional[str] = None,
        era: Optional[str] = None,
        biome: Optional[str] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None
    ):
        """Process a render request and generate an image.
        
        Args:
            request_id: Unique identifier for this request
            prompt: Text prompt for image generation
            depth_map_b64: Optional base64-encoded depth map
            era: Era context for generation
            biome: Biome context for generation
            seed: Random seed for generation
            negative_prompt: Negative prompt for generation
            style: Style hint for generation
        """
        try:
            # Generate the image with full context
            image_path = await self._generate_image(
                prompt=prompt,
                depth_map_b64=depth_map_b64,
                era=era,
                biome=biome,
                seed=seed,
                negative_prompt=negative_prompt,
                style=style
            )
            
            # Publish ImageReady message with context
            await self._publish_image_ready(
                request_id=request_id, 
                image_path=image_path,
                prompt=prompt,
                era=era,
                biome=biome
            )
            
            logger.info(f"Successfully processed RenderRequest {request_id}")
            
        except Exception as e:
            # Record error but don't stop service (non-fatal)
            self.record_error(e, is_fatal=False, custom_message=f"Error processing RenderRequest {request_id}: {e}")
            
            # Publish an error message to notify other services
            await self._publish_image_error(request_id, str(e))
    
    async def _generate_image(
        self,
        prompt: str,
        depth_map_b64: Optional[str] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate an image using the specified strategy.
        
        Args:
            prompt: Text prompt for generation
            depth_map_b64: Optional base64-encoded depth map
            strategy: Generator strategy to use (defaults to configured strategy)
            **kwargs: Additional parameters for the generator
            
        Returns:
            Path to the generated image file
            
        Raises:
            RuntimeError: If image generation fails or times out
        """
        if strategy is None:
            strategy = self.config.generator.default_strategy
        
        # Generate the image with timeout
        try:
            image_path = await asyncio.wait_for(
                self.generator.generate_image(prompt, depth_map_b64, **kwargs),
                timeout=self.config.generator.timeout
            )
            return image_path
        except asyncio.TimeoutError:
            error_msg = f"Image generation timed out after {self.config.generator.timeout} seconds"
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Image generation failed: {e}"
            raise RuntimeError(error_msg) from e
    
    
    async def _publish_image_ready(
        self, 
        request_id: str, 
        image_path: str,
        prompt: Optional[str] = None,
        era: Optional[str] = None,
        biome: Optional[str] = None
    ):
        """Publish an ImageReady message.
        
        Args:
            request_id: The original request ID
            image_path: Path to the generated image
            prompt: The prompt used for generation
            era: Era context for the image
            biome: Biome context for the image
            
        Raises:
            RuntimeError: If publishing the message fails
        """
        try:
            # Convert path to URI format
            image_uri = f"file://{Path(image_path).absolute()}"
            
            # Generate unique image ID
            image_id = str(uuid.uuid4())
            
            # Create ImageReady message with context
            message = {
                "type": MessageType.IMAGE_READY,
                "request_id": request_id,
                "image_id": image_id,
                "uri": image_uri
            }
            
            # Add context fields if provided
            if prompt:
                message["prompt"] = prompt
            if era:
                message["era"] = era
            if biome:
                message["biome"] = biome
            
            # If configured for remote deployment, include base64 data
            if getattr(self.config, 'include_image_data', False):
                try:
                    import base64
                    with open(image_path, 'rb') as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                        message["image_data"] = image_data
                        
                    # Get image dimensions for metadata
                    from PIL import Image
                    with Image.open(image_path) as img:
                        message["width"] = img.width
                        message["height"] = img.height
                        message["format"] = img.format.lower() if img.format else "png"
                except Exception as e:
                    logger.warning(f"Failed to encode image data for {image_path}: {e}")
                    # Continue with URI-only message
            
            # Publish the message using push socket
            success = await self.send_response(message)
            
            if success:
                logger.info(f"Published ImageReady message for request {request_id}")
            else:
                error_msg = f"Failed to publish ImageReady message for request {request_id}"
                # Record non-fatal error but continue service operation
                self.record_error(RuntimeError(error_msg), is_fatal=False, custom_message=error_msg)
        except Exception as e:
            error_msg = f"Error publishing ImageReady message for request {request_id}: {e}"
            # Record non-fatal error but continue service operation
            self.record_error(e, is_fatal=False, custom_message=error_msg)
            raise RuntimeError(error_msg) from e
    
    async def _publish_image_error(self, request_id: str, error_message: str):
        """Publish an error message for a failed request.
        
        Args:
            request_id: The original request ID
            error_message: Description of the error
        """
        try:
            # Create error message (using ALERT type)
            message = {
                "type": MessageType.ALERT,
                "request_id": request_id,
                "severity": "error",
                "message": f"Image generation failed: {error_message}"
            }
            
            # Publish the message
            success = await self.publish_message(message)
            
            if success:
                logger.info(f"Published error message for request {request_id}")
            else:
                log_msg = f"Failed to publish error message for request {request_id}"
                logger.error(log_msg)
                # We don't record this as an error since it's already handling another error
        except Exception as e:
            # Just log this error since we're already in an error handling path
            logger.error(f"Error publishing error message for request {request_id}: {e}", exc_info=True)
    
    async def _cleanup_cache(self):
        """Clean up old cached images if cache size exceeds limit."""
        try:
            # Get cache directory size
            total_size = sum(f.stat().st_size for f in self.config.cache_dir.rglob('*') if f.is_file())
            max_size_bytes = self.config.max_cache_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
            
            if total_size > max_size_bytes:
                logger.info(f"Cache size ({total_size / 1024 / 1024 / 1024:.2f} GB) exceeds limit "
                          f"({self.config.max_cache_size_gb} GB), cleaning up...")
                
                # Get all files sorted by modification time (oldest first)
                files = sorted(
                    [f for f in self.config.cache_dir.rglob('*') if f.is_file()],
                    key=lambda x: x.stat().st_mtime
                )
                
                # Remove files until we're under the limit
                for file in files:
                    file.unlink()
                    total_size -= file.stat().st_size
                    
                    if total_size <= max_size_bytes * 0.8:  # Keep 20% buffer
                        break
                
                logger.info(f"Cache cleanup completed, new size: {total_size / 1024 / 1024 / 1024:.2f} GB")
                
        except Exception as e:
            # Non-fatal error, record but don't stop service
            self.record_error(e, is_fatal=False, custom_message=f"Error during cache cleanup: {e}")
            
    async def _periodic_cache_cleanup(self, interval: float = 600):
        """Run cache cleanup periodically.
        
        Args:
            interval: Time between cleanups in seconds (default: 10 minutes)
        """
        try:
            while self.running:
                await self._cleanup_cache()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("Periodic cache cleanup task cancelled")
            raise
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message=f"Error in periodic cache cleanup: {e}")
            raise
    
    async def stop(self):
        """Stop the image server service."""
        logger.info("Stopping ImageServerService...")
        
        try:
            # Clean up generators
            if hasattr(self, 'generator'):
                await self.generator.stop()
            
            # Stop the base service
            await super().stop()
            
            logger.info("ImageServerService stopped")
        except Exception as e:
            self.record_error(e, is_fatal=True, custom_message=f"Error during ImageServerService shutdown: {e}")
            raise  # Re-raise to ensure calling code knows there was an issue


async def run_image_server_service(
    config_path: str = "config.toml", 
    args: Optional[argparse.Namespace] = None
) -> None:
    """
    Run the Experimance Imager Server Service with CLI integration.
    
    Args:
        config_path: Path to configuration file
        args: CLI arguments from argparse (for config overrides)
    """
    # Create config with CLI overrides
    config = ImageServerConfig.from_overrides(
        config_file=config_path,
        args=args  # CLI args automatically override config values
    )
    
    service = ImageServerService(
        config=config
    )
    await service.start()
    await service.run()
