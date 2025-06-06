#!/usr/bin/env python3
"""
Image Server Service for the Experimance project.

This service subscribes to RenderRequest messages from the events channel
and publishes ImageReady messages to the images channel using ZeroMQ.
It supports multiple image generation strategies including mock, local SDXL,
FAL.AI, and OpenAI DALL-E.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from experimance_common.zmq.pubsub import ZmqPublisherSubscriberService
from experimance_common.zmq.zmq_utils import MessageType
from experimance_common.constants import DEFAULT_PORTS
from experimance_common.config import load_config_with_overrides

from .config import ImageServerConfig
from .generators import create_generator, ImageGenerator

logger = logging.getLogger(__name__)


class ImageServerService(ZmqPublisherSubscriberService):
    """Main image server service that handles render requests and publishes generated images.
    
    This service follows the Experimance star topology pattern:
    - Subscribes to the events channel for RenderRequest messages
    - Publishes ImageReady messages to the images channel
    - Supports multiple image generation strategies
    """
    
    def __init__(
        self,
        service_name: str = "image-server",
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
        args: Optional[Any] = None
    ):
        """Initialize the Image Server Service.
        
        Args:
            service_name: Name of this service instance
            config: Configuration dictionary (overrides config_file)
            config_file: Path to TOML configuration file
            args: Command line arguments parsed by argparse
        """
        # Load and validate configuration using Pydantic
        self.config = self._load_config(service_name, config, config_file, args)
        
        # Create cache directory
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the base service
        super().__init__(
            service_name=self.config.service_name,
            service_type="image-server",
            pub_address=self.config.zmq.images_pub_address,
            sub_address=self.config.zmq.events_sub_address,
            topics=[MessageType.RENDER_REQUEST]
        )
        
        # For compatibility with tests
        self._default_strategy = self.config.generator.default_strategy
        
        # Generator instances (lazy-loaded)
        self._generators: Dict[str, ImageGenerator] = {}
        
        logger.info(f"ImageServerService initialized with strategy: {self.config.generator.default_strategy}")
    
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
        # Register message handlers
        self.register_handler(MessageType.RENDER_REQUEST, self._handle_render_request)
        
        # Start the base service
        await super().start()
        
        logger.info(f"ImageServerService started, listening for {MessageType.RENDER_REQUEST} messages")
    
    async def _handle_render_request(self, message: Dict[str, Any]):
        """Handle incoming RenderRequest messages.
        
        Args:
            message: The RenderRequest message containing generation parameters
        """
        try:
            # Validate required fields
            if not self._validate_render_request(message):
                logger.error(f"Invalid RenderRequest message: {message}")
                return
            
            request_id = message["request_id"]
            era = message["era"]
            biome = message["biome"]
            prompt = message["prompt"]
            depth_map_b64 = message.get("depth_map_png")
            
            logger.info(f"Processing RenderRequest {request_id} for era={era}, biome={biome}")
            
            # Generate image asynchronously
            asyncio.create_task(self._process_render_request(
                request_id, era, biome, prompt, depth_map_b64
            ))
            
        except Exception as e:
            logger.error(f"Error handling RenderRequest: {e}", exc_info=True)
    
    def _validate_render_request(self, message: Dict[str, Any]) -> bool:
        """Validate a RenderRequest message.
        
        Args:
            message: The message to validate
            
        Returns:
            True if message is valid, False otherwise
        """
        required_fields = ["request_id", "era", "biome", "prompt"]
        
        for field in required_fields:
            if field not in message or not message[field]:
                logger.error(f"Missing or empty required field: {field}")
                return False
        
        # Validate message type
        if message.get("type") != MessageType.RENDER_REQUEST:
            logger.error(f"Invalid message type: {message.get('type')}")
            return False
        
        return True
    
    async def _process_render_request(
        self,
        request_id: str,
        era: str,
        biome: str,
        prompt: str,
        depth_map_b64: Optional[str] = None
    ):
        """Process a render request and generate an image.
        
        Args:
            request_id: Unique identifier for this request
            era: Era context for the image
            biome: Biome context for the image
            prompt: Text prompt for image generation
            depth_map_b64: Optional base64-encoded depth map
        """
        try:
            # Generate the image
            image_path = await self._generate_image(
                prompt=prompt,
                depth_map_b64=depth_map_b64,
                era=era,
                biome=biome
            )
            
            # Publish ImageReady message
            await self._publish_image_ready(request_id, image_path)
            
            logger.info(f"Successfully processed RenderRequest {request_id}")
            
        except Exception as e:
            logger.error(f"Error processing RenderRequest {request_id}: {e}", exc_info=True)
            
            # Optionally publish an error message
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
        """
        if strategy is None:
            strategy = self.config.generator.default_strategy
        
        # Get or create generator instance
        generator = self._get_generator(strategy)
        
        # Generate the image with timeout
        try:
            image_path = await asyncio.wait_for(
                generator.generate_image(prompt, depth_map_b64, **kwargs),
                timeout=self.config.generator.timeout_seconds
            )
            return image_path
        except asyncio.TimeoutError:
            raise RuntimeError(f"Image generation timed out after {self.config.generator.timeout_seconds} seconds")
    
    def _get_generator(self, strategy: Optional[str] = None) -> ImageGenerator:
        """Get or create an image generator instance.
        
        Args:
            strategy: Generator strategy to use
            
        Returns:
            ImageGenerator instance
        """
        if strategy is None:
            strategy = self.config.generator.default_strategy
        
        # Create generator if not cached
        if strategy not in self._generators:
            generator_config = self.config.get_generator_config(strategy)
            self._generators[strategy] = create_generator(strategy, **generator_config)
        
        return self._generators[strategy]
    
    async def _publish_image_ready(self, request_id: str, image_path: str):
        """Publish an ImageReady message.
        
        Args:
            request_id: The original request ID
            image_path: Path to the generated image
        """
        # Convert path to URI format
        image_uri = f"file://{Path(image_path).absolute()}"
        
        # Generate unique image ID
        image_id = str(uuid.uuid4())
        
        # Create ImageReady message
        message = {
            "type": MessageType.IMAGE_READY,
            "request_id": request_id,
            "image_id": image_id,
            "uri": image_uri
        }
        
        # Publish the message
        success = await self.publish_message(message)
        
        if success:
            logger.info(f"Published ImageReady message for request {request_id}")
        else:
            logger.error(f"Failed to publish ImageReady message for request {request_id}")
    
    async def _publish_image_error(self, request_id: str, error_message: str):
        """Publish an error message for a failed request.
        
        Args:
            request_id: The original request ID
            error_message: Description of the error
        """
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
            logger.error(f"Failed to publish error message for request {request_id}")
    
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
            logger.error(f"Error during cache cleanup: {e}", exc_info=True)
    
    async def stop(self):
        """Stop the image server service."""
        logger.info("Stopping ImageServerService...")
        
        # Clean up generators
        self._generators.clear()
        
        # Stop the base service
        await super().stop()
        
        logger.info("ImageServerService stopped")


async def main():
    """Main entry point for running the image server service."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Experimance Image Server Service")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (TOML format)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="image-server",
        help="Service instance name"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory to store generated images"
    )
    parser.add_argument(
        "--max-cache-size-gb",
        type=float,
        help="Maximum cache size in GB"
    )
    parser.add_argument(
        "--generator.default_strategy",
        type=str,
        choices=["mock", "sdxl", "falai", "openai"],
        help="Default image generation strategy"
    )
    parser.add_argument(
        "--generator.timeout_seconds",
        type=int,
        help="Timeout for image generation in seconds"
    )
    parser.add_argument(
        "--zmq.events_sub_address",
        type=str,
        help="ZMQ address for subscribing to events"
    )
    parser.add_argument(
        "--zmq.images_pub_address",
        type=str,
        help="ZMQ address for publishing images"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and start the service
    service = ImageServerService(
        service_name=args.name,
        config_file=args.config,
        args=args
    )

    await service.start()
    logger.info(f"Image server service '{args.name}' started successfully")
    
    # Run the service
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
