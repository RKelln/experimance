#!/usr/bin/env python3
"""
Image Server Service for the Experimance project.

This service receives RenderRequest messages via ZMQ worker pattern and 
publishes ImageReady messages. It supports multiple image generation 
strategies including mock, local SDXL, FAL.AI, and OpenAI DALL-E.

Uses the new composition-based ZMQ architecture with BaseService + WorkerService.
"""

import argparse
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Union

from experimance_common.base_service import BaseService
from experimance_common.image_utils import extract_image_as_base64
from experimance_common.schemas import ImageReady, RenderRequest, MessageType
from experimance_common.zmq.services import WorkerService
from experimance_common.zmq.config import MessageDataType
from experimance_common.constants import DEFAULT_PORTS, GENERATED_IMAGES_DIR
from experimance_common.logger import configure_external_loggers
from image_server.generators.config import GENERATOR_NAMES
from pydantic import ValidationError
from typing import get_args

from .config import ImageServerConfig
from .generators.factory import GeneratorManager

from experimance_common.logger import setup_logging

SERVICE_TYPE = "image_server"

logger = setup_logging(__name__, log_filename=f"{SERVICE_TYPE}.log")

class ImageServerService(BaseService):
    """Main image server service that handles render requests and publishes generated images.
    
    This service uses the new composition-based ZMQ architecture:
    - BaseService for lifecycle management and error handling
    - WorkerService for ZMQ worker pattern (PULL/PUSH + PUB/SUB)
    - Supports multiple image generation strategies
    """
    
    def __init__(self, config: ImageServerConfig):
        """Initialize the Image Server Service.
        
        Args:
            config: Service configuration object
            
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            super().__init__(service_name=config.service_name, service_type=SERVICE_TYPE)
            self.config = config

            # Create cache directory
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

            # Initialize ZMQ worker service using composition
            self.zmq_service = WorkerService(self.config.zmq)
            
            # Collect strategy-specific configurations for the generator manager
            default_configs = {}
            for strategy in get_args(GENERATOR_NAMES):
                if hasattr(config, strategy):
                    strategy_config_obj = getattr(config, strategy)
                    if strategy_config_obj:
                        if hasattr(strategy_config_obj, "model_dump"):
                            default_configs[strategy] = strategy_config_obj.model_dump()
                        else:
                            default_configs[strategy] = dict(strategy_config_obj)
            
            # Initialize generator manager for dynamic generator creation
            self.generator_manager = GeneratorManager(
                default_strategy=config.generator.strategy,
                cache_dir=config.cache_dir,
                timeout=getattr(config.generator, "timeout", 120),
                default_configs=default_configs
            )
            
            logger.info(f"ImageServerService initialized with default strategy: {self.config.generator.strategy}")
        except Exception as e:
            error_msg = f"Failed to initialize ImageServerService: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    async def start(self):
        """Start the image server service."""
        try:
            # Set up message handlers for the ZMQ service
            self.zmq_service.set_work_handler(self._handle_render_request)
            
            # FIXME: this is for subscriber, but image requests come through PULL socket
            #self.zmq_service.add_message_handler(str(MessageType.RENDER_REQUEST), self._handle_topic_render_request)
            
            # Start the ZMQ service
            await self.zmq_service.start()

            # Start any generators that need to be pre-warmed
            await self.generator_manager.start()

            # Register periodic cache cleanup task (run every 10 minutes)
            self.add_task(self._periodic_cache_cleanup(600))
            
            # Call parent start (this will set state and handle lifecycle)
            await super().start()
            
            logger.info(f"ImageServerService started, listening for {MessageType.RENDER_REQUEST} messages")
        except Exception as e:
            self.record_error(e, is_fatal=True, custom_message=f"Failed to start ImageServerService: {e}")
            raise

    async def stop(self):
        """Stop the image server service."""
        try:
            # Stop all generators
            if hasattr(self, 'generator_manager'):
                await self.generator_manager.stop_all_generators()
            
            # Stop the ZMQ service
            if hasattr(self, 'zmq_service'):
                await self.zmq_service.stop()
            
            # Call parent stop
            await super().stop()
            
            logger.info("ImageServerService stopped")
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message=f"Error stopping ImageServerService: {e}")

    def _handle_topic_render_request(self, topic: str, message_data: MessageDataType):
        """Handle RenderRequest messages received via topic subscription.
        
        Args:
            topic: The topic the message was received on
            message_data: The message data from the subscription
        """
        # Convert MessageDataType to dict if needed
        if hasattr(message_data, 'model_dump') and callable(getattr(message_data, 'model_dump')):
            data_dict = message_data.model_dump()  # type: ignore
        else:
            data_dict = message_data
            
        # Ensure we have a dict
        if not isinstance(data_dict, dict):
            logger.error(f"Expected dict but got {type(data_dict)}: {data_dict}")
            return
            
        # Schedule the async handler
        asyncio.create_task(self._handle_render_request(data_dict))

    async def _handle_render_request(self, message: MessageDataType):
        """Handle incoming RenderRequest messages.
        
        Args:
            message: The RenderRequest message containing generation parameters
        """
        try:
            logger.debug(f"Received RenderRequest message: {message}")

            # Ensure message is a RenderRequest object using schemas.py
            try:
                request: RenderRequest = RenderRequest.to_message_type(message)  # type: ignore
            except ValidationError as e:
                self.record_error(
                    ValueError(f"Invalid RenderRequest message: {message}"),
                    is_fatal=False,
                    custom_message=f"Invalid RenderRequest message: {message}"
                )
                return

            logger.info(f"Processing RenderRequest {request.request_id}")

            # Schedule the image generation task using the full RenderRequest object
            logger.debug(f"Creating and scheduling task for request {request.request_id}")
            self.add_task(self._process_render_request(request))
            logger.debug(f"Task created and scheduled for request {request.request_id}")

        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message=f"Error handling RenderRequest: {e}")
    
    def _validate_render_request(self, message: RenderRequest) -> bool:
        """Validate a RenderRequest message.
        
        Args:
            message: The RenderRequest object to validate
            
        Returns:
            True if message is valid, False otherwise
        """
        # Check required fields exist and are not empty
        if not hasattr(message, 'request_id') or not message.request_id:
            return False
            
        if not hasattr(message, 'prompt') or not message.prompt:
            return False
        
        # Validate message type
        if not hasattr(message, 'type') or message.type != "RenderRequest":
            return False
        
        return True

    async def _process_render_request(
        self,
        request: RenderRequest,
    ):
        """Process a render request and generate an image.
        
        Args:
            request: RenderRequest object with generation parameters
        """
        try:
            logger.debug(f"Starting _process_render_request for {request.request_id}")

            # Unpack fields from RenderRequest (from schemas.py)
            request_id = request.request_id
            prompt = request.prompt
            
            # Handle optional fields that may be added by project-specific schemas
            era = getattr(request, 'era', None)
            biome = getattr(request, 'biome', None)
            
            # Get generator strategy from request or use default
            generator_strategy = getattr(request, 'generator', None)

            # Standardized image processing using utility functions
            # Extract depth map as base64 if present
            depth_map_b64 = None
            if hasattr(request, 'depth_map') and request.depth_map is not None:
                depth_map_b64 = extract_image_as_base64(request.depth_map, "depth_map")

            # Extract reference image as base64 if present
            reference_image_b64 = None
            if hasattr(request, 'reference_image') and request.reference_image is not None:
                reference_image_b64 = extract_image_as_base64(request.reference_image, "reference_image")

            # Log the request details
            strategy_info = f" using {generator_strategy}" if generator_strategy else ""
            context_info = f" for {era}/{biome}" if era and biome else ""
            reference_info = " (image-to-image)" if reference_image_b64 else ""
            depth_info = " (with depth map)" if depth_map_b64 else ""
            logger.info(f"Processing RenderRequest {request_id}{strategy_info}{context_info}{reference_info}{depth_info}")

            # Generate the image with full context
            logger.debug(f"Calling _generate_image for {request_id}")
            image_path = await self._generate_image(
                request_id=request_id,  # Include request_id for filename
                prompt=prompt,
                negative_prompt = getattr(request, 'negative_prompt', None),
                depth_map_b64=depth_map_b64,
                strategy=generator_strategy,
                reference_image_b64=reference_image_b64,  # Use standardized base64 format
                era=str(era) if era else None,
                biome=str(biome) if biome else None,
                seed=getattr(request, 'seed', None),
                style=getattr(request, 'style', None),
                width=request.get('width', None),
                height=request.get('height', None),
            )
            logger.debug(f"Generated image path: {image_path}")

            # Create symlink to latest generated image for easy access
            try:
                latest_filename = f"latest{Path(image_path).suffix}"
                latest_path = GENERATED_IMAGES_DIR / latest_filename
                if latest_path.exists() or latest_path.is_symlink():
                    latest_path.unlink()
                latest_path.symlink_to(Path(image_path).name)
                logger.debug(f"Created {latest_filename} symlink to {Path(image_path).name}")
            except Exception as e:
                logger.warning(f"Failed to create {latest_filename} symlink: {e}")

            # Publish ImageReady message with context (from schemas.py)
            logger.debug(f"Publishing ImageReady for {request_id}")
            await self._publish_image_ready(
                request_id=request_id,
                image_path=image_path,
                era=era,
                biome=biome,
                prompt=prompt
            )

            logger.info(f"Successfully processed RenderRequest {request_id}")

        except Exception as e:
            # Record error but don't stop service (non-fatal)
            request_id = getattr(request, 'request_id', None)
            self.record_error(e, is_fatal=False, custom_message=f"Error processing RenderRequest {request_id}: {e}")
            # Publish an error message to notify other services
            if request_id:
                await self._publish_image_error(request_id, str(e))
    
    async def _generate_image(
        self,
        prompt: str,
        depth_map_b64: Optional[str] = None,
        reference_image_b64: Optional[str] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate an image using the specified or default strategy.
        
        Args:
            prompt: Text prompt for generation
            depth_map_b64: Optional base64-encoded depth map
            strategy: Generator strategy to use (defaults to configured strategy)
            reference_image_b64: Optional base64-encoded reference image for image-to-image generation
            **kwargs: Additional parameters for the generator
            
        Returns:
            Path to the generated image file
            
        Raises:
            RuntimeError: If image generation fails or times out
        """
        if strategy is None:
            strategy = self.config.generator.strategy
        
        # Get the appropriate generator from the manager
        generator = self.generator_manager.get_generator(strategy)
        
        # Prepare generation parameters based on generator capabilities
        generation_kwargs = kwargs.copy()
        logger.debug(f"Using generator {generator.__class__.__name__} for strategy {strategy} with kwargs: {generation_kwargs}")
        
        generation_kwargs['depth_map_b64'] = depth_map_b64

        # Handle image-to-image generation
        if reference_image_b64 and self.generator_manager.is_image_to_image_generator(strategy):
            logger.info(f"Using image-to-image generation with strategy: {strategy}")
            generation_kwargs['image_b64'] = reference_image_b64
        elif reference_image_b64:
            logger.warning(f"Reference image provided but strategy {strategy} doesn't support image-to-image generation")
        
        error_msg = False
        # Generate the image with timeout
        try:
            image_path = await asyncio.wait_for(
                generator.generate_image(prompt, **generation_kwargs),
                timeout=self.config.generator.timeout
            )
            return image_path
        except asyncio.TimeoutError:
            error_msg = f"Image generation timed out after {self.config.generator.timeout} seconds"
            logger.warning(f"Image generation timeout: {error_msg}")
        except Exception as e:
            error_msg = f"Image generation failed: {e}"
            logger.warning(f"Image generation exception: {error_msg}")
        finally:
            if error_msg:
                mock_generator = self.generator_manager.get_generator("mock")
                if mock_generator:
                    # Fallback to mock generator if available
                    logger.warning(f"Timeout occurred, falling back to mock generator")
                    generation_kwargs['immediate'] = True  # Use immediate mode for mock
                    return await mock_generator.generate_image(prompt, **generation_kwargs)
                else:
                    logger.error("No mock generator available for fallback")
                raise RuntimeError(error_msg)
    
    async def _publish_image_ready(
        self, 
        request_id: str, 
        image_path: str,
        era: Optional[Any] = None,
        biome: Optional[Any] = None,
        prompt: Optional[str] = None,
    ):
        """Publish an ImageReady message.
        
        Args:
            request_id: The original request ID
            image_path: Path to the generated image
            prompt: The prompt used for generation
            era: Era context for the image (project-specific)
            biome: Biome context for the image (project-specific)
            
        Raises:
            RuntimeError: If publishing the message fails
        """
        try:
            # Convert path to URI format
            image_uri = f"file://{Path(image_path).absolute()}"
            
            # Generate unique image ID
            image_id = str(uuid.uuid4())
            
            # Create base message
            message_data = {
                "request_id": request_id,
                "uri": image_uri,
                "prompt": prompt,
            }
            
            # Add era and biome if they exist (project-specific fields)
            if era is not None:
                message_data["era"] = era
            if biome is not None:
                message_data["biome"] = biome
            
            message = ImageReady(**message_data)
            
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
            
            logger.debug(f"Publish ImageReady for request {request_id}")

            # Publish the message using push socket
            await self.zmq_service.send_response(message)            
            logger.info(f"Published ImageReady message for request {request_id}")
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
            
            # Publish the message via the publisher component
            await self.zmq_service.publish(message, str(MessageType.ALERT))            
            logger.info(f"Published error message for request {request_id}")
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
