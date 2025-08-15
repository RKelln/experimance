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
from typing import Optional, Dict, Any, Set
from PIL import Image, ImageDraw, ImageFont

from image_server.generators.config import BaseGeneratorConfig
from experimance_common.logger import configure_external_loggers

# Configure logging
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp']

class GeneratorCapabilities:
    """Defines the capabilities that image generators can support."""
    
    # Core generation capabilities
    IMAGE_TO_IMAGE = "image_to_image"      # img2img generation with reference images
    CONTROLNET = "controlnet"              # ControlNet conditioning (depth maps, etc.)
    LORAS = "loras"                        # LoRA model loading and application
    INPAINTING = "inpainting"              # Inpainting/outpainting
    UPSCALING = "upscaling"                # Image upscaling
    STYLE_TRANSFER = "style_transfer"       # Style transfer between images
    
    # Advanced features
    BATCH_GENERATION = "batch_generation"   # Generate multiple images at once
    CUSTOM_SCHEDULERS = "custom_schedulers" # Support for custom sampling schedulers
    NEGATIVE_PROMPTS = "negative_prompts"   # Negative prompt support
    SEEDS = "seeds"                        # Deterministic generation with seeds
    
    @classmethod
    def all_capabilities(cls) -> Set[str]:
        """Get all defined capabilities."""
        return {
            value for name, value in cls.__dict__.items() 
            if isinstance(value, str) and not name.startswith('_')
        }

class ImageGenerator(ABC):
    """Abstract base class for image generation strategies."""
    
    # Generator capabilities - subclasses should override this
    # Use a set of GeneratorCapabilities constants
    supported_capabilities: Set[str] = set()
    
    def __init__(self, config: BaseGeneratorConfig, output_dir: str = "/tmp", max_concurrent: int = 1, **kwargs):
        """Initialize the image generator.
        
        Args:
            output_dir: Directory to save generated images
            max_concurrent: Maximum number of concurrent generations (default 1 for thread-safety)
            **kwargs: Additional configuration options
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Initialize queuing mechanism for thread-safe generation
        self._generation_queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_task = None
        self._is_running = False
        self._pending_requests = {}  # Track pending requests for cancellation
        
        self._configure(config, **kwargs)
    
    def _configure(self, config, **kwargs):
        """Configure generator-specific settings.
        
        Subclasses can override this to handle their specific configuration.
        """
        pass

    def supports_capability(self, capability: str) -> bool:
        """Check if this generator supports a specific capability.
        
        Args:
            capability: Capability to check (use GeneratorCapabilities constants)
            
        Returns:
            True if the generator supports the capability
        """
        # Check new capabilities system
        if hasattr(self.__class__, 'supported_capabilities'):
            if capability in self.supported_capabilities:
                return True
        
        return False
    
    def get_supported_capabilities(self) -> Set[str]:
        """Get all capabilities supported by this generator.
        
        Returns:
            Set of supported capability strings
        """
        capabilities = set(self.supported_capabilities)
        
        return capabilities
    
    @classmethod
    def supports_capability_class(cls, capability: str) -> bool:
        """Class method to check capability without instantiating.
        
        Args:
            capability: Capability to check (use GeneratorCapabilities constants)
            
        Returns:
            True if the generator class supports the capability
        """
        # Check new capabilities system
        if hasattr(cls, 'supported_capabilities'):
            if capability in cls.supported_capabilities:
                return True
        
        return False

    async def _process_generation_queue(self):
        """Process generation requests from the queue."""
        while self._is_running:
            try:
                # Get next request from queue
                request_data = await self._generation_queue.get()
                
                if request_data is None:  # Shutdown signal
                    break
                
                request_id, prompt, kwargs, future = request_data
                
                # Check if request was cancelled
                if request_id not in self._pending_requests:
                    self._generation_queue.task_done()
                    continue
                
                # Process request with semaphore for concurrency control
                async with self._semaphore:
                    try:
                        if not future.cancelled():
                            result = await self._generate_image_impl(prompt, **kwargs)
                            future.set_result(result)
                    except Exception as e:
                        if not future.cancelled():
                            future.set_exception(e)
                    finally:
                        # Clean up pending request
                        self._pending_requests.pop(request_id, None)
                        self._generation_queue.task_done()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in generation queue processor: {e}")

    async def generate_image(self, prompt: str, **kwargs) -> str:
        """Generate an image using the queue system for thread safety.
        
        This method queues the request and returns when generation is complete.
        """
        if not self._is_running:
            await self.start()
        
        # Create unique request ID and future for this request
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        # Add to pending requests for tracking
        self._pending_requests[request_id] = future
        
        # Queue the request
        await self._generation_queue.put((request_id, prompt, kwargs, future))
        
        # Wait for completion
        try:
            return await future
        except asyncio.CancelledError:
            # Clean up if cancelled
            self._pending_requests.pop(request_id, None)
            raise

    @abstractmethod
    async def _generate_image_impl(self, prompt: str, **kwargs) -> str:
        """Internal implementation of image generation.
        
        This is the method that subclasses should implement instead of generate_image.
        
        Args:
            prompt: Text description of the image to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Path to the generated image file
            
        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails
        """
        pass

    async def start(self):
        """Start the generator and queue processing.
        
        This method can be overridden by subclasses to implement pre-warming logic.
        """
        if not self._is_running:
            self._is_running = True
            self._queue_task = asyncio.create_task(self._process_generation_queue())
            logger.debug(f"{self.__class__.__name__}: Queue processor started")

    async def restart_queue_processor(self):
        """Restart the queue processor after recovery.
        
        This is useful when a generator needs to reset its processing state
        after recovering from errors without a full stop/start cycle.
        """
        if self._is_running and self._queue_task and not self._queue_task.done():
            logger.info(f"{self.__class__.__name__}: Restarting queue processor after recovery")
            
            # Stop current processor
            await self._generation_queue.put(None)  # Signal to stop
            try:
                await asyncio.wait_for(self._queue_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(f"{self.__class__.__name__}: Queue processor didn't stop cleanly, cancelling")
                self._queue_task.cancel()
            
            # Start new processor
            self._queue_task = asyncio.create_task(self._process_generation_queue())
            logger.debug(f"{self.__class__.__name__}: Queue processor restarted")

    async def stop(self):
        """Stop the generator and queue processing.
        
        This method should be extended by subclasses to handle their specific cleanup.
        """
        if self._is_running:
            self._is_running = False
            
            # Cancel all pending requests
            for future in self._pending_requests.values():
                if not future.done():
                    future.cancel()
            self._pending_requests.clear()
            
            # Signal queue processor to stop
            await self._generation_queue.put(None)
            
            # Wait for queue processor to finish
            if self._queue_task and not self._queue_task.done():
                try:
                    await asyncio.wait_for(self._queue_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"{self.__class__.__name__}: Queue processor didn't stop cleanly, cancelling")
                    self._queue_task.cancel()
                    
            logger.debug(f"{self.__class__.__name__}: Generator stopped")

    async def clear_pending_requests(self, error_message: str = "Request cancelled due to generator recovery"):
        """Clear all pending requests in the queue with an error.
        
        This is useful when the generator needs to recover from a bad state
        and wants to fail pending requests immediately rather than process them.
        
        Args:
            error_message: Error message to set on cancelled requests
        """
        logger.warning(f"{self.__class__.__name__}: Clearing {len(self._pending_requests)} pending requests: {error_message}")
        
        # Cancel all pending futures
        cancelled_count = 0
        for request_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.set_exception(RuntimeError(error_message))
                cancelled_count += 1
        
        # Clear the pending requests dict
        self._pending_requests.clear()
        
        # Clear any remaining items in the queue
        queue_size = self._generation_queue.qsize()
        cleared_queue_items = 0
        
        # Create new queue to effectively clear all pending items
        old_queue = self._generation_queue
        self._generation_queue = asyncio.Queue()
        
        # Drain old queue and count items
        try:
            while not old_queue.empty():
                old_queue.get_nowait()
                old_queue.task_done()
                cleared_queue_items += 1
        except asyncio.QueueEmpty:
            pass
            
        logger.info(f"{self.__class__.__name__}: Cleared {cancelled_count} pending futures and {cleared_queue_items} queued items")
        
    def get_queue_stats(self) -> Dict[str, int]:
        """Get current queue statistics.
        
        Returns:
            Dictionary with queue size and pending request count
        """
        return {
            "queue_size": self._generation_queue.qsize(),
            "pending_requests": len(self._pending_requests),
            "is_running": self._is_running
        }

    def _validate_prompt(self, prompt: str):
        """Validate the input prompt."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
    
    def _get_output_path(self, file_or_extension: str = "png", request_id: Optional[str] = None, sub_dir: str = "") -> str:
        """Generate a unique output path for an image.
        
        Args:
            file_or_extension: File extension or full filename
            request_id: Optional request ID to include in filename for traceability
            sub_dir: Optional subdirectory within the output directory
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

        if sub_dir:
            sub_dir_path = self.output_dir / sub_dir
            sub_dir_path.mkdir(parents=True, exist_ok=True)
            return str(sub_dir_path / f"{image_id}{extension}")

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