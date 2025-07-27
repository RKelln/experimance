#!/usr/bin/env python3
"""
Sohkepayin Core Service.

Main service that orchestrates the Sohkepayin installation by:
1. Listening for stories from the agent service
2. Analyzing stories to infer environmental settings
3. Generating base panorama images
4. Creating tiled high-resolution versions
5. Sending images to the display service

State machine: Idle → Listening → BaseImage → Tiles
"""

import asyncio
import argparse
import logging
import uuid
from enum import Enum
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from experimance_common.base_service import BaseService
from experimance_common.config import ConfigError, resolve_path
from experimance_common.constants import IMAGE_TRANSPORT_MODES
from experimance_common.schemas import ImageSource
from experimance_common.zmq.config import MessageDataType
from experimance_common.zmq.services import ControllerService  
from experimance_common.schemas import (
    StoryHeard, UpdateLocation, ImageReady, RenderRequest, DisplayMedia, 
    ContentType, MessageType
)
from experimance_common.zmq.zmq_utils import prepare_image_source

from .config import SohkepayinCoreConfig, ImagePrompt
from .llm import LLMProvider, get_llm_provider
from .llm_prompt_builder import LLMPromptBuilder
from .tiler import PanoramaTiler, TileSpec, create_tiler_from_config

from experimance_common.logger import setup_logging

SERVICE_TYPE = "core"

logger = setup_logging(__name__, log_filename=f"{SERVICE_TYPE}.log")


class CoreState(Enum):
    """Core service states."""
    IDLE = "idle"
    LISTENING = "listening" 
    BASE_IMAGE = "base_image"
    TILES = "tiles"


@dataclass
class ActiveRequest:
    """Tracks an active image generation request."""
    
    request_id: str
    base_prompt: ImagePrompt
    tiles: List[TileSpec] = field(default_factory=list)
    base_image_ready: bool = False
    completed_tiles: Dict[int, str] = field(default_factory=dict)  # tile_index -> image_path
    total_tiles: int = 0


class SohkepayinCoreService(BaseService):
    """
    Core service for the Sohkepayin installation.
    
    Manages the complete pipeline from story to panoramic visualization:
    - Story analysis and location inference
    - Image prompt generation  
    - Base panorama generation
    - Tile-based high-resolution generation
    - Image delivery to display service
    """
    
    def __init__(self, config: SohkepayinCoreConfig):
        """Initialize the Sohkepayin core service."""
        super().__init__(
            service_name=config.service_name,
            service_type=SERVICE_TYPE
        )
        
        self.config = config
        self.core_state = CoreState.IDLE  # services already have state, we track core state
        self.current_request: Optional[ActiveRequest] = None
        self.pending_image_requests: Dict[str, str] = {}  # request_id -> type
        
        # Initialize components
        self.llm = get_llm_provider(**config.llm.model_dump())
        
        print(config.llm.system_prompt_or_file)

        # system prompt
        if config.llm.system_prompt_or_file is not None:
            try:
                system_prompt = resolve_path(config.llm.system_prompt_or_file, hint="project")
                logger.info(f"Using system prompt from: {system_prompt}")
            except ConfigError:
                # its a string or doesn't exist
                system_prompt = config.llm.system_prompt_or_file
                logger.info(f"Could not resolve: {system_prompt}")
        

        self.prompt_builder = LLMPromptBuilder(
            llm=self.llm,
            system_prompt_or_file=system_prompt
        )
        
        self.tiler = create_tiler_from_config(config.tiles)
        
        # ZMQ communication will be initialized in start()
        self.zmq_service: ControllerService = None  # type: ignore # Will be initialized in start()
        
        logger.info("Sohkepayin core service initialized")
    
    async def start(self):
        """Start the service and initialize ZMQ communication."""
        logger.info("Starting Sohkepayin core service")
        
        # Initialize ZMQ service
        self.zmq_service = ControllerService(self.config.zmq)
        
        # Set up message handlers
        self.zmq_service.add_message_handler(MessageType.STORY_HEARD, self._handle_story_heard)
        self.zmq_service.add_message_handler(MessageType.UPDATE_LOCATION, self._handle_story_heard) # for now use the same handler
        
        # Set up worker response handler for image results
        self.zmq_service.add_response_handler(self._handle_worker_response)
        
        # Add periodic tasks
        self.add_task(self._state_monitor_task())
        
        # Start ZMQ service
        await self.zmq_service.start()
        
        # Transition to listening state
        await self._transition_to_state(CoreState.LISTENING)
        
        await super().start()
    
    async def stop(self):
        """Stop the service gracefully."""
        logger.info("Stopping Sohkepayin core service")
        
        if self.zmq_service:
            await self.zmq_service.stop()
        
        await super().stop()
    
    async def _transition_to_state(self, new_state: CoreState):
        """Transition to a new state with logging."""
        old_state = self.core_state
        self.core_state = new_state
        logger.info(f"State transition: {old_state.value} → {new_state.value}")
        
        # State-specific actions
        if new_state == CoreState.LISTENING:
            # Clear any old request when returning to listening
            if self.current_request:
                logger.info("Clearing previous request")
                self.current_request = None
                self.pending_image_requests.clear()
    
    async def _handle_story_heard(self, topic: str, message_data: MessageDataType):
        """
        Handle StoryHeard message - start new visualization pipeline.
        
        Args:
            topic: The message topic
            message_data: Message data (dict or MessageBase)
        """
        try:
            story : StoryHeard = StoryHeard.to_message_type(message_data) # type: ignore[assignment]
            if not story or not isinstance(story, StoryHeard):
                logger.warning("Failed to convert message to StoryHeard type")
                return
        
            logger.info(f"Story heard: {len(str(story.content))} chars")
            
            # Cancel any ongoing requests
            if self.current_request:
                logger.info("Canceling previous request for new story")
                self.pending_image_requests.clear()
            
            # Send clear display message
            await self._send_clear_display()
        

            # Analyze story to infer location
            logger.info("Analyzing story with LLM")
            
            # Create base image prompt
            base_prompt = await self.prompt_builder.build_prompt(
                story.content
            )
            
            # Calculate tiling strategy
            tiles = self.tiler.calculate_tiles(
                self.config.panorama.display_width,
                self.config.panorama.display_height
            )
            
            # Create new active request
            self.current_request = ActiveRequest(
                request_id=str(uuid.uuid4()),
                base_prompt=base_prompt,
                tiles=tiles,
                total_tiles=len(tiles)
            )
            
            logger.info(
                f"New request {self.current_request.request_id}: "
                f"{len(tiles)} tiles"
            )
            
            # Transition to base image generation
            await self._transition_to_state(CoreState.BASE_IMAGE)
            
            # Request base panorama image
            await self._request_base_image()
            
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message="Failed to process story")
            await self._transition_to_state(CoreState.LISTENING)

    
    async def _handle_worker_response(self, worker_name: str, response_data):
        """
        Handle worker response (ImageReady) from image_server.
        
        Args:
            worker_name: Name of the worker ("image_server")
            response_data: Response data from worker
        """
        if worker_name != "image_server":
            logger.debug(f"Ignoring response from unknown worker: {worker_name}")
            return
            
        # Convert to ImageReady object if needed
        if isinstance(response_data, dict):
            message = ImageReady(**response_data)
        else:
            message = response_data
            
        await self._handle_image_ready(message)
    
    async def _handle_image_ready(self, message):
        """
        Handle ImageReady message - process completed images.
        
        Args:
            message: ImageReady message from image_server
        """
        if not hasattr(message, 'request_id') or message.request_id not in self.pending_image_requests:
            logger.debug(f"Ignoring ImageReady for unknown request: {getattr(message, 'request_id', 'None')}")
            return
        
        request_type = self.pending_image_requests.pop(message.request_id)
        logger.info(f"Image ready: {request_type} for request {message.request_id}")
        
        if not self.current_request:
            logger.warning("Received ImageReady but no active request")
            return
        
        try:
            if request_type == "base":
                await self._handle_base_image_ready(message)
            elif request_type.startswith("tile_"):
                tile_index = int(request_type.split("_")[1])
                await self._handle_tile_image_ready(message, tile_index)
                
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message=f"Failed to handle {request_type} image")
    
    async def _request_base_image(self):
        """Request generation of the base panorama image."""
        if not self.current_request:
            return
        
        request_id = f"{self.current_request.request_id}_base"
        
        panorama_prompt = self.prompt_builder.base_prompt_to_panorama_prompt(
            self.current_request.base_prompt,
        )

        render_request = RenderRequest(
            request_id=request_id,
            prompt=panorama_prompt.prompt,
            negative_prompt=panorama_prompt.negative_prompt,
            width=self.config.panorama.generated_width,
            height=self.config.panorama.generated_height,
            # Add other parameters as needed
        )
        
        # Track pending request
        self.pending_image_requests[request_id] = "base"
        
        # Send render request
        await self.zmq_service.send_work_to_worker("image_server", render_request)
        
        logger.info(f"Requested base image: {self.config.panorama.generated_width}x{self.config.panorama.generated_height}")
    
    async def _handle_base_image_ready(self, response: ImageReady):
        """Handle completion of base image generation."""
        if not self.current_request:
            return
        
        logger.info("Base image ready, sending to display")
        
        # Send base image to display (no position = base panorama)
        display_message = DisplayMedia(
            content_type=ContentType.IMAGE,
            uri=response.uri,
            fade_in=2.0  # Base image fade-in duration
        )
        
        await self.zmq_service.publish(display_message, MessageType.DISPLAY_MEDIA)
        
        # Mark base image as ready
        self.current_request.base_image_ready = True
        
        # Transition to tile generation
        await self._transition_to_state(CoreState.TILES)
        
        # Start tile generation
        await self._request_all_tiles()
    
    async def _request_all_tiles(self):
        """Request generation of all tiles."""
        if not self.current_request or not self.current_request.tiles:
            return
        
        logger.info(f"Requesting {len(self.current_request.tiles)} tiles")
        
        for tile_spec in self.current_request.tiles:
            await self._request_tile(tile_spec)
    
    async def _request_tile(self, tile_spec: TileSpec):
        """Request generation of a specific tile."""
        if not self.current_request:
            return
        
        # Build tile-specific prompt
        # TODO: look at the reference image to create prompt?

        tile_prompt = self.prompt_builder.base_prompt_to_tile_prompt(
            self.current_request.base_prompt,
        )
        
        request_id = f"{self.current_request.request_id}_tile_{tile_spec.tile_index}"
        
        render_request = RenderRequest(
            request_id=request_id,
            prompt=tile_prompt.prompt,
            negative_prompt=tile_prompt.negative_prompt,
            width=tile_spec.generated_width,
            height=tile_spec.generated_height,
            # Add reference image and other tile-specific parameters
        )
        
        # Track pending request
        self.pending_image_requests[request_id] = f"tile_{tile_spec.tile_index}"
        
        # Send render request
        await self.zmq_service.send_work_to_worker("image_server", render_request)
        
        logger.debug(f"Requested tile {tile_spec.tile_index}: {tile_spec.generated_width}x{tile_spec.generated_height}")
    
    async def _handle_tile_image_ready(self, response: ImageReady, tile_index: int):
        """Handle completion of tile image generation."""
        if not self.current_request:
            return
        
        logger.info(f"Tile {tile_index} ready")
        
        # Store completed tile
        self.current_request.completed_tiles[tile_index] = response.uri
        
        # Find tile spec for positioning
        tile_spec = None
        for tile in self.current_request.tiles:
            if tile.tile_index == tile_index:
                tile_spec = tile
                break
        
        if not tile_spec:
            logger.error(f"No tile spec found for index {tile_index}")
            return
        
        # if overlap then load and apply blending
        if tile_spec.overlap > 0:
            logger.debug(f"Applying overlap blending for tile {tile_index}")
            # TODO: fix this to use the image path
            # remove file:// prefix if present
            file_path = response.uri.split("file://")[-1]  # Extract filename from URI
            image = self.tiler.apply_edge_blending(
                file_path,
                tile_spec
            )
        
            image_source = prepare_image_source(
                image_data=image,
                request_id=response.request_id,
                transport_mode=IMAGE_TRANSPORT_MODES["FILE_URI"]
            )
        else:
            image_source = ImageSource(
                uri=response.uri,
                image_data=None,  # No image data if using file URI
            )
        
        # Send tile to display with position
        display_message = DisplayMedia(
            content_type=ContentType.IMAGE,
            position=(tile_spec.display_x, tile_spec.display_y),  # Position in panorama space
            fade_in=1.5,  # Tile fade-in duration
            image_data=image_source.image_data,
            uri=image_source.uri,
        )
        logger.debug(f"Sending tile {tile_index} to display at position {tile_spec.display_x}, {tile_spec.display_y}")
        
        await self.zmq_service.publish(display_message)
        
        # Check if all tiles completed
        if len(self.current_request.completed_tiles) >= self.current_request.total_tiles:
            logger.info("All tiles completed, returning to listening state")
            await self._transition_to_state(CoreState.LISTENING)
    
    async def _send_clear_display(self):
        """Send clear message to display service."""
        clear_message = DisplayMedia(
            content_type=ContentType.CLEAR
        )
        
        await self.zmq_service.publish(clear_message)
        logger.info("Sent clear display message")
    
    async def _state_monitor_task(self):
        """Monitor state transitions and timeouts."""
        while self.running:
            # Check for timeouts in various states
            if self.core_state == CoreState.BASE_IMAGE:
                # Could add timeout logic here
                pass
            elif self.core_state == CoreState.TILES:
                # Could add tile timeout logic here  
                pass
            
            await self._sleep_if_running(10.0)  # Check every 10 seconds


async def run_sohkepayin_core_service(
    config_path: str,
    args: Optional[argparse.Namespace] = None
) -> None:
    """
    Run the Sohkepayin core service.
    
    Args:
        config_path: Path to configuration file
        args: Optional command line arguments
    """
    # Load configuration using the from_overrides method
    config = SohkepayinCoreConfig.from_overrides(
        config_file=config_path,
        args=args
    )
    
    # Create and run service
    service = SohkepayinCoreService(config)
    
    await service.start()
    logger.info("Sohkepayin core service started successfully")
    await service.run()
