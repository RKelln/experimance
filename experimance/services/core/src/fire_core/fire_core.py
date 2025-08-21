#!/usr/bin/env python3
"""
Feed the Fires Core Service.

Main service that orchestrates the Fire installation by:
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
import time
from enum import Enum
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from PIL import Image

from experimance_common.base_service import BaseService
from experimance_common.config import ConfigError, resolve_path
from experimance_common.constants import IMAGE_TRANSPORT_MODES, DEFAULT_PORTS, ZMQ_TCP_BIND_PREFIX
from experimance_common.schemas import ImageSource
from experimance_common.zmq.config import MessageDataType, SubscriberConfig
from experimance_common.zmq.services import ControllerService, SubscriberComponent
from experimance_common.schemas import (
    StoryHeard, UpdateLocation, TranscriptUpdate,  # type: ignore
    ImageReady, RenderRequest, DisplayMedia, ContentType, MessageType 
)
from experimance_common.zmq.zmq_utils import prepare_image_source

from .config import FireCoreConfig, ImagePrompt
from .llm import LLMProvider, get_llm_provider
from .llm_prompt_builder import InsufficientContentException, LLMPromptBuilder
from .tiler import PanoramaTiler, TileSpec, create_tiler_from_config

from experimance_common.logger import setup_logging

SERVICE_TYPE = "core"
AGENTS = ['llm', 'agent', 'assistant', 'fire_agent', 'experimance_agent']

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
    base_image: Optional[Image.Image] = None  # Base image for reference image generation
    base_image_path: Optional[str] = None  # Keep path for logging/debugging
    completed_tiles: Dict[int, str] = field(default_factory=dict)  # tile_index -> image_path
    total_tiles: int = 0


@dataclass
class TranscriptAccumulator:
    """Accumulates transcript messages for processing."""
    session_id: Optional[str] = None
    messages: List[TranscriptUpdate] = field(default_factory=list)
    last_update_time: float = field(default_factory=lambda: 0.0)
    processed_count: int = 0  # Number of messages processed by LLM


class FireCoreService(BaseService):
    """
    Core service for the Fire installation.
    
    Manages the complete pipeline from story to panoramic visualization:
    - Story analysis and location inference
    - Image prompt generation  
    - Base panorama generation
    - Tile-based high-resolution generation
    - Image delivery to display service
    """
    
    def __init__(self, config: FireCoreConfig):
        """Initialize the Fire core service."""
        super().__init__(
            service_name=config.service_name,
            service_type=SERVICE_TYPE
        )
        
        self.config = config
        self.core_state = CoreState.IDLE  # services already have state, we track core state
        self.current_request: Optional[ActiveRequest] = None
        self.pending_image_requests: Dict[str, tuple] = {}  # request_id -> (type, timestamp)
        
        # Transcript accumulation for streaming updates
        self.transcript_accumulator = TranscriptAccumulator()
        
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
        
        logger.info("Fire core service initialized")
    
    async def start(self):
        """Start the service and initialize ZMQ communication."""
        logger.info("Starting Fire core service")
        
        # Initialize ZMQ service
        self.zmq_service = ControllerService(self.config.zmq)
        
        # Set up message handlers
        self.zmq_service.add_message_handler(MessageType.STORY_HEARD, self._handle_story_heard)
        self.zmq_service.add_message_handler(MessageType.UPDATE_LOCATION, self._handle_story_heard) # for now use the same handler
        self.zmq_service.add_message_handler(MessageType.TRANSCRIPT_UPDATE, self._handle_transcription_update)

        # Add subscriber for updates on port 5556 (core binds, others publish to us)
        updates_config = SubscriberConfig(
            port=DEFAULT_PORTS['updates'],
            address=ZMQ_TCP_BIND_PREFIX,  # Core binds - other services publish to us
            bind=True,  # Core binds on updates channel
            topics=[""]  # Subscribe to all topics
        )
        self.updates_subscriber = SubscriberComponent(updates_config)
        self.updates_subscriber.set_default_handler(self._handle_update_message)
        await self.updates_subscriber.start()
        
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
        logger.info("Stopping Fire core service")
        
        if hasattr(self, 'updates_subscriber'):
            await self.updates_subscriber.stop()
            
        if self.zmq_service:
            await self.zmq_service.stop()
        
        await super().stop()
    
    async def _transition_to_state(self, new_state: CoreState):
        """Transition to a new state with logging."""
        import time
        old_state = self.core_state
        self.core_state = new_state
        self._state_enter_time = time.time()  # Track when we entered this state
        logger.info(f"🔄 State transition: {old_state.value} → {new_state.value}")
        
        # State-specific actions
        if new_state == CoreState.LISTENING:
            # Clear any old request when returning to listening
            logger.info("👂 Ready to listen for new transcripts")
            self.cancel_current_request()

    def cancel_current_request(self):
        """Cancel the current image request, if any."""
        if self.current_request:
            logger.debug("Canceling current request")
            self.current_request = None
            self.pending_image_requests.clear()

    async def _handle_story_heard(self, topic: str, message_data: MessageDataType):
        """
        Handle StoryHeard message - start new visualization pipeline.
        
        Args:
            topic: The message topic
            message_data: Message data (dict or MessageBase)
        """
        logger.debug(f"Received story heard message '{message_data}'")
        try:
            story : StoryHeard = StoryHeard.to_message_type(message_data) # type: ignore[assignment]
            if not story or not isinstance(story, StoryHeard):
                logger.warning("Failed to convert message to StoryHeard type")
                return
        
            logger.info(f"Story heard: {len(str(story.content))} chars")
            
            self.cancel_current_request()
            
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

    async def _handle_update_message(self, topic: str, message_data: MessageDataType):
        """
        Handle update messages from other services on port 5556.
        
        Args:
            topic: The message topic (e.g., "story", "prompt", "status")
            message_data: Message data
        """
        try:
            logger.debug(f"Received update message on topic '{topic}'")
            
            if topic == "story":
                # Handle story updates - use the same handler as StoryHeard
                await self._handle_story_heard(topic, message_data)
                
            elif topic == "prompt":
                # Handle direct prompt - bypass LLM analysis  
                await self._handle_debug_prompt(topic, message_data)
                
            else:
                logger.debug(f"Ignoring update message on unhandled topic '{topic}'")
                
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message=f"Failed to process update message on topic '{topic}'")

    async def _handle_transcription_update(self, topic: str, message_data: MessageDataType):
        """
        Handle TranscriptUpdate message - ALWAYS process transcripts regardless of state.
        Can interrupt ongoing image generation if new compelling content arrives.
        
        Args:
            topic: The message topic
            message_data: Message data (dict or MessageBase)
        """
        logger.debug(f"🔬 ZMQ HANDLER CALLED: _handle_transcription_update with topic='{topic}', data={message_data}")
        try:
            transcript: TranscriptUpdate = TranscriptUpdate.to_message_type(message_data)  # type: ignore[assignment]
            if not transcript or not isinstance(transcript, TranscriptUpdate):
                logger.warning("Failed to convert message to TranscriptUpdate type")
                return

            # ALWAYS log incoming transcripts so we can see what's being received
            logger.info(f"🎙️  TRANSCRIPT [{transcript.session_id}] {transcript.speaker_id}: '{transcript.content}' [State: {self.core_state.value}]")
            
            # Update accumulator
            current_time = time.time()
            
            # Reset accumulator if new session or significant time gap
            if (self.transcript_accumulator.session_id != transcript.session_id or 
                (current_time - self.transcript_accumulator.last_update_time) > 300):  # 5 minute timeout
                logger.info(f"📝 Starting new transcript accumulation session: {transcript.session_id}")
                self.transcript_accumulator = TranscriptAccumulator(
                    session_id=transcript.session_id,
                    messages=[],
                    last_update_time=current_time,
                    processed_count=0
                )
            
            # Add transcript to accumulator
            self.transcript_accumulator.messages.append(transcript)
            self.transcript_accumulator.last_update_time = current_time
            
            logger.info(f"📊 Accumulator: {len(self.transcript_accumulator.messages)} total, "
                       f"{self.transcript_accumulator.processed_count} processed, "
                       f"{len(self.transcript_accumulator.messages) - self.transcript_accumulator.processed_count} unprocessed")
            
            # Only trigger LLM processing on user messages (not agent responses)
            if transcript.speaker_id.lower() in AGENTS:
                logger.debug(f"⏭️  Skipping LLM processing for agent message from {transcript.speaker_id}")
                return
            
            # SIMPLIFIED LOGIC: Always try to generate a new image for ANY new user content
            # This allows the system to be more responsive and change directions
            logger.debug(f"🔍 New user transcript received, checking if we should generate new image")
            
            # Don't wait for multiple messages - be responsive to any new user content
            # Check if this is genuinely new content (not just re-processing old messages)
            unprocessed_messages = self.transcript_accumulator.messages[self.transcript_accumulator.processed_count:]
            user_messages_count = len([msg for msg in unprocessed_messages 
                                     if msg.speaker_id.lower() not in AGENTS])
            
            logger.debug(f"🔍 Analysis: {len(unprocessed_messages)} unprocessed messages, {user_messages_count} from users")
            
            if user_messages_count < 1:  # Wait for at least 1 new user message
                logger.debug("⏸️  Not enough new user content, waiting for more transcripts")
                return
            
            # Format full conversation context for LLM
            full_context = self._format_transcript_context()
            logger.debug(f"🤖 Querying LLM with {len(full_context)} chars of conversation context")
            
            # Ask LLM to decide if we should generate a prompt using the existing prompt builder
            try:
                logger.debug("🔬 ABOUT TO START LLM CALL - this might block the event loop")
                image_prompt = await self.prompt_builder.build_prompt(full_context)
                logger.debug("🔬 LLM CALL COMPLETED")
                
                # The prompt builder will return a result or raise an exception if insufficient
                logger.info("✅ LLM decided to generate prompt based on accumulated transcripts")
                
                # Cancel any existing request (this allows interrupting ongoing image generation)
                if self.current_request or self.core_state != CoreState.LISTENING:
                    logger.info(f"🔄 Canceling existing request/state ({self.core_state.value}) for new transcript-based image")
                
                self.cancel_current_request()
                
                # Force transition back to listening first, then to base_image
                if self.core_state != CoreState.LISTENING:
                    await self._transition_to_state(CoreState.LISTENING)
                
                # Send clear display message
                await self._send_clear_display()
                
                # Calculate tiling strategy
                tiles = self.tiler.calculate_tiles(
                    self.config.panorama.display_width,
                    self.config.panorama.display_height
                )
                
                # Create new active request
                self.current_request = ActiveRequest(
                    request_id=str(uuid.uuid4()),
                    base_prompt=image_prompt,
                    tiles=tiles,
                    total_tiles=len(tiles)
                )
                
                logger.info(f"🖼️  New transcript-based request {self.current_request.request_id}: {len(tiles)} tiles")
                
                # Mark these messages as processed
                old_processed_count = self.transcript_accumulator.processed_count
                self.transcript_accumulator.processed_count = len(self.transcript_accumulator.messages)
                logger.info(f"📝 Marked {self.transcript_accumulator.processed_count - old_processed_count} messages as processed")
                
                # Transition to base image generation
                await self._transition_to_state(CoreState.BASE_IMAGE)
                
                # Request base panorama image
                await self._request_base_image()
                
            except InsufficientContentException:
                logger.debug("⏸️  LLM decided not to generate prompt yet, waiting for more content")
            
        except Exception as e:
            logger.error(f"🔬 EXCEPTION IN _handle_transcription_update: {e}")
            self.record_error(e, is_fatal=False, custom_message="Failed to process transcript update")

    def _format_transcript_context(self) -> str:
        """Format all accumulated transcripts into conversation context."""
        conversation = []
        
        for msg in self.transcript_accumulator.messages:
            speaker = msg.speaker_display_name or msg.speaker_id
            conversation.append(f"{speaker}: {msg.content}")
        
        return "\n".join(conversation)

    async def _handle_debug_prompt(self, topic: str, message_data: MessageDataType):
        """
        Handle debug prompt - bypass LLM analysis and go directly to image generation.
        
        Args:
            topic: The message topic
            message_data: Message data containing the prompt
        """
        logger.debug(f"Received debug prompt on topic '{topic}': {message_data}")
        try:
            prompt = message_data.get('prompt', '')
            negative_prompt = message_data.get('negative_prompt', 'blurry, low quality, distorted')
                
            logger.info(f"Processing debug prompt: {prompt[:100]}...")
            logger.debug(f"Debug negative prompt: {negative_prompt}")
            
            self.cancel_current_request()
            
            # Send clear display message
            await self._send_clear_display()
            
            # Create direct image prompt (no LLM processing)
            base_prompt = ImagePrompt(
                prompt=prompt,
                negative_prompt=negative_prompt
            )
            
            logger.info(f"Created debug base prompt: {base_prompt.prompt}")
            logger.debug(f"Debug base negative: {base_prompt.negative_prompt}")
            
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
                f"New debug request {self.current_request.request_id}: "
                f"direct prompt with {len(tiles)} tiles"
            )
            
            # Transition to base image generation
            await self._transition_to_state(CoreState.BASE_IMAGE)
            
            # Request base panorama image
            await self._request_base_image()

            logger.debug("Debug prompt processed successfully")
            
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message="Failed to process debug prompt")
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
        
        request_type, _timestamp = self.pending_image_requests.pop(message.request_id)
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
        if self.current_request is None:
            return

        logger.info(f"Requesting base image for prompt: {self.current_request.base_prompt}")

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
        
        # Track pending request with timestamp
        import time
        self.pending_image_requests[request_id] = ("base", time.time())
        
        try:
            # Send render request with timeout protection
            logger.debug(f"🔬 SENDING WORK TO image_server: {request_id}")
            await self.zmq_service.send_work_to_worker("image_server", render_request)
            logger.info(f"Requested base image: {self.config.panorama.generated_width}x{self.config.panorama.generated_height}")
        except Exception as e:
            logger.error(f"🔥 FAILED to send image request to image_server: {e}")
            # Remove the pending request since it failed
            self.pending_image_requests.pop(request_id, None)
            
            # Return to listening state immediately on failure
            logger.warning("Image server appears unavailable - returning to listening state")
            await self._transition_to_state(CoreState.LISTENING)
            self.cancel_current_request()
            raise
    
    async def _handle_base_image_ready(self, response: ImageReady):
        """Handle completion of base image generation."""
        if not self.current_request:
            return
        
        logger.info("Base image ready, loading and sending to display")
        
        # Store base image path for logging/debugging
        base_image_path = response.uri.replace("file://", "") if response.uri.startswith("file://") else response.uri
        self.current_request.base_image_path = base_image_path
        logger.debug(f"Base image path: {base_image_path}")
        # Load and store the base image in memory
        try:
            self.current_request.base_image = Image.open(base_image_path)
            logger.debug(f"Loaded base image: {self.current_request.base_image.size}")
        except Exception as e:
            logger.error(f"Failed to load base image from {base_image_path}: {e}")
            # Continue without base image - tiles will generate without reference
        
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
        tile_prompt = self.prompt_builder.base_prompt_to_tile_prompt(
            self.current_request.base_prompt,
        )
        
        logger.info(f"Tile {tile_spec.tile_index} prompt: {tile_prompt.prompt}")
        logger.debug(f"Tile {tile_spec.tile_index} negative: {tile_prompt.negative_prompt}")
        
        request_id = f"{self.current_request.request_id}_tile_{tile_spec.tile_index}"
        
        # Create reference image if base image is available
        reference_image = None
        if self.current_request.base_image is not None:
            try:
                reference_image_path = self.tiler.prepare_tile_reference_image(
                    self.current_request.base_image,
                    tile_spec,
                    self.config.panorama.display_width,
                    self.config.panorama.display_height,
                    output_dir="/tmp"  # Or use config for this
                )
                # Create ImageSource from the reference image path
                reference_image = ImageSource(
                    uri=f"file://{reference_image_path}",
                    image_data=None  # Use file path, not in-memory data
                )
            except Exception as e:
                logger.warning(f"Failed to create reference image for tile {tile_spec.tile_index}: {e}")
        
        render_request = RenderRequest(
            request_id=request_id,
            prompt=tile_prompt.prompt,
            negative_prompt=tile_prompt.negative_prompt,
            width=tile_spec.generated_width,
            height=tile_spec.generated_height,
            reference_image=reference_image,  # Add reference image
            strength=self.config.rendering.tile_strength  # Use configured strength value
        )
        
        # Track pending request
        # Track pending request with timestamp
        import time
        self.pending_image_requests[request_id] = (f"tile_{tile_spec.tile_index}", time.time())
        
        # Send render request
        await self.zmq_service.send_work_to_worker("image_server", render_request)
        
        logger.debug(f"Requested tile {tile_spec.tile_index}: {tile_spec.generated_width}x{tile_spec.generated_height}" + 
                    (f" with reference" if reference_image else ""))
    
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
        import time
        
        # Image request timeout in seconds (reduced to 10 seconds for faster recovery)
        IMAGE_TIMEOUT = 10.0
        
        while self.running:
            current_time = time.time()
            
            # Check for image request timeouts
            expired_requests = []
            for request_id, (request_type, timestamp) in self.pending_image_requests.items():
                if current_time - timestamp > IMAGE_TIMEOUT:
                    expired_requests.append((request_id, request_type))
            
            # Handle expired requests
            for request_id, request_type in expired_requests:
                logger.warning(f"🔥 Image request timeout: {request_type} (request_id: {request_id}) - probably image_server unavailable")
                self.pending_image_requests.pop(request_id, None)
                
                # If it's a base image timeout and we're in base_image state, fall back to listening
                if request_type == "base" and self.core_state == CoreState.BASE_IMAGE:
                    logger.info("🔥 Base image generation timed out, returning to listening state for new transcripts")
                    await self._transition_to_state(CoreState.LISTENING)
                    self.cancel_current_request()
                    
                    # Send a clear display notification 
                    display_message = DisplayMedia(
                        content_type=ContentType.CLEAR,
                        fade_in=1.0
                    )
                    await self.zmq_service.publish(display_message, MessageType.DISPLAY_MEDIA)
                
                # For tile timeouts, we could implement partial display logic
                elif request_type.startswith("tile_") and self.core_state == CoreState.TILES:
                    logger.warning(f"Tile {request_type} generation timed out")
                    # Could implement logic to continue with available tiles or timeout completely
                    # For now, just log the timeout - tiles may still complete individually
            
            # Check for state-specific timeouts  
            if self.core_state == CoreState.BASE_IMAGE:
                # If we've been in base_image state too long without pending requests, reset
                if not self.pending_image_requests and current_time > getattr(self, '_state_enter_time', 0) + IMAGE_TIMEOUT:
                    logger.warning("🔥 Base image state timeout with no pending requests, returning to listening")
                    await self._transition_to_state(CoreState.LISTENING)
                    self.cancel_current_request()
                    
            elif self.core_state == CoreState.TILES:
                # Similar logic for tiles state
                if not self.pending_image_requests and current_time > getattr(self, '_state_enter_time', 0) + IMAGE_TIMEOUT * 2:  # Longer timeout for tiles
                    logger.warning("🔥 Tiles state timeout with no pending requests, returning to listening")
                    await self._transition_to_state(CoreState.LISTENING)
                    self.cancel_current_request()
            
            await self._sleep_if_running(2.0)  # Check every 2 seconds for faster recovery


async def run_fire_core_service(
    config_path: str,
    args: Optional[argparse.Namespace] = None
) -> None:
    """
    Run the Fire core service.
    
    Args:
        config_path: Path to configuration file
        args: Optional command line arguments
    """
    # Load configuration using the from_overrides method
    config = FireCoreConfig.from_overrides(
        config_file=config_path,
        args=args
    )
    
    # Create and run service
    service = FireCoreService(config)
    
    await service.start()
    logger.info("Fire core service started successfully")
    await service.run()
