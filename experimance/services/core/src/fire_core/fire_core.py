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
    ImageReady, RenderRequest, DisplayMedia, ContentType, MessageType,
    AudioRenderRequest, AudioReady  # type: ignore
)
from experimance_common.zmq.zmq_utils import prepare_image_source

from .config import FireCoreConfig, ImagePrompt, MediaPrompt
from .llm import LLMProvider, get_llm_provider
from .llm_prompt_builder import InsufficientContentException, UnchangedContentException, LLMPromptBuilder
from .tiler import PanoramaTiler, TileSpec, create_tiler_from_config
from .audio_manager import AudioManager

SERVICE_TYPE = "core"
AGENTS = ['llm', 'agent', 'assistant', 'fire_agent', 'experimance_agent']

logger = logging.getLogger(__name__)


class RequestState(Enum):
    """Request lifecycle states."""
    QUEUED = "queued"
    PROCESSING_LLM = "processing_llm"
    WAITING_BASE = "waiting_base"
    BASE_READY = "base_ready"
    WAITING_TILES = "waiting_tiles"
    WAITING_AUDIO = "waiting_audio"
    COMPLETED = "completed"
    CANCELLED = "cancelled"  # Request was cancelled or should be discarded


@dataclass
class ActiveRequest:
    """
    Tracks an active image generation request with proper state management.
    
    Request Lifecycle:
    1. QUEUED - Request created and waiting to be processed
    2. PROCESSING_LLM - LLM is analyzing content (can be interrupted)
    3. WAITING_BASE - Base panorama image is being generated (cannot interrupt)
    4. BASE_READY - Base image completed and sent to display
    5. WAITING_TILES - Tile images are being generated (can cancel tiles only)
    6. WAITING_AUDIO - All images complete, waiting for audio generation
    7. COMPLETED - All images and audio finished, request fully complete
    8. CANCELLED - Request was cancelled/interrupted or should be discarded
    
    Interruption Rules:
    - Base images ALWAYS complete once started (state WAITING_BASE)
    - Tile generation can be cancelled if a new request arrives
    - Audio generation can be cancelled/faded out if a new request arrives
    - Requests in QUEUED or PROCESSING_LLM states can be fully cancelled
    - LLM processing can be allowed to finish but marked as CANCELLED
    
    Priority Behavior:
    - New requests always take priority over queued requests
    - Running base images complete but tiles are cancelled for new requests
    - Audio fades out when new requests start
    - LLM processing can complete but result is discarded if cancelled
    - This ensures responsive visual feedback while minimizing wasted work
    """
    
    request_id: str
    base_prompt: ImagePrompt
    state: RequestState = RequestState.QUEUED
    created_at: float = field(default_factory=time.time)
    
    # Image generation data
    tiles: List[TileSpec] = field(default_factory=list)
    base_image: Optional[Image.Image] = None
    base_image_path: Optional[str] = None
    completed_tiles: Dict[int, str] = field(default_factory=dict)
    total_tiles: int = 0
    
    # Audio generation data
    audio_prompt: Optional[str] = None
    audio_request_id: Optional[str] = None
    audio_request_sent_time: Optional[float] = None
    audio_file_path: Optional[str] = None
    is_audio_playing: bool = False
    
    # Background task management
    processing_task: Optional[asyncio.Task] = None
    
    # Timeout tracking
    base_image_request_time: Optional[float] = None
    tile_requests_sent: Dict[str, float] = field(default_factory=dict)  # request_id -> timestamp
    state_transition_time: float = field(default_factory=time.time)
    
    # Timeouts (in seconds)
    IMAGE_TIMEOUT: float = 25.0
    LLM_TIMEOUT: float = 30.0
    STATE_TIMEOUT: float = 60.0
    
    def can_be_interrupted(self) -> bool:
        """
        Check if this request can be safely interrupted.
        
        Returns:
            True if request can be cancelled without losing significant work
        """
        return self.state in [RequestState.QUEUED, RequestState.PROCESSING_LLM]
    
    def is_generating_images(self) -> bool:
        """
        Check if this request is actively generating images.
        
        Returns:
            True if base or tile images are being generated
        """
        return self.state in [RequestState.WAITING_BASE, RequestState.WAITING_TILES]
    
    def is_generating_base(self) -> bool:
        """
        Check if this request is generating the base panorama image.
        
        Returns:
            True if base image generation is in progress (cannot be interrupted)
        """
        return self.state == RequestState.WAITING_BASE
    
    def is_generating_tiles(self) -> bool:
        """
        Check if this request is generating tile images.
        
        Returns:
            True if tile generation is in progress (tiles can be cancelled)
        """
        return self.state == RequestState.WAITING_TILES
    
    def is_waiting_for_audio(self) -> bool:
        """
        Check if this request is waiting for audio generation.
        
        Returns:
            True if audio generation is in progress
        """
        return self.state == RequestState.WAITING_AUDIO
    
    def is_completed(self) -> bool:
        """
        Check if this request is completed or cancelled.
        
        Returns:
            True if request has reached a terminal state
        """
        return self.state in [RequestState.COMPLETED, RequestState.CANCELLED]
    
    def transition_to_state(self, new_state: RequestState):
        """
        Transition to a new state and update timing.
        
        Args:
            new_state: The new state to transition to
        """
        old_state = self.state
        self.state = new_state
        self.state_transition_time = time.time()
        
        # State-specific timing updates
        if new_state == RequestState.WAITING_BASE:
            self.base_image_request_time = time.time()
    
    def is_timed_out(self, current_time: float) -> bool:
        """
        Check if this request has timed out based on its current state.
        
        Args:
            current_time: The current timestamp
            
        Returns:
            True if the request has timed out
        """
        state_duration = current_time - self.state_transition_time
        
        if self.state == RequestState.PROCESSING_LLM:
            return state_duration > self.LLM_TIMEOUT
        elif self.state == RequestState.WAITING_BASE and self.base_image_request_time:
            return (current_time - self.base_image_request_time) > self.IMAGE_TIMEOUT
        elif self.state == RequestState.WAITING_TILES:
            # Check if any tile requests have timed out
            for request_time in self.tile_requests_sent.values():
                if (current_time - request_time) > self.IMAGE_TIMEOUT:
                    return True
            return False
        elif self.state == RequestState.QUEUED:
            # Queued requests should have a much longer timeout to allow for processing
            return state_duration > (self.STATE_TIMEOUT * 3)  # 3 minutes for queued requests
        elif self.state == RequestState.BASE_READY:
            # Base ready should timeout quickly to move to tiles or completion
            return state_duration > 5.0  # 5 seconds
        
        return False
    
    def get_timed_out_tiles(self, current_time: float) -> List[str]:
        """
        Get list of tile request IDs that have timed out.
        
        Args:
            current_time: The current timestamp
            
        Returns:
            List of timed out tile request IDs
        """
        timed_out = []
        for request_id, request_time in self.tile_requests_sent.items():
            if (current_time - request_time) > self.IMAGE_TIMEOUT:
                timed_out.append(request_id)
        return timed_out
    
    def mark_tile_request_sent(self, tile_request_id: str):
        """
        Mark that a tile request has been sent.
        
        Args:
            tile_request_id: The ID of the tile request
        """
        self.tile_requests_sent[tile_request_id] = time.time()
    
    def mark_tile_completed(self, tile_request_id: str):
        """
        Mark that a tile request has completed.
        
        Args:
            tile_request_id: The ID of the tile request
        """
        self.tile_requests_sent.pop(tile_request_id, None)
    
    def get_pending_tile_count(self) -> int:
        """
        Get the number of pending tile requests.
        
        Returns:
            Number of pending tile requests
        """
        return len(self.tile_requests_sent)
    
    def all_tiles_completed(self) -> bool:
        """
        Check if all tiles have been completed.
        
        Returns:
            True if all tiles are completed
        """
        return len(self.completed_tiles) >= self.total_tiles
    
    # Audio management methods
    
    def has_audio(self) -> bool:
        """
        Check if this request includes audio generation.
        
        Returns:
            True if audio prompt is provided
        """
        return self.audio_prompt is not None and self.audio_prompt.strip() != ""
    
    def is_audio_requested(self) -> bool:
        """
        Check if audio request has been sent to image server.
        
        Returns:
            True if audio request has been sent
        """
        return self.audio_request_id is not None
    
    def is_audio_ready(self) -> bool:
        """
        Check if audio file has been generated and is ready for playback.
        
        Returns:
            True if audio file is available
        """
        return self.audio_file_path is not None
    
    def is_audio_timed_out(self, current_time: float) -> bool:
        """
        Check if audio request has timed out.
        
        Args:
            current_time: The current timestamp
            
        Returns:
            True if audio request has timed out
        """
        if not self.audio_request_sent_time:
            return False
        return (current_time - self.audio_request_sent_time) > self.IMAGE_TIMEOUT  # Use same timeout as images
    
    def mark_audio_request_sent(self, audio_request_id: str):
        """
        Mark that an audio request has been sent.
        
        Args:
            audio_request_id: The ID of the audio request
        """
        self.audio_request_id = audio_request_id
        self.audio_request_sent_time = time.time()
    
    def mark_audio_ready(self, audio_file_path: str):
        """
        Mark that audio file is ready for playback.
        
        Args:
            audio_file_path: Path to the generated audio file
        """
        self.audio_file_path = audio_file_path
        self.audio_request_sent_time = None  # Clear timeout tracking
    
    def mark_audio_playing(self, is_playing: bool = True):
        """
        Mark audio playback status.
        
        Args:
            is_playing: True if audio is currently playing
        """
        self.is_audio_playing = is_playing
    
    def should_wait_for_audio(self, current_time: Optional[float] = None) -> bool:
        """
        Check if request should transition to WAITING_AUDIO instead of COMPLETED.
        
        Args:
            current_time: Current timestamp for timeout checking
            
        Returns:
            True if audio was requested and is still being generated
        """
        if current_time is None:
            current_time = time.time()
            
        return (
            self.has_audio() and 
            self.is_audio_requested() and 
            not self.is_audio_ready() and 
            not self.is_audio_timed_out(current_time)
        )
    
    async def cancel(self, audio_manager=None):
        """
        Cancel this request and clean up any background tasks.
        
        This cancels LLM processing tasks but does not stop image generation
        that may already be in progress on the image server. If audio is currently
        playing, it will be faded out gracefully.
        
        Args:
            audio_manager: Optional AudioManager instance for audio fade-out
        """
        # Cancel LLM processing tasks
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Handle audio fade-out if currently playing
        if self.is_audio_playing and audio_manager:
            try:
                logger.info(f"🎵 Fading out audio for cancelled request {self.request_id}")
                await audio_manager.fade_out_all()
                self.mark_audio_playing(False)
            except Exception as e:
                logger.warning(f"Failed to fade out audio during cancellation: {e}")
        
        self.state = RequestState.CANCELLED


@dataclass
class TranscriptAccumulator:
    """Accumulates simplified transcript messages for processing."""
    session_id: Optional[str] = None
    conversation_lines: List[str] = field(default_factory=list)  # Simple "Speaker: content" strings
    last_update_time: float = field(default_factory=lambda: 0.0)

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
        self.current_request: Optional[ActiveRequest] = None
        self.llm_processing_request: Optional[ActiveRequest] = None  # Tracks LLM processing for transcripts
        self.pending_image_requests: Dict[str, tuple] = {}  # request_id -> (type, timestamp) - tracks images sent to image_server
        self.request_queue: List[ActiveRequest] = []  # Queue for pending requests
        
        # Transcript accumulation for streaming updates
        self.transcript_accumulator = TranscriptAccumulator()
        
        # Track current display session to avoid unnecessary clearing
        self.current_display_session_id: Optional[str] = None
        
        # Track last generated prompt for deduplication
        self.last_generated_prompt: Optional[MediaPrompt] = None
        
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
        
        # Initialize AudioManager for audio playbook
        if config.audio.enabled:
            try:
                from .audio_manager import AudioManager
                self.audio_manager = AudioManager(
                    default_volume=config.audio.default_volume,
                    crossfade_duration=config.audio.crossfade_duration
                )
                logger.info(f"AudioManager initialized (volume: {config.audio.default_volume}, crossfade: {config.audio.crossfade_duration}s)")
            except ImportError as e:
                logger.warning(f"AudioManager not available: {e}")
                self.audio_manager = None
        else:
            logger.info("Audio playback disabled in configuration")
            self.audio_manager = None
        
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
        
        # Service is ready to process requests
        logger.info("🔄 Fire core service ready - idle and waiting for requests")
        
        await super().start()
    
    async def stop(self):
        """Stop the service gracefully."""
        logger.info("Stopping Fire core service")
        
        # Clean up AudioManager
        if hasattr(self, 'audio_manager') and self.audio_manager is not None:
            await self.audio_manager.cleanup()
        
        if hasattr(self, 'updates_subscriber'):
            await self.updates_subscriber.stop()
            
        if self.zmq_service:
            await self.zmq_service.stop()
        
        await super().stop()
    
    async def cancel_current_request(self):
        """
        Cancel the current image request, if any.
        
        This method safely cancels the active request and clears any pending
        image generation tasks. It respects the interruption rules:
        - Cancels background LLM processing tasks
        - Clears pending image server requests 
        - Does NOT stop images already being generated on image server
        - Fades out any currently playing audio
        """
        if self.current_request:
            logger.debug(f"🚫 Canceling current request {self.current_request.request_id}")
            await self.current_request.cancel(self.audio_manager)
            self.current_request = None
            self.pending_image_requests.clear()
            logger.debug("🚫 Current request cancelled and pending image requests cleared")
        else:
            logger.debug("🚫 No current request to cancel")

    async def cancel_tile_generation(self):
        """
        Cancel ongoing tile generation but allow base image to complete.
        
        This implements the "base images always complete" policy:
        - Removes pending tile requests from the queue
        - Allows the base image to finish and display
        - Prepares system for the next request
        
        This is called when a new request arrives while tiles are generating.
        """
        if not self.current_request:
            return
            
        # Remove tile requests from pending_image_requests
        tile_requests_to_remove = []
        for request_id, (request_type, timestamp) in self.pending_image_requests.items():
            if request_type.startswith("tile_"):
                tile_requests_to_remove.append(request_id)
        
        for request_id in tile_requests_to_remove:
            self.pending_image_requests.pop(request_id, None)
            logger.debug(f"Canceled tile request: {request_id}")
        
        if tile_requests_to_remove:
            logger.info(f"Canceled {len(tile_requests_to_remove)} tile generation requests")

    def create_request(self, prompt) -> ActiveRequest:
        """Create a new request with proper initialization.
        
        Args:
            prompt: Either ImagePrompt or MediaPrompt with visual and audio components
        """
        # Extract visual and audio prompts
        if isinstance(prompt, MediaPrompt):
            base_prompt = ImagePrompt(
                prompt=prompt.visual_prompt,
                negative_prompt=prompt.visual_negative_prompt
            )
            audio_prompt = prompt.audio_prompt
        else:
            base_prompt = prompt  # Already ImagePrompt
            audio_prompt = None
        
        # Calculate tiling strategy
        tiles = self.tiler.calculate_tiles(
            self.config.panorama.display_width,
            self.config.panorama.display_height
        )
        
        # Create new request
        request = ActiveRequest(
            request_id=str(uuid.uuid4()),
            base_prompt=base_prompt,
            tiles=tiles,
            total_tiles=len(tiles),
            audio_prompt=audio_prompt
        )
        
        audio_info = f" with audio" if audio_prompt else ""
        logger.info(f"Created new request {request.request_id} with {len(tiles)} tiles{audio_info}")
        return request

    def queue_new_request(self, prompt) -> str:
        """Queue a new image generation request.
        
        Args:
            prompt: Either ImagePrompt or MediaPrompt
        """
        request = self.create_request(prompt)
        self.request_queue.append(request)
        logger.info(f"Queued new request {request.request_id} (queue size: {len(self.request_queue)})")
        return request.request_id

    async def start_request_processing(self, request: ActiveRequest):
        """Start processing a request by transitioning through states."""
        self.current_request = request
        request.transition_to_state(RequestState.WAITING_BASE)
        
        logger.info(f"🚀 STARTING request processing: {request.request_id}")
        
        # Starting base image generation
        logger.info("🔄 Transitioning to base image generation")
        
        # Request base panorama image
        await self._request_base_image()

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
            
            await self.cancel_current_request()

            # Analyze story to infer location
            logger.info("Analyzing story with LLM")
            
            # Create base media prompt
            try:
                # For backwards compatibility, convert last_generated_prompt to MediaPrompt if needed
                previous_media_prompt = self.last_generated_prompt
                if previous_media_prompt is None and hasattr(self, '_legacy_last_prompt'):
                    # Convert legacy ImagePrompt to MediaPrompt if we had one
                    legacy_prompt = getattr(self, '_legacy_last_prompt', None)
                    if legacy_prompt:
                        previous_media_prompt = MediaPrompt(
                            visual_prompt=legacy_prompt.prompt,
                            visual_negative_prompt=legacy_prompt.negative_prompt,
                            audio_prompt=None
                        )
                
                media_prompt = await self.prompt_builder.build_media_prompt(
                    story.content,
                    previous_prompt=previous_media_prompt,
                    transcript_callback=self._handle_curated_transcript
                )
                
                # Store the successfully generated prompt
                self.last_generated_prompt = media_prompt
                
                # Queue the new request
                request_id = self.queue_new_request(media_prompt)
                logger.info(f"Queued story-based request {request_id}")
                
            except UnchangedContentException as e:
                logger.info(f"Story content unchanged, skipping image generation: {e}")
                return  # Don't queue a new request
            except InsufficientContentException as e:
                logger.info(f"Insufficient content for image generation: {e}")
                return  # Don't queue a new request
            
            # Request processing will be handled by _state_monitor_task
            
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message="Failed to process story")

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
                
            elif topic == "audio_ready":
                # Handle AudioReady messages for audio playback testing
                await self._handle_audio_ready_update(topic, message_data)
                
            else:
                logger.debug(f"Ignoring update message on unhandled topic '{topic}'")
                
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message=f"Failed to process update message on topic '{topic}'")

    async def _handle_transcription_update(self, topic: str, message_data: MessageDataType):
        """
        Handle TranscriptUpdate message - ALWAYS process transcripts regardless of state.
        
        This handler MUST return quickly to avoid blocking ZMQ message reception.
        All heavy processing is done in background tasks.
        
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
            logger.info(f"🎙️  TRANSCRIPT [{transcript.session_id}] {transcript.speaker_id}: '{transcript.content}'")
            
            # Update accumulator (quick, synchronous operation)
            current_time = time.time()
            
            # Reset accumulator if new session or significant time gap
            if (self.transcript_accumulator.session_id != transcript.session_id or 
                (current_time - self.transcript_accumulator.last_update_time) > 300):  # 5 minute timeout
                logger.info(f"📝 Starting new transcript accumulation session: {transcript.session_id}")
                
                # Clear display for new conversation session
                if self.current_display_session_id != transcript.session_id:
                    logger.info(f"🖼️  NEW SESSION detected - clearing display (was: {self.current_display_session_id}, now: {transcript.session_id})")
                    await self._send_clear_display()
                    self.current_display_session_id = transcript.session_id
                
                # Reset LLM prompt memory for new session to allow fresh image generation
                if self.last_generated_prompt is not None:
                    logger.info(f"🧠 RESET LLM prompt memory for new session - previous prompt will not block new generation")
                    self.last_generated_prompt = None
                
                self.transcript_accumulator = TranscriptAccumulator(
                    session_id=transcript.session_id,
                    conversation_lines=[],
                    last_update_time=current_time,
                )
            
            # Add transcript to accumulator as simple string
            speaker = transcript.speaker_display_name or transcript.speaker_id
            conversation_line = f"{speaker}: {transcript.content}"
            self.transcript_accumulator.conversation_lines.append(conversation_line)
            self.transcript_accumulator.last_update_time = current_time
            
            logger.info(f"📊 Accumulator: {len(self.transcript_accumulator.conversation_lines)} total ")
            
            # Only trigger LLM processing on user messages (not agent responses)
            if transcript.speaker_id.lower() in AGENTS:
                logger.debug(f"⏭️  Skipping LLM processing for agent message from {transcript.speaker_id}")
                return
            
            # Cancel any existing LLM processing request
            if self.llm_processing_request and not self.llm_processing_request.is_completed():
                logger.info("🚫 Cancelling existing LLM processing request for new transcript")
                await self.llm_processing_request.cancel()
            
            # Create a new LLM processing request
            dummy_prompt = ImagePrompt(prompt="", negative_prompt="")  # Will be replaced by LLM
            self.llm_processing_request = ActiveRequest(
                request_id=f"llm_{uuid.uuid4()}",
                base_prompt=dummy_prompt,
                state=RequestState.PROCESSING_LLM
            )
            
            # Schedule the heavy processing in a background task that doesn't block ZMQ
            task = asyncio.create_task(self._process_transcript_in_background())
            self.llm_processing_request.processing_task = task
            
            # Return immediately so ZMQ can continue receiving messages
            logger.debug("✅ ZMQ handler returning immediately - background processing started")
            
        except Exception as e:
            logger.error(f"🔬 EXCEPTION IN _handle_transcription_update: {e}")
            self.record_error(e, is_fatal=False, custom_message="Failed to process transcript update")

    async def _process_transcript_in_background(self):
        """
        Process accumulated transcripts in background without blocking ZMQ reception.
        
        This method contains all the heavy LLM processing that was previously blocking
        the message handler. It implements the smart interruption logic:
        
        1. LLM processes accumulated transcript conversation
        2. If successful, creates a new request
        3. Intelligently interrupts current request based on its state:
           - QUEUED/PROCESSING_LLM: Cancel completely (minimal work lost)
           - WAITING_BASE: Let base complete, cancel future tiles
           - WAITING_TILES: Cancel remaining tiles (base already displayed)
        4. Queues new request for processing
        
        This ensures responsive behavior while minimizing wasted work.
        """
        try:
            # Format full conversation context for LLM
            full_context = self._format_transcript_context()
            logger.debug(f"🤖 Querying LLM with {len(full_context)} chars of conversation context")
            
            # Ask LLM to decide if we should generate a prompt using the existing prompt builder
            try:
                media_prompt = await self.prompt_builder.build_media_prompt(
                    full_context,
                    previous_prompt=self.last_generated_prompt,
                    audio_prefix=["high quality professional recording", "pristine", "high SNR"],
                    audio_suffix=["air utterly still", "stable ambience"],
                    transcript_callback=self._handle_curated_transcript
                )
                
                # Check if this LLM processing request was cancelled while running
                if self.llm_processing_request and self.llm_processing_request.state == RequestState.CANCELLED:
                    logger.info("🚫 LLM processing request was cancelled while running - discarding result")
                    return
                
                # Compare with previous prompt to detect changes
                visual_changed = (
                    not self.last_generated_prompt or 
                    media_prompt.visual_prompt != self.last_generated_prompt.visual_prompt or
                    media_prompt.visual_negative_prompt != self.last_generated_prompt.visual_negative_prompt
                )
                audio_changed = (
                    media_prompt.audio_prompt is not None and
                    (not self.last_generated_prompt or 
                    media_prompt.audio_prompt != self.last_generated_prompt.audio_prompt)
                )
                
                # Debug logging for prompt comparison
                logger.debug(f"🔍 PROMPT COMPARISON:")
                logger.debug(f"  Previous visual: {self.last_generated_prompt.visual_prompt if self.last_generated_prompt else 'None'}...")
                logger.debug(f"  New visual:      {media_prompt.visual_prompt}...")
                logger.debug(f"  Previous audio:  {self.last_generated_prompt.audio_prompt if self.last_generated_prompt and self.last_generated_prompt.audio_prompt else 'None'}...")
                logger.debug(f"  New audio:       {media_prompt.audio_prompt if media_prompt.audio_prompt else 'None'}...")
                logger.debug(f"  Visual changed:  {visual_changed}, Audio changed: {audio_changed}")
                
                if not visual_changed and not audio_changed:
                    logger.info("🔄 No significant changes in prompts - skipping generation")
                    raise UnchangedContentException("LLM returned unchanged prompts")
                
                if visual_changed:
                    logger.info("🖼️ Visual prompt changed - will generate new images")
                if audio_changed:
                    logger.info("🎵 Audio prompt changed - will generate new audio")
                
                # Store the successfully generated prompt
                self.last_generated_prompt = media_prompt
                
                # Queue the new request
                request_id = self.queue_new_request(media_prompt)
                logger.info(f"🖼️ Queued transcript-based request {request_id} (total queue: {len(self.request_queue)})")
                
                # Mark LLM processing as completed
                if self.llm_processing_request:
                    self.llm_processing_request.state = RequestState.COMPLETED
                
            except UnchangedContentException as e:
                logger.info(f"Transcript content unchanged, skipping image generation: {e}")
                # Mark LLM processing as completed
                if self.llm_processing_request:
                    self.llm_processing_request.state = RequestState.COMPLETED
                
            except InsufficientContentException as e:
                logger.debug("⏸️  LLM decided not to generate prompt yet, waiting for more content")
                # Mark LLM processing as completed even if insufficient content
                if self.llm_processing_request:
                    self.llm_processing_request.state = RequestState.COMPLETED
                
        except Exception as e:
            logger.error(f"🔥 BACKGROUND PROCESSING FAILED: {e}")
            self.record_error(e, is_fatal=False, custom_message="Failed to process transcript in background")
            # Mark LLM processing as failed/cancelled on exception
            if self.llm_processing_request:
                self.llm_processing_request.state = RequestState.CANCELLED
        
        finally:
            # Clear the LLM processing request reference when done
            self.llm_processing_request = None

    def _format_transcript_context(self) -> str:
        """Format all accumulated transcript lines into conversation context."""
        return "\n".join(self.transcript_accumulator.conversation_lines)

    def _parse_and_update_curated_transcript(self, transcript: str) -> None:
        """
        Replace transcript accumulator with LLM-curated transcript text.

        Args:
            transcript: LLM-curated transcript in format "Speaker: content" on separate lines
        """
        try:
            logger.info(f"📖 Replacing transcript accumulator with LLM-curated transcript ({len(transcript)} chars)")

            # Parse the transcript text into simple conversation lines
            new_lines = []
            current_time = time.time()
            
            for line in transcript.strip().split('\n'):
                line = line.strip()
                if not line or ':' not in line:
                    continue
                    
                # Validate the line has both speaker and content
                parts = line.split(':', 1)
                if len(parts) == 2 and parts[1].strip():
                    new_lines.append(line)

            # Replace the accumulator content with curated transcript
            original_count = len(self.transcript_accumulator.conversation_lines)
            self.transcript_accumulator.conversation_lines = new_lines
            self.transcript_accumulator.last_update_time = current_time
            
            logger.info(f"✅ Updated transcript accumulator: {original_count} → {len(new_lines)} lines (all marked as processed)")
            logger.debug(f"📝 Curated transcript preview: {transcript}...")

        except Exception as e:
            logger.error(f"Failed to parse curated transcript text: {e}")
            logger.debug(f"Problematic transcript text: {transcript}")

    def _handle_curated_transcript(self, transcript: str) -> None:
        """
        Callback function for receiving LLM-curated transcript text.

        Args:
            transcript: The cleaned transcript text returned by the LLM
        """
        logger.debug(f"📖 Received curated transcript callback: {len(transcript)} chars")
        self._parse_and_update_curated_transcript(transcript)

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
            audio_prompt = message_data.get('audio_prompt', None)  # Optional audio prompt
                
            logger.info(f"Processing debug prompt: {prompt[:100]}...")
            logger.debug(f"Debug negative prompt: {negative_prompt}")
            if audio_prompt:
                logger.debug(f"Debug audio prompt: {audio_prompt[:100]}...")
            
            await self.cancel_current_request()
            
            # Create direct media prompt (no LLM processing)
            media_prompt = MediaPrompt(
                visual_prompt=prompt,
                visual_negative_prompt=negative_prompt,
                audio_prompt=audio_prompt
            )
            
            logger.info(f"Created debug media prompt: {media_prompt.visual_prompt}")
            logger.debug(f"Debug negative: {media_prompt.visual_negative_prompt}")
            if media_prompt.audio_prompt:
                logger.debug(f"Debug audio: {media_prompt.audio_prompt}")
            
            # Queue the new request
            request_id = self.queue_new_request(media_prompt)
            logger.info(f"Queued debug request {request_id}")
            
            # Request processing will be handled by _state_monitor_task

            logger.debug("Debug prompt processed successfully")
            
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message="Failed to process debug prompt")

    async def _handle_audio_ready_update(self, topic: str, message_data: MessageDataType):
        """
        Handle AudioReady message from CLI testing - play the audio.
        
        This method handles AudioReady messages sent via the updates channel for
        audio playback testing. It uses the AudioManager to play the provided audio file.
        
        Args:
            topic: The message topic ("audio_ready")  
            message_data: AudioReady message data
        """
        logger.debug(f"Received AudioReady on topic '{topic}': {message_data}")
        try:
            # Handle AudioReady message - extract fields safely
            if isinstance(message_data, dict):
                request_id = message_data.get('request_id', 'unknown')
                uri = message_data.get('uri', 'unknown')
                prompt = message_data.get('prompt', 'none')
                is_loop = message_data.get('is_loop', True)
            else:
                # Handle pydantic model
                request_id = getattr(message_data, 'request_id', 'unknown')
                uri = getattr(message_data, 'uri', 'unknown')
                prompt = getattr(message_data, 'prompt', 'none')
                is_loop = getattr(message_data, 'is_loop', True)
                
            logger.info(f"🎵 AudioReady received: {request_id}")
            logger.info(f"   URI: {uri}")
            logger.info(f"   Prompt: {prompt}")
            logger.info(f"   Loop: {is_loop}")
            
            # Play the audio using AudioManager if available
            if self.audio_manager is not None:
                logger.info("🔊 Starting audio playback via AudioManager...")
                success = await self.audio_manager.play_audio(
                    uri, 
                    loop=is_loop,
                    crossfade=True  # Enable crossfading for smooth transitions
                )
                
                if success:
                    logger.info(f"✅ Audio playback started successfully: {uri}")
                    logger.info(f"   Playing with loop={is_loop}, crossfade enabled")
                    
                    # Log current audio status
                    playing_count = self.audio_manager.get_playing_count()
                    logger.info(f"   AudioManager now has {playing_count} active audio tracks")
                else:
                    logger.error(f"❌ Failed to start audio playback: {uri}")
            else:
                logger.warning("⚠️ AudioManager not available - cannot play audio")
                logger.info(f"   Would have played: {uri} (loop={is_loop})")
                
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message="Failed to handle AudioReady message")
    
    async def _handle_worker_response(self, worker_name: str, response_data):
        """
        Handle worker response (ImageReady or AudioReady) from image_server.
        
        Args:
            worker_name: Name of the worker ("image_server")
            response_data: Response data from worker
        """
        if worker_name != "image_server":
            logger.debug(f"Ignoring response from unknown worker: {worker_name}")
            return
        
        # Determine message type and convert appropriately
        if isinstance(response_data, dict):
            # Check request_id pattern or uri extension to distinguish between image and audio
            request_id = response_data.get('request_id', '')
            uri = response_data.get('uri', '')
            
            if request_id.endswith('_audio') or any(uri.endswith(ext) for ext in ['.wav', '.mp3', '.ogg', '.flac']):
                message = AudioReady(**response_data)
                await self._handle_audio_ready(message)
            else:
                message = ImageReady(**response_data)
                await self._handle_image_ready(message)
        else:
            # Already converted message object
            if isinstance(response_data, AudioReady):
                await self._handle_audio_ready(response_data)
            else:
                await self._handle_image_ready(response_data)
    
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
    
    async def _handle_audio_ready(self, message: AudioReady):
        """
        Handle AudioReady message - process completed audio generation.
        
        Args:
            message: AudioReady message from image_server
        """
        if not hasattr(message, 'request_id') or message.request_id not in self.pending_image_requests:
            logger.debug(f"Ignoring AudioReady for unknown request: {getattr(message, 'request_id', 'None')}")
            return
        
        request_type, _timestamp = self.pending_image_requests.pop(message.request_id)
        logger.info(f"🎵 Audio ready: {request_type} for request {message.request_id}")
        
        if not self.current_request:
            logger.warning("Received AudioReady but no active request")
            return
        
        if request_type != "audio":
            logger.warning(f"Expected audio request type but got: {request_type}")
            return
        
        try:
            # Store audio file path and mark as ready
            audio_file_path = message.uri.replace("file://", "") if message.uri.startswith("file://") else message.uri
            self.current_request.mark_audio_ready(audio_file_path)
            
            logger.info(f"🎵 Audio file ready: {audio_file_path}")
            logger.debug(f"🎵 Audio duration: {message.duration_s}s, loop: {message.is_loop}")
            
            # Start playing audio if AudioManager is available
            if self.audio_manager and self.config.audio.enabled:
                try:
                    logger.info("🔊 Starting audio playback via AudioManager...")
                    await self.audio_manager.play_audio(
                        audio_file_path,
                        volume=self.config.audio.default_volume,
                        loop=message.is_loop
                    )
                    self.current_request.mark_audio_playing(True)
                    logger.info(f"🎵 Audio playbook started successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to start audio playback: {e}")
            else:
                logger.warning("⚠️ AudioManager not available - cannot play audio")
                
        except Exception as e:
            self.record_error(e, is_fatal=False, custom_message="Failed to handle audio ready")
    
    async def _request_base_image(self):
        """Request generation of the base panorama image."""
        if self.current_request is None:
            return

        logger.info(f"📸 Requesting base image for prompt: {self.current_request.base_prompt.prompt[:100]}...")

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
        
        # Track pending request with timestamp and update ActiveRequest
        self.current_request.base_image_request_time = time.time()
        self.pending_image_requests[request_id] = ("base", time.time())
        logger.debug(f"🕒 Added base image request to pending: {request_id}")
        
        try:
            # Send render request with timeout protection
            logger.debug(f"🔬 SENDING WORK TO image_server: {request_id}")
            await self.zmq_service.send_work_to_worker("image_server", render_request)
            logger.info(f"✅ Sent base image request: {self.config.panorama.generated_width}x{self.config.panorama.generated_height}")
            
            # Request audio in parallel if available
            await self._request_audio_if_available()
            
        except Exception as e:
            logger.error(f"🔥 FAILED to send image request to image_server: {e}")
            # Remove the pending request since it failed
            self.pending_image_requests.pop(request_id, None)
            self.current_request.base_image_request_time = None
            
            # Return to listening state immediately on failure and try next queued request
            logger.warning("🔄 Image server unavailable - returning to listening state and processing queue")
            # Request processing will be handled by _state_monitor_task
            return
    
    async def _request_audio_if_available(self):
        """Request generation of audio if available in current request."""
        if not self.current_request or not self.current_request.has_audio():
            return
        
        if not self.current_request.audio_prompt:
            return
            
        # Truncate for logging safely
        prompt_preview = self.current_request.audio_prompt[:100] if len(self.current_request.audio_prompt) > 100 else self.current_request.audio_prompt
        logger.info(f"🎵 Requesting audio for prompt: {prompt_preview}...")
        
        # Create audio request ID and track it
        audio_request_id = f"{self.current_request.request_id}_audio"
        
        # Create AudioRenderRequest for audio generation
        audio_render_request = AudioRenderRequest(
            request_id=audio_request_id,
            prompt=self.current_request.audio_prompt,
            # Optional parameters can be added here
            # duration_s=30,  # Default duration
            # style="environmental"  # Style hint
            metadata={'no_cache': self.config.audio.no_cache}  # Use config setting for cache behavior
        )
        
        # Track the audio request
        self.current_request.mark_audio_request_sent(audio_request_id)
        self.pending_image_requests[audio_request_id] = ("audio", time.time())
        
        try:
            logger.debug(f"🔬 SENDING AUDIO WORK TO image_server: {audio_request_id}")
            await self.zmq_service.send_work_to_worker("image_server", audio_render_request)
            logger.info(f"✅ Sent audio request for generation")
        except Exception as e:
            logger.error(f"🔥 FAILED to send audio request to image_server: {e}")
            # Remove the pending request since it failed
            self.pending_image_requests.pop(audio_request_id, None)
            self.current_request.audio_request_id = None
            self.current_request.audio_request_sent_time = None
    
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
            request_id=self.current_request.request_id,  # Include request ID for proper crossfade tracking
            fade_in=5.0,  # Base image fade-in duration
            fade_out=2.0  # Base image fade-out duration
        )
        
        await self.zmq_service.publish(display_message, MessageType.DISPLAY_MEDIA)
        
        # Mark base image as ready - state management handled by _state_monitor_task
        self.current_request.transition_to_state(RequestState.BASE_READY)
        logger.info("📸 Base image complete - state updated to BASE_READY")
    
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
            strength=self.config.rendering.tile_strength # type: ignore (dynamic import)
        )
        
        # Track pending request using ActiveRequest's tracking
        self.current_request.mark_tile_request_sent(request_id)
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
        
        # Mark tile as completed in ActiveRequest tracking
        request_id = f"{self.current_request.request_id}_tile_{tile_index}"
        self.current_request.mark_tile_completed(request_id)
        
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
        
        # Check if this is the final tile
        is_final_tile = (len(self.current_request.completed_tiles) >= self.current_request.total_tiles)
        
        # Send tile to display with position
        display_message = DisplayMedia(
            content_type=ContentType.IMAGE,
            position=(tile_spec.display_x, tile_spec.display_y),  # Position in panorama space
            size=(tile_spec.display_width, tile_spec.display_height),  # Target display size
            request_id=f"{self.current_request.request_id}_tile_{tile_index}",  # Include tile-specific request ID
            fade_in=1.0,  # Tile fade-in duration
            fade_out=0.5,  # Tile fade-out duration
            image_data=image_source.image_data,
            uri=image_source.uri,
            final_tile=is_final_tile,  # Set flag for blur acceleration
        )
        logger.debug(f"Sending tile {tile_index} to display at position ({tile_spec.display_x}, {tile_spec.display_y}) "
                    f"with size ({tile_spec.display_width}, {tile_spec.display_height}) "
                    f"from generated size ({tile_spec.generated_width}, {tile_spec.generated_height}) "
                    f"final_tile={is_final_tile}")
        
        await self.zmq_service.publish(display_message)
        
        # State management handled by _state_monitor_task
        logger.debug(f"Tile {tile_index} completed - {len(self.current_request.completed_tiles)}/{self.current_request.total_tiles} done"
                    + (f" (FINAL TILE)" if is_final_tile else ""))
    
    async def _send_clear_display(self):
        """Send clear message to display service."""
        clear_message = DisplayMedia(
            content_type=ContentType.CLEAR,
            fade_out=10.0   # Duration to fade out/clear content
        )
        
        await self.zmq_service.publish(clear_message)
        logger.info("Sent clear display message with 10.0s fade-out")
    
    async def _state_monitor_task(self):
        """
        Monitor and manage all ActiveRequest state transitions and timeouts.
        
        This is the central hub for request lifecycle management, handling:
        1. Timeout detection for all requests
        2. Request state transitions  
        3. Queueing and processing logic
        4. Cleanup of completed/cancelled requests
        
        The method runs multiple processing loops each cycle:
        - Loop 1: Process timeouts and mark requests as cancelled
        - Loop 2: Remove completed/cancelled requests from queue
        - Loop 3: Process request state transitions
        - Loop 4: Start processing next queued request if ready
        """
        while self.running:
            current_time = time.time()
            
            # Debug logging every 10 seconds 
            # if not hasattr(self, '_last_debug_log') or (current_time - self._last_debug_log > 10.0):
            #     # Check ZMQ service status
            #     zmq_status = "Unknown"
            #     if hasattr(self, 'zmq_service') and self.zmq_service:
            #         zmq_status = f"Running={getattr(self.zmq_service, 'running', 'Unknown')}"
            #         if hasattr(self.zmq_service, 'subscriber') and self.zmq_service.subscriber:
            #             zmq_status += f", Subscriber={getattr(self.zmq_service.subscriber, 'running', 'Unknown')}"
                
            #     current_req_info = f"{self.current_request.request_id}:{self.current_request.state.value}" if self.current_request else "None"
            #     logger.info(f"🔍 STATE DEBUG: Current={current_req_info}, Queue={len(self.request_queue)}, Pending={len(self.pending_image_requests)}, ZMQ={zmq_status}")
            #     self._last_debug_log = current_time

            # ===== LOOP 1: Process timeouts for all requests =====
            await self._process_request_timeouts(current_time)
            
            # ===== LOOP 2: Remove completed/cancelled requests =====
            await self._cleanup_completed_requests()
            
            # ===== LOOP 3: Process current request state transitions =====
            await self._process_current_request_transitions()
            
            # ===== LOOP 4: Start next queued request if ready =====
            await self._process_request_queue()

            await self._sleep_if_running(0.1)  # Check every 0.1 seconds for faster recovery

    async def _process_request_timeouts(self, current_time: float):
        """Process timeouts for all requests and mark them as timed out."""
        
        # Check current request for timeout
        if self.current_request and self.current_request.is_timed_out(current_time):
            logger.warning(f"⏰ Current request {self.current_request.request_id} timed out in state {self.current_request.state.value}")
            
            if self.current_request.state == RequestState.WAITING_BASE:
                logger.info("Base image timeout - clearing request to allow queue processing")
                
                # Clear display and mark request as cancelled
                #display_message = DisplayMedia(content_type=ContentType.CLEAR, fade_in=10.0)
                #await self.zmq_service.publish(display_message, MessageType.DISPLAY_MEDIA)
                await self.current_request.cancel()
                
            elif self.current_request.state == RequestState.WAITING_TILES:
                # Handle individual tile timeouts
                timed_out_tiles = self.current_request.get_timed_out_tiles(current_time)
                for tile_request_id in timed_out_tiles:
                    logger.warning(f"🧩 Tile timeout: {tile_request_id}")
                    self.current_request.mark_tile_completed(tile_request_id)
                    self.pending_image_requests.pop(tile_request_id, None)
                
                # If no more tiles pending or too many timeouts, check if we should wait for audio
                if self.current_request.get_pending_tile_count() == 0:
                    if self.current_request.should_wait_for_audio(current_time):
                        logger.info("🔄 All tiles completed or timed out - transitioning to wait for audio")
                        self.current_request.transition_to_state(RequestState.WAITING_AUDIO)
                    else:
                        logger.info("🔄 All tiles completed or timed out - marking request complete")
                        self.current_request.transition_to_state(RequestState.COMPLETED)
            
            # Mark as cancelled for other timeout cases if not already completed
            if not self.current_request.is_completed():
                await self.current_request.cancel()
                
        # Check queued requests for timeout (LLM processing, etc.)
        for request in self.request_queue:
            if request.is_timed_out(current_time):
                logger.warning(f"⏰ Queued request {request.request_id} timed out in state {request.state.value}")
                await request.cancel()
                
        # Clean up expired pending image requests
        expired_requests = []
        for request_id, (request_type, timestamp) in self.pending_image_requests.items():
            if current_time - timestamp > ActiveRequest.IMAGE_TIMEOUT:
                expired_requests.append(request_id)
        
        for request_id in expired_requests:
            logger.warning(f"⏰ Removing expired pending image request: {request_id}")
            self.pending_image_requests.pop(request_id, None)

    async def _cleanup_completed_requests(self):
        """Remove completed or cancelled requests from the queue."""
        self.request_queue = [req for req in self.request_queue if not req.is_completed()]
        
        # Clear current request if it's completed
        if self.current_request and self.current_request.is_completed():
            logger.info(f"🧹 Cleaning up completed current request {self.current_request.request_id}")
            self.current_request = None

    async def _process_current_request_transitions(self):
        """Process state transitions for the current request."""
        if not self.current_request:
            return
            
        # Handle BASE_READY -> TILES, WAITING_AUDIO, or COMPLETED transition
        if self.current_request.state == RequestState.BASE_READY:
            if self.request_queue:
                # Higher priority requests waiting - skip tiles and complete
                logger.info(f"📸 Base ready but {len(self.request_queue)} requests queued - completing without tiles")
                self.current_request.transition_to_state(RequestState.COMPLETED)
            else:
                # Normal flow - check if we need tiles or just wait for audio
                if self.current_request.total_tiles > 0:
                    logger.info("📸 Base ready - starting tile generation")
                    self.current_request.transition_to_state(RequestState.WAITING_TILES)
                    await self._request_all_tiles()
                elif self.current_request.should_wait_for_audio():
                    logger.info("📸 Base ready - waiting for audio (no tiles needed)")
                    self.current_request.transition_to_state(RequestState.WAITING_AUDIO)
                else:
                    logger.info("📸 Base ready - no tiles or audio needed, completing")
                    self.current_request.transition_to_state(RequestState.COMPLETED)
        
        # Handle WAITING_TILES -> COMPLETED or WAITING_AUDIO transition
        elif self.current_request.state == RequestState.WAITING_TILES:
            if self.current_request.all_tiles_completed():
                if self.current_request.should_wait_for_audio():
                    logger.info("🧩 All tiles completed, transitioning to wait for audio")
                    self.current_request.transition_to_state(RequestState.WAITING_AUDIO)
                else:
                    logger.info("🧩 All tiles completed, marking request as complete")
                    self.current_request.transition_to_state(RequestState.COMPLETED)
        
        # Handle WAITING_AUDIO -> COMPLETED transition
        elif self.current_request.state == RequestState.WAITING_AUDIO:
            if self.current_request.is_audio_ready() or not self.current_request.should_wait_for_audio():
                logger.info("🎵 Audio ready or no longer needed, marking request as complete")
                self.current_request.transition_to_state(RequestState.COMPLETED)

    async def _process_request_queue(self):
        """Start processing the next queued request if conditions are met."""
        if (not self.current_request and self.request_queue):
            # Process next request regardless of core state, as long as no current request
            next_request = self.request_queue.pop(0)
            logger.info(f"🚀 PROCESSING queued request {next_request.request_id} (queue size: {len(self.request_queue)})")
            await self.start_request_processing(next_request)


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
