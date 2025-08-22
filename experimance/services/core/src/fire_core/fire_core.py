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


class RequestState(Enum):
    """Request lifecycle states."""
    QUEUED = "queued"
    PROCESSING_LLM = "processing_llm"
    WAITING_BASE = "waiting_base"
    BASE_READY = "base_ready"
    WAITING_TILES = "waiting_tiles"
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
    6. COMPLETED - All tiles finished, request fully complete
    7. CANCELLED - Request was cancelled/interrupted or should be discarded
    
    Interruption Rules:
    - Base images ALWAYS complete once started (state WAITING_BASE)
    - Tile generation can be cancelled if a new request arrives
    - Requests in QUEUED or PROCESSING_LLM states can be fully cancelled
    - LLM processing can be allowed to finish but marked as CANCELLED
    
    Priority Behavior:
    - New requests always take priority over queued requests
    - Running base images complete but tiles are cancelled for new requests
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
    
    # Background task management
    processing_task: Optional[asyncio.Task] = None
    
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
    
    def is_completed(self) -> bool:
        """
        Check if this request is completed or cancelled.
        
        Returns:
            True if request has reached a terminal state
        """
        return self.state in [RequestState.COMPLETED, RequestState.CANCELLED]
    
    async def cancel(self):
        """
        Cancel this request and clean up any background tasks.
        
        This cancels LLM processing tasks but does not stop image generation
        that may already be in progress on the image server.
        """
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        self.state = RequestState.CANCELLED


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
        self.llm_processing_request: Optional[ActiveRequest] = None  # Tracks LLM processing for transcripts
        self.pending_image_requests: Dict[str, tuple] = {}  # request_id -> (type, timestamp) - tracks images sent to image_server
        self.request_queue: List[ActiveRequest] = []  # Queue for pending requests
        
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
            await self.cancel_current_request()

    async def cancel_current_request(self):
        """
        Cancel the current image request, if any.
        
        This method safely cancels the active request and clears any pending
        image generation tasks. It respects the interruption rules:
        - Cancels background LLM processing tasks
        - Clears pending image server requests 
        - Does NOT stop images already being generated on image server
        """
        if self.current_request:
            logger.debug(f"🚫 Canceling current request {self.current_request.request_id}")
            await self.current_request.cancel()
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

    def create_request(self, base_prompt: ImagePrompt) -> ActiveRequest:
        """Create a new request with proper initialization."""
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
            total_tiles=len(tiles)
        )
        
        logger.info(f"Created new request {request.request_id} with {len(tiles)} tiles")
        return request

    def queue_new_request(self, base_prompt: ImagePrompt) -> str:
        """Queue a new image generation request."""
        request = self.create_request(base_prompt)
        self.request_queue.append(request)
        logger.info(f"Queued new request {request.request_id} (queue size: {len(self.request_queue)})")
        return request.request_id

    async def start_request_processing(self, request: ActiveRequest):
        """Start processing a request by transitioning through states."""
        self.current_request = request
        request.state = RequestState.WAITING_BASE
        
        logger.info(f"🚀 STARTING request processing: {request.request_id}")
        
        # Send clear display message
        await self._send_clear_display()
        
        # Transition to base image generation
        await self._transition_to_state(CoreState.BASE_IMAGE)
        
        # Request base panorama image
        await self._request_base_image()

    async def process_next_queued_request(self):
        """Process the next queued request if we're in listening state."""
        if self.core_state != CoreState.LISTENING or not self.request_queue:
            if self.request_queue:
                logger.debug(f"Cannot process queued request - wrong state: {self.core_state.value}")
            return
            
        if self.current_request is not None:
            logger.warning("Cannot process queued request - current request still active")
            return
        
        # Get the next request from queue
        next_request = self.request_queue.pop(0)
        logger.info(f"🚀 PROCESSING queued request {next_request.request_id} (queue size: {len(self.request_queue)})")
        
        # Start processing this request
        await self.start_request_processing(next_request)

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
            
            # Send clear display message
            await self._send_clear_display()

            # Analyze story to infer location
            logger.info("Analyzing story with LLM")
            
            # Create base image prompt
            base_prompt = await self.prompt_builder.build_prompt(
                story.content
            )
            
            # Queue the new request
            request_id = self.queue_new_request(base_prompt)
            logger.info(f"Queued story-based request {request_id}")
            
            # Try to process the queued request
            await self.process_next_queued_request()
            
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
            logger.info(f"🎙️  TRANSCRIPT [{transcript.session_id}] {transcript.speaker_id}: '{transcript.content}' [State: {self.core_state.value}]")
            
            # Update accumulator (quick, synchronous operation)
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
            
            # Check if this is genuinely new content (not just re-processing old messages)
            unprocessed_messages = self.transcript_accumulator.messages[self.transcript_accumulator.processed_count:]
            user_messages_count = len([msg for msg in unprocessed_messages 
                                     if msg.speaker_id.lower() not in AGENTS])
            
            logger.debug(f"🔍 Analysis: {len(unprocessed_messages)} unprocessed messages, {user_messages_count} from users")
            
            if user_messages_count < 1:  # Wait for at least 1 new user message
                logger.debug("⏸️  Not enough new user content, waiting for more transcripts")
                return
            
            # *** CRITICAL FIX: Start background processing but return immediately ***
            logger.debug("🚀 SCHEDULING background transcript processing - handler will return immediately")
            
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
            logger.debug("🔬 BACKGROUND PROCESSING: Starting transcript processing")
            
            # Format full conversation context for LLM
            full_context = self._format_transcript_context()
            logger.debug(f"🤖 Querying LLM with {len(full_context)} chars of conversation context")
            
            # Ask LLM to decide if we should generate a prompt using the existing prompt builder
            try:
                logger.debug("🤖 BACKGROUND: Starting LLM call")
                image_prompt = await self.prompt_builder.build_prompt(full_context)
                logger.debug("✅ BACKGROUND: LLM call completed")
                
                # Check if this LLM processing request was cancelled while running
                if self.llm_processing_request and self.llm_processing_request.state == RequestState.CANCELLED:
                    logger.info("🚫 LLM processing request was cancelled while running - discarding result")
                    return
                
                logger.info("✅ LLM decided to generate prompt based on accumulated transcripts")
                
                # *** SMART INTERRUPTION LOGIC ***
                # Implements priority system: new requests take precedence
                # but "base images always complete" rule is respected
                
                if self.current_request or self.core_state != CoreState.LISTENING:
                    logger.info(f"🔄 Interrupting existing request/state ({self.core_state.value}) for new transcript-based image")
                    
                    if self.current_request and self.current_request.can_be_interrupted():
                        logger.info("🚫 Current request can be safely interrupted - cancelling completely")
                        await self.cancel_current_request()
                        
                    elif self.current_request and self.current_request.is_generating_base():
                        logger.info("📸 Base image generating - will complete but cancel future tiles")
                        await self.cancel_tile_generation()
                        
                    elif self.current_request and self.current_request.is_generating_tiles():
                        logger.info("🧩 Tiles generating - cancelling tiles, base already displayed")
                        await self.cancel_tile_generation()
                        
                    else:
                        logger.debug("🟢 Ready to process transcript - no interruption needed")
                else:
                    logger.debug("🟢 System ready - no current request to interrupt")
                
                # Queue the new request
                request_id = self.queue_new_request(image_prompt)
                logger.info(f"🖼️ Queued transcript-based request {request_id} (total queue: {len(self.request_queue)})")
                
                # Mark these messages as processed
                old_processed_count = self.transcript_accumulator.processed_count
                self.transcript_accumulator.processed_count = len(self.transcript_accumulator.messages)
                logger.info(f"📝 Marked {self.transcript_accumulator.processed_count - old_processed_count} messages as processed")
                
                # Try to process the queued request
                await self.process_next_queued_request()
                
                logger.debug("✅ BACKGROUND PROCESSING: Transcript processing completed successfully")
                
                # Mark LLM processing as completed
                if self.llm_processing_request:
                    self.llm_processing_request.state = RequestState.COMPLETED
                
            except InsufficientContentException:
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
            
            await self.cancel_current_request()
            
            # Send clear display message
            await self._send_clear_display()
            
            # Create direct image prompt (no LLM processing)
            base_prompt = ImagePrompt(
                prompt=prompt,
                negative_prompt=negative_prompt
            )
            
            logger.info(f"Created debug base prompt: {base_prompt.prompt}")
            logger.debug(f"Debug base negative: {base_prompt.negative_prompt}")
            
            # Queue the new request
            request_id = self.queue_new_request(base_prompt)
            logger.info(f"Queued debug request {request_id}")
            
            # Try to process the queued request
            await self.process_next_queued_request()

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
        
        # Track pending request with timestamp
        import time
        self.pending_image_requests[request_id] = ("base", time.time())
        logger.debug(f"🕒 Added base image request to pending: {request_id}")
        
        try:
            # Send render request with timeout protection
            logger.debug(f"🔬 SENDING WORK TO image_server: {request_id}")
            await self.zmq_service.send_work_to_worker("image_server", render_request)
            logger.info(f"✅ Sent base image request: {self.config.panorama.generated_width}x{self.config.panorama.generated_height}")
        except Exception as e:
            logger.error(f"🔥 FAILED to send image request to image_server: {e}")
            # Remove the pending request since it failed
            self.pending_image_requests.pop(request_id, None)
            
            # Return to listening state immediately on failure and try next queued request
            logger.warning("🔄 Image server unavailable - returning to listening state and processing queue")
            await self._transition_to_state(CoreState.LISTENING)
            # Process next queued request if available
            await self.process_next_queued_request()
            return
    
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
        
        # Mark base image as ready - update request state
        self.current_request.state = RequestState.BASE_READY
        
        # Check if we should proceed with tiles or if there are queued requests
        if self.request_queue:
            logger.info(f"📸 Base image complete but {len(self.request_queue)} requests queued - showing base only and processing next")
            # Don't generate tiles, just show the base and prepare for next request
            self.current_request.state = RequestState.COMPLETED
            await self._transition_to_state(CoreState.LISTENING)
            await self.process_next_queued_request()
        else:
            # Normal flow - transition to tile generation
            logger.info("📸 Base image complete - proceeding to tile generation")
            self.current_request.state = RequestState.WAITING_TILES
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
            strength=self.config.rendering.tile_strength # type: ignore (dynamic import)
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
            logger.info("🧩 All tiles completed, marking request as complete")
            self.current_request.state = RequestState.COMPLETED
            await self._transition_to_state(CoreState.LISTENING)
            
            # Process next queued request if any
            if self.request_queue:
                logger.info(f"🔄 Processing next queued request ({len(self.request_queue)} remaining)")
            await self.process_next_queued_request()
    
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
        
        # Much shorter timeout for faster recovery when image server is unavailable
        IMAGE_TIMEOUT = 20.0

        while self.running:
            current_time = time.time()
            
            # Debug logging every 10 seconds to show current state AND ZMQ status
            if hasattr(self, '_last_debug_log'):
                if current_time - self._last_debug_log > 10.0:
                    # Check ZMQ service status
                    zmq_status = "Unknown"
                    if hasattr(self, 'zmq_service') and self.zmq_service:
                        zmq_status = f"Running={getattr(self.zmq_service, 'running', 'Unknown')}"
                        if hasattr(self.zmq_service, 'subscriber') and self.zmq_service.subscriber:
                            zmq_status += f", Subscriber={getattr(self.zmq_service.subscriber, 'running', 'Unknown')}"
                    
                    logger.info(f"🔍 STATE DEBUG: {self.core_state.value}, current_request: {'Yes' if self.current_request else 'None'}, queue: {len(self.request_queue)}, pending: {len(self.pending_image_requests)}, ZMQ: {zmq_status}")
                    self._last_debug_log = current_time
            else:
                self._last_debug_log = current_time
            
            # Check for image request timeouts
            expired_requests = []
            for request_id, (request_type, timestamp) in self.pending_image_requests.items():
                if current_time - timestamp > IMAGE_TIMEOUT:
                    expired_requests.append((request_id, request_type))
            
            # Handle expired requests
            for request_id, request_type in expired_requests:
                logger.warning(f"⏰ Image request TIMEOUT: {request_type} (request_id: {request_id}) after {IMAGE_TIMEOUT}s - image_server likely unavailable")
                self.pending_image_requests.pop(request_id, None)
                
                # If it's a base image timeout and we're in base_image state, fall back to listening
                if request_type == "base" and self.core_state == CoreState.BASE_IMAGE:
                    logger.info("Base image timeout - returning to listening and processing queue")

                    # CRITICAL FIX: Reset processed count so new transcripts can trigger LLM
                    if hasattr(self, 'transcript_accumulator') and self.transcript_accumulator.processed_count > 0:
                        old_count = self.transcript_accumulator.processed_count
                        self.transcript_accumulator.processed_count = 0
                        logger.warning(f"🔄 RESET processed_count from {old_count} to 0 - new transcripts can now trigger LLM")

                    await self._transition_to_state(CoreState.LISTENING)
                    # Process next queued request if available
                    await self.process_next_queued_request()
                    
                    # Send a clear display message
                    display_message = DisplayMedia(
                        content_type=ContentType.CLEAR,
                        fade_in=10.0
                    )
                    await self.zmq_service.publish(display_message, MessageType.DISPLAY_MEDIA)
                
                # For tile timeouts, continue with other tiles or timeout completely
                elif request_type.startswith("tile_") and self.core_state == CoreState.TILES:
                    logger.warning(f"🧩 Tile timeout: {request_type}")
                    # If too many tiles timeout, give up and return to listening
                    remaining_tile_requests = [req_id for req_id, (req_type, _) in self.pending_image_requests.items() if req_type.startswith("tile_")]
                    if len(remaining_tile_requests) == 0:
                        logger.info("🔄 All tiles timed out - returning to listening and processing queue")
                        await self._transition_to_state(CoreState.LISTENING)
                        await self.process_next_queued_request()
            
            # Check for state-specific timeouts with shorter durations
            if self.core_state == CoreState.BASE_IMAGE:
                # If we've been in base_image state too long without pending requests, reset
                if not self.pending_image_requests and current_time > getattr(self, '_state_enter_time', 0) + IMAGE_TIMEOUT:
                    logger.warning("Base image state timeout with no pending requests - returning to listening")
                    await self._transition_to_state(CoreState.LISTENING)
                    await self.process_next_queued_request()
                    
            elif self.core_state == CoreState.TILES:
                # Similar logic for tiles state with slightly longer timeout
                if not self.pending_image_requests and current_time > getattr(self, '_state_enter_time', 0) + IMAGE_TIMEOUT * 1.5:
                    logger.warning("Tiles state timeout with no pending requests - returning to listening")
                    await self._transition_to_state(CoreState.LISTENING)
                    await self.process_next_queued_request()
            
            await self._sleep_if_running(1.0)  # Check every 1 second for faster recovery


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
