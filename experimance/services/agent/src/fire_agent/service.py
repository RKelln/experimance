from __future__ import annotations
import asyncio
import logging
import os
import time
from typing import Any, Dict, cast, Optional

from experimance_common.transcript_manager import TranscriptMessage, TranscriptMessageType
from experimance_common.schemas import AudiencePresent, MessageType  # type: ignore
from agent import AgentServiceBase
from .config import FireAgentServiceConfig
from agent.vision.reolink_detector import ReolinkDetector
from agent.vision.reolink_frame_detector import ReolinkFrameDetector
from agent.vision.yolo_person_detector import YOLO11PersonDetector
from agent.tools import create_zmq_tool
from agent.backends.base import AgentBackendEvent

# Use the original pythonosc for simple, non-blocking UDP sends
from pythonosc.udp_client import SimpleUDPClient

# Get the logger configured by the CLI system (avoids creating duplicate loggers)
logger = logging.getLogger(__name__)

class FireAgentService(AgentServiceBase):
    """Minimal voice-only agent for the Fire project using Reolink camera detection."""

    def __init__(self, config: FireAgentServiceConfig):
        # Let the CLI logging system handle all logging - no separate logger needed
        super().__init__(config=config)
        
        # Cast config to the correct type for linter
        self.config = cast(FireAgentServiceConfig, config)

        self.audience_detector = None
        self.current_presence = None

        # OSC client
        self.osc_client: Optional[SimpleUDPClient] = None
        self.osc_presence_address = self._full_osc_address(self.config.osc.presence_address)
        self.osc_bot_address = self._full_osc_address(self.config.osc.bot_speak_address)
        self.osc_person_address = self._full_osc_address(self.config.osc.person_speak_address)

    def _full_osc_address(self, suffix:str) -> str:
        if self.config.osc.address_prefix is None and self.config.osc.address_prefix == "":
            addr = f"/{suffix}/"
        else:
            addr =  f"/{self.config.osc.address_prefix}/{suffix}/"
        return addr.replace("//", "/")

    def register_project_handlers(self) -> None:
        """Register Fire-specific message handlers."""
        # No project-specific handlers yet for Fire
        logger.info("Fire agent handlers registered")
        return

    def post_backend_startup(self) -> None:
        """Perform actions after the backend has started."""
        if hasattr(self.current_backend, "transcript_manager") and self.current_backend.transcript_manager:  # type: ignore
            self.current_backend.transcript_manager.register_async_callback(self._on_transcript_message) # type: ignore

    def register_project_tools(self) -> None:
        """Register Fire-specific tools with the backend."""
        if not self.current_backend:
            logger.warning("No backend available for tool registration")
            return
        
        return # no tools for now

        logger.info("Registering Fire project tools")
        
        # Create display_location tool
        display_location_tool = create_zmq_tool(
            tool_name="display_location",
            message_type="StoryHeard",
            zmq_service=self.zmq_service,
            transcript_manager=self.current_backend.transcript_manager,
        )
        
        # Create update_location tool  
        update_location_tool = create_zmq_tool(
            tool_name="update_location", 
            message_type="UpdateLocation",
            zmq_service=self.zmq_service,
            transcript_manager=self.current_backend.transcript_manager,
        )
        
        # Define tool schemas for Fire project
        display_location_schema = {
            "type": "function",
            "function": {
                "name": "display_location",
                "description": "Call this when you have heard enough story to illustrate the first scene. Sends the story to the fire core service to generate visuals.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Optional custom content to send (if not provided, uses current transcript)"
                        }
                    },
                    "required": []
                }
            }
        }
        
        update_location_schema = {
            "type": "function",
            "function": {
                "name": "update_location",
                "description": "Call this when the guest adds new or clarifying visual details to their story. Updates the existing visual scene.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Optional custom content to send (if not provided, uses current transcript)"
                        },
                        "update_type": {
                            "type": "string",
                            "description": "Type of update (e.g., 'clarification', 'addition')",
                            "enum": ["clarification", "addition", "correction", "detail"]
                        }
                    },
                    "required": []
                }
            }
        }
        
        # Register tools with backend including schemas
        self.current_backend.register_tool(
            "display_location", 
            display_location_tool,
            "Call this when you have heard enough story to illustrate the first scene. Sends the story to the fire core service to generate visuals.",
            schema=display_location_schema
        )
        
        self.current_backend.register_tool(
            "update_location",
            update_location_tool, 
            "Call this when the guest adds new or clarifying visual details to their story. Updates the existing visual scene.",
            schema=update_location_schema
        )

        logger.info("Fire agent tools registered: display_location, update_location")
        logger.info("Fire project tools registered successfully")

    async def _initialize_background_tasks(self):
        """Initialize backend when audience is detected."""
        logger.info("Audience detected - starting Fire agent conversation")

        # Initialize OSC client if enabled
        if self.config.osc.enabled:
            try:
                # Simple UDP client - no connection needed, just create it
                self.osc_client = SimpleUDPClient(self.config.osc.host, self.config.osc.port)
                logger.info(f"OSC client initialized - {self.config.osc.host}:{self.config.osc.port}")
            except Exception as e:
                logger.error(f"Failed to initialize OSC client: {e}")
                self.osc_client = None

        # start audience detector
        # Automatic discovery with optional known IP
        if self.config.vision.audience_detection_enabled and self.config.reolink.enabled:
            # Get credentials from environment variables
            reolink_user = os.getenv("REOLINK_USER", self.config.reolink.user)
            reolink_password = os.getenv("REOLINK_PASSWORD")
            
            if not reolink_password:
                logger.error("REOLINK_PASSWORD environment variable is required for camera authentication")
                raise ValueError("Reolink password not configured - set REOLINK_PASSWORD in .env file")
            
            # Use hybrid detection: camera AI trigger + YOLO precision
            # Auto-discover camera if host not specified or use known IP as hint
            known_ip = self.config.reolink.host if self.config.reolink.host else None
            
            logger.info(f"Initializing hybrid Reolink detector with discovery (known_ip: {known_ip})")
            
            # Create YOLO configuration from reolink config  
            yolo_config = {}
            if hasattr(self.config.reolink, 'yolo') and self.config.reolink.yolo:
                # Convert Pydantic model to dict for YOLO detector
                yolo_config = self.config.reolink.yolo.dict()
                logger.debug(f"YOLO config loaded from reolink.yolo: confidence_threshold={self.config.reolink.yolo.confidence_threshold}")
            else:
                logger.warning(f"No YOLO config found in reolink config, using defaults")
            
            self.audience_detector = await ReolinkDetector.create_with_discovery(
                known_ip=known_ip,
                user=reolink_user,
                password=reolink_password,
                https=self.config.reolink.https,
                channel=self.config.reolink.channel,
                timeout=self.config.reolink.timeout,
                hysteresis_present=self.config.reolink.hysteresis_present,
                hysteresis_absent=self.config.reolink.hysteresis_absent,
                hybrid_mode=True,  # Enable hybrid camera AI + YOLO detection
                yolo_config=yolo_config,
                yolo_absent_threshold=5,  # YOLO absent readings before switching back to monitoring
                yolo_check_interval=1.0   # Seconds between YOLO checks in active mode
            )
            
            # Start the detector (initialize HTTP session and login)
            await self.audience_detector.start()
            
            logger.info("Hybrid Reolink detector initialized successfully")
            
            # start polling camera
            self.add_task(self._audience_detection_loop())

            # Send initial OSC signal (no audience present at startup)
            if self.osc_client:
                self._send_osc_presence(0)
        else: # no audience detection
            logger.warning("Audience detection is disabled, no audience monitoring will occur")
            # start voice chat backend, since it won't be started by audience detection
            logger.info("Starting conversation backend immediately since audience detection is disabled")
            await self.audience_detected(1)  # assume 1 person present

        # send other initial osc messages
        if self.osc_client:
            self._send_osc_speaking(self.config.osc.bot_speak_address, False)
            self._send_osc_speaking(self.config.osc.person_speak_address, False)

        logger.info("Fire agent conversation started")
    
    async def _stop_background_tasks(self):
        if self.audience_detector:
            await self.audience_detector.stop()
        
        # Send final absence signal before stopping OSC
        if self.osc_client:
            try:
                self._send_osc_presence(0)
                logger.debug("Final absence signal sent")
            except Exception as e:
                logger.error(f"Error sending final OSC signal: {e}")
            finally:
                # SimpleUDPClient doesn't need explicit closing
                self.osc_client = None
                logger.info("OSC client stopped")

    async def _publish_audience_present(self, person_count: int = 0):
        """Publish audience presence detection to fire_core."""
        if not self.running or not self.zmq_service.is_running:
            logger.debug(f"Skipping audience present publish (service running: {self.running}, zmq running: {self.zmq_service.is_running})")
            return
            
        message = AudiencePresent(
            person_count=person_count
        )
        try:
            await self.zmq_service.publish(message, MessageType.AUDIENCE_PRESENT)
            logger.debug(f"Published audience present: (people: {person_count})")
        except Exception as e:
            logger.error(f"Failed to publish audience present: {e}")

    async def _audience_detection_loop(self):
        """Continuously monitor for audience presence using hybrid detection."""
        last_person_count = None  # Track person count changes for vision messages
        
        # Send initial presence (0 people detected) when starting detection
        if self.running and self.zmq_service.is_running:
            await self._publish_audience_present(person_count=0)
        
        while self.running:
            # only check presence when no one is speaking so not to interrupt (audio) processing
            if self.audience_detector:
                if not self.any_speaking:
                    try:
                        # Use hybrid detection (camera AI + YOLO)
                        presence = await self.audience_detector.check_audience_present()

                        current_count = 1 if presence else 0 # simple presence is boolean, but we can detect number of people using yolo below

                        # Get detection stats for logging
                        stats = self.audience_detector.get_stats()
                        
                        # Send vision context messages when person count changes (only in YOLO mode)
                        if (stats.get('detection_mode') == 'active' and 
                            self.current_backend and 
                            'current_person_count' in stats):
                            
                            current_count = stats['current_person_count']
                            
                            # Only send vision message if person count has changed
                            if current_count != last_person_count:
                                # Format vision context message
                                if current_count == 0:
                                    vision_context = "<vision: No people detected>"
                                elif current_count == 1:
                                    vision_context = "<vision: One person detected>"
                                else:
                                    vision_context = f"<vision: {current_count} people detected>"
                                
                                # Send vision context to LLM
                                await self.current_backend.send_message(vision_context, speaker="system")
                                logger.debug(f"Vision update: {vision_context}")
                                
                                # Update tracking
                                last_person_count = current_count

                        # Debug logging with detection info
                        logger.debug(f"Hybrid detection - Presence: {presence}, "
                                f"Mode: {stats.get('detection_mode', 'simple')}, "
                                f"Checks: {stats['total_checks']}, "
                                f"Switches: {stats.get('mode_switches', 0)}")
                        
                        # Check for state changes
                        if presence != self.current_presence:
                            if presence:
                                logger.info("Hybrid detector: Audience detected")
                                await self.audience_detected(current_count)
                            else:
                                logger.info("Hybrid detector: Audience left")
                                await self._audience_left()
                                # Reset person count tracking when audience leaves
                                last_person_count = None
                            
                            # Update current presence after processing
                            self.current_presence = presence
                            
                    except Exception as e:
                        logger.error(f"Error in hybrid detection: {e}")
                        await asyncio.sleep(1.0)  # Brief pause on error
                        continue 
            else: # no detector
                logger.warning("Audience detector not initialized, skipping detection loop")
                await asyncio.sleep(5)

            if not await self._sleep_if_running(self.config.vision.audience_detection_interval): break

        # Send final presence (0 people detected) when stopping detection
        if self.running and self.zmq_service.is_running:
            await self._publish_audience_present(person_count=0)

    def _send_osc_presence(self, person_count: int) -> None:
        """Send OSC presence signal using simple UDP client (fire-and-forget)."""
        if not self.osc_client:
            return
        
        try:
            self.osc_client.send_message(self.osc_presence_address, person_count)

            logger.info(f"OSC presence signal sent: {self.osc_presence_address} = {person_count}")
        except Exception as e:
            logger.error(f"Failed to send OSC presence signal: {e}")

    async def _send_proactive_greeting(self) -> None:
        """Send a proactive greeting to visitors after a short delay."""
        try:
            if self.config.vision.audience_detection_enabled:
                # Wait for the configured greeting delay if we've seen the audience
                await asyncio.sleep(self.config.greeting_delay)

            # Trigger the backend to generate a proactive greeting
            greeting_prompt = self.config.greeting_prompt
            logger.info("Triggering proactive greeting from backend")
            if not self.current_backend:
                logger.warning("No backend available to send proactive greeting")
                # need to start up the backend if not already started
                await self._start_backend_for_conversation()

            if self.current_backend:
                # Use backend's trigger_response method to generate immediate LLM response
                await self.current_backend.trigger_response(greeting_prompt)
            else:
                logger.error("Failed to send proactive greeting - no backend available")
                
        except Exception as e:
            logger.error(f"Failed to send proactive greeting: {e}")

    async def audience_detected(self, person_count: int) -> None:
        """Handle audience detection event."""
        logger.info("Audience detected")
        # Send OSC signal as simple fire-and-forget
        self._send_osc_presence(person_count)

        # Publish presence to fire_core via ZMQ
        await self._publish_audience_present(person_count)

        await self._start_backend_for_conversation()
        
        # Send proactive greeting if enabled (run in background)
        if self.config.proactive_greeting_enabled:
            asyncio.create_task(self._send_proactive_greeting())

    async def _audience_left(self) -> None:
        """Handle audience left event."""
        logger.info("Audience left")
        # Send OSC signal as simple fire-and-forget  
        self._send_osc_presence(0)
        if self.current_backend:
            await self.current_backend.graceful_shutdown()

        # Publish presence to fire_core via ZMQ
        await self._publish_audience_present(0)

    async def _on_transcription_received(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle new transcription data."""
        logger.debug(f"Transcription received: {data}")

    async def _on_transcript_message(self, message: TranscriptMessage):
        """Handle new transcript message."""
        
        # Only process final (non-partial) user utterances
        if message.is_partial:
            logger.debug(f"Skipping transcript message - partial: {message.is_partial}, type: {message.message_type}")
            return
            
        # Get the latest user utterance content
        content = message.content.strip()
        if not content or len(content) <= 1:  # Skip very short utterances
            logger.debug(f"Skipping short utterance: '{content}'")
            return
            
        logger.debug(f"Streaming transcript to fire_core: '{content[:100]}{'...' if len(content) > 100 else ''}'")
        
        # Stream the transcript to fire_core for processing
        # Let fire_core decide when to generate prompts based on accumulated transcripts
        try:
            # Ensure we have a speaker_id (required field)
            speaker_id = message.speaker_id or "unknown"
            
            from experimance_common.schemas import TranscriptUpdate, MessageType # type: ignore (dynamic import)
            transcript_message = TranscriptUpdate(
                content=content,
                speaker_id=speaker_id,
                speaker_display_name=message.speaker_display_name,
                session_id=message.session_id,
                turn_id=message.turn_id,
                confidence=message.confidence,
                timestamp=str(message.timestamp),
                is_partial=message.is_partial,
                duration=message.duration
            )
            
            await self.zmq_service.publish(transcript_message, MessageType.TRANSCRIPT_UPDATE)
            logger.debug(f"Successfully streamed transcript to fire_core")
            
        except Exception as e:
            logger.error(f"Failed to stream transcript to fire_core: {e}")
            self.record_error(e, is_fatal=False, custom_message="Failed to stream transcript to fire_core")

    def _send_osc_speaking(self, speaker: str, is_speaking: bool) -> None:
        """Send OSC speaking signal using simple UDP client (fire-and-forget)."""
        if not self.osc_client:
            return
        
        try:
            # Send 1 for speaking, 0 for not speaking
            value = 1 if is_speaking else 0

            if speaker == "agent" or speaker == "bot":
                osc_speaking_address = self.osc_bot_address
            elif speaker == "user" or speaker == "person" or speaker == "human":
                osc_speaking_address = self.osc_person_address

                # FIXME: when human interrupts bot it doesnt send bot stopped speaking
                self.osc_client.send_message(self.osc_bot_address, 0)
            else:
                logger.warning(f"Unknown speaker: {speaker}")
                return
 
            # Simple fire-and-forget UDP send
            self.osc_client.send_message(osc_speaking_address, value)

            logger.info(f"OSC speaking signal sent: {osc_speaking_address} = {value} ({'speaking' if is_speaking else 'not speaking'})")
        except Exception as e:
            logger.error(f"Failed to send OSC speaking signal: {e}")

    async def _publish_speech_detected(self, is_speaking: bool, speaker_type: str = "agent"):
        """Publish speech detection for conversation tracking."""
        if not self.running:
            return
        # don't call super to avoid sending on ZMQ (no one is listening)
        # at this point we just want to send OSC signals
        self._send_osc_speaking(speaker=speaker_type, is_speaking=is_speaking)
