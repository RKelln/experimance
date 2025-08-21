from __future__ import annotations
import asyncio
import os
from typing import Any, Dict

from experimance_common.logger import setup_logging
from experimance_common.transcript_manager import TranscriptMessage, TranscriptMessageType
from agent import AgentServiceBase, SERVICE_TYPE
from agent.config import AgentServiceConfig
from agent.vision.reolink_detector import ReolinkDetector
from agent.tools import create_zmq_tool
from agent.backends.base import AgentBackendEvent

SERVICE_TYPE = "fire_agent"
logger = setup_logging(__name__, log_filename=f"{SERVICE_TYPE}.log")

class FireAgentService(AgentServiceBase):
    """Minimal voice-only agent for the Fire project using Reolink camera detection."""

    def __init__(self, config: AgentServiceConfig):
        super().__init__(config=config)

        self.audience_detector = None
        self.current_presence = None

    def register_project_handlers(self) -> None:
        """Register Fire-specific message handlers."""
        # No project-specific handlers yet for Fire
        logger.info("Fire agent handlers registered")
        return

    def post_backend_startup(self) -> None:
        """Perform actions after the backend has started."""
        if hasattr(self.current_backend, "transcript_manager") and self.current_backend.transcript_manager:
            self.current_backend.transcript_manager.register_async_callback(self._on_transcript_message)

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

        # start audience detector
        # Automatic discovery with optional known IP
        if self.config.vision.audience_detection_enabled and self.config.vision.reolink_enabled:
            # Get credentials from environment variables
            reolink_user = os.getenv("REOLINK_USER", self.config.vision.reolink_user)
            reolink_password = os.getenv("REOLINK_PASSWORD")
            
            if not reolink_password:
                logger.error("REOLINK_PASSWORD environment variable is required for camera authentication")
                raise ValueError("Reolink password not configured - set REOLINK_PASSWORD in .env file")
            
            # start reolink camera
            self.audience_detector = await ReolinkDetector.create_with_discovery(
                known_ip=self.config.vision.reolink_host,  # Optional for fastest path
                user=reolink_user,
                password=reolink_password,
                https=self.config.vision.reolink_https,
                channel=self.config.vision.reolink_channel,
                timeout=self.config.vision.reolink_timeout
            )
            # Start the detector (initialize session and authenticate)
            await self.audience_detector.start()
            logger.info("Reolink detector started successfully")
            
            # start polling camera
            self.add_task(self._audience_detection_loop())
        else:
            logger.warning("Audience detection is disabled, no audience monitoring will occur")
            # start voice chat backend, since it won't be started by audience detection
            logger.info("Starting conversation backend immediately since audience detection is disabled")
            await self._start_backend_for_conversation()

        logger.info("Fire agent conversation started")
    
    async def _stop_background_tasks(self):
        if self.audience_detector:
            await self.audience_detector.stop()

    async def _audience_detection_loop(self):
        """Continuously monitor for audience presence."""
        while self.running:
            # if presence detected, send signal and start conversation
            if self.audience_detector:
                presence = await self.audience_detector.check_audience_present()
                if presence != self.current_presence:
                    self.current_presence = presence
                    if presence:
                        logger.info("Audience detected, starting conversation backend")
                        await self._start_backend_for_conversation()
                    else:
                        logger.info("No audience detected, stopping conversation backend")
                        if self.current_backend:
                            await self.current_backend.graceful_shutdown()
            else:
                logger.warning("Audience detector not initialized, skipping detection loop")
            
            if not await self._sleep_if_running(self.config.vision.audience_detection_interval): break

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
            
            from experimance_common.schemas import TranscriptUpdate, MessageType
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