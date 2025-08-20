from __future__ import annotations
import asyncio
import os

from experimance_common.logger import setup_logging
from agent import AgentServiceBase, SERVICE_TYPE
from agent.config import AgentServiceConfig
from agent.vision.reolink_detector import ReolinkDetector
from agent.tools import create_zmq_tool

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

    def register_project_tools(self) -> None:
        """Register Fire-specific tools with the backend."""
        if not self.current_backend:
            logger.warning("No backend available for tool registration")
            return
        
        logger.info("Registering Fire project tools")
        
        # Create display_location tool
        display_location_tool = create_zmq_tool(
            tool_name="display_location",
            message_type="StoryHeard",
            zmq_service=self.zmq_service,
            transcript_manager=self.current_backend.transcript_manager,
            content_transformer=self._clean_story_content
        )
        
        # Create update_location tool  
        update_location_tool = create_zmq_tool(
            tool_name="update_location", 
            message_type="UpdateLocation",
            zmq_service=self.zmq_service,
            transcript_manager=self.current_backend.transcript_manager,
            content_transformer=self._clean_story_content
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

    def _clean_story_content(self, content: str) -> str:
        """
        Clean and prepare story content for the fire core service.
        
        Args:
            content: Raw transcript content
            
        Returns:
            Cleaned content suitable for visual generation
        """
        # Remove excessive whitespace
        content = " ".join(content.split())
        
        # Remove common filler words and conversational artifacts
        filler_words = [
            "um", "uh", "like", "you know", "I mean", "actually", 
            "basically", "literally", "so", "well", "okay", "alright"
        ]
        
        words = content.split()
        filtered_words = []
        
        for word in words:
            # Remove filler words (case insensitive)
            clean_word = word.lower().strip(".,!?;:")
            if clean_word not in filler_words:
                filtered_words.append(word)
        
        cleaned_content = " ".join(filtered_words)
        
        # Ensure minimum content length
        if len(cleaned_content.strip()) < 10:
            return content  # Return original if cleaning removed too much
        
        return cleaned_content

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

    async def _initialize_backend(self):
        """Initialize the selected agent backend."""
        backend_name = getattr(self.config, 'agent_backend', 'pipecat').lower()
        
        logger.info(f"Initializing {backend_name} backend for Fire agent...")

        try:
            if backend_name == "pipecat":
                from agent.backends.pipecat_backend import PipecatBackend
                self.current_backend = PipecatBackend(self.config)
                
                # Register project-specific tools with the backend
                self.register_project_tools()
                
                await self.current_backend.start()
                logger.info("Pipecat backend started successfully")
            else:
                raise ValueError(f"Unsupported backend: {backend_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {backend_name} backend: {e}")
            raise

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
