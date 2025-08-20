from __future__ import annotations
import asyncio
import os

from experimance_common.logger import setup_logging
from agent import AgentServiceBase, SERVICE_TYPE
from agent.config import AgentServiceConfig
from agent.vision.reolink_detector import ReolinkDetector

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