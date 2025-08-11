from __future__ import annotations

import asyncio
import copy
import random
import time
from typing import Any, Dict, Optional, cast

from experimance_agent.config import ExperimanceAgentServiceConfig
from experimance_common.logger import setup_logging
from experimance_common.schemas import MessageType
from experimance_common.zmq.config import MessageDataType
# Note: Biome and RequestBiome are dynamically imported where needed
from experimance_common.schemas import AudiencePresent  # type: ignore

from agent import AgentServiceBase, SERVICE_TYPE
from .deep_thoughts import DEEP_THOUGHTS

logger = setup_logging(__name__, log_filename=f"{SERVICE_TYPE}.log")

class ExperimanceAgentService(AgentServiceBase):
    """Experimance-specific specialization of the Agent service."""

    def __init__(self, config: ExperimanceAgentServiceConfig):
        super().__init__(config=config)
        
        # Cast config to the correct type for linter
        self.config = cast(ExperimanceAgentServiceConfig, config)

        # Deep thoughts state (per-conversation)
        self._available_deep_thoughts: Dict[str, Dict[str, str]] = {}
        self._conversation_start_time: Optional[float] = None
        self._last_speech_end_time: Optional[float] = None  # Track when speech last ended
        
        # Space-time tracking (to avoid duplicate projection messages)
        self._current_era: Optional[str] = None
        self._current_biome: Optional[str] = None

        # Vision components (will be initialized if enabled)
        self.webcam_manager = None
        self.vlm_processor = None
        self.audience_detector = None
        self._detector_profile = None  # Store loaded detector profile
        self._person_count = -1  # Track number of people currently detected

    def register_project_handlers(self) -> None:
        self.zmq_service.add_message_handler(MessageType.SPACE_TIME_UPDATE, self._handle_space_time_update)
        self.zmq_service.add_message_handler(MessageType.PRESENCE_STATUS, self._handle_audience_present)

    async def _initialize_background_tasks(self):
        # Initialize vision processing if enabled (this runs immediately)
        if self.config.vision.webcam_enabled:
            await self._initialize_vision()
        
        # Register background tasks
        if self.config.vision.audience_detection_enabled:
            self.add_task(self._audience_detection_loop())
        else:
            logger.warning("Audience detection is disabled, no audience monitoring will occur")
            # start voice chat backend, since it won't be started by audience detection
            logger.info("Starting conversation backend immediately since audience detection is disabled")
            await self._start_backend_for_conversation()

        if self.config.vision.vlm_enabled:
            self.add_task(self._vision_analysis_loop())
    
    async def _stop_background_tasks(self):
        # Clean up vision components
        if self.audience_detector:
            await self.audience_detector.stop()
        if self.vlm_processor:
            await self.vlm_processor.stop()
        if self.webcam_manager:
            await self.webcam_manager.stop()
    
    def _on_conversation_started(self):
        """Handle logic when a conversation is started."""
        # Reset deep thoughts for this conversation - make a deep copy
        self._available_deep_thoughts = copy.deepcopy(DEEP_THOUGHTS)

        # Reset space-time tracking to ensure first projection message is sent
        self._current_era = None
        self._current_biome = None

    # =========================================================================
    # ZMQ Message Handlers
    # =========================================================================
    
    async def _handle_space_time_update(self, message_data: MessageDataType):
        """Handle space-time update messages from core."""
        # Extract data from MessageDataType (could be dict or MessageBase)
        if isinstance(message_data, dict):
            data = message_data
        else:
            # It's a MessageBase object
            data = message_data.model_dump()
            
        era = data.get("era")
        biome = data.get("biome")
        logger.debug(f"Received space-time update: era={era}, biome={biome}")
        
        # Validate that we have valid era and biome values
        if not era or not biome:
            logger.warning(f"Invalid space-time update - era: {era}, biome: {biome}")
            return
        
        # Update agent context with current era/biome information
        if self.current_backend and self.current_backend.is_connected:
            # Only send space-time updates when in explorer node
            current_node = self.current_backend.get_current_node()
            if current_node != "explorer":
                logger.debug(f"Skipping space-time update - not in explorer node (current: {current_node})")
                return
            
            # Check if era or biome has changed since last update
            era_changed = era != self._current_era
            biome_changed = biome != self._current_biome
            
            if not era_changed and not biome_changed:
                logger.debug(f"No change in era/biome ({era}/{biome}), skipping projection message")
                return
            
            # Update our tracking state
            self._current_era = era
            self._current_biome = biome
            
            # Send projection update since something changed
            context_msg = f"<projection: currently displaying {self.era_to_description(biome, era)}.>"
            
            # Try to get a deep thought for this biome/era combination
            deep_thought = self._get_deep_thought(biome, era)
            if deep_thought:
                context_msg += f"\n<thought: {deep_thought}>"
                await self.current_backend.send_message(deep_thought, speaker="system", say_tts=True)
                logger.debug(f"Sent deep thought: {deep_thought[:50]}...")
            else:
                logger.debug("No deep thought retrieved - check debug logs for reason")

            logger.debug(f"Sent projection update for era/biome change: {era}/{biome}")
            await self.current_backend.send_message(context_msg, speaker="system")


    def era_to_description(self, biome: str, era: str) -> str:
        """Convert era string to human-readable description."""
        str_biome = str(biome).replace("_", " ").lower()
        era_descriptions = {
            "wilderness": f"a {str_biome} landscape untouched by humans",
            "pre_industrial": f"a {str_biome} landscape with an ancient civilization",
            "early_industrial": f"a {str_biome} landscape as industry begins to emerge",
            "late_industrial": f"a {str_biome} landscape dominated by industry",
            "modern": f"a {str_biome} landscape in late 20th century",
            "current": f"a {str_biome} landscape in the present day",
            "future": f"a {str_biome} landscape in the near future if things go well",
            "dystopia": f"a {str_biome} landscape in a dystopian future where things have gone wrong",
            "ruins": f"a future {str_biome} landscape with remnants of our civilization",
        }
        if era is None or era == "":
            logger.warning(f"Received empty era in space-time update: {era}")
        if era not in era_descriptions:
            logger.warning(f"Unknown era '{era}' in space-time update, using default description")
            # Fallback to a generic description if era is unknown
        return era_descriptions.get(era, f"a {str_biome} landscape")
    
    def _get_deep_thought(self, biome: str, era: str) -> Optional[str]:
        """Get a deep thought for the given biome and era, removing it from available thoughts.
        
        Returns None if no thought available or if it's too early in the conversation.
        """
        logger.info(f"Retrieving deep thought for {era}/{biome}...")

        # Check if we're in a conversation and if enough time has passed
        if not self._conversation_start_time:
            # If no conversation start time is set, return None
            logger.debug("No conversation start time set, skipping deep thought")
            return None
        
        now = time.time()
        time_elapsed = now - self._conversation_start_time
        if time_elapsed < self.config.deep_thoughts_min_delay:
            logger.debug(f"Too soon: {time_elapsed:.2f}s since start, waiting for {self.config.deep_thoughts_min_delay}s")
            return None
        
        # Check for quiet period - only share deep thoughts during reflection time
        if not self._last_speech_end_time:
            logger.debug("No quiet period detected (speech may be active), skipping deep thought")
            return None

        # Calculate quiet time since last speech ended
        quiet_elapsed = now - self._last_speech_end_time
        if quiet_elapsed < self.config.deep_thoughts_quiet_delay:
            logger.debug(f"Not quiet long enough: {quiet_elapsed:.2f}s since speech ended, waiting for {self.config.deep_thoughts_quiet_delay}s")
            return None
        
        # Check if we have thoughts available for this era
        if era not in self._available_deep_thoughts:
            logger.debug(f"No deep thoughts available for era: {era}")
            return None
            
        # Check if we have thoughts available for this biome within the era
        era_thoughts = self._available_deep_thoughts[era]
        if biome not in era_thoughts:
            logger.debug(f"No deep thoughts available for biome: {biome} in era: {era}")
            return None

        # random chance to not give a thought
        if random.random() > self.config.deep_thoughts_chance:
            logger.debug(f"Skipping deep thought for {era}/{biome} due to random chance")
            return None
        
        # Get the thought and remove it from available thoughts
        thought = era_thoughts.pop(biome)
        
        # Clean up empty era if no more thoughts remain
        if not era_thoughts:
            del self._available_deep_thoughts[era]
        
        logger.info(f"Retrieved deep thought after {quiet_elapsed:.1f}s of quiet time: {thought[:50]}...")
        return thought
    
    async def _handle_audience_present(self, message_data: MessageDataType):
        """Handle audience presence updates."""
        # Don't process presence updates if service is shutting down
        if not self.running: return
            
        logger.debug(f"Audience presence update received: {message_data}")

        audience_present = message_data.get("present", True)
        idle = message_data.get("idle", False)
        
        # First, handle backend startup for anyone present (regardless of our person count)
        if audience_present:
            # if we are inactive then we need to start the backend
            if not self.current_backend or not self.current_backend.is_connected:
                logger.info("Audience detected, starting conversation backend...")
                await self._start_backend_for_conversation()
                return  # Early return to avoid flow transitions during startup
        
        # Then handle flow transitions only if we have a backend running
        if not self.current_backend or not self.current_backend.is_connected:
            return
            
        # double check that we don't have a newer person count internally, no transition if people detected
        # NOTE: self._person_count starts at -1, so if we have no one detected, it will be 0
        if self._person_count == 0 and (not audience_present or idle):

            # Handle flow transitions based on presence and idle state
            if hasattr(self.current_backend, 'transition_to_node'):
                current_node = self.current_backend.get_current_node()
                if current_node is None:
                    logger.warning("Current node is None, cannot transition")
                    return
                
                # idle overrides not being present
                if idle and current_node != "goodbye":
                    logger.info("Idle state detected, ending conversation gracefully")
                    await self.current_backend.graceful_shutdown()

    # =========================================================================
    # Vision Processing Initialization
    # =========================================================================
    
    async def _initialize_vision(self):
        """Initialize vision processing components."""
        try:
            from agent.vision import WebcamManager, VLMProcessor, AudienceDetector, CPUAudienceDetector, load_profile
            
            # Load detector profile first to get camera settings
            detector_profile = None
            if hasattr(self.config.vision, 'detector_profile') and self.config.vision.detector_profile:
                try:
                    detector_profile = load_profile(self.config.vision.detector_profile)
                    logger.info(f"Loaded detector profile: {detector_profile.name}")
                except Exception as e:
                    logger.warning(f"Failed to load detector profile '{self.config.vision.detector_profile}': {e}")
            
            # Initialize webcam manager
            self.webcam_manager = WebcamManager(self.config.vision)
            await self.webcam_manager.start()
            
            # Apply camera profile settings if available
            if detector_profile and detector_profile.camera:
                logger.info(f"Applying camera profile: {detector_profile.camera.name}")
                success = self.webcam_manager.apply_camera_profile(detector_profile.camera)
                if success:
                    logger.info(f"Successfully applied camera profile: {detector_profile.camera.name}")
                else:
                    logger.warning(f"Camera profile '{detector_profile.camera.name}' had limited success")
            
            # Initialize VLM processor if enabled and needed
            if (self.config.vision.vlm_enabled and 
                self.config.vision.detection_method in ["vlm", "hybrid"]):
                self.vlm_processor = VLMProcessor(self.config.vision)
                await self.vlm_processor.start()
            
            # Initialize appropriate audience detector based on detection method
            detection_method = self.config.vision.detection_method
            
            if detection_method == "cpu":
                # Use fast CPU-only detection with intelligent face prioritization
                self.audience_detector = CPUAudienceDetector(self.config.vision)
                await self.audience_detector.start()
                
                # Set performance mode
                performance_mode = self.config.vision.cpu_performance_mode
                self.audience_detector.set_performance_mode(performance_mode)
                
                logger.info(f"Using CPU audience detection (mode: {performance_mode}) with built-in intelligence")
                
            elif detection_method in ["vlm", "hybrid"]:
                # Use VLM-based detection (original detector)
                self.audience_detector = AudienceDetector(self.config.vision)
                await self.audience_detector.start()
                
                logger.info(f"Using {detection_method} audience detection with VLM")
            
            # Store the detector profile for potential use in debug info
            if detector_profile:
                self._detector_profile = detector_profile
            
            logger.info("Vision processing initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vision processing: {e}")
            # Continue without vision processing
            self.webcam_manager = None
            self.vlm_processor = None
            self.audience_detector = None

    # =========================================================================
    # Background Task Loops
    # =========================================================================
    
    async def _audience_detection_loop(self):
        """Monitor audience presence and publish to presence system."""
        # Adaptive timing: faster checks when detector reports instability
        normal_interval = self.config.vision.audience_detection_interval
        # frame_duration = 1.0 / self.config.vision.webcam_fps
        # rapid_interval = max(frame_duration, normal_interval / 5)  # 5x faster, but at least 1 frame
        rapid_interval = max(0.2, 0.8 / self.config.vision.stable_readings_required)  # try to get stable readings every second
        current_interval = normal_interval
        last_publish_time = time.monotonic() # used to ensure we report at least every report_min_interval seconds
        report_min_interval = 30.0  # used so if core service restarts we update it every 30 seconds regardless of changes

        # send 0 people detected to start
        if self.running and self.zmq_service.is_running:
            await self._publish_audience_present(person_count=0)

        while self.running:
            try:
                if (self.config.vision.audience_detection_enabled and 
                    self.audience_detector and 
                    self.webcam_manager and 
                    self.webcam_manager.is_active):
                    
                    # Capture frame for analysis
                    frame = await self.webcam_manager.capture_frame()
                    if frame is not None:
                        # Let the detector handle all the complex logic
                        detection_result = await self.audience_detector.detect_audience(frame)
                        
                        if detection_result.get("success", False):
                            # Use detector's stability assessment for adaptive timing
                            is_stable = detection_result.get("stable", False)
                            
                            # Publish when detector reports stable state
                            if is_stable:
                                current_interval = normal_interval
                                person_count = detection_result.get("person_count", 0)
                                now = time.monotonic()
                                # Only publish if we have a significant change or it's time to report
                                if self._person_count != person_count or now - last_publish_time > report_min_interval:
                                    if self._person_count != person_count:
                                        old_count = self._person_count
                                        self._person_count = person_count
                                        
                                        # Track audience absence during cooldown for early cooldown end
                                        if (self.conversation_end_time is not None and 
                                            self.config.cancel_cooldown_on_absence and
                                            old_count > 0 and person_count == 0):
                                            self.audience_went_to_zero_after_conversation = True
                                            logger.info("Audience left during cooldown - cooldown will end when someone returns")
                                        
                                        # If audience returns after leaving during cooldown, end cooldown early
                                        if (self.conversation_end_time is not None and 
                                            self.config.cancel_cooldown_on_absence and
                                            self.audience_went_to_zero_after_conversation and
                                            old_count == 0 and person_count > 0):
                                            logger.info("Audience returned after leaving - ending cooldown early")
                                            # The cooldown check will handle this automatically
                                        
                                        # update the LLM backend
                                        await self._backend_update_person_count(person_count)

                                    last_publish_time = now
                                    if self.running and self.zmq_service.is_running:
                                        await self._publish_audience_present(
                                            person_count=person_count
                                        )
                                    
                                    # Check if cooldown just ended and we should start a conversation
                                    if (person_count > 0 and 
                                        not self.current_backend and 
                                        not self._is_in_conversation_cooldown()):
                                        logger.info("Cooldown ended with audience present - starting conversation")
                                        await self._start_backend_for_conversation()

                            else:
                                current_interval = rapid_interval
                                logger.debug(f"Detector reports instability, using rapid checks ({rapid_interval}s)")

                        else:
                            logger.warning(f"Audience detection failed: {detection_result.get('error', 'Unknown error')}")
                            # On error, use normal interval
                            current_interval = normal_interval
                
                if not await self._sleep_if_running(current_interval):
                    break
                
            except asyncio.CancelledError:
                # Task was cancelled during shutdown - exit gracefully
                logger.debug("Audience detection loop cancelled")
                
                break
            except Exception as e:
                self.record_error(e, is_fatal=False)
                # Reset to normal interval on error and back off more
                current_interval = normal_interval
                await self._sleep_if_running(10.0)  # Back off on error

        # report 0 on shutdown - but only if ZMQ service is still running
        if self.running and self.zmq_service.is_running:
            await self._publish_audience_present(person_count=0) 

    async def _vision_analysis_loop(self):
        """Perform periodic vision analysis for scene understanding."""

        while self.running:
            try:
                if (self.config.vision.vlm_enabled and 
                    self.vlm_processor and 
                    self.vlm_processor.is_loaded and
                    self.webcam_manager and
                    self.webcam_manager.is_active):
                    
                    # Capture frame for VLM analysis
                    frame = await self.webcam_manager.capture_frame()
                    if frame is not None:
                        # Preprocess frame for VLM
                        rgb_frame = self.webcam_manager.preprocess_for_vlm(frame)
                        
                        # Perform scene analysis
                        analysis = await self.vlm_processor.analyze_scene(rgb_frame, "scene_description")
                        
                        if analysis.get("success", False):
                            await self._process_vision_analysis(analysis)
                        else:
                            logger.warning(f"VLM analysis failed: {analysis.get('error', 'Unknown error')}")
                
                await self._sleep_if_running(self.config.vision.vlm_analysis_interval)
                
            except asyncio.CancelledError:
                # Task was cancelled during shutdown - exit gracefully
                logger.debug("Vision analysis loop cancelled")
                break
            except Exception as e:
                self.record_error(e, is_fatal=False)
                await self._sleep_if_running(30.0)  # Back off on error
    
    async def _process_vision_analysis(self, analysis: Dict[str, Any]):
        """Process VLM analysis results and update agent context."""
        try:
            description = analysis.get("description", "")
            analysis_type = analysis.get("analysis_type", "unknown")
            
            logger.debug(f"VLM Analysis ({analysis_type}): {description}")
            
            # Send scene context to the agent backend if conversation is active
            if (self.current_backend and 
                self.current_backend.is_connected and 
                self.is_conversation_active and 
                description.strip()):
                
                # Format as system message for context
                context_msg = f"<vision: {description}>"
                await self.current_backend.send_message(context_msg, speaker="system")
                
                logger.info(f"Updated agent context with scene analysis: {description[:100]}...")
                
        except Exception as e:
            logger.error(f"Failed to process vision analysis: {e}")

    # =========================================================================
    # Publishing Methods
    # =========================================================================
    
    async def _publish_audience_present(self, person_count: int = 0):
        """Publish audience presence detection."""
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

    async def _publish_request_biome(self, requested_biome: str):
        """Publish biome change request based on conversation context."""
        try:
            # Convert string to Biome enum if needed
            if isinstance(requested_biome, str):
                # Import dynamically to avoid linter errors
                from experimance_common.schemas import Biome  # type: ignore
                # Handle both underscore and space formats
                biome_value = requested_biome.lower().replace(" ", "_")
                biome_enum = Biome(biome_value)
            else:
                biome_enum = requested_biome
                
            # Create RequestBiome message - using dynamic import to avoid linter errors
            from experimance_common.schemas import RequestBiome, MessageType  # type: ignore
            message = RequestBiome(
                biome=biome_enum.value  # Field name expected by core service
            )
            
            if not self.running or not self.zmq_service.is_running:
                logger.debug(f"Skipping biome request publish (service running: {self.running}, zmq running: {self.zmq_service.is_running})")
                return
                
            await self.zmq_service.publish(message.model_dump(), MessageType.REQUEST_BIOME)  # type: ignore
            logger.info(f"Published biome request: {biome_enum.value}")
        except ValueError as e:
            logger.warning(f"Invalid biome request '{requested_biome}': {e}")
        except Exception as e:
            logger.error(f"Failed to publish biome request: {e}")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _backend_update_person_count(self, person_count: int):
        """Send audience presence update to the LLM service."""
        if self.current_backend and self.current_backend.is_connected:
            text = ""
            if person_count == 0:
                text = "No people"
            elif person_count == 1:
                text = "One person"
            elif person_count > 1:
                text = f"{person_count} people"
            await self.current_backend.send_message(
                f"<vision: {text} detected>",
                speaker="system"
            )