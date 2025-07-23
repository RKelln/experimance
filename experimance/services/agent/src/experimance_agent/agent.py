"""
Main Agent Service for the Experimance interactive art installation.

This service handles speech-to-speech conversation with the audience, integrates webcam feeds 
for audience detection and scene understanding, and provides tool calling capabilities for 
controlling other services.
"""

import asyncio
import time
import json
import logging
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path

from experimance_common.base_service import BaseService
from experimance_common.health import HealthStatus
from experimance_common.zmq.services import PubSubService
from experimance_common.schemas import (
    RequestBiome, AudiencePresent, SpeechDetected,
    DisplayText, RemoveText, MessageType, Biome
)
from experimance_common.constants import TICK, DEFAULT_PORTS
from experimance_common.zmq.config import MessageDataType

from .config import AgentServiceConfig
from .backends.base import AgentBackend, AgentBackendEvent, ConversationTurn, ToolCall

logger = logging.getLogger(__name__)


class AgentService(BaseService):
    """
    Main agent service that orchestrates conversation AI, vision processing, and tool integration.
    
    This service acts as a coordinator between various components:
    - Agent backends (Pipecat, etc.)
    - Vision processing (webcam, audience detection, VLM)
    - Transcript management and display
    - Tool calling for system control
    """
    
    def __init__(self, config: AgentServiceConfig):
        super().__init__(service_type="agent", service_name=config.service_name)
        
        # Store immutable config
        self.config = config
        
        # Runtime state
        self.current_backend: Optional[AgentBackend] = None
        self.is_conversation_active = False
        self.agent_speaking = False
        
        # Display text tracking (for transcript display)
        self.displayed_text_ids: set[str] = set()
        
        # ZMQ service for communication
        self.zmq_service = PubSubService(config.zmq)
        
        # Vision components (will be initialized if enabled)
        self.webcam_manager = None
        self.vlm_processor = None
        self.audience_detector = None
        self._detector_profile = None  # Store loaded detector profile
        self._person_count = 0  # Track number of people currently detected
        
        # Transcript display handler (will be initialized if enabled)
        self.transcript_display_handler = None
        
    async def start(self):
        """Initialize and start the agent service."""
        logger.info(f"Starting {self.service_name} with backend: {self.config.agent_backend}")
        
        # Set up message handlers before starting ZMQ service
        self.zmq_service.add_message_handler(MessageType.SPACE_TIME_UPDATE, self._handle_space_time_update)
        
        # Start ZMQ service
        await self.zmq_service.start()
        
        # Initialize agent backend
        await self._initialize_backend()
        
        # Initialize vision processing if enabled
        if self.config.vision.webcam_enabled:
            await self._initialize_vision()
        
        # Initialize transcript handling if enabled
        if self.config.transcript.display_transcripts:
            await self._initialize_transcript_display_handler()
        
        # Register background tasks
        #self.add_task(self._conversation_monitor_loop())
        self.add_task(self._audience_detection_loop())
        if self.config.vision.vlm_enabled:
            self.add_task(self._vision_analysis_loop())
        
        # ALWAYS call super().start() LAST - this starts health monitoring automatically
        await super().start()
        
        logger.info(f"{self.service_name} started successfully")
    
    async def stop(self):
        """Clean up resources and stop the agent service."""
        logger.info(f"Stopping {self.service_name}")
        
        # ALWAYS call super().stop() FIRST - this stops health monitoring automatically
        await super().stop()
        
        # Stop agent backend
        if self.current_backend:
            await self.current_backend.stop()
        
        # Stop ZMQ service
        await self.zmq_service.stop()
        
        # Clean up vision components
        if self.audience_detector:
            await self.audience_detector.stop()
        
        if self.vlm_processor:
            await self.vlm_processor.stop()
            
        if self.webcam_manager:
            await self.webcam_manager.stop()
        
        # Clear any displayed text
        await self._clear_all_displayed_text()
        
        logger.info(f"{self.service_name} stopped")
    
    # =========================================================================
    # Backend Management
    # =========================================================================
    
    async def _initialize_backend(self):
        """Initialize the selected agent backend."""
        backend_name = self.config.agent_backend.lower()
        
        try:
            if backend_name == "pipecat":
                from .backends.pipecat_backend import PipecatBackend
                self.current_backend = PipecatBackend(self.config)
            else:
                raise ValueError(f"Unknown agent backend: {backend_name}")
            
            # Register event callbacks and tools only if backend is available
            if self.current_backend is not None:
                backend = self.current_backend  # For type checking
                # Register event callbacks
                backend.add_event_callback(
                    AgentBackendEvent.CONVERSATION_STARTED, self._on_conversation_started
                )
                backend.add_event_callback(
                    AgentBackendEvent.CONVERSATION_ENDED, self._on_conversation_ended
                )
                backend.add_event_callback(
                    AgentBackendEvent.SPEECH_DETECTED, self._on_speech_detected
                )
                backend.add_event_callback(
                    AgentBackendEvent.SPEECH_ENDED, self._on_speech_ended
                )
                backend.add_event_callback(
                    AgentBackendEvent.BOT_STARTED_SPEAKING, self._on_speech_detected
                )
                backend.add_event_callback(
                    AgentBackendEvent.BOT_STOPPED_SPEAKING, self._on_speech_ended
                )
                backend.add_event_callback(
                    AgentBackendEvent.TRANSCRIPTION_RECEIVED, self._on_transcription_received
                )
                backend.add_event_callback(
                    AgentBackendEvent.TOOL_CALLED, self._on_tool_called
                )
                
                # Register available tools
                await self._register_tools()
                
                # Start the backend
                await backend.start()  # type: ignore
                
                # Add pipeline task to service task management if available
                if hasattr(backend, 'get_pipeline_task'):
                    pipeline_task = backend.get_pipeline_task()
                    if pipeline_task:
                        self.add_task(pipeline_task)
                        logger.info("Added pipeline task to service task management")
                
                # Debug output
                if hasattr(backend, 'get_debug_status'):
                    print(json.dumps(backend.get_debug_status(), indent=2))
                                
                logger.info(f"Successfully initialized {backend_name} backend")
            else:
                logger.info(f"Agent service started in placeholder mode (no {backend_name} backend)")
            
        except Exception as e:
            self.record_error(e, is_fatal=True)
            raise
    
    async def _register_tools(self):
        """Register available tools with the agent backend."""
        if not self.current_backend:
            return
        
        # Register biome suggestion tool
        if self.config.biome_suggestions_enabled:
            self.current_backend.register_tool(
                "suggest_biome",
                self._suggest_biome_tool,
                "Suggest a new biome for the installation based on conversation context"
            )
        
        # Register audience interaction tools
        self.current_backend.register_tool(
            "get_audience_status",
            self._get_audience_status_tool,
            "Check if audience members are currently present"
        )
        
        logger.info("Registered tools with agent backend")
    
    # =========================================================================
    # Vision Processing Initialization
    # =========================================================================
    
    async def _initialize_vision(self):
        """Initialize vision processing components."""
        try:
            from .vision import WebcamManager, VLMProcessor, AudienceDetector, CPUAudienceDetector, load_profile
            
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
    
    async def _initialize_transcript_display_handler(self):
        """Initialize transcript display handling."""
        if not self.current_backend:
            logger.warning("Cannot initialize transcript display handler: no backend available")
            return
            
        # Register display callback with the backend's transcript manager
        self.current_backend.transcript_manager.add_display_callback(
            self._display_transcript_callback
        )
        
        logger.info("Transcript display handler initialized")
    
    async def _display_transcript_callback(self, message):
        """Callback for displaying transcript messages."""
        from experimance_common.transcript_manager import TranscriptMessageType
        
        # Only display speech messages, not system messages or tool calls
        if message.message_type not in [TranscriptMessageType.USER_SPEECH, TranscriptMessageType.AGENT_RESPONSE]:
            return
            
        # Skip partial messages for display
        if message.is_partial:
            return
        
        # Display the transcript text
        await self._display_transcript_text(message.content, message.display_name)
    
    # =========================================================================
    # Background Task Loops
    # =========================================================================
    
    async def _conversation_monitor_loop(self):
        """Monitor conversation state and manage system interactions."""
        while self.running:
            try:
                if self.current_backend and self.current_backend.is_connected:
                    # Check if conversation is active based on recent messages
                    if hasattr(self.current_backend, 'transcript_manager'):
                        recent_messages = self.current_backend.transcript_manager.get_messages(limit=5)
                        if recent_messages:
                            # Update conversation active state based on recent activity
                            last_message_time = recent_messages[-1].timestamp
                            import time
                            time_since_last = time.time() - last_message_time
                            self.is_conversation_active = time_since_last < 30.0  # Active if message in last 30 seconds
                
                await self._sleep_if_running(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.record_error(e, is_fatal=False)
                await self._sleep_if_running(10.0)  # Back off on error
    
    async def _audience_detection_loop(self):
        """Monitor audience presence and publish to presence system."""
        # Adaptive timing: faster checks when detector reports instability
        normal_interval = self.config.vision.audience_detection_interval
        frame_duration = 1.0 / self.config.vision.webcam_fps
        rapid_interval = max(frame_duration, normal_interval / 5)  # 5x faster, but at least 1 frame
        current_interval = normal_interval
        last_publish_time = time.monotonic() # used to ensure we report at least every report_min_interval seconds
        report_min_interval = 30.0  # used so if core service restarts we update it every 30 seconds regardless of changes

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
                                if self._person_count != person_count or now - last_publish_time > report_min_interval:
                                    last_publish_time = now
                                    self._person_count = person_count
                                    await self._publish_audience_present(
                                        person_count=person_count
                                    )
                                    
                            else:
                                current_interval = rapid_interval
                                logger.debug(f"Detector reports instability, using rapid checks ({rapid_interval}s)")

                        else:
                            logger.warning(f"Audience detection failed: {detection_result.get('error', 'Unknown error')}")
                            # On error, use normal interval
                            current_interval = normal_interval
                
                await self._sleep_if_running(current_interval)
                
            except Exception as e:
                self.record_error(e, is_fatal=False)
                # Reset to normal interval on error and back off more
                current_interval = normal_interval
                await self._sleep_if_running(10.0)  # Back off on error    async def _vision_analysis_loop(self):
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
                context_msg = f"<System: seen by the installation camera: {description}>"
                await self.current_backend.send_message(context_msg, speaker="system")
                
                logger.info(f"Updated agent context with scene analysis: {description[:100]}...")
                
        except Exception as e:
            logger.error(f"Failed to process vision analysis: {e}")
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    async def _on_conversation_started(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle conversation started event."""
        self.is_conversation_active = True
        logger.info("Conversation started with audience")
        
        # Optionally send welcome message
        # if self.current_backend:
        #     await self.current_backend.send_message(
        #         "Hello! I'm the spirit of this installation. Feel free to interact with the sand while we talk.",
        #         speaker="system"
        #     )
    
    async def _on_conversation_ended(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle conversation ended event."""
        self.is_conversation_active = False
        logger.info("Conversation ended")
        
        # Clear displayed transcripts after a delay
        await asyncio.sleep(5.0)
        await self._clear_all_displayed_text()
    
    async def _on_speech_detected(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle speech detection event from pipecat backend."""
        logger.debug("_on_speech_detected: event=%s, data=%s", event, data)

        speaker = data.get("speaker", "unknown") if data else "unknown"
        
        # Update internal state
        if speaker == "agent":
            self.agent_speaking = True
        
        # Publish to presence system for conversation tracking
        if self.config.speech_detection_enabled:
            speaker_type = "agent" if speaker == "agent" else "human"
            await self._publish_speech_detected(is_speaking=True, speaker_type=speaker_type)
            logger.debug(f"Speech detected: {speaker_type}")
    
    async def _on_speech_ended(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle speech ended event from pipecat backend."""
        logger.debug("_on_speech_ended: event=%s, data=%s", event, data)
        speaker = data.get("speaker", "unknown") if data else "unknown"
        
        # Update internal state
        if speaker == "agent":
            self.agent_speaking = False
        
        # Publish to presence system for conversation tracking
        if self.config.speech_detection_enabled:
            speaker_type = "agent" if speaker == "agent" else "human"
            await self._publish_speech_detected(is_speaking=False, speaker_type=speaker_type)
            logger.debug(f"Speech ended: {speaker_type}")
    
    async def _on_transcription_received(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle new transcription data."""
        if data and self.config.transcript.display_transcripts:
            content = data.get("content", "")
            speaker = data.get("speaker", "unknown")
            
            if content.strip():
                await self._display_transcript_text(content, speaker)
    
    async def _on_tool_called(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle tool call from agent."""
        if data:
            tool_name = data.get("tool_name")
            parameters = data.get("parameters", {})
            logger.info(f"Agent called tool: {tool_name} with parameters: {parameters}")
    
    # =========================================================================
    # Tool Implementation
    # =========================================================================
    
    async def _suggest_biome_tool(self, biome_name: str, reason: str = "") -> Dict[str, Any]:
        """Tool for suggesting biome changes."""
        try:
            # Validate biome name
            biome = Biome(biome_name.lower())
            
            # Publish biome suggestion using new schema
            await self._publish_suggest_biome(biome, confidence=0.8)
            
            logger.info(f"Agent suggested biome change to: {biome}")
            return {"success": True, "biome": biome, "reason": reason}
            
        except ValueError as e:
            logger.error(f"Invalid biome suggestion: {biome_name}")
            return {"success": False, "error": f"Invalid biome: {biome_name}"}
    
    async def _get_audience_status_tool(self) -> Dict[str, Any]:
        """Tool for checking conversation and agent status."""
        # Note: We no longer track audience_present directly - that's handled by the presence system
        return {
            "conversation_active": self.is_conversation_active,
            "agent_speaking": self.agent_speaking,
            "note": "Audience presence is now managed by the core service presence system"
        }
    
    # =========================================================================
    # Message Processing
    # =========================================================================
    
    async def _process_conversation_turn(self, turn: ConversationTurn):
        """Process a new conversation turn."""
        logger.debug(f"Processing conversation turn: {turn.speaker} - {turn.content[:50]}...")
        
        # The transcript manager in the backend handles saving automatically,
        # we just need to handle any additional display logic here
        
        # Update conversation state
        if turn.speaker == "human":
            self.is_conversation_active = True
        elif turn.speaker == "agent":
            self.agent_speaking = True
            
        # Publish speech events if needed  
        if turn.speaker == "human":
            await self._publish_speech_detected(is_speaking=True, speaker_type="human")
    
    async def _display_transcript_text(self, content: str, speaker: str):
        """Display transcript text on the visual interface."""
        import uuid
        
        text_id = str(uuid.uuid4())
        speaker_name = (
            self.config.transcript.agent_speaker_name if speaker == "agent"
            else self.config.transcript.human_speaker_name if speaker == "human"
            else speaker
        )
        
        # Create display text message
        display_msg = DisplayText(
            text_id=text_id,
            content=content,
            speaker=speaker_name,
            duration=self.config.transcript.transcript_line_duration,
            fade_duration=self.config.transcript.transcript_fade_duration
        )
        
        # Send to display service (TODO: implement push to display)
        # await self.display_push_service.send(display_msg)
        
        # Track displayed text
        self.displayed_text_ids.add(text_id)
        
        # Schedule removal if duration is set
        if display_msg.duration:
            asyncio.create_task(self._schedule_text_removal(text_id, display_msg.duration))
        
        logger.debug(f"Displayed transcript: {speaker_name} - {content}")
    
    async def _schedule_text_removal(self, text_id: str, delay: float):
        """Schedule removal of displayed text after a delay."""
        await asyncio.sleep(delay)
        if text_id in self.displayed_text_ids:
            await self._remove_displayed_text(text_id)
    
    async def _remove_displayed_text(self, text_id: str):
        """Remove displayed text from the interface."""
        if text_id in self.displayed_text_ids:
            remove_msg = RemoveText(text_id=text_id)
            
            # Send to display service (TODO: implement push to display)
            # await self.display_push_service.send(remove_msg)
            
            self.displayed_text_ids.remove(text_id)
            logger.debug(f"Removed displayed text: {text_id}")
    
    async def _clear_all_displayed_text(self):
        """Clear all currently displayed text."""
        for text_id in list(self.displayed_text_ids):
            await self._remove_displayed_text(text_id)
    
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
        
        # TODO: Update agent context with current era/biome information
        if self.current_backend and self.current_backend.is_connected:
            context_msg = f"<System: the installation is currently showing {era} era in a {biome} biome.>"
            await self.current_backend.send_message(context_msg, speaker="system")
    
    
    # =========================================================================
    # Publishing Methods
    # =========================================================================
    
    async def _publish_audience_present(self, person_count: int = 0):
        """Publish audience presence detection."""
        message = AudiencePresent(
            person_count=person_count
        )
        await self.zmq_service.publish(message, MessageType.AUDIENCE_PRESENT)
        logger.debug(f"Published audience present: (people: {person_count})")

    async def _publish_speech_detected(self, is_speaking: bool, speaker_type: str = "agent"):
        """Publish speech detection for conversation tracking."""
        message = SpeechDetected(
            is_speaking=is_speaking,
            speaker=speaker_type  # Convert to enum value
        )
        await self.zmq_service.publish(message, MessageType.SPEECH_DETECTED)
        logger.debug(f"Published speech detected: {speaker_type} speaking={is_speaking}")
    
    async def _publish_suggest_biome(self, suggested_biome: str, confidence: float = 0.8):
        """Publish biome suggestion based on conversation context."""
        try:
            # Convert string to Biome enum if needed
            if isinstance(suggested_biome, str):
                biome_enum = Biome(suggested_biome.lower())
            else:
                biome_enum = suggested_biome
                
            message = RequestBiome(
                biome=biome_enum.value  # Field name expected by core service
            )
            await self.zmq_service.publish(message.model_dump(), MessageType.REQUEST_BIOME)
            logger.info(f"Published biome suggestion: {biome_enum.value}")
        except ValueError as e:
            logger.warning(f"Invalid biome suggestion '{suggested_biome}': {e}")
    
    async def _publish_audience_status(self, present: bool):
        """Publish audience presence status."""
        # This is kept for backward compatibility but redirects to new method
        await self._publish_audience_present(present)
    
    async def get_debug_status(self) -> Dict[str, Any]:
        """Get comprehensive debug status from the agent service."""
        status = {
            "service": {
                "name": self.service_name,
                "status": self.status.value if self.status else "unknown",
                "is_conversation_active": self.is_conversation_active,
                "agent_speaking": self.agent_speaking,
                "displayed_text_count": len(self.displayed_text_ids),
                "note": "audience_present is now managed by the core service presence system"
            },
            "backend": None,
            "vision": {},
            "config": {
                "agent_backend": self.config.agent_backend,
                "vision_enabled": self.config.vision.webcam_enabled,
                "transcript_enabled": self.config.transcript.display_transcripts,
                "tool_calling_enabled": self.config.tool_calling_enabled,
                "biome_suggestions_enabled": self.config.biome_suggestions_enabled,
            }
        }
        
        # Add vision component status
        if self.webcam_manager:
            status["vision"]["webcam"] = self.webcam_manager.get_capture_info()
        
        if self.vlm_processor:
            status["vision"]["vlm"] = self.vlm_processor.get_status()
            
        if self.audience_detector:
            status["vision"]["audience_detection"] = self.audience_detector.get_detection_stats()
            # Add detection method info
            if hasattr(self.audience_detector, 'set_performance_mode'):
                status["vision"]["detection_method"] = "cpu"
                status["vision"]["performance_mode"] = self.config.vision.cpu_performance_mode
            else:
                status["vision"]["detection_method"] = self.config.vision.detection_method
        else:
            status["vision"]["detection_method"] = self.config.vision.detection_method
            status["vision"]["audience_detection"] = {"enabled": False}
        
        # Add detector profile information if available
        if hasattr(self, '_detector_profile') and self._detector_profile:
            profile = self._detector_profile
            status["vision"]["detector_profile"] = {
                "name": profile.name,
                "description": profile.description,
                "environment": profile.environment,
                "lighting": profile.lighting,
                "camera_profile_applied": profile.camera is not None,
                "camera_profile_name": profile.camera.name if profile.camera else None
            }
        elif hasattr(self.config.vision, 'detector_profile'):
            status["vision"]["detector_profile"] = {
                "name": self.config.vision.detector_profile,
                "status": "not_loaded"
            }
        
        # Add transcript manager status if available
        if self.current_backend and hasattr(self.current_backend, 'transcript_manager'):
            transcript_stats = self.current_backend.transcript_manager.get_session_stats()
            status["transcript"] = transcript_stats
            status["service"]["conversation_history_length"] = transcript_stats.get("total_messages", 0)
        
        # Get backend debug info if available
        if self.current_backend and hasattr(self.current_backend, 'get_debug_status'):
            status["backend"] = self.current_backend.get_debug_status()
        elif self.current_backend:
            status["backend"] = self.current_backend.get_status()
        
        return status
    
    # =========================================================================
    # Transcript Access Methods
    # =========================================================================
    
    def get_conversation_history(self) -> List[ConversationTurn]:
        """Get the current conversation history from the backend's transcript manager."""
        if not self.current_backend:
            return []
        return self.current_backend.get_conversation_history()
    
    def get_transcript_messages(self, limit: Optional[int] = None):
        """Get transcript messages from the backend's transcript manager."""
        if not self.current_backend:
            return []
        return self.current_backend.get_transcript_messages(limit=limit)
    
    def get_transcript_session_stats(self) -> Dict[str, Any]:
        """Get transcript session statistics."""
        if not self.current_backend or not hasattr(self.current_backend, 'transcript_manager'):
            return {}
        return self.current_backend.transcript_manager.get_session_stats()

