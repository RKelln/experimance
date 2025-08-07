"""
Main Agent Service for the Experimance interactive art installation.

This service handles speech-to-speech conversation with the audience, integrates webcam feeds 
for audience detection and scene understanding, and provides tool calling capabilities for 
controlling other services.
"""

import asyncio
import gc
import json
import logging
import os
import random
import threading
import time
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

from experimance_common.base_service import BaseService
from experimance_common.service_state import ServiceState
from experimance_common.health import HealthStatus
from experimance_common.zmq.services import PubSubService
from experimance_common.schemas import (
    AudiencePresent, SpeechDetected,
    DisplayText, RemoveText, MessageType, Biome
)
from experimance_common.constants import TICK, DEFAULT_PORTS
from experimance_common.zmq.config import MessageDataType

from .config import AgentServiceConfig
from .backends.base import AgentBackend, AgentBackendEvent, ConversationTurn, ToolCall
from .deep_thoughts import DEEP_THOUGHTS

from experimance_common.logger import setup_logging

SERVICE_TYPE = "agent"

logger = setup_logging(__name__, log_filename=f"{SERVICE_TYPE}.log")


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
        super().__init__(service_type=SERVICE_TYPE, service_name=config.service_name)

        # Store immutable config
        self.config = config
        
        # Runtime state
        self.current_backend: Optional[AgentBackend] = None
        self.is_conversation_active = False
        self.agent_speaking = False
        
        # Conversation cooldown state
        self.conversation_end_time: Optional[float] = None
        self.audience_went_to_zero_after_conversation = False
        
        # Audio output monitoring
        self._audio_output_issues_count = 0
        self._last_audio_issue_time = None
        self._audio_issue_threshold = 3  # Number of issues before attempting recovery
        self._audio_issue_window = 300  # 5 minutes window for counting issues
        
        # Display text tracking (for transcript display)
        self.displayed_text_ids: set[str] = set()
        
        # Deep thoughts state (per-conversation)
        self._available_deep_thoughts: Dict[str, Dict[str, str]] = {}
        self._conversation_start_time: Optional[float] = None
        self._last_speech_end_time: Optional[float] = None  # Track when speech last ended
        
        # Space-time tracking (to avoid duplicate projection messages)
        self._current_era: Optional[str] = None
        self._current_biome: Optional[str] = None
        
        # ZMQ service for communication
        self.zmq_service = PubSubService(config.zmq)
        
        # Vision components (will be initialized if enabled)
        self.webcam_manager = None
        self.vlm_processor = None
        self.audience_detector = None
        self._detector_profile = None  # Store loaded detector profile
        self._person_count = -1  # Track number of people currently detected
        
        # Transcript display handler (will be initialized if enabled)
        self.transcript_display_handler = None
        
        # Pipeline task reference (not managed by service, just for monitoring)
        self._pipeline_task_ref = None
        
    async def start(self):
        """Initialize and start the agent service."""
        logger.info(f"Starting {self.service_name}")
        
        # Set up message handlers before starting ZMQ service
        self.zmq_service.add_message_handler(MessageType.SPACE_TIME_UPDATE, self._handle_space_time_update)
        self.zmq_service.add_message_handler(MessageType.PRESENCE_STATUS, self._handle_audience_present)

        # Start ZMQ service
        await self.zmq_service.start()
        
        # DON'T initialize agent backend yet - wait for person detection
        # await self._initialize_backend()
        
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
        
        # ALWAYS call super().start() LAST - this starts health monitoring automatically
        await super().start()
        
        logger.info(f"{self.service_name} started successfully")
    
    async def stop(self):
        """Clean up resources and stop the agent service."""
        logger.info(f"Stopping {self.service_name}")
        
        # Stop agent backend if it exists - force immediate shutdown only for signal-based shutdowns
        if self.current_backend:
            try:
                if self.state == ServiceState.STOPPING:
                    # Mark as forced shutdown for signal-based shutdowns
                    setattr(self.current_backend, '_shutdown_reason', "forced")
                    logger.debug("Marked backend for forced shutdown due to signal")

                # Add a timeout to prevent hanging
                await asyncio.wait_for(self.current_backend.stop(), timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("Backend stop timed out after 3s, performing immediate aggressive cleanup")
                # Perform aggressive cleanup immediately if backend hangs
                await self._perform_aggressive_cleanup()
            except Exception as e:
                logger.error(f"Error stopping backend: {e}")
            finally:
                # Always clear references
                self.current_backend = None
                self._pipeline_task_ref = None

        # Clean up vision components
        if self.audience_detector:
            await self.audience_detector.stop()
        if self.vlm_processor:
            await self.vlm_processor.stop()
        if self.webcam_manager:
            await self.webcam_manager.stop()
        
        # Clear any displayed text
        await self._clear_all_displayed_text()

        # Stop ZMQ service
        await self.zmq_service.stop()
        
        # Clean up audio resources
        try:
            from experimance_common.audio_utils import cleanup_audio_resources
            cleanup_audio_resources()
            logger.debug("Audio resources cleaned up")
        except Exception as e:
            logger.debug(f"Error cleaning up audio resources: {e}")
        
        logger.info(f"{self.service_name} stopped")
        
        # Call super().stop() LAST
        await super().stop() # always in STOPPING state after this

        # Shutdown diagnostics and aggressive cleanup
        #await self._perform_shutdown_diagnostics()
        await self._perform_aggressive_cleanup()

    async def _perform_shutdown_diagnostics(self):
        """Perform diagnostic logging before shutdown."""
        logger.info("=== SHUTDOWN DIAGNOSTICS ===")
        
        # Force garbage collection
        gc.collect()
        
        # Show running asyncio tasks
        try:
            running_tasks = [task for task in asyncio.all_tasks() if not task.done()]
            logger.info(f"Active asyncio tasks: {len(running_tasks)}")
            for i, task in enumerate(running_tasks[:5]):  # Show first 5 tasks
                task_name = getattr(task, 'get_name', lambda: 'unnamed')()
                logger.info(f"  Task {i+1}: {task_name}")
            if len(running_tasks) > 5:
                logger.info(f"  ... and {len(running_tasks) - 5} more tasks")
        except Exception as e:
            logger.info(f"Could not get asyncio tasks: {e}")
        
        # Show active threads
        try:
            active_threads = threading.enumerate()
            logger.info(f"Active threads: {len(active_threads)}")
            for thread in active_threads:
                logger.info(f"  Thread: {thread.name} (daemon: {thread.daemon})")
        except Exception as e:
            logger.info(f"Could not get thread info: {e}")
        
        # Show process info if psutil is available
        if psutil:
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"Memory usage: RSS={memory_info.rss // 1024 // 1024}MB")
                
                children = process.children(recursive=True)
                if children:
                    logger.info(f"Child processes: {len(children)}")
                else:
                    logger.info("No child processes")
            except Exception as e:
                logger.info(f"Could not get process info: {e}")
        
        logger.info("=== END DIAGNOSTICS ===")

    async def _perform_aggressive_cleanup(self):
        """Perform aggressive cleanup of hanging tasks."""
        logger.info("Performing aggressive cleanup...")
        
        try:
            # Get current task to avoid cancelling the shutdown sequence itself
            current_task = asyncio.current_task()
            remaining_tasks = [
                task for task in asyncio.all_tasks() 
                if not task.done() and task != current_task
            ]
            
            if remaining_tasks:
                logger.info(f"Force-cancelling {len(remaining_tasks)} remaining tasks (excluding current shutdown task)")
                
                # Log task details for debugging
                for i, task in enumerate(remaining_tasks[:10]):  # Log first 10 tasks
                    task_name = getattr(task, 'get_name', lambda: 'unnamed')()
                    task_repr = repr(task).replace('\n', ' ')[:100]
                    logger.debug(f"  Task {i+1}: {task_name} - {task_repr}")
                
                # Cancel all remaining tasks
                for task in remaining_tasks:
                    if not task.cancelled():
                        task.cancel()
                
                # Give tasks a brief moment to cancel, but don't wait too long
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*remaining_tasks, return_exceptions=True),
                        timeout=1.0  # Reduced from 0.5s to 1.0s but still aggressive
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Some tasks didn't cancel within 1.0s - this may indicate hanging WebSocket connections or other external I/O")
                    
                    # Check which tasks are still running
                    still_running = [task for task in remaining_tasks if not task.done()]
                    if still_running:
                        logger.warning(f"{len(still_running)} tasks still running after cancellation timeout")
                        for i, task in enumerate(still_running[:5]):  # Show first 5 still running
                            task_name = getattr(task, 'get_name', lambda: 'unnamed')()
                            task_repr = repr(task).replace('\n', ' ')[:100]
                            logger.warning(f"  Still running {i+1}: {task_name} - {task_repr}")
                        
                except Exception as e:
                    logger.warning(f"Task cancellation error: {e}")
            else:
                logger.info("No tasks to cancel (excluding current shutdown task)")
        except Exception as e:
            logger.error(f"Aggressive cleanup error: {e}")
        
        # Give logs a moment to flush before final exit
        await asyncio.sleep(0.1)
    
    # =========================================================================
    # Backend Management
    # =========================================================================
    
    def _is_in_conversation_cooldown(self) -> bool:
        """Check if we're currently in a conversation cooldown period."""
        if self.conversation_end_time is None:
            return False
            
        # Check if cooldown time has passed
        time_elapsed = time.time() - self.conversation_end_time
        cooldown_expired = time_elapsed >= self.config.conversation_cooldown_duration
        
        # If audience absence override is enabled and audience left and returned, end cooldown early
        if (self.config.cancel_cooldown_on_absence and 
            self.audience_went_to_zero_after_conversation):
            return False  # Cooldown ends early when audience changes
        
        # Otherwise, cooldown continues until time expires
        return not cooldown_expired
    
    def _clear_conversation_cooldown(self):
        """Clear the conversation cooldown state."""
        self.conversation_end_time = None
        self.audience_went_to_zero_after_conversation = False
        logger.debug("Conversation cooldown cleared")
    
    async def _initialize_backend(self):
        """Initialize the selected agent backend."""
        backend_name = self.config.agent_backend.lower()
        
        logger.debug(f"Initializing {backend_name} backend...")

        try:
            if backend_name == "pipecat":
                from .backends.pipecat_backend import PipecatBackend
                # Pass self (agent service) to backend
                self.current_backend = PipecatBackend(self.config, agent_service=self)
            else:
                raise ValueError(f"Unknown agent backend: {backend_name}")
            
            # Register event callbacks and tools only if backend is available
            if self.current_backend is not None:
                backend = self.current_backend  # For type checking
                # Register event callbacks
                # backend.add_event_callback(
                #     AgentBackendEvent.CONVERSATION_STARTED, self._on_conversation_started
                # )
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
                backend.add_event_callback(
                    AgentBackendEvent.CANCEL, self._on_cancel_backend_pipeline
                )
                backend.add_event_callback(
                    AgentBackendEvent.AUDIO_OUTPUT_ISSUE_DETECTED, self._on_audio_output_issue
                )

                # Start the backend
                await backend.start()  # type: ignore
                
                # Add pipeline task to service task management if available
                if hasattr(backend, 'get_pipeline_task'):
                    pipeline_task = backend.get_pipeline_task()
                    if pipeline_task:
                        self._pipeline_task_ref = pipeline_task
                
                logger.info(f"Successfully initialized {backend_name} backend")
            else:
                logger.info(f"Agent service started in placeholder mode (no {backend_name} backend)")
            
        except Exception as e:
            self.record_error(e, is_fatal=True)
            raise

    async def _start_backend_for_conversation(self):
        """Start the agent backend when a person is detected."""
        if self.current_backend is not None:
            logger.debug("Backend already exists, returning True")
            return True
        
        # Allow backend startup during service initialization (STARTING state)
        # Only block if service is stopped/stopping
        if self.state in [ServiceState.STOPPED, ServiceState.STOPPING]: 
            logger.debug(f"Service state is {self.state}, cannot start backend")
            return False

        # Check if we're in cooldown period
        if self._is_in_conversation_cooldown():
            cooldown_remaining = max(0, (self.conversation_end_time + self.config.conversation_cooldown_duration) - time.time()) if self.conversation_end_time else 0
            
            if self.config.cancel_cooldown_on_absence:
                logger.info(f"Person detected but conversation cooldown active ({cooldown_remaining:.1f}s remaining, or until audience changes)")
            else:
                logger.info(f"Person detected but conversation cooldown active ({cooldown_remaining:.1f}s remaining)")
            return False

        try:
            await self._initialize_backend()
            
            # Initialize transcript handling if enabled
            # if self.config.transcript.display_transcripts:
            #     await self._initialize_transcript_display_handler()
                
            # Clear cooldown state when successfully starting a new conversation
            self._clear_conversation_cooldown()
            
            # Initialize conversation state and deep thoughts
            self.is_conversation_active = True
            self._conversation_start_time = time.time()
            
            # Reset deep thoughts for this conversation - make a deep copy
            import copy
            self._available_deep_thoughts = copy.deepcopy(DEEP_THOUGHTS)
            
            # Reset speech tracking for quiet period detection
            self._last_speech_end_time = None
            
            # Reset space-time tracking to ensure first projection message is sent
            self._current_era = None
            self._current_biome = None
            
            logger.info("Conversation backend started successfully - deep thoughts initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start conversation backend: {e}")
            self.record_error(e, is_fatal=True)
            return False
    
    async def _stop_backend_after_conversation(self):
        """Stop the agent backend after conversation ends."""
        if self.current_backend is None:
            return
        
        # Check if backend is already shutting down
        if (hasattr(self.current_backend, '_shutdown_state') and 
            getattr(self.current_backend, '_shutdown_state') != "running"):
            self.current_backend = None
            self._pipeline_task_ref = None
            return
        
        if not self.running: 
            return

        logger.info("Conversation ended, stopping backend...")
        
        try:
            await self.current_backend.stop()
            self.current_backend = None
            self._pipeline_task_ref = None

            # Clear any displayed text
            await self._clear_all_displayed_text()
            
            # Reset conversation state
            self.is_conversation_active = False
            self.agent_speaking = False
            
            logger.info("Conversation backend stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop conversation backend: {e}")
            self.record_error(e, is_fatal=False)

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

        # report 0 on shutdown
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
    # Event Handlers
    # =========================================================================
    
    # async def _on_conversation_started(self, event: AgentBackendEvent, data: Dict[str, Any]):
    #     """Handle conversation started event."""
    #     self.is_conversation_active = True
    #     self._conversation_start_time = time.time()
        
    #     # Reset deep thoughts for this conversation - make a deep copy
    #     import copy
    #     self._available_deep_thoughts = copy.deepcopy(DEEP_THOUGHTS)
        
    #     logger.info("Conversation started with audience - deep thoughts reset")

    async def _on_conversation_ended(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle conversation ended event."""
        logger.info("Conversation ended")
        self.is_conversation_active = False
        self._conversation_start_time = None  # Clear conversation start time
        
        # Start cooldown timer
        self.conversation_end_time = time.time()
        if self.config.cancel_cooldown_on_absence:
            self.audience_went_to_zero_after_conversation = False
            logger.info(f"Conversation cooldown started ({self.config.conversation_cooldown_duration}s, or until audience changes)")
        else:
            logger.info(f"Conversation cooldown started ({self.config.conversation_cooldown_duration}s)")
        
        # Check if this is a natural shutdown (flow ended gracefully)
        reason = data.get("reason", "unknown") if data else "unknown"
        if reason in ["pipeline_ended", "idle_timeout"]:
            # This is a natural shutdown from the flow ending (goodbye node) or idle timeout
            # The backend will handle its own shutdown gracefully
            if reason == "idle_timeout":
                logger.info("Idle timeout detected, treating as natural conversation end")
            else:
                logger.info("Natural conversation end detected, backend will handle shutdown")
            
            # Clear displayed text after a short delay
            if self.displayed_text_ids and len(self.displayed_text_ids) > 0:
                await asyncio.sleep(2.0)
                await self._clear_all_displayed_text()
                
            # Wait a moment for the backend to finish naturally, then clean up reference
            await asyncio.sleep(1.0)
            if (self.current_backend and 
                hasattr(self.current_backend, '_shutdown_reason') and 
                getattr(self.current_backend, '_shutdown_reason') == "natural"):
                # Backend is shutting down naturally, just clear our reference
                self.current_backend = None
                self._pipeline_task_ref = None
                logger.info("Cleared backend reference after natural shutdown")
        else:
            # This is a forced or unexpected shutdown, stop the backend normally
            await self._stop_backend_after_conversation()

            if self.displayed_text_ids and len(self.displayed_text_ids) > 0:
                # Clear displayed transcripts after a delay
                await asyncio.sleep(2.0)
                await self._clear_all_displayed_text()

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
    
    async def _on_speech_detected(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle speech detection event from pipecat backend."""
        speaker = data.get("speaker", "unknown") if data else "unknown"
        
        # Update internal state
        if speaker == "agent":
            self.agent_speaking = True
        
        # Reset speech end time when new speech starts (interrupts quiet period)
        self._last_speech_end_time = None
        logger.debug(f"Speech detected for {speaker}, resetting quiet period")
        
        # Publish to presence system for conversation tracking
        if self.config.speech_detection_enabled:
            speaker_type = "agent" if speaker == "agent" else "human"
            await self._publish_speech_detected(is_speaking=True, speaker_type=speaker_type)
    
    async def _on_speech_ended(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle speech ended event from pipecat backend."""
        speaker = data.get("speaker", "unknown") if data else "unknown"
        
        # Update internal state
        if speaker == "agent":
            self.agent_speaking = False
        
        # Track when speech ends for deep thoughts timing
        self._last_speech_end_time = time.time()
        logger.debug(f"Speech ended for {speaker}, tracking quiet period for deep thoughts")
        
        # Publish to presence system for conversation tracking
        if self.config.speech_detection_enabled:
            speaker_type = "agent" if speaker == "agent" else "human"
            await self._publish_speech_detected(is_speaking=False, speaker_type=speaker_type)
    
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
    

    # ONLY for use if pipecat sigint handling is on (currently off by default)
    async def _on_cancel_backend_pipeline(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle pipeline shutdown signal from backend.
        
        This is triggered when the backend detects a shutdown signal (like Ctrl-C).
        We trigger immediate service shutdown - the stop() method handles pipeline coordination.
        """
        if not self.current_backend or not self.running:
            return
        
        # Check if we're already stopping to avoid duplicate stop calls
        if self.state == ServiceState.STOPPING:
            logger.debug("Backend shutdown signal received, but service already stopping")
            return
        
        logger.info("Backend shutdown signal received, stopping service...")
        
        # Trigger immediate service shutdown - stop() method will coordinate with pipeline
        asyncio.create_task(self.stop())

    async def _on_audio_output_issue(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle audio output issues detected by the backend.
        
        This is triggered when the backend detects potential audio output failures,
        such as TTS duration anomalies or other audio pipeline issues.
        """
        if not self.current_backend or not self.running:
            return
            
        self._audio_output_issues_count += 1
        issue_type = data.get('issue_type', 'unknown')
        details = data.get('details', {})
        
        logger.warning(
            f"Audio output issue detected (#{self._audio_output_issues_count}): {issue_type}",
            extra={"details": details}
        )
        
        # Report to health system with appropriate severity
        health_status = HealthStatus.WARNING if self._audio_output_issues_count < self._audio_issue_threshold else HealthStatus.ERROR
        
        # For now, just report general audio output issues to the health system
        # TODO: Add specific detection for TTS billing issues when we can capture HTTP 402 errors
        # Currently, pipecat library errors (like Cartesia HTTP 402) aren't directly accessible
        
        # Report all audio issues with escalating severity
        if self._audio_output_issues_count >= self._audio_issue_threshold:
            # Multiple failures suggest a systemic issue - report as FATAL
            self.record_health_check(
                "tts_audio_pipeline",
                HealthStatus.FATAL,
                f"Multiple audio output failures detected ({self._audio_output_issues_count}) - possible TTS service issue",
                metadata={
                    "error_count": self._audio_output_issues_count,
                    "issue_type": issue_type,
                    "details": details,
                    "threshold": self._audio_issue_threshold,
                    "note": "Check logs for HTTP 402 or payment errors from TTS service"
                }
            )
            logger.error("FATAL: Multiple audio output failures - likely TTS service billing/connection issue")
        else:
            # Single or few failures - report as WARNING/ERROR
            self.record_health_check(
                "tts_audio_pipeline",
                health_status,
                f"Audio output issue detected: {issue_type}",
                metadata={
                    "error_count": self._audio_output_issues_count,
                    "issue_type": issue_type,
                    "details": details,
                    "threshold": self._audio_issue_threshold
                }
            )
        
        # If we've had multiple issues in a short period, attempt recovery
        if self._audio_output_issues_count >= self._audio_issue_threshold:
            logger.error(
                f"Multiple audio output issues ({self._audio_output_issues_count}), attempting recovery..."
            )
            
            try:
                # Import here to avoid circular imports
                from experimance_common.audio_utils import reset_audio_device_by_name
                
                # Get the current audio device from config
                audio_device = self.config.backend_config.pipecat.audio_input_device_name
                if audio_device:
                    logger.info(f"Attempting to reset audio device: {audio_device}")
                    success = reset_audio_device_by_name(audio_device)
                    
                    if success:
                        # Reset the counter after successful recovery
                        self._audio_output_issues_count = 0
                        logger.info("Audio device reset completed, counter reset")
                    else:
                        logger.warning("Audio device reset failed")
                else:
                    logger.warning("No audio device name configured for reset")
                    
            except Exception as e:
                logger.error(f"Failed to reset audio device: {e}")
                # Don't reset counter if recovery failed - let it accumulate

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
                    logger.info("Idle state detected, transitioning to goodbye node")
                    #await self.current_backend.transition_to_node("goodbye")
                    await self.current_backend.stop()
    
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
    
    async def _publish_request_biome(self, requested_biome: str):
        """Publish biome change request based on conversation context."""
        try:
            # Convert string to Biome enum if needed
            if isinstance(requested_biome, str):
                # Handle both underscore and space formats
                biome_value = requested_biome.lower().replace(" ", "_")
                biome_enum = Biome(biome_value)
            else:
                biome_enum = requested_biome
                
            # Create RequestBiome message - using dynamic import to avoid linter errors
            from experimance_common.schemas import RequestBiome, MessageType
            message = RequestBiome(
                biome=biome_enum.value  # Field name expected by core service
            )
            await self.zmq_service.publish(message.model_dump(), MessageType.REQUEST_BIOME)
            logger.info(f"Published biome request: {biome_enum.value}")
        except ValueError as e:
            logger.warning(f"Invalid biome request '{requested_biome}': {e}")
        except Exception as e:
            logger.error(f"Failed to publish biome request: {e}")
    
    async def get_debug_status(self) -> Dict[str, Any]:
        """Get comprehensive debug status from the agent service."""
        status = {
            "service": {
                "name": self.service_name,
                "status": self.status.value if self.status else "unknown",
                "is_conversation_active": self.is_conversation_active,
                "agent_speaking": self.agent_speaking,
                "displayed_text_count": len(self.displayed_text_ids),
                "note": "audience_present is now managed by the core service presence system",
                "cooldown": {
                    "active": self._is_in_conversation_cooldown(),
                    "end_time": self.conversation_end_time,
                    "audience_went_to_zero": self.audience_went_to_zero_after_conversation,
                    "cooldown_duration": self.config.conversation_cooldown_duration,
                    "cancel_cooldown_on_absence": self.config.cancel_cooldown_on_absence
                }
            },
            "backend": None,
            "vision": {},
            "config": {
                "agent_backend": self.config.agent_backend,
                "vision_enabled": self.config.vision.webcam_enabled,
                "transcript_enabled": self.config.transcript.display_transcripts,
                "tool_calling_enabled": self.config.tool_calling_enabled,
                "biome_suggestions_enabled": self.config.biome_suggestions_enabled,
                "conversation_cooldown_duration": self.config.conversation_cooldown_duration,
                "cancel_cooldown_on_absence": self.config.cancel_cooldown_on_absence,
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

