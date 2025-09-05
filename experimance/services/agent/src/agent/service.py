"""
Shared AgentServiceBase that implements generic lifecycle, ZMQ wiring,
optional vision/presence toggles, and transcript handling. Subclass this in
project-specific packages (experimance_agent, fire_agent).
"""

import asyncio
import gc
import os
import threading
import time
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
from experimance_common.schemas import ( # type: ignore
    AudiencePresent, SpeechDetected,
    DisplayText, RemoveText, MessageType
)

from .config import AgentServiceConfig
from .backends.base import AgentBackend, AgentBackendEvent, ConversationTurn

from experimance_common.base_service import BaseService
from experimance_common.zmq.services import PubSubService

from experimance_common.service_state import ServiceState

import logging

SERVICE_TYPE = "agent"

# Module-level logger for the base class
logger = logging.getLogger(__name__)

class AgentServiceBase(BaseService):
    """Base Agent service with shared lifecycle and hooks for specialization."""

    def __init__(self, config: AgentServiceConfig):
        super().__init__(service_type=SERVICE_TYPE, service_name=getattr(config, "service_name", "agent"))
        self.config = config
    
        self.zmq_service = PubSubService(config.zmq)

        # Runtime state
        self.current_backend: Optional[AgentBackend] = None
        self.is_conversation_active = False
        self.agent_speaking = False
        self.user_speaking = False
        
        # Conversation cooldown state
        self.conversation_end_time: Optional[float] = None
        self.audience_went_to_zero_after_conversation = False
        
        # Audio output monitoring
        self._audio_output_issues_count = 0
        self._last_audio_issue_time = None
        self._audio_issue_threshold = 3  # Number of issues before attempting recovery
        self._audio_issue_window = 300  # 5 minutes window for counting issues

    @property
    def any_speaking(self) -> bool:
        """True if either the agent or user is currently speaking."""
        return self.agent_speaking or self.user_speaking

    # ---------------- Subclass hooks ----------------

    def register_project_handlers(self) -> None:
        """Subclass can register project-specific ZMQ handlers."""
        return

    def register_project_tools(self) -> None:
        """Subclass can register project-specific tools with the backend."""
        return

    def post_backend_startup(self) -> None:
        """Subclass can perform actions after the backend has started."""
        return

    # ---------------- Lifecycle ----------------

    async def start(self) -> None:
        logger.info("Starting AgentServiceBase")
        # Register generic handlers first if needed, then project handlers
        self.register_project_handlers()
        await self.zmq_service.start()

        await self._initialize_background_tasks()

        await super().start()
        logger.info("AgentServiceBase started")

    async def stop(self) -> None:
        logger.info("Stopping AgentServiceBase")

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

        await self._stop_background_tasks()

        # Stop background tasks BEFORE stopping ZMQ to prevent publish errors
        # This will cancel the audience detection and vision analysis loops
        try:
            # Give background tasks a moment to exit their loops cleanly
            await asyncio.sleep(0.1)

            logger.debug("Background tasks signaled to stop before ZMQ shutdown")
        except Exception as e:
            logger.debug(f"Error stopping background tasks: {e}")

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

        # Shutdown diagnostics and aggressive cleanup (fixes for bugs in websocket shutdown in pipecat components)
        #await self._perform_shutdown_diagnostics()
        await self._perform_aggressive_cleanup()

    async def _initialize_background_tasks(self):
        """Initialize background tasks. By default this just starts the voice agent."""
        await self._start_backend_for_conversation()
    
    async def _stop_background_tasks(self):
        pass

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
                from .backends.pipecat.backend import PipecatBackend
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
                    AgentBackendEvent.CANCEL, self._on_cancel_backend_pipeline
                )
                backend.add_event_callback(
                    AgentBackendEvent.AUDIO_OUTPUT_ISSUE_DETECTED, self._on_audio_output_issue
                )

                # Register project-specific tools with the backend
                self.register_project_tools()

                # Start the backend
                await backend.start()  # type: ignore

                self.post_backend_startup()

                logger.info(f"Successfully initialized {backend_name} backend")
            else:
                logger.info(f"Agent service started in placeholder mode (no {backend_name} backend)")
            
        except Exception as e:
            self.record_error(e, is_fatal=True)
            raise

    async def _start_backend_for_conversation(self):
        """Start the agent backend when a person is detected."""
        if self.current_backend is not None and self.current_backend.is_connected:
            logger.debug("Backend already exists and is connected, returning True")
            return True
        
        # Clear disconnected backend reference
        if self.current_backend is not None and not self.current_backend.is_connected:
            logger.debug("Clearing disconnected backend reference")
            self.current_backend = None
        
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
            
            # Reset speech tracking for quiet period detection
            self._last_speech_end_time = None

            self._on_conversation_started()
            
            logger.info("Conversation backend started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start conversation backend: {e}")
            self.record_error(e, is_fatal=True)
            return False
        
    def _on_conversation_started(self):
        """Handle logic when a conversation is started."""
        pass

    async def _stop_backend_after_conversation(self):
        """Stop the agent backend after conversation ends."""
        if self.current_backend is None:
            return
        
        # Check if backend is already shutting down
        if (hasattr(self.current_backend, '_shutdown_state') and 
            getattr(self.current_backend, '_shutdown_state') != "running"):
            self.current_backend = None
            return
        
        if not self.running: 
            return

        logger.info("Conversation ended, stopping backend...")
        
        try:
            await self.current_backend.stop()
            self.current_backend = None
            
            # Reset conversation state
            self.is_conversation_active = False
            self.agent_speaking = False
            self.user_speaking = False
            
            logger.info("Conversation backend stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop conversation backend: {e}")
            self.record_error(e, is_fatal=False)


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

    # =========================================================================
    # Publishing Methods
    # =========================================================================

    async def _publish_speech_detected(self, is_speaking: bool, speaker_type: str = "agent"):
        """Publish speech detection for conversation tracking."""
        if not self.running or not self.zmq_service.is_running:
            logger.debug(f"Skipping speech detected publish (service running: {self.running}, zmq running: {self.zmq_service.is_running})")
            return
            
        message = SpeechDetected(
            is_speaking=is_speaking,
            speaker=speaker_type  # Convert to enum value
        )
        try:
            await self.zmq_service.publish(message, MessageType.SPEECH_DETECTED)
            logger.debug(f"Published speech detected: {speaker_type} speaking={is_speaking}")
        except Exception as e:
            logger.debug(f"Failed to publish speech detected: {e}")
            # Don't raise the exception to prevent shutdown issues

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_transcription_received(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle new transcription data."""
        pass


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

    async def _on_speech_detected(self, event: AgentBackendEvent, data: Dict[str, Any]):
        """Handle speech detection event from pipecat backend."""
        speaker = data.get("speaker", "unknown") if data else "unknown"
        
        # Update internal state
        if speaker == "agent":
            self.agent_speaking = True
        else:  # Human/user speech
            self.user_speaking = True
        
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
        else:  # Human/user speech
            self.user_speaking = False
        
        # Track when speech ends for deep thoughts timing
        self._last_speech_end_time = time.time()
        logger.debug(f"Speech ended for {speaker}, tracking quiet period for deep thoughts")
        
        # Publish to presence system for conversation tracking
        if self.config.speech_detection_enabled:
            speaker_type = "agent" if speaker == "agent" else "human"
            await self._publish_speech_detected(is_speaking=False, speaker_type=speaker_type)

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
