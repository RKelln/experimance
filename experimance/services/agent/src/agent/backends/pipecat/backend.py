"""
Pipecat backend v2 for the Experimance agent service.

This module implements the AgentBackend interface using Pipecat's local audio pipeline,
providing speech-to-text, LLM conversation, and text-to-speech capabilities in a single process.

Based on the working ensemble implementation from flows_test.py with proper shutdown handling.

Function Calling Approach:
- Tools are passed to OpenAI LLM service via the `tools` parameter (standard Pipecat approach)
- PipecatFunctionCallProcessor handles execution of FunctionCallInProgressFrame events
- This processor properly handles StartFrame and other pipeline events to avoid frame processing errors
- Pipecat handles the OpenAI function calling protocol, we just execute the functions

Audio Health Monitoring:
- If audio crashes occur, set `_enable_audio_health_monitoring = True` in __init__ method
- This enables comprehensive health checks and recovery mechanisms
- Default is False for production stability
"""
import asyncio
import logging
import os
import sys
import time
import importlib.util
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable
from pathlib import Path

from agent.config import AgentServiceConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from .multi_channel_transport import MultiChannelAudioTransport, MultiChannelAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.services.whisper.stt import WhisperSTTService

# Apply VAD monkey patch to fix sample rate override issues
from agent.vad_patch import apply_silero_vad_patch
apply_silero_vad_patch()
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai_realtime_beta.openai import OpenAIRealtimeBetaLLMService
from pipecat.services.openai_realtime_beta.events import SessionProperties, TurnDetection
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transcriptions.language import Language
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.frames.frames import (
    AudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TTSStartedFrame,
    TTSStoppedFrame, Frame, TranscriptionFrame,
    UserStoppedSpeakingFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
    SystemFrame, CancelFrame, EndFrame, FunctionCallInProgressFrame, FunctionCallResultFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat_flows import FlowManager, FlowArgs, FlowResult, FlowConfig

from experimance_common.audio_utils import resolve_audio_device_index
from experimance_common.constants import AGENT_SERVICE_DIR
from ..base import AgentBackend, AgentBackendEvent, ConversationTurn, ToolCall, UserContext, load_prompt

logger = logging.getLogger(__name__)


class ResampleFilter(BaseAudioFilter):
    """
    Audio filter that resamples audio from input rate to 16kHz using synchronous resampling.
    This is needed when the audio device operates at non 16kHz, which is expected by VAD and STT services.
    
    DESIGN RATIONALE:
    After extensive testing with AIRHUG USB device (48kHz → 16kHz conversion), this approach was chosen because:
    
    ✅ CHUNK-BASED SYNCHRONOUS RESAMPLING (this implementation):
       - 0% empty chunks - perfect reliability 
       - Zero latency - processes each chunk immediately
       - Fast processing (~11ms for 3s audio, 17.5% faster without normalization)
       - Works perfectly with real speech audio
       - Minor clicking artifacts (less noticeable with real speech audio)

    ❌ STREAM-BASED RESAMPLING (SOXRStreamAudioResampler):
       - 71% empty chunks with 20ms chunks
       - Requires 80ms+ buffering to work properly (4+ chunks accumulated)
       - Higher latency for real-time applications
       - Complex state management
    
    OPTIMIZATION: Direct int16→float64→int16 conversion without normalization provides identical 
    audio quality but 17.5% better performance compared to normalized float32 approach.
    
    For 48kHz→16kHz conversion with 20ms chunks, this approach is optimal for real-time speech processing.
    """

    def __init__(self, in_rate: int = 48000, out_rate: int = 16000, mode="soxr"):
        super().__init__()
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.mode = mode
        self._resampler_func = None
        
    async def start(self, sample_rate: int) -> None:
        """Initialize the resampler with the actual transport sample rate."""
        try:
            # Use the actual sample rate from the transport
            #self.in_rate = sample_rate
            
            # Set up synchronous resampling function
            if self.mode == "soxr":
                try:
                    import soxr
                    self._resampler_func = lambda x, sr_orig, sr_new: soxr.resample(x, sr_orig, sr_new)
                    logger.info(f"Resample16kFilter: Using soxr for {self.in_rate}Hz -> {self.out_rate}Hz")
                except ImportError:
                    try:
                        import resampy
                        self._resampler_func = resampy.resample
                        logger.info(f"Resample16kFilter: Using resampy for {self.in_rate}Hz -> {self.out_rate}Hz")
                    except ImportError:
                        raise ImportError("Neither soxr nor resampy available for resampling")
            elif self.mode == "resampy":
                try:
                    import resampy
                    self._resampler_func = resampy.resample
                    logger.info(f"Resample16kFilter: Using resampy for {self.in_rate}Hz -> {self.out_rate}Hz")
                except ImportError:
                    raise ImportError("resampy not available for resampling")

        except Exception as e:
            logger.error(f"Failed to initialize resampler: {e}")
            raise
    
    async def stop(self) -> None:
        """Clean up the resampler."""
        self._resampler_func = None
        logger.debug("Resample16kFilter stopped")
        
    async def filter(self, audio: bytes) -> bytes:
        """Resample audio from input rate to 16kHz."""
        if self._resampler_func is None:
            logger.warning("Resampler not initialized, returning original audio")
            return audio
            
        # Skip resampling if no data
        if not audio:
            return audio
            
        try:
            import numpy as np
            
            # Convert bytes to numpy array (int16)
            audio_array = np.frombuffer(audio, dtype=np.int16)
            
            # Direct resampling with float64 for soxr compatibility
            # (Testing showed this is 17.5% faster than normalization with identical results)
            audio_float = audio_array.astype(np.float64)
            
            # Resample using the synchronous resampler
            resampled_array = self._resampler_func(
                audio_float,
                sr_orig=self.in_rate, 
                sr_new=self.out_rate
            )
            
            # Convert directly back to int16
            resampled_bytes = resampled_array.astype(np.int16).tobytes()
            
            return resampled_bytes
            
        except Exception as e:
            logger.warning(f"Audio resampling failed: {e}, returning original audio")
            return audio
    
    async def process_frame(self, frame) -> None:
        """Process frames - required by BaseAudioFilter interface."""
        # This method is required by the interface but may not be used
        # depending on how the filter is integrated
        pass


class SimpleResample16kFilter(SOXRStreamAudioResampler):

    def __init__(self, in_rate: int, out_rate: int):
        super().__init__()
        self._in_rate = in_rate
        self._out_rate = out_rate

    async def start(self, sample_rate: int):
        pass

    async def stop(self) -> None:
        pass

    async def filter(self, audio: bytes) -> bytes:
        """Resample audio from input rate to 16kHz."""
        if not audio:
            return audio
        audio = await self.resample(audio, self._in_rate, self._out_rate) # type: ignore
        if not audio:
            # return empty bytes
            return b""
        return audio

    async def process_frame(self, frame) -> None:
        """Process frames - required by BaseAudioFilter interface."""
        # This method is required by the interface but may not be used
        # depending on how the filter is integrated
        pass

class PipecatFunctionCallProcessor(FrameProcessor):
    """
    Processor that handles function calls from the OpenAI LLM API.
    """
    
    def __init__(self, backend: 'PipecatBackend', **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames to handle function calls."""
        # Always call parent first to handle StartFrame and other pipeline frames
        await super().process_frame(frame, direction)
        
        if isinstance(frame, FunctionCallInProgressFrame):
            logger.info(f"Function call in progress: {frame.function_name} with args: {frame.arguments}")
            
            # Create ToolCall object
            tool_call = ToolCall(
                tool_name=frame.function_name,
                parameters=frame.arguments
            )
            
            # Execute the tool call
            result = await self.backend.call_tool(tool_call)
            
            # Send result back to the pipeline
            result_frame = FunctionCallResultFrame(
                function_name=frame.function_name,
                result=result,
                tool_call_id=getattr(frame, 'tool_call_id', frame.function_name),
                arguments=frame.arguments
            )
            await self.push_frame(result_frame, direction)
            logger.info(f"Function call completed: {tool_call.tool_name}")
            
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)


class PipecatEventProcessor(FrameProcessor):
    """
    Event processor that captures conversation events and forwards them to the backend.
    """
    
    def __init__(self, backend: 'PipecatBackend'):
        super().__init__()
        self.backend = backend
        self.user_speaking = False
        self.bot_speaking = False
        self._conversation_ending = False  # Track if we're in normal conversation end sequence
        
        # Audio output monitoring
        self._last_tts_start_time = None
        self._expected_audio_output = False
        self._audio_output_timeout = 10.0  # seconds to wait for audio after TTS starts
        
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames and extract conversation events."""
        
        # Call parent class to handle StartFrame properly
        await super().process_frame(frame, direction)
        
        # Skip event processing if backend is shutting down (except for shutdown frames)
        if (self.backend._shutdown_state != "running" and 
            not isinstance(frame, (EndFrame, CancelFrame))):
            await self.push_frame(frame, direction)
            return
        
        # Handle user speaking state
        if isinstance(frame, UserStartedSpeakingFrame):
            self.user_speaking = True
            await self.backend.emit_event(AgentBackendEvent.SPEECH_DETECTED, {
                "speaker": "user"
            })
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_speaking = False
            await self.backend.emit_event(AgentBackendEvent.SPEECH_ENDED, {
                "speaker": "user"
            })
            
        # Handle bot speaking state  
        elif isinstance(frame, TTSStartedFrame):
            self.bot_speaking = True
            self._last_tts_start_time = time.time()
            self._expected_audio_output = True
            await self.backend.emit_event(AgentBackendEvent.BOT_STARTED_SPEAKING, {
                "speaker": "agent"
            })
            
        elif isinstance(frame, TTSStoppedFrame):
            self.bot_speaking = False
            self._expected_audio_output = False
            await self.backend.emit_event(AgentBackendEvent.BOT_STOPPED_SPEAKING, {
                "speaker": "agent"
            })
            
            # Check if we had a reasonable TTS duration (audio output health check)
            if self._last_tts_start_time:
                tts_duration = time.time() - self._last_tts_start_time
                if tts_duration < 0.1:  # Very short TTS might indicate audio output issues
                    logger.warning(f"Very short TTS duration: {tts_duration:.2f}s - possible audio output issue")
                    await self.backend.emit_event(AgentBackendEvent.AUDIO_OUTPUT_ISSUE_DETECTED, {
                        "tts_duration": tts_duration,
                        "timestamp": time.time()
                    })
                self._last_tts_start_time = None
        # Handle pipeline shutdown - EndFrame → CancelFrame sequence
        elif isinstance(frame, EndFrame):
            logger.info("Pipeline EndFrame received, starting conversation end sequence")
            self._conversation_ending = True
            # Mark this as a natural shutdown in the backend
            self.backend._shutdown_reason = "natural"
            await self.backend.emit_event(AgentBackendEvent.CONVERSATION_ENDED, {
                "reason": "pipeline_ended"
            })
            
        elif isinstance(frame, CancelFrame):
            if self._conversation_ending:
                logger.info("Pipeline CancelFrame received after EndFrame (normal conversation end)")
                self._conversation_ending = False
            else:
                # Check if this is an idle timeout cancellation (natural end) vs forced shutdown
                # If the backend is marked for forced shutdown, always treat as forced
                logger.info("Pipeline CancelFrame received without EndFrame - checking shutdown reason")
                
                if (self.backend._shutdown_reason == "forced" or 
                    self.backend._shutdown_state != "running"):
                    logger.info("Pipeline CancelFrame during forced shutdown")
                    await self.backend.emit_event(AgentBackendEvent.CANCEL, {
                        "reason": "pipeline_cancelled"
                    })
                else:
                    # Only treat as idle timeout if we're still in running state AND no forced shutdown
                    logger.info("Treating CancelFrame as idle timeout (natural conversation end)")
                    self.backend._shutdown_reason = "natural"
                    await self.backend.emit_event(AgentBackendEvent.CONVERSATION_ENDED, {
                        "reason": "idle_timeout"
                    })
        # Note: transcription handled through transcription manager
            
        # Forward the frame to the next processor
        await self.push_frame(frame, direction)


class PipecatBackend(AgentBackend):
    """
    Pipecat backend implementation using the working ensemble approach from flows_test.py.
    """
    
    def __init__(self, config: AgentServiceConfig, user_context: Optional[UserContext] = None, agent_service=None):
        super().__init__(config)
        self.pipecat_config = self.config.backend_config.pipecat
        self.user_context = user_context or UserContext()
        self.agent_service = agent_service  # Store reference to the main service
        
        # Pipeline components
        self.pipeline: Optional[Pipeline] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        self.transport: Optional[LocalAudioTransport | MultiChannelAudioTransport] = None
        self.flow_manager: Optional[FlowManager] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._flow_config: Optional[FlowConfig] = None
        
        # Shutdown state management
        self._shutdown_state = "running"  # "running", "stopping", "stopped"
        self._shutdown_reason = None  # "natural", "forced", "error"
        self._shutdown_lock = asyncio.Lock()  # Prevent concurrent shutdown operations
        
        # Event handling
        self.event_processor: Optional[PipecatEventProcessor] = None
        
        # Audio health monitoring - controlled by debug flag
        self._enable_audio_health_monitoring = self.config.audio_health_monitoring  # Set to True if audio crashes occur again
        self._audio_health_monitor_task: Optional[asyncio.Task] = None
        self._last_audio_health_check = 0
        self._audio_health_check_interval = 30  # seconds, 0 = off
        

    async def _log_system_resources(self, stage: str) -> None:
        """Log system resource usage at different pipeline stages."""
        try:
            import psutil
            
            # Get memory info
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            
            # Get CPU info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get process info
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)
            
            logger.info(f"[{stage}] System: {memory_mb:.1f}MB ({memory_percent:.1f}%) RAM, {cpu_percent:.1f}% CPU | Process: {process_memory:.1f}MB")
            
        except ImportError:
            # psutil not available, use basic logging
            logger.debug(f"[{stage}] System resource monitoring unavailable (psutil not installed)")
        except Exception as e:
            logger.warning(f"[{stage}] Failed to log system resources: {e}")

    async def _test_audio_output(self) -> bool:
        """Test if audio output is working to detect conflicts before pipeline starts."""
        try:
            logger.info("Testing audio output availability...")
            
            # If we have a transport, try to access the audio output device
            if self.transport:
                transport_output = getattr(self.transport, '_output', None)
                if transport_output:
                    # Just check if the device is accessible, don't actually play anything
                    logger.info("Audio output device appears accessible")
                    return True
                else:
                    logger.warning("Transport created but no output device found")
                    return False
            else:
                # No transport yet - this is expected during standalone testing
                # Just return True since we can't test without a transport
                logger.debug("No transport available for audio output test (expected during standalone testing)")
                return True
            
        except Exception as e:
            logger.error(f"Audio output test failed: {e}")
            return False

    async def _comprehensive_audio_health_check(self) -> bool:
        """Comprehensive audio device health check to prevent crashes."""
        try:
            logger.info("Running comprehensive audio health check...")
            
            # Check 1: Device indices are valid
            if hasattr(self.pipecat_config, 'audio_input_device_index') and self.pipecat_config.audio_input_device_index is not None:
                input_idx = self.pipecat_config.audio_input_device_index
                output_idx = self.pipecat_config.audio_output_device_index
                logger.info(f"Checking configured device indices: input={input_idx}, output={output_idx}")
            
            # Check 2: Try to enumerate devices to see if USB devices are responsive
            try:
                from experimance_common.audio_utils import list_audio_devices
                devices = list_audio_devices()
                yealink_found = any("Yealink" in dev.get('name', '') for dev in devices)
                logger.info(f"Audio device enumeration successful. Yealink found: {yealink_found}")
            except Exception as enum_error:
                logger.warning(f"Audio device enumeration failed: {enum_error}")
                return False
            
            # Check 3: Basic transport accessibility
            basic_test = await self._test_audio_output()
            if not basic_test:
                logger.warning("Basic audio output test failed")
                return False
            
            logger.info("Audio health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Audio health check failed: {e}")
            return False

    async def _attempt_audio_recovery(self) -> bool:
        """Attempt to recover from audio issues before they cause crashes."""
        try:
            logger.warning("Attempting audio recovery...")
            
            # Try the audio reset utility
            try:
                from experimance_common.audio_utils import reset_audio_device_by_name
                reset_audio_device_by_name("Yealink")
                logger.info("Audio device reset completed")
                
                # Wait for device to stabilize
                await asyncio.sleep(2)
                
                # Re-test after recovery
                health_ok = await self._comprehensive_audio_health_check()
                if health_ok:
                    logger.info("Audio recovery successful")
                    return True
                else:
                    logger.warning("Audio recovery attempted but health check still fails")
                    return False
                    
            except Exception as recovery_error:
                logger.error(f"Audio recovery failed: {recovery_error}")
                return False
                
        except Exception as e:
            logger.error(f"Audio recovery attempt failed: {e}")
            return False

    async def _audio_health_monitor(self) -> None:
        """Periodic audio health monitoring to detect issues before they cause crashes."""
        try:
            while self._shutdown_state == "running":
                await asyncio.sleep(self._audio_health_check_interval)
                
                if self._shutdown_state != "running":
                    break
                    
                try:
                    # Quick health check
                    current_time = time.time()
                    if self._audio_health_check_interval > 0 and current_time - self._last_audio_health_check > self._audio_health_check_interval:
                        logger.debug("Running periodic audio health check...")
                        
                        # Basic check - just see if devices are still enumerable
                        try:
                            from experimance_common.audio_utils import list_audio_devices
                            devices = list_audio_devices()
                            yealink_found = any("Yealink" in dev.get('name', '') for dev in devices)
                            
                            if not yealink_found:
                                logger.warning("Yealink device not found during health check - may need recovery")
                                await self.emit_event(AgentBackendEvent.AUDIO_OUTPUT_ISSUE_DETECTED, {
                                    "issue": "yealink_device_missing",
                                    "timestamp": current_time
                                })
                            else:
                                logger.debug("Audio health check passed")
                                
                        except Exception as health_error:
                            logger.warning(f"Audio health check failed: {health_error}")
                            await self.emit_event(AgentBackendEvent.AUDIO_OUTPUT_ISSUE_DETECTED, {
                                "issue": "audio_health_check_error",
                                "error": str(health_error),
                                "timestamp": current_time
                            })
                        
                        self._last_audio_health_check = current_time
                        
                except Exception as monitor_error:
                    logger.error(f"Audio health monitor error: {monitor_error}")
                    
        except asyncio.CancelledError:
            logger.debug("Audio health monitor cancelled")
        except Exception as e:
            logger.error(f"Audio health monitor failed: {e}")

    def _load_flow_config(self) -> FlowConfig:
        """Load flow configuration from the specified flow file."""
        flow_file = self.pipecat_config.flow_file
        if not flow_file:
            raise ValueError("Flow file must be specified to load flow configuration")
        flow_path = Path(flow_file)
        if not flow_path.exists():
            flow_path = AGENT_SERVICE_DIR / flow_file
            if not flow_path.exists():
                flow_path = AGENT_SERVICE_DIR / "flows" / flow_file
                if not flow_path.exists():
                    raise FileNotFoundError(f"Flow file not found: {flow_file}")
            
        # Import the flow module dynamically
        spec = importlib.util.spec_from_file_location("flow_config", flow_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load flow file: {flow_path}")
            
        flow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flow_module)
        
        # Get the flow configuration
        if hasattr(flow_module, 'flow_config'):
            return flow_module.flow_config
        else:
            raise ValueError(f"Flow file {flow_path} does not contain 'flow_config'")
    
    async def start(self) -> None:
        """Start the Pipecat backend."""
        try:
            logger.info("Starting Pipecat backend v2...")
            
            # Call parent class to initialize transcript manager
            await super().start()

            logger.debug("Creating audio transport...")
            try:
                self.transport = self._create_transport()
                assert self.transport is not None, "Transport must be created successfully"
                logger.info("Audio transport created successfully")
            except Exception as e:
                logger.error(f"Failed to create transport: {e}")
                raise
            
            # Create pipeline based on mode
            if self.config.backend_config.pipecat.mode == "realtime":
                await self._create_realtime_pipeline()
            else:
                await self._create_ensemble_pipeline()
            
            assert self.pipeline is not None, "Pipeline must be created successfully"
            assert self.task is not None, "Pipeline task must be created successfully"

            logger.debug("Creating pipeline runner...")
            
            # runner
            self.runner = PipelineRunner(
                handle_sigint=False,  # handle these ourselves
                #handle_sigterm=False, # handle these ourselves (newer version of pipecat)
            )
            
            logger.debug("Pipecat backend v2 started successfully")

            # Start the pipeline as a background task - this doesn't block
            if self.runner and self.task:
                logger.debug("Starting pipeline task...")
                
                self._pipeline_task = asyncio.create_task(
                    self.runner.run(self.task),
                    name="pipecat-pipeline"
                )
                
                logger.debug("Connecting to pipeline...")
                
                # Only run comprehensive audio health check if monitoring is enabled
                if self._enable_audio_health_monitoring:
                    logger.debug("Running pre-startup audio health check...")
                    audio_health_ok = await self._comprehensive_audio_health_check()
                    
                    if not audio_health_ok:
                        logger.warning("Audio health check failed - attempting recovery...")
                        recovery_success = await self._attempt_audio_recovery()
                        
                        if not recovery_success:
                            logger.error("Audio recovery failed - startup may be unstable")
                            # Continue anyway but warn user
                            await self.emit_event(AgentBackendEvent.AUDIO_OUTPUT_ISSUE_DETECTED, {
                                "issue": "audio_health_check_failed",
                                "recovery_attempted": True,
                                "recovery_success": False
                            })
                        else:
                            logger.info("Audio recovery successful - continuing startup")
                
                await self.connect()
                
                # Only start audio health monitoring if enabled
                if self._enable_audio_health_monitoring and self._shutdown_state == "running":
                    self._audio_health_monitor_task = asyncio.create_task(
                        self._audio_health_monitor(),
                        name="audio-health-monitor"
                    )
                    logger.info("Started audio health monitoring")
                
                logger.debug("Pipecat pipeline started as background task")
            
        except Exception as e:
            logger.error(f"Error starting Pipecat backend: {e}")
            raise
    
    async def _on_transcript_update(self, processor, frame):
        """Handle transcript updates from Pipecat's TranscriptProcessor."""
        # Skip processing if we're shutting down or not connected
        if self._shutdown_state != "running" or not self.is_connected:
            return
            
        try:
            # Add timeout to prevent hanging during shutdown
            async with asyncio.timeout(1.0):
                await self._process_transcript_frame(frame)
        except asyncio.TimeoutError:
            logger.warning("Transcript update timed out during shutdown")
        except Exception as e:
            logger.error(f"Error processing transcript update: {e}")
    
    async def _process_transcript_frame(self, frame):
        """Process transcript frame messages."""
        if hasattr(frame, 'messages'):
            for message in frame.messages:
                if hasattr(message, 'role') and hasattr(message, 'content'):
                    if message.role == "user":
                        await self.add_user_speech(message.content)
                        await self.emit_event(AgentBackendEvent.TRANSCRIPTION_RECEIVED, {
                            "text": message.content
                        })
                    elif message.role == "assistant":
                        await self.add_agent_response(message.content)
                        await self.emit_event(AgentBackendEvent.RESPONSE_GENERATED, {
                            "text": message.content
                        })

    async def disconnect(self) -> None:
        """Disconnect the Pipecat backend gracefully."""
        async with self._shutdown_lock:
            if self._shutdown_state != "running":
                return
                
            self._shutdown_state = "stopping"
            logger.info("Disconnecting Pipecat backend...")
            
            try:
                # Check if this is a natural conversation end (flow triggered)
                is_natural_end = self._shutdown_reason == "natural"
                
                # Stop audio health monitoring
                if self._audio_health_monitor_task and not self._audio_health_monitor_task.done():
                    logger.debug("Stopping audio health monitor...")
                    self._audio_health_monitor_task.cancel()
                    try:
                        await asyncio.wait_for(self._audio_health_monitor_task, timeout=1.0)  # Reduced timeout
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass  # Expected when cancelling
                
                # Stop audio tasks FIRST to prevent race conditions with stream cleanup
                if self.transport:
                    try:
                        # First, stop any running audio tasks from the transport streams
                        transport_input = getattr(self.transport, '_input', None)
                        transport_output = getattr(self.transport, '_output', None)
                        
                        task_timeout = 0.5 if self._shutdown_reason == "forced" else 1.0
                        
                        # Cancel audio tasks first to prevent them from writing to streams during cleanup
                        if transport_output and hasattr(transport_output, '_media_sender'):
                            media_sender = transport_output._media_sender
                            if media_sender and hasattr(media_sender, '_cancel_audio_task'):
                                logger.debug("Cancelling output audio task...")
                                try:
                                    await asyncio.wait_for(media_sender._cancel_audio_task(), timeout=task_timeout)
                                except (asyncio.TimeoutError, Exception) as e:
                                    logger.debug(f"Output audio task cancellation error: {e}")
                        
                        if transport_input and hasattr(transport_input, '_media_sender'):
                            media_sender = transport_input._media_sender
                            if media_sender and hasattr(media_sender, '_cancel_audio_task'):
                                logger.debug("Cancelling input audio task...")
                                try:
                                    await asyncio.wait_for(media_sender._cancel_audio_task(), timeout=task_timeout)
                                except (asyncio.TimeoutError, Exception) as e:
                                    logger.debug(f"Input audio task cancellation error: {e}")
                        
                    except Exception as e:
                        logger.debug(f"Audio task cancellation error: {e}")
                
                # Now handle pipeline shutdown
                if is_natural_end and self._pipeline_task and not self._pipeline_task.done():
                    logger.debug("Natural conversation end - waiting for pipeline to complete")
                    try:
                        await asyncio.wait_for(self._pipeline_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning("Pipeline graceful shutdown timed out, forcing cancellation")
                        await self._force_cancel_pipeline()
                    except Exception as e:
                        logger.debug(f"Pipeline completed with error: {e}")
                else:
                    # Forced shutdown or error - cancel immediately with no waiting
                    logger.debug("Forced shutdown detected - cancelling pipeline immediately")
                    await self._force_cancel_pipeline()
                
                # Finally cleanup transport streams safely
                if self.transport:
                    try:
                        # Now cleanup transport streams safely
                        cleanup_timeout = 1.0 if self._shutdown_reason == "forced" else 2.0  # Faster for forced shutdown
                        await asyncio.wait_for(self.transport.cleanup(), timeout=cleanup_timeout)
                        
                        transport_input = getattr(self.transport, '_input', None)
                        transport_output = getattr(self.transport, '_output', None)
                        
                        stream_timeout = 0.5 if self._shutdown_reason == "forced" else 1.0  # Faster for forced shutdown
                        if transport_input and hasattr(transport_input, 'cleanup'):
                            await asyncio.wait_for(transport_input.cleanup(), timeout=stream_timeout)
                        if transport_output and hasattr(transport_output, 'cleanup'):
                            await asyncio.wait_for(transport_output.cleanup(), timeout=stream_timeout)
                            
                    except asyncio.TimeoutError:
                        logger.warning("Transport cleanup timed out - audio streams may not have closed gracefully")
                    except Exception as e:
                        logger.debug(f"Transport cleanup error: {e}")
                    
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
                self._shutdown_reason = "error"
            finally:
                self._shutdown_state = "stopped"
                # Call parent class to disconnect transcript manager
                await super().disconnect()
                logger.info("Pipecat backend disconnected successfully")

    async def _force_cancel_pipeline(self) -> None:
        """Force cancel the pipeline and runner without waiting to prevent recursion."""
        try:
            # Check if pipeline task is already done - if so, skip cancellation
            if self._pipeline_task and self._pipeline_task.done():
                return
                
            # Cancel our pipeline task and wait for it to finish
            if self._pipeline_task and not self._pipeline_task.done():
                self._pipeline_task.cancel()
                
                # Wait for the cancelled task to actually finish with reduced timeout
                # Use even shorter timeout for forced shutdowns
                timeout = 1.0 if self._shutdown_reason == "forced" else 2.0
                try:
                    await asyncio.wait_for(self._pipeline_task, timeout=timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as e:
                    logger.debug(f"Pipeline task cancellation (timeout={timeout}s): {e}")
                    # If it still doesn't cancel after timeout, we'll let the agent service's
                    # aggressive cleanup handle any remaining tasks
                
        except Exception as e:
            logger.debug(f"Error during force cancel: {e}")

    async def stop(self) -> None:
        """Stop the Pipecat backend."""
        if self._shutdown_state != "running":
            return
            
        logger.info("Stopping Pipecat backend...")
        
        # Mark as forced shutdown if not already set
        if self._shutdown_reason is None:
            self._shutdown_reason = "forced"
        
        try:
            await self.disconnect()  # This handles the actual shutdown logic
            logger.info("Pipecat backend stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Pipecat backend: {e}")
        finally:
            # Always call parent class stop
            await super().stop()
    
    def _create_transport(self) -> LocalAudioTransport | MultiChannelAudioTransport:
        """Create audio transport with retry logic for device resolution."""
        import time
        
        # Retry audio device resolution with backoff
        max_attempts = getattr(self.pipecat_config, 'audio_device_retry_attempts', 3)
        retry_delay = getattr(self.pipecat_config, 'audio_device_retry_delay', 1.0)
        
        audio_in_device = None
        audio_out_device = None
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Resolving audio devices (attempt {attempt + 1}/{max_attempts})...")
                
                # Use fast mode if we have device indices to avoid slow enumeration
                use_fast_mode = (
                    self.pipecat_config.audio_input_device_index is not None and 
                    self.pipecat_config.audio_output_device_index is not None and
                    not self.pipecat_config.audio_input_device_name and
                    not self.pipecat_config.audio_output_device_name
                )
                
                if use_fast_mode:
                    logger.info("Using fast mode - bypassing device enumeration")
                
                # Time the resolution process to detect issues
                start_time = time.time()
                
                # Resolve audio device indices
                audio_in_device = resolve_audio_device_index(
                    self.pipecat_config.audio_input_device_index,
                    self.pipecat_config.audio_input_device_name,
                    input_device=True,
                    fast_mode=use_fast_mode
                )
                audio_out_device = resolve_audio_device_index(
                    self.pipecat_config.audio_output_device_index,
                    self.pipecat_config.audio_output_device_name,
                    input_device=False,
                    fast_mode=use_fast_mode
                )
                
                resolution_time = time.time() - start_time
                
                # If resolution was slow and we're not in fast mode, it might indicate a problem
                if resolution_time > 2.0 and not use_fast_mode:
                    logger.warning(f"Audio device resolution was slow ({resolution_time:.2f}s), system may need recovery")
                    # Could trigger recovery here if needed
                
                logger.info(f"Audio devices resolved: input={audio_in_device}, output={audio_out_device} (took {resolution_time:.2f}s)")
                break
                
            except Exception as e:
                logger.warning(f"Audio device resolution attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("All audio device resolution attempts failed")
                    # On final failure, try a recovery if we have specific device indices
                    if (self.pipecat_config.audio_input_device_index is not None and 
                        self.pipecat_config.audio_output_device_index is not None):
                        
                        logger.info("Attempting audio recovery before using default devices...")
                        try:
                            from experimance_common.audio_utils import reset_audio_device_by_name
                            reset_audio_device_by_name("Yealink")
                            time.sleep(3)
                            
                            # Try one more time with the expected indices
                            audio_in_device = self.pipecat_config.audio_input_device_index
                            audio_out_device = self.pipecat_config.audio_output_device_index
                            logger.info(f"Using device indices after recovery: input={audio_in_device}, output={audio_out_device}")
                        except Exception as recovery_error:
                            logger.error(f"Audio recovery failed: {recovery_error}")
                            audio_in_device = None
                            audio_out_device = None
                    else:
                        audio_in_device = None
                        audio_out_device = None
        
        # Create transport
        input_rate = 16000 # NOTE: this is the only value that works with VAD and Assembly
        
        transport_params = LocalAudioTransportParams(
            audio_in_enabled=self.pipecat_config.audio_in_enabled,
            audio_out_enabled=self.pipecat_config.audio_out_enabled,
            audio_in_sample_rate=self.pipecat_config.audio_in_sample_rate,
            audio_out_sample_rate=self.pipecat_config.audio_out_sample_rate,
            input_device_index=audio_in_device,
            output_device_index=audio_out_device,
        )

        # Add audio input filter for resampling if needed
        if self.pipecat_config.audio_in_sample_rate != input_rate:
            logger.info(f"Adding Resample16kFilter: {self.pipecat_config.audio_in_sample_rate}Hz -> {input_rate}Hz for STT compatibility")
            transport_params.audio_in_filter = ResampleFilter(in_rate=self.pipecat_config.audio_in_sample_rate, out_rate=input_rate)
            # Set transport to use mono if device is stereo for consistent processing
            transport_params.audio_in_channels = 1

        # Handle VAD configuration 
        if self.pipecat_config.vad_enabled and self.pipecat_config.mode == "ensemble":
            # Create VAD for 16kHz (our monkey patch prevents sample rate override)
            transport_params.vad_analyzer = SileroVADAnalyzer(sample_rate=input_rate)
            logger.info(f"VAD enabled at {input_rate}Hz")

        logger.info(f"Creating transport with device rate: {self.pipecat_config.audio_in_sample_rate}Hz → pipeline rate: {input_rate}Hz")
        
        # Check if multi-channel output is configured
        if hasattr(self.pipecat_config, 'multi_channel_output') and self.pipecat_config.multi_channel_output:
            transport = self._create_multi_channel_transport(transport_params, input_rate)
        else:
            transport = LocalAudioTransport(transport_params)
        
        return transport
        
    def _create_multi_channel_transport(self, standard_params: LocalAudioTransportParams, input_rate: int) -> MultiChannelAudioTransport:
        """Create multi-channel audio transport with delay support.
        
        Args:
            standard_params: Standard transport parameters to copy from.
            input_rate: Input audio sample rate.
            
        Returns:
            Configured multi-channel audio transport.
        """
        logger.info("Creating multi-channel audio transport...")
        
        # Create multi-channel transport parameters
        mc_params = MultiChannelAudioTransportParams(
            # Copy standard audio parameters
            audio_in_enabled=standard_params.audio_in_enabled,
            audio_out_enabled=standard_params.audio_out_enabled,
            audio_in_channels=standard_params.audio_in_channels,
            audio_out_channels=standard_params.audio_out_channels,
            audio_in_sample_rate=standard_params.audio_in_sample_rate,
            audio_out_sample_rate=standard_params.audio_out_sample_rate,
            audio_in_filter=standard_params.audio_in_filter,
            vad_analyzer=standard_params.vad_analyzer,
            
            # Input device configuration (reuse standard input)
            input_device_index=standard_params.input_device_index,
            
            # Multi-channel output configuration - use standard audio output device fields
            aggregate_device_index=getattr(self.pipecat_config, 'audio_output_device_index', None),
            aggregate_device_name=getattr(self.pipecat_config, 'audio_output_device_name', None),
            output_channels=getattr(self.pipecat_config, 'output_channels', 4),
            channel_delays=getattr(self.pipecat_config, 'channel_delays', {}),
            channel_volumes=getattr(self.pipecat_config, 'channel_volumes', {}),
            max_delay_seconds=getattr(self.pipecat_config, 'max_delay_seconds', 1.0),
        )
        
        logger.info(f"Multi-channel config: {mc_params.output_channels} channels, "
                   f"delays: {mc_params.channel_delays}, volumes: {mc_params.channel_volumes}")
        
        return MultiChannelAudioTransport(mc_params)
        
    def _create_stt_mute_processor(self) -> STTMuteFilter:
        """Create STT mute processor with configurable strategies."""
        if not self.pipecat_config.stt_mute_enabled:
            logger.info("STT mute filter disabled by configuration")
            # Return a pass-through processor or handle this case appropriately
            # For now, we'll create an empty STTMuteFilter
            return STTMuteFilter(config=STTMuteConfig(strategies=set()))
        
        # Map string strategy names to STTMuteStrategy enum values
        strategy_mapping = {
            "always": STTMuteStrategy.ALWAYS,
            "custom": STTMuteStrategy.CUSTOM,
            "first_speech": STTMuteStrategy.FIRST_SPEECH,
            "function_call": STTMuteStrategy.FUNCTION_CALL,
            "mute_until_first_bot_complete": STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE,
        }
        
        # Convert configured strategies to enum values
        strategies = set()
        for strategy_name in self.pipecat_config.stt_mute_strategies:
            strategy_enum = strategy_mapping.get(strategy_name.lower())
            if strategy_enum:
                strategies.add(strategy_enum)
            else:
                logger.warning(f"Unknown STT mute strategy: {strategy_name}")
        
        logger.info(f"Creating STT mute processor with strategies: {[s.name for s in strategies]}")
        
        return STTMuteFilter(
            config=STTMuteConfig(strategies=strategies)
        )

    async def _create_ensemble_pipeline(self) -> None:
        """Create ensemble pipeline with separate STT/LLM/TTS services."""
        logger.info("Creating ensemble pipeline...")
        
        # Create STT service
        if self.pipecat_config.ensemble.stt == "assemblyai":
            if os.getenv("ASSEMBLYAI_API_KEY") is None:
                raise ValueError("AssemblyAI API key is required for ensemble mode")
            
            stt = AssemblyAISTTService(
                api_key=os.getenv("ASSEMBLYAI_API_KEY", "failed"),
                vad_force_turn_endpoint=False,
                connection_params=AssemblyAIConnectionParams(
                    end_of_turn_confidence_threshold=0.7,
                    min_end_of_turn_silence_when_confident=160,
                    max_turn_silence=2400,
                )
            )
        
        # Create LLM service  
        if self.pipecat_config.ensemble.llm == "openai":
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError("OpenAI API key is required")
            
            logger.info("Creating OpenAI LLM service...")
            
            try:
                # Get available tools if not using flows
                available_tools = None
                if not self.pipecat_config.flow_file and self._available_tools:
                    available_tools = self.get_available_tools()
                    logger.info(f"Configuring OpenAI LLM with {len(available_tools)} tools")
                elif not self.pipecat_config.flow_file:
                    logger.info("No tools available for OpenAI LLM")
                else:
                    logger.info("Using flows - tools will be handled by flow manager")
                
                # Create LLM service with tools (if available) - let Pipecat handle function calling natively
                llm = OpenAILLMService(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=self.pipecat_config.openai_model or "gpt-4o-mini",
                    tools=available_tools  # Pipecat handles function calling internally
                )
                logger.info("OpenAI LLM service created successfully")
            except Exception as e:
                logger.error(f"Failed to create OpenAI LLM service: {e}")
                raise

            # Create context aggregator
            context = OpenAILLMContext()
            context_aggregator = llm.create_context_aggregator(context)
        
        # Create TTS service
        if self.pipecat_config.ensemble.tts == "cartesia":
            if os.getenv("CARTESIA_API_KEY") is None:
                raise ValueError("Cartesia API key is required for ensemble mode")
            
            logger.debug("Creating Cartesia TTS service...")
            
            try:
                # Configure Cartesia to output at the device's output sample rate
                output_sample_rate = self.pipecat_config.audio_out_sample_rate
                logger.info(f"Configuring Cartesia TTS for {output_sample_rate}Hz output to match device")
                
                tts = CartesiaTTSService(
                    api_key=os.getenv("CARTESIA_API_KEY", "failed to load"),
                    voice_id=self.config.cartesia_voice_id,
                    model="sonic-2",
                    sample_rate=output_sample_rate,  # Set TTS output to match device rate
                    params=CartesiaTTSService.InputParams(
                        language=Language.EN,
                        speed="fast",  # Options: "fast", "normal", "slow"
                    )
                )
                logger.info(f"Cartesia TTS service created successfully at {output_sample_rate}Hz")
            except Exception as e:
                logger.error(f"Failed to create Cartesia TTS service: {e}")
                raise
        
        # Create event processor
        self.event_processor = PipecatEventProcessor(self)
        
        # Create transcript processor for capturing complete conversations
        transcript_processor = TranscriptProcessor()
        
        # Register transcript event handler as instance method
        transcript_processor.event_handler("on_transcript_update")(self._on_transcript_update)
        
        assert self.transport is not None, "Transport must be created successfully"

        logger.info("Creating pipeline with audio components...")
        
        # Build pipeline components list - audio resampling now handled by transport filter
        pipeline_components = [self.transport.input()]

        # muting during bot speech
        if self.pipecat_config.stt_mute_enabled:
            # Create STT mute processor with configurable strategies
            stt_mute_processor = self._create_stt_mute_processor()
            pipeline_components.append(stt_mute_processor)

        pipeline_components.extend([
            stt,
            transcript_processor.user(),
            context_aggregator.user(),
        ])
        
        # Add simplified function call processor if tools are available and not using flows
        if not self.pipecat_config.flow_file and self._available_tools:
            logger.info("Adding function call processor for tool execution")
            self.function_call_processor = PipecatFunctionCallProcessor(self)
            pipeline_components.append(self.function_call_processor)
        
        # Continue with LLM and remaining components
        # Note: Tools are passed to OpenAI service, processor only handles execution
        pipeline_components.extend([
            llm,
            tts,
            self.transport.output(),
            transcript_processor.assistant(),
            context_aggregator.assistant(),
            self.event_processor
        ])
        
        # Create pipeline
        self.pipeline = Pipeline(pipeline_components)
        
        logger.info("Pipeline created successfully")

        # Create task and runner
        logger.info("Creating pipeline task...")
        self.task = PipelineTask(self.pipeline)
        logger.info("Pipeline task created successfully")
        
        # Check if we should use flows or direct tool calling
        if self.pipecat_config.flow_file:
            # Create flow manager for flow-based conversation
            logger.info("Flow file specified, creating flow manager...")
            flow_config = self._load_flow_config()
            self._flow_config = flow_config  # Store for later use in transitions
            
            self.flow_manager = FlowManager(
                task=self.task,
                llm=llm,
                context_aggregator=context_aggregator,
                flow_config=flow_config
            )

            # Store agent service reference in flow manager state for flow functions to access
            if hasattr(self, 'agent_service') and self.agent_service:
                self.flow_manager.state["_agent_service"] = self.agent_service
                logger.debug("Agent service reference stored in flow manager state")
            else:
                logger.warning("No agent service reference available for flow manager")

            # Initialize the flow manager with the initial node
            try:
                await self.flow_manager.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize flow manager: {e}")
        else:
            # No flow file - set up direct tool calling similar to realtime mode
            logger.info("No flow file specified, setting up direct tool calling...")
            
            # Load system prompt if available
            if self.config.backend_config.prompt_path is not None:
                prompt = load_prompt(self.config.backend_config.prompt_path)
                logger.info(f"Loaded system prompt from {self.config.backend_config.prompt_path}")
            else:
                prompt = "You are a helpful AI assistant. Please assist the user with their questions."
                logger.warning("No prompt file configured, using default prompt")
            
            # Set the system prompt in the context
            try:
                # Add system message to the context
                context.messages.append({
                    "role": "system",
                    "content": prompt
                })
                logger.info("System prompt added to context successfully")
            except Exception as e:
                logger.error(f"Failed to set system prompt: {e}")
                # Fallback - let the LLM handle it without explicit system context
                logger.warning("Continuing without explicit system prompt in context")
            
            # Get available tools and configure LLM for tool calling
            available_tools = self.get_available_tools()
            if available_tools:
                logger.info(f"Tools available for ensemble mode: {len(available_tools)}")
                # Tools are passed to OpenAI service, PipecatFunctionCallProcessor handles execution
            else:
                logger.info("No tools available, ensemble mode will work without tool calling")

    
    async def _create_realtime_pipeline(self) -> None:
        """Create realtime pipeline with OpenAI Realtime Beta."""
        logger.info("Creating realtime pipeline...")
        
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OpenAI API key is required for realtime mode")
        
        if self.config.backend_config.prompt_path is not None:
            prompt = load_prompt(self.config.backend_config.prompt_path)
        else:   
            prompt = "Tell the user something has gone wrong with loading the agent configuration."

        # Get available tools
        available_tools = self.get_available_tools()
        
        logger.info(f"Creating OpenAI Realtime service with {len(available_tools) if available_tools else 0} tools")
        if available_tools:
            logger.debug(f"Tool schemas being sent to OpenAI: {[tool['function']['name'] for tool in available_tools]}")
            # Log the full schemas for debugging
            for tool in available_tools:
                logger.debug(f"Full schema for {tool['function']['name']}: {tool}")
        
        realtime_service = OpenAIRealtimeBetaLLMService(
            api_key=os.getenv("OPENAI_API_KEY", "failed"),
            session_properties=SessionProperties(
                instructions=prompt,
                voice="alloy",
                turn_detection=TurnDetection(type="server_vad")
            ),
            # Add tools if available
            tools=available_tools if available_tools else None
        )
        
        logger.info("OpenAI Realtime service created successfully")
        
        # Create event processor
        self.event_processor = PipecatEventProcessor(self)
        
        # Add simplified function call processor if tools are available
        if available_tools:
            logger.info("Adding function call processor for realtime mode")
            self.function_call_processor = PipecatFunctionCallProcessor(self)
        else:
            self.function_call_processor = None
        
        # For realtime mode, we might need to handle transcription differently
        # since the OpenAI Realtime API handles conversation internally
        # TODO: Investigate how to get transcription events from realtime service
        
        assert self.transport is not None, "Transport must be created successfully"
        
        # Create pipeline - add function call processor if tools are available
        pipeline_components = [self.transport.input()]
        
        if self.function_call_processor:
            pipeline_components.append(self.function_call_processor)
            
        pipeline_components.extend([
            realtime_service,  # Realtime service handles function calls internally
            self.transport.output(),
            self.event_processor
        ])
        
        self.pipeline = Pipeline(pipeline_components)

        self.task = PipelineTask(self.pipeline)
    
    def _handle_backend_event(self, event: AgentBackendEvent) -> None:
        """Handle events from the event processor."""
        # This method is no longer needed as we use the base class emit_event
        pass
    
    async def send_message(self, message: str, speaker: str = "system", say_tts: bool = False) -> None:
        """Send a message to the conversation.
        
        Args:
            message: The message text to send
            speaker: The speaker role ("system", "user", "agent")
            say_tts: If True, send as TextFrame for immediate TTS output instead of context message
        """
        if not self.is_connected:
            logger.warning("Backend not connected, cannot send message")
            return
            
        try:
            if self.task:
                if say_tts:
                    # Send directly to TTS for immediate speech output
                    from pipecat.frames.frames import TextFrame
                    tts_frame = TextFrame(text=message)
                    await self.task.queue_frame(tts_frame)
                    logger.debug(f"Sent TTS frame: {message}")
                else:
                    # Use LLMMessagesAppendFrame to add messages to the context
                    from pipecat.frames.frames import LLMMessagesAppendFrame
                    
                    if speaker == "system":
                        # Add system message to context
                        llm_message = {"role": "system", "content": message}
                    else:
                        # Add user message to context
                        llm_message = {"role": "user", "content": message}
                    
                    # Send the message frame to the pipeline
                    append_frame = LLMMessagesAppendFrame(messages=[llm_message])
                    await self.task.queue_frame(append_frame)
                    logger.debug(f"Sent context message from {speaker}: {message}")
                    
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def trigger_response(self, prompt: str) -> None:
        """Trigger the LLM to generate an immediate response by simulating user input.
        
        This method simulates a user transcription to trigger the LLM pipeline to process
        the prompt and generate a spoken response. This is the proper way to make the
        agent speak proactively in Pipecat.
        
        Args:
            prompt: The prompt to trigger the LLM response with
        """
        if not self.is_connected:
            logger.warning("Backend not connected, cannot trigger response")
            return
            
        try:
            if self.task:
                # Simulate a transcription frame to trigger LLM processing
                from pipecat.frames.frames import TranscriptionFrame
                import time
                
                # Create a transcription frame that looks like the user spoke the prompt
                transcription_frame = TranscriptionFrame(
                    text=prompt,
                    user_id="visitor",
                    timestamp=str(int(time.time())),
                    language=Language.EN
                )
                
                # Send the transcription frame to trigger LLM processing
                await self.task.queue_frame(transcription_frame)
                logger.debug(f"Sent transcription frame to trigger response: {prompt}")
                
        except Exception as e:
            logger.error(f"Error triggering response: {e}")
    
    async def interrupt_bot(self) -> None:
        """Interrupt the bot if it's currently speaking."""
        if self.task and self.is_connected:
            try:
                # Send cancel frame to interrupt
                await self.task.queue_frame(CancelFrame())
                logger.debug("Bot interrupted")
            except Exception as e:
                logger.error(f"Error interrupting bot: {e}")
    
    async def end_conversation_naturally(self) -> None:
        """End the conversation naturally by sending an EndFrame.
        
        This triggers the natural shutdown sequence that the goodbye node would use,
        allowing the conversation to end gracefully rather than being forcibly stopped.
        """
        if self.task and self.is_connected:
            try:
                # Send EndFrame to signal natural conversation end
                await self.task.queue_frame(EndFrame())
                logger.info("Sent EndFrame to end conversation naturally")
            except Exception as e:
                logger.error(f"Error ending conversation naturally: {e}")
    
    async def graceful_shutdown(self, goodbye_message: Optional[str] = None) -> None:
        """
        Gracefully shutdown the conversation pipeline, allowing any final messages to be processed before terminating.
        This is the proper way to end a conversation when the user leaves or the session should end naturally.
        
        Args:
            goodbye_message: Optional goodbye message to say before shutting down (not yet implemented)
        """
        logger.info("Starting graceful shutdown of Pipecat backend")
        
        # For now, just end the conversation naturally - in the future we could implement
        # saying a goodbye message first if provided
        if goodbye_message:
            logger.debug(f"TODO: Say goodbye message before shutdown: {goodbye_message}")
            # await self.send_message(goodbye_message, speaker="agent", say_tts=True)
            # await asyncio.sleep(2.0)  # Give time for TTS to complete
        
        await self.end_conversation_naturally()
    
    async def say_goodbye_and_shutdown(self, goodbye_message: str = "Thank you for visiting Experimance. Have a wonderful day!") -> None:
        """
        Say a goodbye message and then gracefully shutdown the conversation.
        
        Args:
            goodbye_message: The goodbye message to speak before shutting down
        """
        logger.info(f"Saying goodbye and shutting down: {goodbye_message}")
        
        try:
            # Send the goodbye message with TTS
            await self.send_message(goodbye_message, speaker="agent", say_tts=True)
            
            # Give some time for the TTS to complete
            await asyncio.sleep(3.0)
            
            # Then end the conversation naturally
            await self.end_conversation_naturally()
            
        except Exception as e:
            logger.error(f"Error during goodbye and shutdown: {e}")
            # Fallback to just ending naturally
            await self.end_conversation_naturally()
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        super().clear_conversation_history()  # Call base class method
        
        # Also clear transcript manager if available
        if hasattr(self, 'transcript_manager'):
            self.transcript_manager._messages.clear()
            
        # Reset context using frames instead of direct access to aggregator
        if self.task and self.is_connected:
            try:
                from pipecat.frames.frames import LLMMessagesUpdateFrame
                # Clear context by sending an empty messages list
                reset_frame = LLMMessagesUpdateFrame(messages=[])
                asyncio.create_task(self.task.queue_frame(reset_frame))
            except Exception as e:
                logger.warning(f"Failed to reset context: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information."""
        return {
            "is_connected": self.is_connected,
            "mode": self.config.backend_config.pipecat.mode,
            "conversation_turns": len(self._conversation_history),
            "user_context": self.user_context.__dict__ if self.user_context else None
        }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools for the current LLM service."""
        logger.info(f"get_available_tools called. Registered tools: {list(self._available_tools.keys())}")
        
        # Convert registered tools to OpenAI function schema format
        tools = []
        for tool_name, tool_func in self._available_tools.items():
            logger.debug(f"Creating tool schema for: {tool_name}")
            
            # Use provided schema if available, otherwise create a generic one
            if tool_name in self._tool_schemas and self._tool_schemas[tool_name]:
                tool_schema = self._tool_schemas[tool_name]
                logger.debug(f"Using provided schema for {tool_name}")
            else:
                # Generic tool schema fallback
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": getattr(tool_func, "__doc__", f"Execute {tool_name}"),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
                logger.debug(f"Using generic schema for {tool_name}")
            
            tools.append(tool_schema)
            logger.debug(f"Added tool schema for {tool_name}")
        
        logger.info(f"Returning {len(tools)} tool schemas to OpenAI")
        return tools

    async def call_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call."""
        tool_name = tool_call.tool_name
        parameters = tool_call.parameters
        
        logger.debug(f"Executing tool call: {tool_name} with parameters: {parameters}")
        
        if tool_name not in self._available_tools:
            error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self._available_tools.keys())}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # Execute the tool function
            tool_func = self._available_tools[tool_name]
            
            # Call the function with parameters
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**parameters)
            else:
                result = tool_func(**parameters)
            
            logger.debug(f"Tool '{tool_name}' executed successfully")
            
            # Emit tool called event
            await self.emit_event(AgentBackendEvent.TOOL_CALLED, {
                "tool_name": tool_name,
                "parameters": parameters,
                "result": result
            })
            
            # Add to transcript
            await self.add_tool_call(tool_name, parameters, result)
            
            return {"result": result}
            
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}
    
    def get_pipeline_task(self) -> Optional[asyncio.Task]:
        """Get the pipeline task for external management."""
        return self._pipeline_task
    
    async def transition_to_node(self, node_name: str) -> bool:
        """Transition to a specific node in the flow."""
        if not self.flow_manager:
            logger.warning("Cannot transition to node: flow manager not available")
            return False
            
        try:
            # For static flows, we need to create the node config from the flow_config
            if hasattr(self, '_flow_config') and self._flow_config:
                nodes = self._flow_config.get("nodes", {})
                if node_name in nodes:
                    node_config = nodes[node_name]
                    await self.flow_manager.set_node_from_config(node_config)
                    logger.info(f"Successfully transitioned to node: {node_name}")
                    return True
                else:
                    logger.error(f"Node '{node_name}' not found in flow configuration")
                    return False
            else:
                logger.error("Flow configuration not available for node transition")
                return False
                
        except Exception as e:
            logger.error(f"Failed to transition to node '{node_name}': {e}")
            return False
    
    def get_current_node(self) -> Optional[str]:
        """Get the current active node name."""
        if self.flow_manager and hasattr(self.flow_manager, 'current_node'):
            return self.flow_manager.current_node
        return None
    
    def is_conversation_active(self) -> bool:
        """Check if conversation is currently active (not in search or goodbye)."""
        current_node = self.get_current_node()
        return current_node not in ["search", "goodbye"] if current_node else False
