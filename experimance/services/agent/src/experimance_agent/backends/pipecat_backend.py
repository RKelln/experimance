"""
Pipecat backend v2 for the Experimance agent service.

This module implements the AgentBackend interface using Pipecat's local audio pipeline,
providing speech-to-text, LLM conversation, and text-to-speech capabilities in a single process.

Based on the working ensemble implementation from flows_test.py with proper shutdown handling.
"""
import asyncio
import logging
import os
import sys
import time
import importlib.util
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable
from pathlib import Path

from experimance_agent.config import AgentServiceConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.whisper.stt import WhisperSTTService
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
from pipecat.frames.frames import (
    Frame, AudioRawFrame, TextFrame, TranscriptionFrame, 
    TTSStartedFrame, TTSStoppedFrame, UserStartedSpeakingFrame, 
    UserStoppedSpeakingFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
    SystemFrame, CancelFrame, EndFrame
)
from pipecat_flows import FlowManager, FlowArgs, FlowResult, FlowConfig

from experimance_common.audio_utils import resolve_audio_device_index
from experimance_common.constants import AGENT_SERVICE_DIR
from .base import AgentBackend, AgentBackendEvent, ConversationTurn, ToolCall, UserContext, load_prompt

logger = logging.getLogger(__name__)


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
                logger.info("Pipeline CancelFrame received without EndFrame (forced shutdown)")
                await self.backend.emit_event(AgentBackendEvent.CANCEL, {
                    "reason": "pipeline_cancelled"
                })
        
        # Handle transcription frames
        # elif isinstance(frame, TranscriptionFrame):
        #     if frame.text and frame.text.strip():
        #         await self.backend.emit_event(AgentBackendEvent.TRANSCRIPTION_RECEIVED, {
        #             "content": frame.text,
        #             "speaker": "user",  # TranscriptionFrame is typically from user speech
        #             "is_partial": getattr(frame, 'is_partial', False)
        #         })
        
        # # Handle agent text responses for transcription
        # elif isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
        #     if frame.text and frame.text.strip():
        #         await self.backend.emit_event(AgentBackendEvent.TRANSCRIPTION_RECEIVED, {
        #             "content": frame.text,
        #             "speaker": "agent",  # TextFrame from LLM is agent response
        #             "is_partial": False
        #         })
            
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
        self.transport: Optional[LocalAudioTransport] = None
        self.flow_manager: Optional[FlowManager] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._flow_config: Optional[FlowConfig] = None
        
        # Shutdown state management
        self._shutdown_state = "running"  # "running", "stopping", "stopped"
        self._shutdown_reason = None  # "natural", "forced", "error"
        self._shutdown_lock = asyncio.Lock()  # Prevent concurrent shutdown operations
        
        # Event handling
        self.event_processor: Optional[PipecatEventProcessor] = None
        

    def _load_flow_config(self) -> FlowConfig:
        """Load flow configuration from the specified flow file."""
        flow_file = self.pipecat_config.flow_file
        if not flow_file:
            raise ValueError("Flow file must be specified in the configuration")
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

            self.transport = self._create_transport()
            assert self.transport is not None, "Transport must be created successfully"
            
            # Create pipeline based on mode
            if self.config.backend_config.pipecat.mode == "realtime":
                await self._create_realtime_pipeline()
            else:
                await self._create_ensemble_pipeline()
            
            assert self.pipeline is not None, "Pipeline must be created successfully"
            assert self.task is not None, "Pipeline task must be created successfully"

            # runner
            self.runner = PipelineRunner(
                handle_sigint=False,  # handle these ourselves
                #handle_sigterm=False, # handle these ourselves (newer version of pipecat)
            )
            
            logger.info("Pipecat backend v2 started successfully")

            # Start the pipeline as a background task - this doesn't block
            if self.runner and self.task:
                self._pipeline_task = asyncio.create_task(
                    self.runner.run(self.task),
                    name="pipecat-pipeline"
                )
                await self.connect()
                logger.info("Pipecat pipeline started as background task")
            
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
            async with asyncio.timeout(2.0):
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
                    # Forced shutdown or error - cancel immediately
                    await self._force_cancel_pipeline()
                
                # Cleanup transport
                if self.transport:
                    try:
                        await self.transport.cleanup()
                        
                        # Clean up transport streams if available
                        transport_input = getattr(self.transport, '_input', None)
                        transport_output = getattr(self.transport, '_output', None)
                        
                        if transport_input and hasattr(transport_input, 'cleanup'):
                            await transport_input.cleanup()
                        if transport_output and hasattr(transport_output, 'cleanup'):
                            await transport_output.cleanup()
                            
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
                
                # Wait for the cancelled task to actually finish
                try:
                    await asyncio.wait_for(self._pipeline_task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as e:
                    logger.debug(f"Pipeline task cancellation: {e}")
                
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
    
    def _create_transport(self) -> LocalAudioTransport:
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
        transport_params = LocalAudioTransportParams(
            audio_in_enabled=self.pipecat_config.audio_in_enabled,
            audio_out_enabled=self.pipecat_config.audio_out_enabled,
            audio_in_sample_rate=self.pipecat_config.audio_in_sample_rate,
            audio_out_sample_rate=self.pipecat_config.audio_out_sample_rate,
            input_device_index=audio_in_device,
            output_device_index=audio_out_device,
        )

        if self.pipecat_config.vad_enabled and self.pipecat_config.mode == "ensemble":
            transport_params.vad_analyzer = SileroVADAnalyzer()

        return LocalAudioTransport(transport_params)
        

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
            
            llm = OpenAILLMService(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=self.pipecat_config.openai_model or "gpt-4o-mini"
            )

            # Create context aggregator
            context = OpenAILLMContext()
            context_aggregator = llm.create_context_aggregator(context)
        
        # Create TTS service
        if self.pipecat_config.ensemble.tts == "cartesia":
            if os.getenv("CARTESIA_API_KEY") is None:
                raise ValueError("Cartesia API key is required for ensemble mode")
            
            tts = CartesiaTTSService(
                api_key=os.getenv("CARTESIA_API_KEY", "failed to load"),
                voice_id=self.config.cartesia_voice_id,
                model="sonic-2",
                params=CartesiaTTSService.InputParams(
                    language=Language.EN,
                    speed="fast",  # Options: "fast", "normal", "slow"
                )
            )
        
        # Create event processor
        self.event_processor = PipecatEventProcessor(self)
        
        # Create transcript processor for capturing complete conversations
        transcript_processor = TranscriptProcessor()
        
        # Register transcript event handler as instance method
        transcript_processor.event_handler("on_transcript_update")(self._on_transcript_update)
        
        # Mute user during function calls
        stt_mute_processor = STTMuteFilter(
            config=STTMuteConfig(
                strategies={
                    STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE,
                    STTMuteStrategy.FUNCTION_CALL,
                }
            ),
        )

        assert self.transport is not None, "Transport must be created successfully"

        # Create pipeline
        self.pipeline = Pipeline([
            self.transport.input(),
            stt_mute_processor,
            stt,
            transcript_processor.user(),
            context_aggregator.user(),
            llm,
            tts,
            self.transport.output(),
            transcript_processor.assistant(),
            context_aggregator.assistant(),
            self.event_processor
        ])

        # Create task and runner
        self.task = PipelineTask(self.pipeline)
        
        # Create flow manager
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

    
    async def _create_realtime_pipeline(self) -> None:
        """Create realtime pipeline with OpenAI Realtime Beta."""
        logger.info("Creating realtime pipeline...")
        
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OpenAI API key is required for realtime mode")
        
        if self.config.backend_config.prompt_path is not None:
            prompt = load_prompt(self.config.backend_config.prompt_path)
        else:   
            prompt = "Tell the user something has gone wrong with loading the agent configuration."

        realtime_service = OpenAIRealtimeBetaLLMService(
            api_key=os.getenv("OPENAI_API_KEY", "failed"),
            session_properties=SessionProperties(
                instructions=prompt,
                voice="alloy",
                turn_detection=TurnDetection(type="server_vad")
            )
        )
        
        # Create event processor
        self.event_processor = PipecatEventProcessor(self)
        
        # For realtime mode, we might need to handle transcription differently
        # since the OpenAI Realtime API handles conversation internally
        # TODO: Investigate how to get transcription events from realtime service
        
        assert self.transport is not None, "Transport must be created successfully"
        
        # Create pipeline
        self.pipeline = Pipeline([
            self.transport.input(),
            realtime_service,
            self.transport.output(),
            self.event_processor
        ])

        self.task = PipelineTask(self.pipeline)
    
    def _handle_backend_event(self, event: AgentBackendEvent) -> None:
        """Handle events from the event processor."""
        # This method is no longer needed as we use the base class emit_event
        pass
    
    async def send_message(self, text: str, speaker: str = "system") -> None:
        """Send a message to the conversation."""
        if not self.is_connected:
            logger.warning("Backend not connected, cannot send message")
            return
            
        try:
            # Use LLMMessagesAppendFrame to add messages to the context
            if self.task:
                from pipecat.frames.frames import LLMMessagesAppendFrame
                
                if speaker == "system":
                    # Add system message to context
                    message = {"role": "system", "content": text}
                else:
                    # Add user message to context
                    message = {"role": "user", "content": text}
                
                # Send the message frame to the pipeline
                append_frame = LLMMessagesAppendFrame(messages=[message])
                await self.task.queue_frame(append_frame)
                    
            logger.debug(f"Sent message from {speaker}: {text}")
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def interrupt_bot(self) -> None:
        """Interrupt the bot if it's currently speaking."""
        if self.task and self.is_connected:
            try:
                # Send cancel frame to interrupt
                await self.task.queue_frame(CancelFrame())
                logger.debug("Bot interrupted")
            except Exception as e:
                logger.error(f"Error interrupting bot: {e}")
    
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
        """Get available tools for the current flow."""
        # For now, return empty list - tools would be defined in flow config
        return []
    
    async def call_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call."""
        # Tool calling would be handled by the flow manager
        # For now, return empty result
        return {"result": "Tool calling not yet implemented"}
    
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
