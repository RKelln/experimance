"""
Pipecat backend for the Experimance agent service.

This module implements the AgentBackend interface using Pipecat's local audio pipeline,
providing speech-to-text, LLM conversation, and text-to-speech capabilities in a single process.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, AsyncGenerator

from dataclasses import dataclass

from experimance_agent.config import AgentServiceConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai_realtime_beta import OpenAIRealtimeBetaLLMService
from pipecat.services.openai_realtime_beta.events import SessionProperties, TurnDetection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame, 
    TextFrame,
    TranscriptionFrame,
    TranscriptionMessage,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    EndFrame,
    EndTaskFrame,
)

from .base import (
    AgentBackend, 
    AgentBackendEvent, 
    ConversationTurn, 
    ToolCall, 
    UserContext,
    load_prompt
)

from experimance_common.audio_utils import resolve_audio_device_index
from openai.types.chat import ChatCompletionMessageParam  # Add this import

logger = logging.getLogger(__name__)


class PipecatEventProcessor(FrameProcessor):
    """Frame processor that captures Pipecat events and forwards them to the backend."""
    
    def __init__(self, backend: 'PipecatBackend'):
        super().__init__()
        self.backend = backend
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and emit appropriate events."""
        await super().process_frame(frame, direction)
        
        # Handle transcription frames (user speech)
        if isinstance(frame, TranscriptionFrame):
            turn = ConversationTurn(
                speaker="human",
                content=frame.text,
                timestamp=time.time(),
                metadata={"frame_type": "transcription"}
            )
            self.backend._conversation_history.append(turn)
            await self.backend.emit_event(
                AgentBackendEvent.TRANSCRIPTION_RECEIVED, 
                {"transcription": frame.text, "turn": turn}
            )
            
        # Handle LLM response frames (agent speech)
        elif isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            turn = ConversationTurn(
                speaker="agent",
                content=frame.text,
                timestamp=time.time(),
                metadata={"frame_type": "llm_response"}
            )
            self.backend._conversation_history.append(turn)
            await self.backend.emit_event(
                AgentBackendEvent.RESPONSE_GENERATED,
                {"response": frame.text, "turn": turn}
            )
            
        # Handle speech detection events
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self.backend.emit_event(AgentBackendEvent.SPEECH_DETECTED)
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.backend.emit_event(AgentBackendEvent.SPEECH_ENDED)
            
        elif isinstance(frame, BotStartedSpeakingFrame):
            if not self.backend.conversation_started:
                # Emit conversation started event only once
                self.backend.conversation_started = True
                await self.backend.emit_event(
                    AgentBackendEvent.CONVERSATION_STARTED,
                    {"speaker": "agent"}
                )
            await self.backend.emit_event(AgentBackendEvent.BOT_STARTED_SPEAKING)
            
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Don't emit CONVERSATION_ENDED here as the conversation continues
            await self.backend.emit_event(AgentBackendEvent.BOT_STOPPED_SPEAKING)
            
        # Forward the frame downstream
        await self.push_frame(frame, direction)


class PipecatToolProcessor(FrameProcessor):
    """Frame processor that handles tool calls from the LLM."""
    
    def __init__(self, backend: 'PipecatBackend'):
        super().__init__()
        self.backend = backend
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle tool calls if present."""
        await super().process_frame(frame, direction)
        
        # TODO: Implement tool call detection and handling
        # Pipecat's OpenAI LLM service supports function calling
        # We would need to detect function call frames and route them through handle_tool_call
        
        await self.push_frame(frame, direction)


class PipecatBackend(AgentBackend):
    """
    Pipecat-based agent backend using local audio transport.
    
    This backend runs the entire speech-to-text, LLM, and text-to-speech pipeline
    in a single process using Pipecat's LocalAudioTransport.
    
    Shutdown Methods:
    - stop(): Immediate shutdown using task cancellation - for force stops
    - graceful_shutdown(): Graceful shutdown using EndFrame - for when user leaves
    - say_goodbye_and_shutdown(): Says goodbye message then graceful shutdown
    - disconnect(): Graceful disconnect while keeping backend ready to reconnect
    """
    
    def __init__(self, config: AgentServiceConfig):
        """Initialize the Pipecat backend."""
        super().__init__(config)
        
        # Parse configuration
        self.pipecat_config = config.backend_config.pipecat
            
        # Pipeline components
        self.transport: Optional[LocalAudioTransport] = None
        self.pipeline: Optional[Pipeline] = None
        self.pipeline_task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        self.llm_context: Optional[OpenAILLMContext] = None
        
        # Control
        self._pipeline_running = False
        self._shutdown_event = asyncio.Event()
        
        # User context
        self.user_context = UserContext()

        # Audio device indices
        self.input_device_index: Optional[int] = None
        self.output_device_index: Optional[int] = None
        
    async def start(self) -> None:
        """Start the Pipecat backend and initialize the pipeline."""
        logger.info(f"Starting Pipecat backend in {self.pipecat_config.mode} mode...")
        
        try:
            # Create audio transport with updated VAD configuration
            # Resolve device indices from names if provided
            self.input_device_index = resolve_audio_device_index(
                self.pipecat_config.audio_input_device_index,
                self.pipecat_config.audio_input_device_name,
                input_device=True
            )
            self.output_device_index = resolve_audio_device_index(
                self.pipecat_config.audio_output_device_index,
                self.pipecat_config.audio_output_device_name,
                input_device=False
            )
            
            transport_params = LocalAudioTransportParams(
                audio_in_enabled=self.pipecat_config.audio_in_enabled,
                audio_out_enabled=self.pipecat_config.audio_out_enabled,
                audio_in_sample_rate=self.pipecat_config.audio_in_sample_rate,
                audio_out_sample_rate=self.pipecat_config.audio_out_sample_rate,
                input_device_index=self.input_device_index,
                output_device_index=self.output_device_index,
            )
            
            # Set up VAD analyzer (replacing deprecated vad_enabled parameter)
            if self.pipecat_config.vad_enabled:
                transport_params.vad_analyzer = SileroVADAnalyzer()
            
            self.transport = LocalAudioTransport(transport_params)
            
            # Create our custom processors
            event_processor = PipecatEventProcessor(self)
            tool_processor = PipecatToolProcessor(self)
            
            # Build pipeline based on mode
            if self.pipecat_config.mode == "realtime":
                await self._start_realtime_mode(event_processor, tool_processor)
            else:  # ensemble mode
                await self._start_ensemble_mode(event_processor, tool_processor)
            
            # Create pipeline task and runner
            assert self.pipeline, "Pipeline must be initialized before starting task"
            self.pipeline_task = PipelineTask(self.pipeline)
            self.runner = PipelineRunner()
            
            self.is_active = True
            logger.info(f"Pipecat backend started successfully in {self.pipecat_config.mode} mode")
            
        except Exception as e:
            logger.error(f"Failed to start Pipecat backend: {e}")
            await self.stop()
            raise
            
    async def _start_realtime_mode(self, event_processor: 'PipecatEventProcessor', tool_processor: 'PipecatToolProcessor') -> None:
        """Initialize pipeline for OpenAI Realtime Beta mode."""
        logger.info("Initializing OpenAI Realtime Beta pipeline...")
        
        assert self.transport, "Transport must be initialized before starting pipeline"

        # Create OpenAI Realtime service
        realtime_service = OpenAIRealtimeBetaLLMService(
            api_key=os.getenv("OPENAI_API_KEY", "failed to load"),
            model=self.pipecat_config.openai_realtime_model,
            session_properties=SessionProperties(
                modalities=["audio", "text"],
                instructions=load_prompt(self.pipecat_config.system_prompt),
                voice=self.pipecat_config.openai_voice,
                turn_detection=TurnDetection(
                    threshold=self.pipecat_config.turn_detection_threshold,
                    silence_duration_ms=self.pipecat_config.turn_detection_silence_ms
                ),
                temperature=0.7,
                #tools=[]
            )
        )

        transcript = TranscriptProcessor()
        self.llm_context = OpenAILLMContext()
        context_aggregator = realtime_service.create_context_aggregator(self.llm_context)

        # Build simplified pipeline for realtime mode
        self.pipeline = Pipeline([
            self.transport.input(),      # Audio input (microphone)
            context_aggregator.user(),   # Add user message to context
            realtime_service,            # OpenAI Realtime Beta (STT + LLM + TTS in one)
            transcript.user(),           # Placed after the LLM, as LLM pushes TranscriptionFrames downstream
            event_processor,             # Capture events
            tool_processor,              # Handle tool calls (if supported)
            self.transport.output(),     # Audio output (speakers)
            transcript.assistant(),      # After the transcript output, to time with the audio output
            context_aggregator.assistant()  # Add assistant response to context
        ])

        @transcript.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            for msg in frame.messages:
                if isinstance(msg, TranscriptionMessage):
                    timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                    line = f"{timestamp}{msg.role}: {msg.content}"
                    logger.info(f"Transcript: {line}")
        
    async def _start_ensemble_mode(self, event_processor: 'PipecatEventProcessor', tool_processor: 'PipecatToolProcessor') -> None:
        """Initialize pipeline for ensemble mode with separate STT/LLM/TTS services."""
        logger.info("Initializing ensemble pipeline with separate STT/LLM/TTS services...")
        
        assert self.transport, "Transport must be initialized before starting pipeline" 

        # Create STT service
        stt = WhisperSTTService(model=self.pipecat_config.whisper_model)
        
        # Create LLM service and context
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY", "failed to load"),
            model=self.pipecat_config.openai_model
        )
        
        system_prompt = load_prompt(self.pipecat_config.system_prompt)
        self.llm_context = OpenAILLMContext(
            messages=[{
                "role": "system",
                "content": system_prompt
            }],
            #tools=tools
        )
        context_aggregator = llm.create_context_aggregator(self.llm_context)
        
        # Create TTS service  
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY", "failed to load"),
            voice_id=self.pipecat_config.elevenlabs_voice_id
        )
        
        # Create sentence aggregator for better speech flow
        sentence_aggregator = SentenceAggregator()
        
        # Build full ensemble pipeline
        self.pipeline = Pipeline([
            self.transport.input(),       # Audio input (microphone)
            stt,                         # Speech-to-text
            context_aggregator.user(),   # Add user message to context
            llm,                         # Language model
            tool_processor,              # Handle tool calls
            sentence_aggregator,         # Aggregate into sentences
            event_processor,             # Capture events
            tts,                         # Text-to-speech
            self.transport.output(),     # Audio output (speakers)
            context_aggregator.assistant(), # Add assistant response to context
        ])
    
    async def stop(self) -> None:
        """Stop the Pipecat backend immediately and clean up resources."""
        logger.info("Stopping Pipecat backend immediately...")
        
        self.is_active = False
        self.is_connected = False
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Immediate shutdown using task cancellation (as per Pipecat docs)
        if self.pipeline_task:
            try:
                logger.info("Cancelling pipeline task for immediate shutdown...")
                await self.pipeline_task.cancel()
            except Exception as e:
                # Suppress common errors when pipeline hasn't fully started yet
                error_msg = str(e).lower()
                if any(phrase in error_msg for phrase in [
                    "startframe not received yet",
                    "trying to process cancelframe",
                    "but startframe not received"
                ]):
                    logger.debug(f"Pipeline task cancelled before full startup (this is normal): {e}")
                else:
                    logger.error(f"Error cancelling pipeline task: {e}")
        
        # Clean up components
        self.transport = None
        self.pipeline = None
        self.pipeline_task = None
        self.runner = None
        self.llm_context = None
        self._pipeline_running = False
        
        logger.info("Pipecat backend stopped")
        
    async def graceful_shutdown(self, goodbye_message: Optional[str] = None) -> None:
        """
        Gracefully shutdown the conversation when user leaves.
        
        This method uses EndFrame to allow any final messages to be processed
        before terminating the pipeline. Should be called when vision detects
        no users are present or core service indicates the session should end.
        
        Args:
            goodbye_message: Optional goodbye message to say before shutting down
        """
        if not self.is_connected or not self.pipeline_task:
            logger.warning("Cannot gracefully shutdown: pipeline not connected")
            return
            
        logger.info("Initiating graceful shutdown of conversation...")
        
        try:
            # If we have a goodbye message, say it first
            if goodbye_message:
                logger.info(f"Saying goodbye: {goodbye_message}")
                # We could inject the goodbye message into the pipeline here
                # This would depend on having a way to inject TextFrames
                pass
            
            # Queue an EndFrame for graceful shutdown (as per Pipecat docs)
            logger.info("Queueing EndFrame for graceful shutdown...")
            await self.pipeline_task.queue_frame(EndFrame())
            
            # Update connection status but keep backend active for potential reconnection
            self.is_connected = False
            self._pipeline_running = False
            
            # Emit disconnection event
            await self.emit_event(AgentBackendEvent.DISCONNECTED)
            
            logger.info("Graceful shutdown initiated - pipeline will end after processing current frames")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            # Fall back to immediate shutdown if graceful fails
            await self.stop()
    
    async def say_goodbye_and_shutdown(self, goodbye_message: str = "Thank you for visiting Experimance. Have a wonderful day!") -> None:
        """
        Say a goodbye message and then gracefully shutdown.
        
        This is the recommended way to end a conversation when the user leaves,
        as detected by vision or signaled by the core service.
        
        Args:
            goodbye_message: The goodbye message to speak before shutting down
        """
        if not self.is_connected or not self.pipeline_task:
            logger.warning("Cannot say goodbye: pipeline not connected")
            return
            
        logger.info(f"Saying goodbye and shutting down: {goodbye_message}")
        
        try:
            # Send the goodbye message to the conversation
            await self.send_message(goodbye_message, speaker="agent")
            
            # Wait a moment for the message to be processed
            await asyncio.sleep(1.0)
            
            # Now initiate graceful shutdown
            await self.graceful_shutdown()
            
        except Exception as e:
            logger.error(f"Error during goodbye and shutdown: {e}")
            # Fall back to immediate shutdown
            await self.stop()
        
    async def connect(self) -> None:
        """Connect and start the conversation pipeline."""
        if not self.is_active:
            raise RuntimeError("Backend must be started before connecting")
            
        logger.info("Connecting Pipecat pipeline...")
        
        try:
            # Start the pipeline in the background
            if self.runner and self.pipeline_task:
                self._pipeline_running = True
                
                # Start pipeline as background task
                async def run_pipeline():
                    try:
                        await self.runner.run(self.pipeline_task) # type: ignore
                    except Exception as e:
                        logger.error(f"Pipeline error: {e}")
                        await self.emit_event(AgentBackendEvent.ERROR, {"error": str(e)})
                    finally:
                        self._pipeline_running = False
                        
                asyncio.create_task(run_pipeline())
                
                # Give the pipeline a moment to initialize
                await asyncio.sleep(0.5)
                
                self.is_connected = True
                await self.emit_event(AgentBackendEvent.CONNECTED)
                logger.info("Pipecat pipeline connected and running")
            else:
                raise RuntimeError("Pipeline components not initialized")
                
        except Exception as e:
            logger.error(f"Failed to connect Pipecat pipeline: {e}")
            raise
            
    async def disconnect(self) -> None:
        """Disconnect the pipeline while keeping backend active."""
        logger.info("Disconnecting Pipecat pipeline...")
        
        self.is_connected = False
        
        # Stop the pipeline but keep the backend active
        if self.pipeline:
            try:
                # For disconnect, we should use graceful shutdown instead
                if self.pipeline_task:
                    await self.graceful_shutdown()
                    return
            except Exception as e:
                logger.error(f"Error disconnecting pipeline: {e}")
                
        self._pipeline_running = False
        await self.emit_event(AgentBackendEvent.DISCONNECTED)
        logger.info("Pipecat pipeline disconnected")
        
    async def send_message(self, message: str, speaker: str = "system") -> None:
        """Send a message to the conversation."""
        if not self.is_connected:
            raise RuntimeError("Backend must be connected to send messages")
            
        # Add message to conversation history
        turn = ConversationTurn(
            speaker=speaker,
            content=message,
            timestamp=time.time(),
            metadata={"injected": True}
        )
        self._conversation_history.append(turn)
        
        # If this is a system message, we can inject it into the LLM context
        if speaker == "system" and self.llm_context:
            # For system messages, we could update the context
            # This is a simplified approach - in practice you might want more sophisticated context management
            pass
            
        logger.info(f"Message sent from {speaker}: {message}")
        
    async def get_conversation_history(self) -> List[ConversationTurn]:
        """Get the current conversation history."""
        return self._conversation_history.copy()
        
    async def handle_tool_call(self, tool_call: ToolCall) -> Any:
        """Handle a tool call from the agent."""
        logger.info(f"Handling tool call: {tool_call.tool_name}")
        
        # Check if tool is available
        if tool_call.tool_name not in self._available_tools:
            error_msg = f"Tool '{tool_call.tool_name}' not found"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Execute the tool
        try:
            tool_func = self._available_tools[tool_call.tool_name]
            
            # Emit tool call event
            await self.emit_event(
                AgentBackendEvent.TOOL_CALLED,
                {"tool_name": tool_call.tool_name, "parameters": tool_call.parameters}
            )
            
            # Call the tool function
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**tool_call.parameters)
            else:
                result = tool_func(**tool_call.parameters)
                
            logger.info(f"Tool call completed: {tool_call.tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"Tool call failed: {tool_call.tool_name} - {e}")
            await self.emit_event(AgentBackendEvent.ERROR, {"error": str(e)})
            raise
            
    async def set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt for the agent."""
        logger.info("Updating system prompt")
        
        # Update the configuration
        self.pipecat_config.system_prompt = prompt
        
        # If we have an active context, update it
        if self.llm_context and self.llm_context.messages:
            self.llm_context.messages[0]["content"] = prompt
            logger.info("System prompt updated in active context")
        else:
            logger.warning("No active LLM context to update")
            
    async def process_image(self, image_data: bytes, prompt: Optional[str] = None) -> Optional[str]:
        """Process an image through the agent (not implemented in base Pipecat setup)."""
        logger.warning("Image processing not implemented in basic Pipecat backend")
        return None
        
    async def get_transcript_stream(self) -> AsyncGenerator[ConversationTurn, None]:
        """Get a real-time stream of conversation turns."""
        # This is a simplified implementation
        # In a real scenario, you might want to use asyncio.Queue or similar
        last_count = 0
        
        while self.is_connected:
            current_count = len(self._conversation_history)
            if current_count > last_count:
                # Yield new turns
                for turn in self._conversation_history[last_count:current_count]:
                    yield turn
                last_count = current_count
                
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
    def get_debug_status(self) -> Dict[str, Any]:
        """Get detailed debug status information."""
        base_status = super().get_debug_status()
        base_status.update({
            "pipeline_running": self._pipeline_running,
            "transport_connected": self.transport is not None,
            "pipeline_initialized": self.pipeline is not None,
            "user_context": {
                "is_identified": self.user_context.is_identified,
                "full_name": self.user_context.full_name,
                "location": self.user_context.location,
            },
            "config": {
                "mode": self.pipecat_config.mode,
                "whisper_model": self.pipecat_config.whisper_model,
                "openai_model": self.pipecat_config.openai_model,
                "vad_enabled": self.pipecat_config.vad_enabled,
                "audio_in_sample_rate": self.pipecat_config.audio_in_sample_rate,
                "audio_out_sample_rate": self.pipecat_config.audio_out_sample_rate,
                "audio_input_device_index": self.input_device_index,
                "audio_output_device_index": self.output_device_index,
                "audio_input_device_name": self.pipecat_config.audio_input_device_name,
                "audio_output_device_name": self.pipecat_config.audio_output_device_name,
            }
        })
        return base_status
