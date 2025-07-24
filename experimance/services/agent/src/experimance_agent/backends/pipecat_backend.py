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
        
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames and extract conversation events."""
        
        # Call parent class to handle StartFrame properly
        await super().process_frame(frame, direction)
        
        # Handle user speaking state
        if isinstance(frame, UserStartedSpeakingFrame):
            self.user_speaking = True
            await self.backend.emit_event(AgentBackendEvent.SPEECH_DETECTED, {
                "speaker": "user"
            })
            logger.debug("User started speaking")
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_speaking = False
            await self.backend.emit_event(AgentBackendEvent.SPEECH_ENDED, {
                "speaker": "user"
            })
            logger.debug("User stopped speaking")
            
        # Handle bot speaking state  
        elif isinstance(frame, TTSStartedFrame):
            self.bot_speaking = True
            await self.backend.emit_event(AgentBackendEvent.BOT_STARTED_SPEAKING, {
                "speaker": "agent"
            })
            logger.debug("Bot started speaking")
            
        elif isinstance(frame, TTSStoppedFrame):
            self.bot_speaking = False
            await self.backend.emit_event(AgentBackendEvent.BOT_STOPPED_SPEAKING, {
                "speaker": "agent"})
            logger.debug("Bot stopped speaking")
            
        # Handle pipeline shutdown
        elif isinstance(frame, EndFrame):
            logger.info("Pipeline EndFrame received, triggering conversation ended")
            await self.backend.emit_event(AgentBackendEvent.CONVERSATION_ENDED, {
                "reason": "pipeline_ended"
            })
            
        elif isinstance(frame, CancelFrame):
            logger.info("Pipeline CancelFrame received, triggering conversation ended")
            # CancelFrame can be sent during SIGINT shutdown, so treat as conversation end
            await self.backend.emit_event(AgentBackendEvent.CANCEL, {
                "reason": "pipeline_cancelled"
            })
            
        # Forward the frame to the next processor
        await self.push_frame(frame, direction)


class PipecatBackend(AgentBackend):
    """
    Pipecat backend implementation using the working ensemble approach from flows_test.py.
    """
    
    def __init__(self, config: AgentServiceConfig, user_context: Optional[UserContext] = None):
        super().__init__(config)
        self.pipecat_config = self.config.backend_config.pipecat
        self.user_context = user_context or UserContext()
        
        # Pipeline components
        self.pipeline: Optional[Pipeline] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        self.transport: Optional[LocalAudioTransport] = None
        self.flow_manager: Optional[FlowManager] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._flow_config: Optional[FlowConfig] = None
        self._stopping = False  # Flag to prevent multiple stop calls
        
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
            self.runner = PipelineRunner()
            
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
        if hasattr(frame, 'messages'):
            for message in frame.messages:
                # TranscriptionMessage has role, content, timestamp
                if hasattr(message, 'role') and hasattr(message, 'content'):
                    if message.role == "user":
                        await self.add_user_speech(message.content)
                        await self.emit_event(AgentBackendEvent.TRANSCRIPTION_RECEIVED, {
                            "text": message.content
                        })
                        logger.debug(f"User transcription: {message.content}")
                    elif message.role == "assistant":
                        await self.add_agent_response(message.content)
                        await self.emit_event(AgentBackendEvent.RESPONSE_GENERATED, {
                            "text": message.content
                        })
                        logger.debug(f"Assistant response: {message.content}")

    async def stop(self) -> None:
        """Stop the Pipecat backend."""
        if self._stopping or not self.is_connected:
            logger.debug("Backend already stopping or stopped, skipping")
            return
            
        self._stopping = True  # Set flag to prevent multiple calls
        logger.info("Stopping Pipecat backend v2...")
        
        # Emit conversation ended event first (only once)
        await self.emit_event(AgentBackendEvent.CONVERSATION_ENDED, {
            "reason": "backend_stopped"
        })
        
        await self.disconnect()  # Ensure we disconnect first
        
        try:
            # Stop Pipecat components first
            # Cancel the pipeline task if it's running
            if self._pipeline_task and not self._pipeline_task.done():
                logger.debug("Cancelling pipeline task...")
                self._pipeline_task.cancel()
                try:
                    await asyncio.wait_for(self._pipeline_task, timeout=5.0)
                except asyncio.CancelledError:
                    logger.debug("Pipeline task cancelled successfully")
                except asyncio.TimeoutError:
                    logger.warning("Pipeline task cancellation timed out")
                except Exception as e:
                    logger.warning(f"Error while cancelling pipeline task: {e}")
            
            # Stop the runner
            if self.runner:
                await self.runner.cleanup()
                
            # Stop the task
            if self.task:
                await self.task.cleanup()
                
            # Stop the transport
            if self.transport:
                await self.transport.cleanup()

            # Call parent class to stop transcript manager
            await super().stop()
                
            logger.info("Pipecat backend v2 stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Pipecat backend: {e}")
    
    def _create_transport(self) -> LocalAudioTransport:

        # Resolve audio device indices
        audio_in_device = resolve_audio_device_index(
            self.pipecat_config.audio_input_device_index,
            self.pipecat_config.audio_input_device_name,
            input_device=True
        )
        audio_out_device = resolve_audio_device_index(
            self.pipecat_config.audio_output_device_index,
            self.pipecat_config.audio_output_device_name,
            input_device=False
        )
        
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
                    # STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE,
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
            # Load prompt from file if specified
            prompt = load_prompt(self.config.backend_config.prompt_path)
        else:   
            prompt = "Tell the user something has gone wrong with loading the agent configuration."

        # TODO: get prompt 
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
                    await self.flow_manager.set_node(node_name, node_config)
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
