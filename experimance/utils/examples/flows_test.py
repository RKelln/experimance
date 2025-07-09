#!/usr/bin/env python3
"""
Standalone Pipecat Flows Test Application

This script creates an independent voice agent using Pipecat flows to test
the multi-persona conversation system outside of the main agent service.

Usage:
    uv run utils/examples/flows_test.py
"""

import asyncio
import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Add the agent service to the path so we can reuse config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services" / "agent" / "src"))

from experimance_common.audio_utils import resolve_audio_device_index
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame, 
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
)

# Pipecat Flows imports
from pipecat_flows import FlowConfig, FlowManager, FlowArgs, FlowResult, NodeConfig

# Import flow configuration
from experimance_test_flow import flow_config

# Import config from agent service
from experimance_agent.config import AgentServiceConfig, PipecatBackendConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DebugEventProcessor(FrameProcessor):
    """Frame processor that logs events for debugging."""
    
    def __init__(self, test_app: 'FlowsTestApp'):
        super().__init__()
        self.test_app = test_app
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and log events for debugging."""
        await super().process_frame(frame, direction)
        
        # Log key events
        if isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                logger.info(f"[USER SPEECH] {frame.text}")
                
        elif isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            logger.info(f"[BOT RESPONSE] {frame.text}")
            
        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.info("[EVENT] User started speaking")
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("[EVENT] User stopped speaking")
            
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info("[EVENT] Bot started speaking")
            
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info("[EVENT] Bot stopped speaking")
        
        # Forward the frame downstream
        await self.push_frame(frame, direction)


class FlowsTestApp:
    """Main test application for Pipecat flows."""
    
    def __init__(self, config: AgentServiceConfig):
        self.config = config
        self.pipecat_config = config.backend_config.pipecat
        
        # Pipeline components
        self.transport: Optional[LocalAudioTransport] = None
        self.pipeline: Optional[Pipeline] = None
        self.pipeline_task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        
        # Flow manager
        self.flow_manager: Optional[FlowManager] = None
        
        # Control
        self._running = False
        
    async def start(self):
        """Start the flows test application."""
        logger.info("Starting Pipecat Flows Test Application...")
        
        try:
            # Set up audio transport
            await self._setup_transport()
            
            # Build pipeline
            await self._build_pipeline()
            
            # Start pipeline
            await self._start_pipeline()
            
        except Exception as e:
            logger.error(f"Failed to start flows test application: {e}")
            raise
    
    async def _setup_transport(self):
        """Set up the audio transport."""

        input_device_index = resolve_audio_device_index(
            self.pipecat_config.audio_input_device_index,
            self.pipecat_config.audio_input_device_name,
            input_device=True
        )
        output_device_index = resolve_audio_device_index(
            self.pipecat_config.audio_output_device_index,
            self.pipecat_config.audio_output_device_name,
            input_device=False
        )
        
        transport_params = LocalAudioTransportParams(
            audio_in_enabled=self.pipecat_config.audio_in_enabled,
            audio_out_enabled=self.pipecat_config.audio_out_enabled,
            audio_in_sample_rate=self.pipecat_config.audio_in_sample_rate,
            audio_out_sample_rate=self.pipecat_config.audio_out_sample_rate,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
        )

        if self.pipecat_config.vad_enabled:
            transport_params.vad_analyzer = SileroVADAnalyzer()
         
        self.transport = LocalAudioTransport(transport_params)
        
    async def _build_pipeline(self):
        """Build the ensemble pipeline with flows."""
        assert self.transport, "Transport must be set up before building pipeline"
        
        # Create services
        stt = AssemblyAISTTService(
            api_key=os.getenv("ASSEMBLYAI_API_KEY", "failed to load"),
            vad_force_turn_endpoint=False,  # Use AssemblyAI's STT-based turn detection
            connection_params=AssemblyAIConnectionParams(
                end_of_turn_confidence_threshold=0.7,
                min_end_of_turn_silence_when_confident=160,  # in ms
                max_turn_silence=2400,  # in ms
            )
        )
        
        # Use Cartesia for TTS (faster than ElevenLabs for testing)
        cartesia_api_key = os.getenv("CARTESIA_API_KEY")
        if not cartesia_api_key:
            raise ValueError("CARTESIA_API_KEY environment variable is required")
            
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY", "failed to load"),
            voice_id=self.pipecat_config.cartesia_voice_id,
            model="sonic-2",
            params=CartesiaTTSService.InputParams(
                language=Language.EN,
                speed="fast",  # Options: "fast", "normal", "slow"
            )
        )
        
        # LLM service
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        llm = OpenAILLMService(
            api_key=openai_api_key,
            model=self.pipecat_config.openai_model,
        )
        
        # Context aggregator
        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)
        # Processors
        debug_processor = DebugEventProcessor(self)
        #transcript = TranscriptProcessor()
        
        #sentence_aggregator = SentenceAggregator()

        # Use the imported flow configuration directly
    

        self.pipeline = Pipeline([
            self.transport.input(),                           # Audio input (microphone)
            stt,                                             # Speech-to-text
            #transcript.user(),
            context_aggregator.user(),             # Add user message to context
            llm,                                     # Language model or Flow Manager
            #sentence_aggregator,                             # Aggregate into sentences                           
            tts,                                             # Text-to-speech
            debug_processor,                                  # Handle tool calls
            self.transport.output(),                         # Audio output (speakers)
            #transcript.assistant_tts(),
            context_aggregator.assistant(),        # Add assistant response to context
        ])
        
        # Create pipeline task
        self.pipeline_task = PipelineTask(self.pipeline)
        
        # Create flow manager with the flow configuration
        self.flow_manager = FlowManager(
            task=self.pipeline_task,
            llm=llm,
            context_aggregator=context_aggregator,
            flow_config=flow_config,
        )
        
        # Initialize with the welcome node
        welcome_node = flow_config["nodes"]["welcome"]
        logger.info(f"Initializing flow manager with welcome node: {welcome_node}")
            
    async def _start_pipeline(self):
        """Start the pipeline."""
        assert self.pipeline, "Pipeline must be built before starting"
        assert self.transport, "Transport must be set up before starting pipeline"
        assert self.pipeline_task, "Pipeline task must be created before starting"
        
        # Set the task in the flow manager (pipeline task should already exist)
        if self.flow_manager and self.pipeline_task:
            self.flow_manager.task = self.pipeline_task
            
            # Initialize the flow manager with the welcome node
            try:
                welcome_node = flow_config["nodes"]["welcome"]
                await self.flow_manager.initialize(welcome_node)
                logger.info("Flow manager initialized with welcome node")
            except Exception as e:
                logger.error(f"Failed to initialize flow manager: {e}")
        
        self.runner = PipelineRunner()
        
        # Start the runner
        try:
            self._running = True
            await self.runner.run(self.pipeline_task)
        except asyncio.CancelledError:
            logger.info("Pipeline runner was cancelled")
            self._running = False
        except KeyboardInterrupt:
            logger.info("Pipeline runner interrupted by user")
            await self.stop()
        except Exception as e:
            logger.error(f"Failed to run pipeline: {e}")
            self._running = False
            raise
        
    async def stop(self):
        """Stop the flows test application."""
        logger.info("Stopping flows test application...")
        
        self._running = False
        
        if self.pipeline_task:
            # Use the task's cancel method if available
            if hasattr(self.pipeline_task, 'cancel'):
                await self.pipeline_task.cancel()
            
        if self.runner:
            # PipelineRunner doesn't have a stop method, just let it finish naturally
            logger.info("Pipeline runner will finish naturally")
            
        logger.info("Flows test application stopped.")
        
    def is_running(self) -> bool:
        """Check if the application is running."""
        return self._running


async def main():
    """Main entry point for the flows test application."""
    
    # Check required environment variables
    required_env_vars = ["OPENAI_API_KEY", "CARTESIA_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Create configuration using proper types
    from experimance_agent.config import PipecatBackendConfig, BackendConfig
    
    pipecat_config = PipecatBackendConfig(
        mode="ensemble",
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate = 16000,
        audio_out_sample_rate = 16000,
        audio_input_device_name = "Yealink",  # USB Conference microphone
        audio_output_device_name = "Yealink",  # USB Conference speaker
        vad_enabled=True,
        whisper_model="tiny",
        openai_model="gpt-4o-mini",
        use_flows=True,
        system_prompt="You are testing Pipecat flows for the Experimance installation.",
        cartesia_voice_id="bf0a246a-8642-498a-9950-80c35e9276b5"  # Default voice
    )
    
    backend_config = BackendConfig(pipecat=pipecat_config)
    
    config = AgentServiceConfig(backend_config=backend_config)
    
    # Create and start the test application
    app = FlowsTestApp(config)
    
    try:
        await app.start()
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal...")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
