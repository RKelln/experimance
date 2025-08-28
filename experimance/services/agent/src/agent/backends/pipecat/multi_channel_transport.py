"""Multi-channel audio transport for Pipecat with per-channel delays.

This module extends Pipecat's LocalAudioTransport to support multi-channel
audio output with configurable per-channel delays for echo cancellation.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

try:
    import pyaudio
except ModuleNotFoundError as e:
    logging.error(f"PyAudio not available: {e}")
    raise Exception(f"Missing module: {e}")

from pipecat.frames.frames import OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.local.audio import LocalAudioInputTransport

from experimance_common.audio_utils import (
    resolve_audio_device_index,
    suppress_audio_errors
)
from .audio_utils import (
    DelayBuffer,
    create_delay_buffers,
    audio_to_multi_channel,
    validate_channel_config,
    estimate_latency
)

logger = logging.getLogger(__name__)


class MultiChannelAudioTransportParams(TransportParams):
    """Configuration parameters for multi-channel audio transport.
    
    Parameters:
        aggregate_device_index: PyAudio device index for multi-channel aggregate device.
        aggregate_device_name: Device name to search for if index not provided.
        output_channels: Total number of output channels.
        channel_delays: Dict mapping channel index to delay in seconds.
        channel_volumes: Dict mapping channel index to volume (0.0-1.0).
        max_delay_seconds: Maximum delay buffer size in seconds.
        input_device_index: Optional input device index (for compatibility).
        input_device_name: Optional input device name (for compatibility).
    """
    
    aggregate_device_index: Optional[int] = None
    aggregate_device_name: Optional[str] = None
    output_channels: int = 4
    channel_delays: Dict[int, float] = {}
    channel_volumes: Dict[int, float] = {}
    max_delay_seconds: float = 1.0
    
    # For compatibility with standard LocalAudioTransport
    input_device_index: Optional[int] = None
    input_device_name: Optional[str] = None


class MultiChannelAudioOutputTransport(BaseOutputTransport):
    """Multi-channel audio output transport with per-channel delays.
    
    Extends Pipecat's BaseOutputTransport to support multi-channel output
    with configurable delays and volumes per channel for echo cancellation.
    """

    _params: MultiChannelAudioTransportParams

    def __init__(self, py_audio: pyaudio.PyAudio, params: MultiChannelAudioTransportParams):
        """Initialize the multi-channel audio output transport.

        Args:
            py_audio: PyAudio instance for audio device management.
            params: Transport configuration parameters.
        """
        super().__init__(params)
        self._py_audio = py_audio
        self._params = params

        self._out_stream = None
        self._sample_rate = 0
        self._delay_buffers: Dict[int, DelayBuffer] = {}

        # Single thread executor for audio output
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        # Validate configuration
        if not validate_channel_config(
            params.channel_delays,
            params.channel_volumes, 
            params.output_channels
        ):
            raise ValueError("Invalid channel configuration")

    async def start(self, frame: StartFrame):
        """Start the multi-channel audio output stream.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._out_stream:
            return

        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        
        # Resolve output device
        output_device_index = resolve_audio_device_index(
            self._params.aggregate_device_index,
            self._params.aggregate_device_name,
            input_device=False
        )
        
        if output_device_index is not None:
            logger.info(f"Using multi-channel output device index: {output_device_index}")
        else:
            logger.info("Using default multi-channel output device")

        # Create delay buffers for channels that need them
        self._delay_buffers = create_delay_buffers(
            self._params.channel_delays,
            self._sample_rate,
            self._params.max_delay_seconds
        )

        # Open PyAudio stream with multi-channel output
        try:
            with suppress_audio_errors():
                self._out_stream = self._py_audio.open(
                    format=self._py_audio.get_format_from_width(2),  # 16-bit
                    channels=self._params.output_channels,
                    rate=self._sample_rate,
                    output=True,
                    output_device_index=output_device_index,
                )
                self._out_stream.start_stream()
                
                # Log latency estimate
                buffer_size = 1024  # Typical PyAudio buffer size
                latency = estimate_latency(
                    self._sample_rate, 
                    buffer_size, 
                    self._params.output_channels
                )
                
                logger.info(f"Multi-channel audio output started: "
                           f"{self._params.output_channels} channels @ {self._sample_rate}Hz, "
                           f"estimated latency: {latency*1000:.1f}ms")
                
        except Exception as e:
            logger.error(f"Failed to open multi-channel audio output stream: {e}")
            raise

        await self.set_transport_ready(frame)

    async def cleanup(self):
        """Stop and cleanup the audio output stream."""
        await super().cleanup()
        if self._out_stream:
            try:
                self._out_stream.stop_stream()
                self._out_stream.close()
                logger.info("Multi-channel audio output stream stopped")
            except Exception as e:
                logger.error(f"Error stopping multi-channel audio stream: {e}")
            finally:
                self._out_stream = None

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """Write an audio frame to the multi-channel output stream.

        Args:
            frame: The mono audio frame to convert to multi-channel output.
        """
        if not self._out_stream:
            return
            
        try:
            # Convert mono or stereo to multi-channel with delays and volumes
            # Pass the frame's channel count to avoid incorrect stereo detection
            multi_channel_audio = audio_to_multi_channel(
                frame.audio,
                self._params.output_channels,
                self._params.channel_delays,
                self._params.channel_volumes,
                self._sample_rate,
                self._delay_buffers,
                input_channels=frame.num_channels  # Pass actual channel count from frame
            )
            
            # Write to PyAudio stream
            await self.get_event_loop().run_in_executor(
                self._executor, self._out_stream.write, multi_channel_audio
            )
            
        except Exception as e:
            logger.error(f"Error writing multi-channel audio frame: {e}")


class MultiChannelAudioTransport(BaseTransport):
    """Complete multi-channel audio transport with input and output capabilities.

    Provides a unified interface for audio I/O using PyAudio, with multi-channel
    output support and standard mono input.
    """

    def __init__(self, params: MultiChannelAudioTransportParams):
        """Initialize the multi-channel audio transport.

        Args:
            params: Transport configuration parameters.
        """
        super().__init__()
        self._params = params
        
        with suppress_audio_errors():
            self._pyaudio = pyaudio.PyAudio()

        self._input: Optional[LocalAudioInputTransport] = None
        self._output: Optional[MultiChannelAudioOutputTransport] = None

    def input(self) -> FrameProcessor:
        """Get the input frame processor for this transport.
        
        Uses standard LocalAudioInputTransport for mono input.

        Returns:
            The audio input transport processor.
        """
        if not self._input:
            # Create compatible params for input transport
            from pipecat.transports.local.audio import LocalAudioTransportParams
            input_params = LocalAudioTransportParams(
                input_device_index=self._params.input_device_index,
                # Copy other relevant params from base TransportParams
                audio_in_enabled=self._params.audio_in_enabled,
                audio_in_channels=self._params.audio_in_channels,
                audio_in_sample_rate=self._params.audio_in_sample_rate,
            )
            
            self._input = LocalAudioInputTransport(self._pyaudio, input_params)
        return self._input

    def output(self) -> FrameProcessor:
        """Get the multi-channel output frame processor for this transport.

        Returns:
            The multi-channel audio output transport processor.
        """
        if not self._output:
            self._output = MultiChannelAudioOutputTransport(self._pyaudio, self._params)
        return self._output
    
    async def cleanup(self):
        """Cleanup PyAudio resources."""
        try:
            if self._input:
                await self._input.cleanup()
            if self._output:
                await self._output.cleanup()
        finally:
            if hasattr(self, '_pyaudio'):
                with suppress_audio_errors():
                    self._pyaudio.terminate()