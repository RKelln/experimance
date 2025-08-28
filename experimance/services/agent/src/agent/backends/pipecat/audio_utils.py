"""Audio processing utilities for Pipecat multi-channel backends.

This module provides utilities specific to multi-channel audio processing
for Pipecat transports, including delay buffers and channel mapping.
It reuses the comprehensive audio device utilities from experimance_common.
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as e:
    logging.error(f"NumPy required for audio processing: {e}")
    raise

logger = logging.getLogger(__name__)


class DelayBuffer:
    """Circular buffer for implementing per-channel audio delays."""
    
    def __init__(self, max_delay_samples: int):
        """Initialize delay buffer.
        
        Args:
            max_delay_samples: Maximum delay in samples.
        """
        self.buffer = deque(maxlen=max_delay_samples)
        self.delay_samples = 0
        
        # Pre-fill with zeros
        for _ in range(max_delay_samples):
            self.buffer.append(0.0)
    
    def set_delay(self, delay_samples: int):
        """Set the current delay in samples.
        
        Args:
            delay_samples: Delay in samples (must be <= max_delay_samples).
        """
        self.delay_samples = min(delay_samples, len(self.buffer))
    
    def process_sample(self, input_sample: float) -> float:
        """Process a single sample through the delay buffer.
        
        Args:
            input_sample: Input audio sample.
            
        Returns:
            Delayed audio sample.
        """
        # Add new sample to buffer (oldest sample is automatically removed)
        self.buffer.append(input_sample)
        
        # Return sample from delay_samples ago
        if self.delay_samples == 0:
            return input_sample
        else:
            delay_index = len(self.buffer) - 1 - self.delay_samples
            return self.buffer[delay_index]
    
    def process_chunk(self, input_chunk: np.ndarray) -> np.ndarray:
        """Process a chunk of audio samples.
        
        Args:
            input_chunk: Input audio chunk as numpy array.
            
        Returns:
            Delayed audio chunk as numpy array.
        """
        output_chunk = np.zeros_like(input_chunk)
        
        for i, sample in enumerate(input_chunk):
            output_chunk[i] = self.process_sample(sample)
        
        return output_chunk


def audio_to_multi_channel(audio_data: bytes, num_channels: int, 
                          channel_delays: Dict[int, float], 
                          channel_volumes: Dict[int, float],
                          sample_rate: int,
                          delay_buffers: Dict[int, DelayBuffer],
                          input_channels: Optional[int] = None) -> bytes:
    """Convert mono or stereo audio to multi-channel with per-channel delays and volumes.
    
    Args:
        audio_data: Input audio as bytes (int16). Can be mono or stereo.
        num_channels: Total number of output channels.
        channel_delays: Dict mapping channel index to delay in seconds.
        channel_volumes: Dict mapping channel index to volume (0.0-1.0).
        sample_rate: Audio sample rate.
        delay_buffers: Dict mapping channel index to DelayBuffer instances.
        input_channels: Number of input channels (1=mono, 2=stereo). If None, auto-detect.
        
    Returns:
        Multi-channel audio as interleaved bytes (int16).
    """
    # Convert bytes to numpy array
    input_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    
    # Use provided input_channels or attempt to detect
    if input_channels is not None:
        is_stereo = (input_channels == 2)
        logger.debug(f"Using provided channel count: {input_channels} ({'stereo' if is_stereo else 'mono'})")
    else:
        # Fallback to old detection method (not reliable)
        is_stereo = len(input_samples) % 2 == 0
        logger.debug(f"Auto-detecting: {'stereo' if is_stereo else 'mono'} (len={len(input_samples)})")
    
    if is_stereo:
        # De-interleave stereo data
        left_samples = input_samples[0::2]   # Even indices: L, L, L, ...
        right_samples = input_samples[1::2]  # Odd indices:  R, R, R, ...
        num_samples = len(left_samples)
        
        logger.debug(f"Processing stereo input: {num_samples} samples per channel")
        
        # Verify stereo detection by checking if L and R are different
        if num_samples > 10:
            diff = np.mean(np.abs(left_samples[:10] - right_samples[:10]))
            logger.debug(f"L/R difference: {diff:.3f} (>0.01 confirms stereo)")
    else:
        # Mono input
        mono_samples = input_samples
        num_samples = len(mono_samples)
        
        logger.debug(f"Processing mono input: {num_samples} samples")
    
    # Create multi-channel output array
    multi_channel_samples = np.zeros((num_samples, num_channels), dtype=np.float32)
    
    # Process each channel
    for channel in range(num_channels):
        # Get volume for this channel (default 1.0 if not specified)
        volume = channel_volumes.get(channel, 1.0)
        
        # Get delay buffer for this channel
        delay_buffer = delay_buffers.get(channel)
        
        if volume <= 0.0:
            continue  # Skip silent channels
            
        # Choose source audio based on channel and input type
        if is_stereo:
            # For stereo input, map channels to L/R:
            # Even channels (0, 2, 4, ...) get left channel
            # Odd channels (1, 3, 5, ...) get right channel
            if channel % 2 == 0:
                source_samples = left_samples
            else:
                source_samples = right_samples
        else:
            # For mono input, all channels get the same source
            source_samples = mono_samples
        
        # Apply delay and volume
        if delay_buffer is not None:
            delayed_samples = delay_buffer.process_chunk(source_samples)
            multi_channel_samples[:, channel] = delayed_samples * volume
        else:
            # No delay buffer, just apply volume
            multi_channel_samples[:, channel] = source_samples * volume
        # else: channel is muted (volume 0.0 or no delay buffer)
    
    # Convert back to interleaved int16 bytes
    # Flatten to interleaved format: [L0, R0, C0, LFE0, L1, R1, C1, LFE1, ...]
    interleaved_samples = multi_channel_samples.flatten()
    
    # Clamp to int16 range and convert
    np.clip(interleaved_samples, -32768, 32767, out=interleaved_samples)
    output_bytes = interleaved_samples.astype(np.int16).tobytes()
    
    return output_bytes


def create_delay_buffers(channel_delays: Dict[int, float], 
                        sample_rate: int,
                        max_delay_seconds: float = 1.0) -> Dict[int, DelayBuffer]:
    """Create delay buffers for each channel.
    
    Args:
        channel_delays: Dict mapping channel index to delay in seconds.
        sample_rate: Audio sample rate.
        max_delay_seconds: Maximum delay to support in seconds.
        
    Returns:
        Dict mapping channel index to DelayBuffer instances.
    """
    max_delay_samples = int(max_delay_seconds * sample_rate)
    delay_buffers = {}
    
    for channel, delay_seconds in channel_delays.items():
        delay_samples = int(delay_seconds * sample_rate)
        
        buffer = DelayBuffer(max_delay_samples)
        buffer.set_delay(delay_samples)
        delay_buffers[channel] = buffer
        
        logger.info(f"Created delay buffer for channel {channel}: "
                   f"{delay_seconds:.3f}s ({delay_samples} samples)")
    
    return delay_buffers


def calculate_delay_samples(delay_seconds: float, sample_rate: int) -> int:
    """Calculate delay in samples for a given time delay and sample rate.
    
    Args:
        delay_seconds: Delay time in seconds.
        sample_rate: Audio sample rate in Hz.
        
    Returns:
        Number of samples for the delay.
    """
    return int(delay_seconds * sample_rate)


def suggest_echo_cancellation_delays(distances: Dict[int, float], 
                                     sound_speed: float = 343.0) -> Dict[int, float]:
    """Suggest delay values for echo cancellation based on speaker distances.
    
    Args:
        distances: Dict mapping channel indices to distances from microphone in meters.
        sound_speed: Speed of sound in m/s (default: 343.0 at 20°C).
        
    Returns:
        Dict mapping channel indices to suggested delay values in seconds.
    """
    if not distances:
        return {}
    
    # Find the shortest distance (reference)
    min_distance = min(distances.values())
    
    delays = {}
    for channel, distance in distances.items():
        # Calculate additional delay needed to align with shortest distance
        additional_distance = distance - min_distance
        delay_seconds = additional_distance / sound_speed
        delays[channel] = max(0.0, delay_seconds)  # No negative delays
        
        logger.info(f"Channel {channel}: {distance:.2f}m distance, {delay_seconds*1000:.1f}ms delay")
    
    return delays


def validate_channel_config(channel_delays: Dict[int, float], 
                           channel_volumes: Dict[int, float],
                           max_channels: int) -> bool:
    """Validate channel configuration parameters.
    
    Args:
        channel_delays: Channel delay configuration.
        channel_volumes: Channel volume configuration.
        max_channels: Maximum number of channels supported.
        
    Returns:
        True if configuration is valid, False otherwise.
    """
    # Check channel indices are within range
    all_channels = set(channel_delays.keys()) | set(channel_volumes.keys())
    if any(ch < 0 or ch >= max_channels for ch in all_channels):
        logger.error(f"Channel indices must be 0-{max_channels-1}, got: {sorted(all_channels)}")
        return False
    
    # Check delay values are reasonable (0-1 second)
    for channel, delay in channel_delays.items():
        if delay < 0 or delay > 1.0:
            logger.error(f"Channel {channel} delay {delay:.3f}s is outside reasonable range (0-1.0s)")
            return False
    
    # Check volume values are in valid range
    for channel, volume in channel_volumes.items():
        if volume < 0.0 or volume > 1.0:
            logger.error(f"Channel {channel} volume {volume:.2f} is outside range (0.0-1.0)")
            return False
    
    logger.info("Channel configuration validated successfully")
    return True


def estimate_latency(sample_rate: int, buffer_size: int, num_channels: int) -> float:
    """Estimate total audio latency for multi-channel output.
    
    Args:
        sample_rate: Audio sample rate in Hz.
        buffer_size: PyAudio buffer size in samples per channel.
        num_channels: Number of output channels.
        
    Returns:
        Estimated latency in seconds.
    """
    # Buffer latency (time to fill one buffer)
    buffer_latency = buffer_size / sample_rate
    
    # Processing overhead (estimated)
    processing_overhead = 0.005  # 5ms
    
    # Multi-channel processing overhead
    channel_overhead = num_channels * 0.001  # 1ms per channel
    
    total_latency = buffer_latency + processing_overhead + channel_overhead
    
    logger.info(f"Estimated latency: {total_latency*1000:.1f}ms "
               f"(buffer: {buffer_latency*1000:.1f}ms, processing: {(processing_overhead + channel_overhead)*1000:.1f}ms)")
    
    return total_latency
