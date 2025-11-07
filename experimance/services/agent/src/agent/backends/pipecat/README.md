# Multi-Channel Audio Transport for Pipecat

This implementation provides multi-channel audio output support for the Experimance agent service, enabling dual speaker setups with echo cancellation through configurable per-channel delays.

## Features

- **Multi-Channel Output**: Support for any number of output channels (2, 4, 6, 8, etc.)
- **Per-Channel Delays**: Individual delay buffers for each channel to compensate for echo and processing latency
- **Per-Channel Volume Control**: Individual volume control for each channel
- **Echo Cancellation**: Delay-based echo cancellation for dual speaker configurations
- **macOS Aggregate Device Support**: Designed to work with macOS Aggregate Devices for clock sync
- **Backward Compatibility**: Falls back to standard LocalAudioTransport when not enabled

## Architecture

### File Structure
```
services/agent/src/agent/backends/pipecat/
├── __init__.py                      # Module exports
├── backend.py                       # Main PipecatBackend (moved from pipecat_backend.py)
├── multi_channel_transport.py       # Multi-channel transport classes
└── audio_utils.py                   # Audio processing utilities (uses experimance_common)
```

### Key Components

1. **MultiChannelAudioTransportParams**: Configuration class with channel delays, volumes, and device settings
2. **MultiChannelAudioOutputTransport**: Extends Pipecat's BaseOutputTransport for multi-channel output
3. **MultiChannelAudioTransport**: Complete transport with multi-channel output and standard mono input
4. **DelayBuffer**: Circular buffer implementation for per-channel audio delays
5. **Audio Processing Functions**: Convert mono to multi-channel with delays and volumes

## Configuration

Enable multi-channel output in your agent configuration:

```toml
[backend_config.pipecat]
# Enable multi-channel output
multi_channel_output = true
aggregate_device_name = "Aggregate Device"  # macOS Aggregate Device
output_channels = 4

# Per-channel delays for echo cancellation (seconds)
[backend_config.pipecat.channel_delays]
0 = 0.0    # USB conference speaker (reference)
1 = 0.0    # USB conference speaker (reference)
2 = 0.02   # Audio port speaker (20ms delay)
3 = 0.02   # Audio port speaker (20ms delay)

# Per-channel volumes (0.0 to 1.0)
[backend_config.pipecat.channel_volumes]
0 = 1.0    # Full volume USB
1 = 1.0    # Full volume USB
2 = 0.8    # Lower volume for audio port
3 = 0.8    # Lower volume for audio port
```

## Use Cases

### Dual Speaker Setup
- **USB Conference Speaker/Mic**: Channels 0-1 (no delay, full volume)
- **Audio Port Speaker**: Channels 2-3 (delayed to compensate for USB processing)
- **Echo Cancellation**: USB device handles echo cancellation when delays are properly aligned

### Multi-Speaker Installation
- Support for 4, 6, or 8 speaker configurations
- Individual delay tuning for each speaker position
- Volume balancing across all speakers

## Implementation Notes

### Audio Processing
- Converts mono TTS output to multi-channel interleaved format
- Uses circular delay buffers for real-time processing
- Maintains low latency through efficient NumPy operations

### Device Compatibility
- Reuses existing audio device resolution from `experimance_common`
- Supports both device names and indices
- Works with macOS Aggregate Devices for clock synchronization

### Performance
- Single-threaded audio output (ThreadPoolExecutor with max_workers=1)
- Efficient delay buffer implementation
- Estimated latency logging for monitoring

## Testing

1. **Standard Mode**: Set `multi_channel_output = false` (default)
2. **Multi-Channel Mode**: Enable and configure channel delays/volumes
3. **Device Testing**: Use `uv run python scripts/list_audio_devices.py` to find devices
4. **Delay Tuning**: Start with small delays (10-50ms) and adjust by ear

## Future Enhancements

- Runtime delay adjustment via ZMQ messages
- Auto-calibration using audio loopback
- Support for different delay strategies (distance-based, acoustic-based)
- Integration with room correction systems

## Dependencies

- **NumPy**: Audio array processing
- **PyAudio**: Multi-channel audio output
- **Pipecat**: Base transport classes
- **experimance_common**: Audio device utilities
