# Multi-Channel Audio Delay System

## Overview

The Experimance multi-channel audio system provides sophisticated echo cancellation and audio routing capabilities by supporting multiple speaker outputs with configurable per-channel delays and volumes. This is particularly useful for installations with multiple speakers where you need to prevent echo/chorus effects by carefully timing audio output.

## Architecture

### Components

1. **MultiChannelAudioTransport** - Extends Pipecat's audio transport for multi-channel output
2. **DelayBuffer** - Per-channel delay processing with configurable delay times  
3. **PipeWire Virtual-Multi Sink** - Linux audio routing to multiple physical devices
4. **Interactive Calibration Tool** - Real-time delay adjustment with voice audio testing

### Audio Flow

```
Cartesia TTS (mono, 44100Hz) 
    ↓
MultiChannelAudioTransport
    ↓
Per-channel DelayBuffers (0-4 channels)
    ↓
PipeWire Virtual-Multi Sink (4 channels)
    ↓
Physical Outputs:
  - Channels 0,1 → Laptop Speakers (120ms delay)  
  - Channels 2,3 → Bluetooth Speakers (0ms delay)
```

## Configuration

### Fire Agent Configuration

In `projects/fire/agent.toml`:

```toml
# Enable multi-channel output
multi_channel_output = true
output_channels = 4

# Audio device selection - use PipeWire aggregate device
audio_output_device_name = "pipewire"

# Per-channel delays (in seconds) for echo cancellation
[backend_config.pipecat.channel_delays]
0 = 0.120    # Laptop speakers (left)
1 = 0.120    # Laptop speakers (right) 
2 = 0.000    # Bluetooth speakers (left)
3 = 0.000    # Bluetooth speakers (right)

# Per-channel volumes (0.0 to 1.0)
[backend_config.pipecat.channel_volumes]
0 = 1.0      # Full volume laptop left
1 = 1.0      # Full volume laptop right
2 = 0.8      # Slightly lower Bluetooth left
3 = 0.8      # Slightly lower Bluetooth right
```

### PipeWire Setup

The system requires a PipeWire Virtual-Multi sink to route 4-channel audio to multiple physical devices:

```bash
# Create Virtual-Multi sink (done automatically by setup scripts)
pw-loopback -P "{ audio.rate=48000 audio.channels=4 }" \
  --capture-props="{ node.name=Virtual-Multi-capture }" \
  --playback-props="{ node.name=Virtual-Multi }"

# Link channels to physical devices
pw-link Virtual-Multi:output_FL alsa_output.pci-0000_00_1f.3.analog-stereo:playback_FL  # Laptop
pw-link Virtual-Multi:output_FR alsa_output.pci-0000_00_1f.3.analog-stereo:playback_FR  # Laptop  
pw-link Virtual-Multi:output_RL bluetooth_output_device:playback_FL                      # BT Left
pw-link Virtual-Multi:output_RR bluetooth_output_device:playback_FR                      # BT Right
```

## Delay Calibration

### Why Delays are Needed

Different audio devices have different latencies:
- **Laptop speakers**: Lower latency (faster)
- **Bluetooth speakers**: Higher latency (slower) 

Without delay compensation, audio reaches listeners at different times, creating an echo/chorus effect.

### Calibration Process

1. **Estimate initial delays**: Bluetooth typically ~120ms slower than laptop speakers
2. **Use interactive calibration tool** for fine-tuning with real voice audio
3. **Test with actual agent** to verify results

### Interactive Calibration Tool

The calibration tool provides real-time delay adjustment:

```bash
# Start with voice audio loaded
uv run python scripts/test_multi_channel_audio.py --file media/audio/cartesia_sophie.wav

# Set initial delays (milliseconds)  
> d 0 120
> d 1 120

# Start interactive mode for channels 0,1 (laptop speakers)
> i 0,1

# Interactive controls:
# ,/.  - Adjust ±1ms 
# </>  - Adjust ±5ms
# v    - Toggle voice/clicks mode
# ESC  - Exit
```

### Fine-Tuning Guidelines

- **Listen for echo**: If you hear the same voice twice slightly offset, delays need adjustment
- **Start with estimates**: Bluetooth ~120ms, wired ~0ms
- **Fine-tune iteratively**: Use ±1ms adjustments for precision
- **Test with real content**: Voice audio is more revealing than click tracks
- **Verify in actual usage**: Test with the Fire agent after calibration

## Technical Details

### DelayBuffer Implementation

Each channel uses a circular buffer to implement sample-accurate delays:

```python
class DelayBuffer:
    def __init__(self, max_delay_samples: int):
        self.max_delay_samples = max_delay_samples
        self.buffer = collections.deque([0.0] * max_delay_samples, maxlen=max_delay_samples)
        self.delay_samples = 0
    
    def set_delay(self, delay_samples: int):
        """Set delay in samples (delay_seconds * sample_rate)"""
        self.delay_samples = min(delay_samples, self.max_delay_samples - 1)
    
    def process_sample(self, input_sample: float) -> float:
        """Process single sample with delay"""
        self.buffer.append(input_sample)
        if self.delay_samples == 0:
            return input_sample
        else:
            delay_index = len(self.buffer) - 1 - self.delay_samples
            return self.buffer[delay_index]
```

### Sample Rate Considerations

- **Cartesia TTS**: 44100Hz output
- **Transport**: 44100Hz processing
- **Delay precision**: ~0.023ms per sample at 44100Hz
- **Typical delays**: 120ms = ~5292 samples

### Channel Mapping

The system uses a standardized 4-channel layout:
- **Channel 0**: Front Left (laptop left speaker)
- **Channel 1**: Front Right (laptop right speaker)  
- **Channel 2**: Rear Left (Bluetooth left)
- **Channel 3**: Rear Right (Bluetooth right)

Mono TTS audio is duplicated to all channels with individual delays and volumes applied.

## Troubleshooting

### Common Issues

1. **No audio output**
   - Check PipeWire Virtual-Multi sink is created and linked
   - Verify device name "pipewire" is recognized  
   - Check both speakers are connected and unmuted

2. **Chipmunk/fast audio**
   - Sample rate mismatch between components
   - Verify all components use 44100Hz
   - Check TTS service configuration

3. **Still hearing echo**
   - Delays need further adjustment
   - Use interactive calibration tool for fine-tuning
   - Consider room acoustics and speaker placement

4. **Audio cutting out** 
   - Buffer underruns due to high latency
   - Check system audio performance
   - Consider reducing delay buffer sizes

### Debug Commands

```bash
# List audio devices
uv run python scripts/test_multi_channel_audio.py --list-devices

# Check PipeWire status  
pw-dump | grep -A10 -B5 "Virtual-Multi"

# Monitor audio links
pw-link -l | grep Virtual-Multi

# Test with debug logging
PROJECT_ENV=fire uv run -m fire_agent --log-level debug
```

## Performance Notes

- **CPU overhead**: Minimal - delay processing is highly optimized
- **Memory usage**: ~5KB per channel for 120ms delay buffer at 44100Hz
- **Latency impact**: Adds configured delay time (e.g., 120ms for laptop speakers)
- **Audio quality**: No degradation - uses floating-point processing with proper clipping

## Future Enhancements

- **Auto-calibration**: Microphone-based automatic delay detection
- **Room correction**: Frequency response compensation per channel  
- **Dynamic adjustment**: Real-time delay adaptation based on network conditions
- **Multi-room support**: Extended channel count for larger installations
