# Experimance Agent Service

The Agent Service is the core conversational AI component of the Experimance installation. It orchestrates conversation AI, vision processing, and tool integration to create interactive experiences with visitors.

## Overview

The agent service acts as a coordinator between:
- **Conversation AI**: Pipecat backend with STT/LLM/TTS pipeline
- **Vision Processing**: Webcam capture, audience detection, and VLM analysis
- **Tool Integration**: Dynamic biome changes and system control
- **Transcript Management**: Real-time conversation logging and display

## Quick Start

### Basic Setup

```bash
# From the project root
uv run -m experimance_agent
```

### Reolink Camera Setup (Fire Project)

For the Fire project using Reolink cameras for audience detection:

```bash
# find reolink  cameras
uv run python scripts/list_cameras.py
```

Configure camera in `projects/fire/.env`:
```env
REOLINK_HOST=192.168.2.229
REOLINK_USER=admin  
REOLINK_PASSWORD=your_password
```

### Testing Without External Dependencies

For development and testing without a webcam or core service:

```bash
# Option 1: Command line override
uv run -m experimance_agent --no-vision-webcam_enabled --no_vision_audience_detection_enabled

# Option 2: Edit projects/experimance/agent.toml
[vision]
webcam_enabled = false
audience_detection_enabled = false
```

### Mock Audience Testing (Fire Project)

For testing presence detection without a camera, use the mock audience detector.
Update the `agent.toml` for shorter timeouts and mock presence detector:

```toml
# Timeout - Reduced for testing (normally 300s for production)
idle_timeout_secs = 30                 # 30 seconds for testing (was 300)

# Faster testing timings
conversation_cooldown_duration = 5.0   # seconds (reduced from 12.0)
audience_detection_interval = 1.0      # seconds (reduced from 1.5)

# Enable mock detector (overrides camera)
[mock_detector]
enabled = true                   # Enable mock detector
control_method = "file"          # Use file-based control
control_dir = "/tmp/mock_detector"
initial_state = false            # Start with no audience
initial_count = 0
```

```bash
# Start fire agent
uv run -m fire_agent

# Control presence interactively (in another terminal)
uv run python scripts/mock_control.py -i

uv run python scripts/mock_control.py --test-scenarios
```

For complete mock detector setup and usage, see [Mock Detector Testing Guide](../../docs/mock_detector_testing.md).

This allows you to test the conversation AI independently without requiring:
- Physical webcam hardware
- Core service for audience presence coordination
- Vision processing dependencies

## Configuration

Main configuration is in `projects/experimance/agent.toml`:

```toml
# Backend selection
agent_backend = "pipecat"  # Currently only supported backend

# Audio devices (use partial names or indices)
[backend_config.pipecat]
audio_input_device_name = "Torch Streaming Microphone"
audio_output_device_name = "Torch Streaming Microphone"

# Conversation behavior
conversation_cooldown_duration = 12.0  # Wait between conversations
cancel_cooldown_on_absence = true      # End cooldown if audience changes

# Vision processing (see README_VISION.md)
[vision]
webcam_enabled = false           # Set to true for vision features
audience_detection_enabled = false

# Mock detector for testing (Fire project)
[mock_detector]
enabled = false                  # Enable for testing without camera
control_method = "file"          # "file" or "keyboard" control
control_dir = "/tmp/mock_detector"
```

## Audio Setup

List available audio devices:
```bash
uv run python scripts/list_audio_devices.py
```

The agent supports partial device name matching for easier configuration:
- `"Torch"` matches `"Torch Streaming Microphone"`
- `"Yealink"` matches `"Yealink SP92 Speaker"`

## Architecture

### Conversation Flow
1. **Audience Detection**: Vision system detects people (if enabled)
2. **Backend Initialization**: Pipecat pipeline starts when audience present
3. **Conversation Management**: STT → LLM → TTS with tool calling
4. **Cooldown Period**: Prevents immediate restart after conversations end

### Tool Integration
The agent can call tools to:
- Request biome changes based on conversation context
- Control system behavior
- Provide dynamic responses to visitor interactions

### State Management
- **Service States**: Starting → Running → Stopping → Stopped
- **Conversation States**: Active/Inactive with cooldown management
- **Audio States**: Speaking/Silent tracking for timing features

## Backend Details

### Pipecat Backend
- **Ensemble Mode**: Separate STT/LLM/TTS components (default)
- **Realtime Mode**: OpenAI Realtime API (experimental)
- **Flow Files**: Python-based conversation flows in `flows/`
- **Audio Processing**: Real-time VAD and noise suppression

### Audio Configuration
- **Sample Rates**: 16kHz for optimal performance
- **VAD**: Voice Activity Detection enabled by default
- **Error Handling**: Automatic device retry and recovery

## Vision Integration

For vision processing capabilities, see [README_VISION.md](README_VISION.md).

Key vision features:
- Webcam capture and preprocessing
- Audience detection with multiple methods
- VLM-based scene understanding
- Real-time performance optimization

### Vision System Setup

The agent service includes vision capabilities for audience detection. To set up and test webcam functionality:

```bash
# Check available webcams
uv run python scripts/list_webcams.py

# Test vision system components
cd services/agent && uv run python tests/test_vision_imports.py

# Test CPU-based audience detection
cd services/agent && uv run python tests/test_cpu_detection.py
```

## Testing

### Basic Functionality
```bash
# Test without dependencies
uv run -m experimance_agent --no-vision-webcam_enabled --no_vision_audience_detection_enabled

# Test with vision (requires webcam)
uv run -m experimance_agent
```

### Mock Audience Testing (Fire Project)
```bash
# Test presence detection without camera
cp projects/fire/config_mock_detector.toml projects/fire/config.toml
uv run -m fire_agent

# Interactive control (in another terminal)
uv run python scripts/mock_control.py -i
```

For detailed mock testing procedures and scenarios, see [Mock Detector Testing Guide](../../docs/mock_detector_testing.md).

### Audio Testing
```bash
# List and test audio devices
uv run python scripts/list_audio_devices.py
uv run python scripts/audio_recovery.py test
```

### Vision Testing (if enabled)
```bash
# Test vision components
uv run python tests/test_vision.py

# Interactive detection testing
uv run python tests/test_cpu_live_detection.py
```

## Development

### Environment Variables
Override any config setting:
```bash
export EXPERIMANCE_VISION_WEBCAM_ENABLED=false
export EXPERIMANCE_BACKEND_CONFIG_PIPECAT_AUDIO_INPUT_DEVICE_NAME="USB Audio"
```

### Debugging
```bash
# Enable debug logging
export EXPERIMANCE_LOG_LEVEL=DEBUG
uv run -m experimance_agent
# or:
uv run -m experimance_agent --log-level=debug
```

### Signal Handling
The agent supports graceful shutdown:
- `Ctrl+C` (SIGINT): Fast shutdown (~1 second)
- Natural conversation end: Preserves Pipecat's idle timeout behavior

## Integration

The agent integrates with other Experimance services via ZMQ:
- **Core Service**: Receives space-time updates, publishes audience presence
- **Display Service**: Sends transcript text for visual display (currently disabled)
- **Health Service**: Reports status and errors

## Troubleshooting

### Common Issues

**Audio not working:**
- Check device names with `uv run scripts/list_audio_devices.py`
- Verify device permissions (`usermod -a -G audio $USER`)
- Try different sample rates or devices

**Vision not detecting:**
- Ensure webcam permissions and availability
- Check lighting conditions and camera positioning
- Use interactive tuner: `uv run python scripts/tune_detector.py`
- For detailed vision troubleshooting, see [README_VISION.md](README_VISION.md).

**Backend startup fails:**
- Verify OpenAI API key and model access
- Check audio device availability
- Review flow file syntax

