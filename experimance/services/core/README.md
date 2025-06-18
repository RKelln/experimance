# Experimance Core Service

The **Core Service** is the central coordinator for the Experimance interactive art installation. It manages the experience state machine, processes depth camera data for interaction detection, and coordinates all other services through a sophisticated event-driven architecture.

## Overview

Experimance is an interactive art installation that responds to human presence and interaction, creating a dynamic narrative journey through different eras of human development. The Core Service acts as the "brain" of the installation, interpreting user interactions through depth sensing and orchestrating audio-visual responses across multiple connected services.

### Key Features

- 🎭 **Era-based Narrative**: Progresses through human history from wilderness to AI/future
- 👋 **Interaction Detection**: Real-time hand/presence detection via Intel RealSense cameras
- 🔄 **Event Coordination**: ZMQ-based pub/sub architecture for service coordination
- 🎨 **Image Generation**: Coordinates AI image generation based on interactions
- 🔊 **Environmental Audio**: Manages spatial audio that responds to era and biome
- 🤖 **AI Agent Integration**: Conversational AI that can influence the experience
- 🛡️ **Robust Error Handling**: Automatic recovery from hardware and network failures

## Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   Depth Camera  │───▶│   Core Service   │
│   (RealSense)   │    │   (Coordinator)  │
└─────────────────┘    └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  ZMQ Event Bus   │───▶│  Other Services │
                       │   (tcp://5555)   │    │ Audio/Display/  │
                       └──────────────────┘    │ Image/Agent/etc │
                                               └─────────────────┘
```

### Service Type
- **Publisher/Subscriber Service**: Extends `ZMQPublisherSubscriberService`
- **Async Processing**: Built on `asyncio` for concurrent operations
- **Event-Driven**: Responds to user interactions and service coordination events

## Quick Start

### Prerequisites

- Python 3.11+
- Intel RealSense camera (D415, D435, D455, etc.)
- `uv` package manager

### Installation

```bash
# Navigate to the core service directory
cd services/core

# Install dependencies
uv sync
```

### Running the Service

```bash
# Start the core service
uv run -m experimance_core

# Or run with specific config
uv run -m experimance_core --config config.toml

# Run in development mode (more verbose logging)
uv run -m experimance_core --dev
```

### Testing Camera Setup

```bash
# Test camera connection and basic functionality
uv run python tests/test_camera.py --info

# Test with real camera (press Ctrl+C to stop)
uv run python tests/test_camera.py --real

# Test with mock camera (no hardware required)
uv run python tests/test_camera.py --mock

# Visual debugging with real-time processing display
uv run python tests/test_camera.py --visualize
```

## Era System

The experience progresses through different eras based on user interaction:

```
Wilderness → Pre-industrial → Early Industrial → Late Industrial → 
Early Modern → Modern → AI/Future ⟷ Post-apocalyptic → Ruins → [back to Wilderness]
```

### Era Progression Logic

- **Forward Movement**: Triggered by sustained user interaction (hands in sand, touching surfaces)
- **Idle Drift**: Gradually returns toward Wilderness after periods of inactivity
- **Agent Influence**: Conversational AI can suggest biome changes and influence progression
- **Branching Paths**: AI era can transition to Post-apocalyptic scenarios or loop within AI themes

### Biome System

Each era can be experienced in different environmental contexts:
- **Natural**: `forest`, `desert`, `mountain`, `coastal`, `tundra`
- **Human**: `urban`, `farmland`, `industrial`, `residential`
- **Future**: `space_station`, `underwater`, `virtual`, `post_apocalyptic`

## Interaction Detection

The Core Service uses Intel RealSense depth cameras for sophisticated interaction detection:

### Supported Detection Methods

1. **Hand Detection**: Recognizes hands entering the interaction space
2. **Change Detection**: Tracks movement and changes in the depth field
3. **Presence Detection**: Determines if people are present in the space
4. **Future**: Touch sensors in sand table (planned integration)

### Camera Integration

The service uses the `robust_camera.py` module for reliable depth processing:

- **Automatic Error Recovery**: Handles camera disconnections and USB issues
- **Mock Support**: Development and testing without hardware
- **Performance Optimized**: 30 FPS processing with configurable resolution
- **Type-Safe**: Modern Python with full type hints and async/await

For detailed camera setup and troubleshooting, see [README_DEPTH.md](README_DEPTH.md).

## Configuration

### Main Configuration (`config.toml`)

```toml
[experimance_core]
name = "experimance_core"
publish_port = 5555           # ZMQ event publishing port
heartbeat_interval = 3.0      # Service health broadcast interval

[state_machine] 
idle_timeout = 45.0           # Seconds before idle drift starts
wilderness_reset = 300.0      # Seconds to full reset to wilderness
interaction_threshold = 0.3   # Minimum score for era advancement
era_min_duration = 10.0       # Minimum time in era before advancement

[depth_processing]
camera_config_path = ""       # Optional RealSense advanced config JSON
resolution = [640, 480]       # Camera resolution (validated working)
fps = 30                      # Target frame rate
change_threshold = 50         # Motion detection sensitivity
min_depth = 0.49             # Minimum depth range (meters)
max_depth = 0.56             # Maximum depth range (meters)
output_size = [1024, 1024]   # Processed output resolution

[audio]
zmq_address = "tcp://*:5560"  # Audio service coordination
interaction_sound_duration = 2.0  # Duration for interaction sounds

[prompting]
data_path = "data/"           # Location and development data
locations_file = "locations.json"
developments_file = "anthropocene.json"
```

### Camera-Specific Configuration

The camera system supports extensive configuration for different scenarios:

```python
# Production configuration (recommended)
production_config = CameraConfig(
    resolution=(640, 480),      # Validated working resolution
    fps=30,                     # Smooth operation
    max_retries=5,              # Robust error recovery
    detect_hands=True,          # Enable interaction detection
    crop_to_content=True,       # Optimize processing
    verbose_performance=False   # Minimal logging overhead
)

# Development configuration
dev_config = CameraConfig(
    resolution=(320, 240),      # Faster processing
    fps=15,                     # Lower CPU usage
    max_retries=2,              # Fail fast for debugging
    lightweight_mode=True,      # Skip expensive operations
    verbose_performance=True    # Detailed performance logs
)
```

## Event System

The Core Service communicates with other services through ZMQ events published on port 5555.

### Published Events

#### EraChanged
Notifies all services when the experience transitions between eras:
```json
{
  "event_type": "EraChanged",
  "timestamp": "2025-06-15T10:30:00Z",
  "era": "modern",
  "biome": "urban", 
  "user_interaction_score": 0.7,
  "transition_reason": "user_interaction"
}
```

#### RenderRequest
Triggers AI image generation based on current state:
```json
{
  "event_type": "RenderRequest",
  "timestamp": "2025-06-15T10:30:00Z",
  "request_id": "uuid-string",
  "prompt": "A bustling modern cityscape with glass towers...",
  "era": "modern",
  "biome": "urban",
  "seed": 12345
}
```

#### AudioCommand
Controls environmental audio and interaction sounds:
```json
{
  "event_type": "AudioCommand",
  "timestamp": "2025-06-15T10:30:00Z",
  "command_type": "spacetime",
  "era": "modern",
  "biome": "urban",
  "tags_to_include": ["urban", "traffic", "city"],
  "tags_to_exclude": ["birds", "forest", "wind"]
}
```

#### VideoMask
Provides real-time interaction visualization:
```json
{
  "event_type": "VideoMask",
  "timestamp": "2025-06-15T10:30:00Z", 
  "mask_type": "depth_difference",
  "interaction_score": 0.8,
  "depth_map_base64": "base64-encoded-data"
}
```

#### DisplayMedia
Sends coordinated display content with intelligent transitions:
```json
{
  "event_type": "DisplayMedia",
  "content_type": "image",
  "image_data": "<transport_optimized_data>",
  "fade_in": 0.5,
  "fade_out": 0.3,
  "era": "modern",
  "biome": "urban",
  "source_request_id": "uuid-string"
}
```

### Subscribed Events

The service listens for coordination events from other services:

- **ImageReady**: Confirms image generation completion, triggers DisplayMedia publishing
- **AgentControl**: AI agent presence detection and biome suggestions  
- **AudioStatus**: Audio system status and coordination

## Development

### Project Structure

```
services/core/
├── README.md                    # This file
├── DESIGN.md                    # Detailed architecture documentation
├── README_DEPTH.md              # Camera setup and troubleshooting
├── config.toml                  # Service configuration
├── pyproject.toml              # Python dependencies and scripts
├── src/
│   └── experimance_core/
│       ├── __init__.py
│       ├── experimance_core.py  # Main service class
│       ├── robust_camera.py     # Modern camera interface
│       ├── camera_utils.py      # Camera diagnostics and utilities
│       ├── mock_depth_processor.py  # Mock camera for development
│       ├── depth_factory.py     # Camera factory function
│       ├── config.py           # Configuration classes
│       ├── prompter.py         # Text prompt generation
│       └── depth_finder.py     # Legacy camera module (deprecated)
├── tests/
│   ├── test_camera.py          # Camera testing and visualization
│   ├── test_integration.py     # Service integration tests
│   └── test_depth_integration.py  # Depth processing tests
├── data/
│   ├── locations.json          # Geographic location data
│   └── anthropocene.json      # Era development data
└── saved_data/                 # State persistence
```

### Running Tests

```bash
# Test camera functionality
uv run python tests/test_camera.py --mock --duration 10

# Test service integration
uv run -m pytest tests/test_integration.py

# Test depth processing
uv run -m pytest tests/test_depth_integration.py

# Run all tests
uv run -m pytest
```

### Development Workflow

1. **Start with Mock Camera**: Use `--mock` mode for initial development
2. **Test Camera Setup**: Use visualization mode to verify camera configuration
3. **Integration Testing**: Test with other services using ZMQ events
4. **Performance Validation**: Monitor FPS and processing times
5. **Error Recovery Testing**: Simulate camera disconnections and failures

### Mock Mode Development

For development without hardware, use the mock camera system:

```python
from experimance_core.mock_depth_processor import MockDepthProcessor
from experimance_core.config import CameraConfig

# Create mock processor
config = CameraConfig(fps=10)  # Faster for testing
processor = MockDepthProcessor(config)

await processor.initialize()

# Process mock frames with realistic interaction patterns
async for frame in processor.stream_frames():
    # Your processing logic here
    if frame.has_interaction:
        print(f"Mock interaction detected: {frame.change_score}")
```

## Service Integration

### Audio Service Integration
- **Environmental Audio**: Coordinates biome-appropriate soundscapes
- **Interaction Sounds**: Triggers audio feedback for user interactions
- **Era Transitions**: Manages audio transitions between time periods

### Display Service Integration  
- **Video Overlays**: Real-time interaction visualization
- **Era Visuals**: Coordinates visual themes with audio
- **Idle States**: Visual feedback during wilderness reset

### Image Server Integration
- **AI Generation**: Triggers image generation based on interactions
- **Context Passing**: Provides era/biome context for relevant imagery
- **Timing Coordination**: Manages image transition timing

### Agent Service Integration
- **Presence Detection**: Audience detection for conversation initiation
- **Biome Influence**: AI can suggest biome changes through conversation
- **Context Awareness**: Provides interaction context to AI responses

## Monitoring and Debugging

### Logging

The service provides comprehensive logging at multiple levels:

```bash
# Normal operation (INFO level)
uv run -m experimance_core

# Verbose debugging (DEBUG level)
uv run -m experimance_core --verbose

# Performance monitoring
uv run -m experimance_core --performance

# Camera-specific debugging
uv run python tests/test_camera.py --verbose --visualize
```

### Health Monitoring

The service publishes health information via ZMQ heartbeats:

- **Service Status**: Running, error, or stopped states
- **Camera Status**: Connected, disconnected, or error states  
- **Performance Metrics**: FPS, processing latency, error rates
- **Interaction Metrics**: User presence, interaction frequency

### Troubleshooting

#### Camera Issues
```bash
# Check camera connection
uv run python tests/test_camera.py --info

# Test basic functionality
uv run python tests/test_camera.py --mock --duration 5

# Visual debugging
uv run python tests/test_camera.py --visualize
```

#### Service Issues
```bash
# Check ZMQ port availability
netstat -ln | grep 5555

# Test service startup
uv run -m experimance_core --test

# Check configuration
uv run python -c "from experimance_core.config import CoreServiceConfig; print(CoreServiceConfig.load())"
```

#### Performance Issues
```bash
# Monitor resource usage
top -p $(pgrep -f experimance_core)

# Test with reduced settings
uv run python tests/test_camera.py --mock --duration 30 --verbose
```

## Deployment

### Production Setup

1. **Hardware Requirements**:
   - Intel RealSense D415/D435/D455 camera
   - USB 3.0 port with stable power
   - Linux system with proper USB permissions

2. **System Configuration**:
   ```bash
   # Add user to camera permissions
   sudo usermod -a -G dialout $USER
   
   # Install RealSense drivers
   sudo apt install librealsense2-*
   
   # Verify camera detection
   rs-enumerate-devices
   ```

3. **Service Configuration**:
   ```bash
   # Create production config
   cp config.toml.example config.toml
   
   # Edit for production settings
   nano config.toml
   
   # Test configuration
   uv run -m experimance_core --test
   ```

4. **Process Management**:
   ```bash
   # Using systemd (recommended)
   sudo cp experimance-core.service /etc/systemd/system/
   sudo systemctl enable experimance-core
   sudo systemctl start experimance-core
   
   # Monitor service
   sudo systemctl status experimance-core
   journalctl -u experimance-core -f
   ```

### Docker Deployment

```dockerfile
FROM python:3.11

# Install RealSense libraries
RUN apt-get update && apt-get install -y \
    librealsense2-dev \
    librealsense2-utils

# Install uv and dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

# Copy service code
COPY src/ src/
COPY config.toml data/ ./

# Run service
CMD ["uv", "run", "-m", "experimance_core"]
```

## Performance Characteristics

### Hardware Requirements
- **CPU**: Multi-core recommended for concurrent processing
- **Memory**: 2-4GB RAM for normal operation
- **USB**: USB 3.0 required for RealSense cameras
- **Network**: Minimal bandwidth for ZMQ coordination

### Performance Metrics
- **Camera Processing**: 30 FPS at 640x480 resolution
- **Interaction Response**: < 100ms from detection to event publishing
- **Memory Usage**: ~200-500MB steady state
- **CPU Usage**: 15-30% on modern multi-core systems

### Scaling Considerations
- Multiple camera support (planned)
- Distributed deployment across multiple machines
- Redis-based state persistence for multi-instance setups
- Load balancing for high-traffic installations

## Contributing

### Development Setup

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd services/core
   uv sync --dev
   ```

2. **Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

3. **Testing**:
   ```bash
   uv run -m pytest
   uv run python tests/test_camera.py --mock
   ```

### Code Standards

- **Type Hints**: Required for all public APIs
- **Async/Await**: Use for all I/O operations
- **Error Handling**: Comprehensive with recovery strategies
- **Logging**: Structured logging with appropriate levels
- **Testing**: Unit tests and integration tests required

### Architecture Guidelines

- **Modularity**: Keep camera, state machine, and coordination separate
- **Event-Driven**: Use ZMQ events for all service coordination
- **Configuration**: All behavior configurable via `config.toml`
- **Monitoring**: Include metrics and health checks
- **Documentation**: Keep README and design docs updated

## Related Documentation

- **[DESIGN.md](DESIGN.md)**: Detailed architecture and implementation design
- **[README_DEPTH.md](README_DEPTH.md)**: Camera setup, configuration, and troubleshooting
- **[TODO.md](TODO.md)**: Current development tasks and future plans

For questions, issues, or contributions, please refer to the project documentation or create an issue in the repository.



