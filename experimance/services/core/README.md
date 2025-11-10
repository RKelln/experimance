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
- 🎯 **Presence Management**: Multi-modal audience detection with idle state coordination
- 🛡️ **Robust Error Handling**: Automatic recovery from hardware and network failures

## Architecture

```
┌─────────────────┐    ┌────────────────────────────────────────────┐
│   Depth Camera  │───▶│            Core Service                    │
│   (RealSense)   │    │  ┌───────────────────────────────────────┐ │
└─────────────────┘    │  │      ControllerService                │ │
                       │  │  ┌─────────────┬─────────────────┐    │ │
                       │  │  │ Publisher   │  Subscriber     │    │ │
                       │  │  │ (events)    │  (coordination) │    │ │
                       │  │  └─────────────┴─────────────────┘    │ │
                       │  │  ┌─────────────────────────────────┐  │ │
                       │  │  │         Workers                 │  │ │
                       │  │  │  ┌─────────┬─────────┬───────┐  │  │ │
                       │  │  │  │ Image   │ Audio   │Display│  │  │ │
                       │  │  │  │Server   │Service  │Service│  │  │ │
                       │  │  │  │Push/Pull│Push/Pull│Push/  │  │  │ │
                       │  │  │  │         │         │Pull   │  │  │ │
                       │  │  │  └─────────┴─────────┴───────┘  │  │ │
                       │  │  └─────────────────────────────────┘  │ │
                       │  └───────────────────────────────────────┘ │
                       └────────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────────────┐
                      │             ZMQ Communication                 │
                      │   ┌─────────────┬─────────────┬─────────────┐ │
                      │   │PubSub Events│Push/Pull    │Worker Result│ │
                      │   │(tcp://5555) │Work Distrib │(tcp://556x) │ │
                      │   │             │(tcp://556x) │             │ │
                      │   └─────────────┴─────────────┴─────────────┘ │
                      └───────────────────────────────────────────────┘
```

### Service Type
- **Controller Service**: Uses `BaseService` + `ControllerService` composition pattern
- **Multi-Worker Coordination**: Manages push/pull workers for image, audio, and display services
- **Async Processing**: Built on `asyncio` for concurrent operations
- **Event-Driven**: Responds to user interactions and service coordination events

## Quick Start

### Prerequisites

- Python 3.11 (max if RealSense is used, until Intel updates to later Python)
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
# Set the active project (do this once)
uv run set-project experimance

# Start the core service (from project root)
uv run -m experimance_core

# With mock depth camera for testing without hardware
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth

# Force presence always-present for testing
uv run -m experimance_core \
  --presence-always-present

# Combine mock depth and always-present
uv run -m experimance_core \
  --presence-always-present \
  --depth-processing-mock-depth-images-path media/images/mocks/depth

# Enable visualization for debugging
uv run -m experimance_core --visualize

# See all command line options
uv run -m experimance_core --help
```

### Command Line Arguments

The service automatically generates CLI arguments from the configuration schema. All nested config fields can be overridden via `--section-name-field-name` format:

- `--visualize`: Enable depth camera visualization window
- `--presence-always-present`: Keep presence active (for testing)
- `--depth-processing-mock-depth-images-path PATH`: Use mock depth images from directory
- `--camera-debug-depth`: Send debug depth maps to display service
- `--state-machine-era-min-duration SECONDS`: Minimum era duration
- And many more... (use `--help` to see all options)

## Era System

The experience progresses through different eras based on user interaction:

```
Wilderness → Pre-industrial → Early Industrial → Late Industrial → Modern (1970s) → Current (2020s) 
→ Future (AI) or
→ Post-apocalyptic → Ruins → [back to Wilderness]
```

### Era Progression Logic

- **Forward Movement**: Triggered by sustained user interaction (hands in sand, touching surfaces)
- **Idle Drift**: Gradually returns toward Wilderness after periods of inactivity
- **Agent Influence**: Conversational AI can suggest biome changes and influence progression
- **Branching Paths**: AI era can transition to Post-apocalyptic scenarios or loop within AI themes

## Interaction Detection

The Core Service uses Intel RealSense depth cameras for sophisticated interaction detection:

### Supported Detection Methods

1. **Hand Detection**: Recognizes hands entering the interaction space
2. **Change Detection**: Tracks movement and changes in the depth field
3. **Presence Detection**: Determines if people are present in the space
4. **Touch Detection**: One-shot triggers for audio SFX (no persistent state)
5. **Future**: Touch sensors in sand table (planned integration)

### Presence Management System

The service includes a comprehensive presence detection system that coordinates audience interaction states across all services:

#### PresenceStatus Components
- **`idle`**: Core's decision on whether system should idle (cost optimization)
- **`present`**: Audience is detected in the space (vision, hand, voice)
- **`hand`**: Hand detected over the sand bowl (depth camera)
- **`voice`**: Audience is speaking (audio detection)
- **`touch`**: Momentary interaction trigger for audio SFX
- **`conversation`**: Either agent or human is speaking (audio ducking control)
- **`person_count`**: Number of people detected by vision system

#### Hysteresis Logic
- **Presence confirmation**: Requires 1+ seconds of consistent detection
- **Absence confirmation**: Requires 2+ seconds of no detection
- **Prevents flapping**: Debounces rapid state changes for stable operation

#### Service Integration
- **Audio Service**: Uses `conversation` field for audio ducking during speech
- **Agent Service**: Receives presence updates for conversation management
- **Display Service**: Coordinates visual feedback based on interaction state
- **Image Server**: Presence influences generation timing and content

### Camera Integration

The service uses the `realsense_camera.py` module for reliable depth processing:

- **Automatic Error Recovery**: Handles camera disconnections and USB issues
- **Mock Support**: Development and testing without hardware
- **Performance Optimized**: 30 FPS processing with configurable resolution
- **Type-Safe**: Modern Python with full type hints and async/await

For detailed camera setup and troubleshooting, see [README_DEPTH.md](README_DEPTH.md).

## Configuration

Configuration is managed through project-specific TOML files in `projects/experimance/core.toml`.

### Main Configuration

```toml
# Debugging and visualization options
visualize = false            # enable visual debugging - display depth image and processing flags

[experimance_core]
name = "experimance_core"
change_smoothing_queue_size = 4   # size of change score queue for smoothing (reduces hand artifacts)
render_request_cooldown = 2.0     # minimum interval between render requests in seconds (throttling)
#seed = 1                          # start seed for image and prompt generation

[state_machine]
entire_surface_intensity = 0.7 
interaction_threshold = 0.9  # threshold for user interaction detection (1.0 = the entire surface area once)
interaction_modifier = 0.1   # added to each interaction
era_min_duration = 0         # minimum time in seconds before era can change
era_max_duration = 180       # maximum time in seconds before era will change

[presence]
# Hysteresis/debouncing thresholds for presence detection
presence_threshold = 0.5     # seconds of presence detection before considering audience present
absence_threshold = 20.0     # seconds of no presence detection before considering audience gone
idle_threshold = 45.0        # seconds of no presence detection before considering audience gone
presence_publish_interval = 20.0  # interval in seconds to publish presence status updates
conversation_timeout = 7.0   # time after a agent or human speaker finishes that conversation is considered complete
always_present=false         # for debugging, to keep presence alive

[camera]
# Core camera settings
resolution = [1280, 720]     # depth camera resolution (width, height)
fps = 30                     # camera frames per second 
# for sand a range of about 7 cm works well
min_depth = 0.49             # minimum depth value in meters
max_depth = 0.56             # maximum depth value in meters
align_frames = true          # use color frames to help align the depth frames
colorizer_scheme = 2         # white to black

# Depth map settings
flip_horizontal = true
flip_vertical = true
circular_crop = true
blur_depth = true

# Processing parameters
output_resolution = [1024, 1024]      # processed output size (width, height)
change_threshold = 10                 # threshold for depth differences per pixel
significant_change_threshold = 0.006  # threshold for depth change detection per image
detect_hands = true          # enable hand detection
crop_to_content = true       # crop output to content area
lightweight_mode = false     # skip some processing for higher FPS
verbose_performance = false  # show detailed performance timing
debug_mode = false           # include intermediate images for visualization

[depth_processing]
# Mock depth camera for testing without hardware
mock_depth_images_path = ""  # path to directory with mock depth images (empty = use real camera)
```

### ZMQ Configuration

ZMQ ports and addresses are configured in `libs/common/src/experimance_common/constants.py`:

```python
DEFAULT_PORTS = {
    "events": 5555,           # Unified pub/sub channel
    "image_server_push": 5564,  # Core -> Image Server work
    "image_server_pull": 5565,  # Image Server -> Core results
    "audio_push": 5566,         # Core -> Audio work
    "audio_pull": 5567,         # Audio -> Core results
    "display_push": 5568,       # Core -> Display work
    "display_pull": 5569,       # Display -> Core results
}
```

The Core Service automatically configures these ports through the `ControllerService` composition pattern.

## Event System

The Core Service communicates with other services through a modern ZMQ architecture using the composition pattern:

### ZMQ Architecture

- **ControllerService Composition**: Uses `BaseService` + `ControllerService` from `experimance_common.zmq.services`
- **Publisher/Subscriber**: Broadcasts events and receives coordination messages on port 5555
- **Push/Pull Workers**: Distributes work to and receives results from worker services
- **Type-Safe Configuration**: All ZMQ settings defined via `ControllerServiceConfig` from `experimance_common.zmq.config`

### Worker Communication Patterns

The Core Service uses push/pull sockets to distribute work to worker services and receive results.

#### Sending Work to Workers

```python
# Send render request to image server worker
await self.zmq_service.send_work_to_worker("image_server", render_request)

# The render_request is a RenderRequest Pydantic model that gets serialized
```

#### Receiving Worker Responses

Worker responses are handled via the registered response handler:

```python
async def _handle_worker_response(self, worker_name: str, response_data: MessageDataType):
    """Handle responses from worker services."""
    if worker_name == "image_server":
        # Handle ImageReady message
        await self._handle_image_ready(response_data)
```

The service automatically routes responses based on their message type and worker name.

### Published Events

The Core Service publishes events via ZMQ pub/sub on the unified events channel (port 5555).

#### PresenceStatus
Publishes comprehensive audience detection state for service coordination:
```json
{
  "type": "PresenceStatus",
  "idle": false,
  "present": true,
  "hand": true,
  "voice": false,
  "touch": false,
  "conversation": true,
  "person_count": 1,
  "presence_duration": 45.3,
  "hand_duration": 12.5,
  "voice_duration": 3.2,
  "touch_duration": 0.0,
  "timestamp": "2025-11-10T12:30:00.123456"
}
```

#### SpaceTimeUpdate
Notifies all services when era/biome changes (replaces old EraChanged event):
```json
{
  "type": "SpaceTimeUpdate",
  "era": "modern",
  "biome": "urban",
  "tags": ["urban", "traffic", "city"],
  "timestamp": "2025-11-10T12:30:00Z"
}
```

#### RenderRequest
Sent to image_server worker via push/pull to trigger AI image generation:
```json
{
  "type": "RenderRequest",
  "request_id": "uuid-string",
  "era": "modern",
  "biome": "urban",
  "prompt": "A bustling modern cityscape with glass towers...",
  "negative_prompt": "blurry, low quality",
  "seed": 12345,
  "width": 1024,
  "height": 1024,
  "depth_map": {
    "image_data": "base64-encoded-png",
    "uri": null
  }
}
```

#### CHANGE_MAP (MessageType.CHANGE_MAP)
Provides real-time interaction visualization to display service:
```json
{
  "type": "CHANGE_MAP",
  "change_score": 0.8,
  "has_change_map": true,
  "image_data": "base64-encoded-binary-mask",
  "timestamp": "2025-11-10T12:30:00Z",
  "mask_id": "change_map_1699632000123"
}
```

#### DisplayMedia
Sends coordinated display content to display service:
```json
{
  "type": "DisplayMedia",
  "content_type": "image",
  "request_id": "uuid-string",
  "uri": "file:///path/to/image.png",
  "era": "modern",
  "biome": "urban",
  "fade_in": 0.5,
  "fade_out": 0.3
}
```

### Subscribed Events

The service listens for these message types via ZMQ subscriber on port 5555:

- **REQUEST_BIOME** (`RequestBiome`): Agent service requests specific biome change
- **AUDIENCE_PRESENT** (`AudiencePresent`): Vision system reports people count
- **SPEECH_DETECTED** (`SpeechDetected`): Audio system reports human or agent speech

### Worker Responses

The service receives responses via pull sockets from worker services:

- **ImageReady**: From image_server worker when image generation completes (port 5565)

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
│       ├── experimance_core.py  # Main service class with ControllerService composition
│       ├── config.py           # Configuration classes using ControllerServiceConfig
│       ├── depth_processor.py   # Modern camera interface
│       ├── depth_visualizer.py  # Camera visualization and debugging
│       ├── depth_factory.py     # Camera factory function
│       └── prompter.py         # Text prompt generation
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
# Run all core service tests
cd services/core
uv run -m pytest

# Run specific test file
uv run -m pytest tests/test_presence_manager.py

# Run tests with verbose output
uv run -m pytest -v

# Run tests matching a pattern
uv run -m pytest -k "presence"
```

### Available Test Files

The core service includes comprehensive test coverage:

- `test_config.py`: Configuration loading and validation
- `test_core_service.py`: Core service functionality
- `test_presence_manager.py`: Presence detection and hysteresis logic
- `test_experimance_core_state_management.py`: Era progression and state machine
- `test_core_image_ready_handling.py`: Image ready message handling
- `test_depth_integration.py`: Depth processing integration
- `test_queue_smoothing.py`: Change score smoothing algorithms

### Development Workflow

1. **Start with Mock Depth Camera**: Use `--depth-processing-mock-depth-images-path` for development without hardware
2. **Enable Visualization**: Use `--visualize` flag to see depth processing in real-time
3. **Use Always-Present Mode**: Add `--presence-always-present` to bypass presence detection during testing
4. **Integration Testing**: Test with other services using the unified events channel (port 5555)
5. **Monitor Logs**: Watch for state transitions, render requests, and worker responses
6. **Test State Machine**: Interact with the depth camera or use mock data to test era progression

### Mock Mode Development

For development without hardware, use the mock depth camera system:

```bash
# Set the active project (if not already set)
uv run set-project experimance

# Create a directory with mock depth images (grayscale PNG files)
mkdir -p media/images/mocks/depth

# Start core with mock depth camera
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth \
  --presence-always-present
```

The mock depth processor will cycle through images in the specified directory, simulating camera input.

**Note**: The active project is stored in `projects/.project` file. You can also manually edit this file or use the `set-project` CLI tool to switch between projects (e.g., `experimance`, `fire`, `sohkepayin`).

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

The service uses the unified health monitoring system via `BaseService`:

- **Automatic Health Reporting**: Health status written to JSON files for system monitoring
- **Service Status**: Tracks running, error, and performance states automatically
- **Camera Status**: Reports camera connection and processing health
- **Integration Metrics**: ZMQ communication health and worker coordination status

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

- **Composition over Inheritance**: Uses `BaseService` + `ControllerService` composition pattern
- **Standard Configuration**: All ZMQ config via `ControllerServiceConfig` from common library
- **Event-Driven**: Use ZMQ events for all service coordination
- **Type Safety**: Full type hints with Pydantic configuration validation
- **Worker Coordination**: Push/pull patterns for distributed work processing
- **Configuration**: All behavior configurable via `config.toml`
- **Monitoring**: Include metrics and health checks
- **Documentation**: Keep README and design docs updated

### Benefits of New ZMQ Architecture

- **Standardization**: Uses proven patterns from `experimance_common` library
- **Type Safety**: Pydantic models ensure configuration correctness
- **Maintainability**: Clear separation between service logic and ZMQ communication
- **Scalability**: Worker patterns support distributed processing
- **Reliability**: Built-in error handling and recovery patterns
- **Testing**: Easy to mock and test with standard interfaces

## Related Documentation

- **[DESIGN.md](DESIGN.md)**: Detailed architecture and implementation design
- **[README_DEPTH.md](README_DEPTH.md)**: Camera setup, configuration, and troubleshooting
- **[TODO.md](TODO.md)**: Current development tasks and future plans

For questions, issues, or contributions, please refer to the project documentation or create an issue in the repository.



