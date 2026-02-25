# Configuration Guide

This guide explains all configuration files, their relationships, and how to customize Experimance for your installation needs.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Project Configuration](#project-configuration)
3. [Service Configuration](#service-configuration)
4. [Environment Variables](#environment-variables)
5. [Hardware Configuration](#hardware-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Security Configuration](#security-configuration)
8. [Configuration Examples](#configuration-examples)

## Configuration Overview

Experimance uses a hierarchical configuration system that allows for flexible customization while maintaining sensible defaults.

### Configuration Hierarchy

```
1. Command Line Arguments (highest priority)
2. Environment Variables
3. Project-specific Configuration Files
4. Service Default Configuration
5. Base Default Configuration (lowest priority)
```

### Configuration File Structure

```
projects/
├── .project                    # Active project selector
├── experimance/               # Default project
│   ├── .env                   # Environment variables
│   ├── config.toml           # Main project configuration
│   ├── constants.py          # Project constants
│   ├── schemas.py            # Custom schemas
│   ├── core.toml             # Core service config
│   ├── display.toml          # Display service config
│   ├── image_server.toml     # Image server config
│   ├── audio.toml            # Audio service config
│   ├── agent.toml            # Agent service config
│   └── health.toml           # Health monitor config
└── your_project/             # Custom project
    └── ... (same structure)
```

## Project Configuration

### Main Configuration (`config.toml`)

The main project configuration file defines global settings:

```toml
# projects/experimance/config.toml

[project]
name = "experimance"
version = "1.0.0"
description = "Interactive sand table installation"

[installation]
# Physical installation settings
location = "gallery_space"
timezone = "America/Toronto"
display_count = 1
audio_channels = 8

[experience]
# Experience behavior settings
era_progression_speed = 1.0
interaction_sensitivity = 0.7
idle_timeout_minutes = 5
reset_to_wilderness_minutes = 15

[zmq]
# ZeroMQ communication settings
events_port = 5555
timeout_ms = 5000
high_water_mark = 1000

[logging]
# Logging configuration
level = "INFO"
file_rotation_mb = 100
max_files = 10
```

### Project Constants (`constants.py`)

Define project-specific constants and enums:

```python
# projects/experimance/constants.py
from enum import Enum

class Era(str, Enum):
    """Time periods for the experience."""
    WILDERNESS = "wilderness"
    PRE_INDUSTRIAL = "pre_industrial"
    EARLY_INDUSTRIAL = "early_industrial"
    LATE_INDUSTRIAL = "late_industrial"
    MODERN = "modern"
    CURRENT = "current"
    AI_FUTURE = "ai_future"
    POST_APOCALYPTIC = "post_apocalyptic"
    RUINS = "ruins"

class Biome(str, Enum):
    """Environmental themes."""
    FOREST = "forest"
    DESERT = "desert"
    GRASSLAND = "grassland"
    TUNDRA = "tundra"
    WETLAND = "wetland"
    URBAN = "urban"
    INDUSTRIAL = "industrial"
    WASTELAND = "wasteland"

# Project-specific settings
DEFAULT_ERA = Era.WILDERNESS
DEFAULT_BIOME = Biome.FOREST
INTERACTION_THRESHOLD = 0.5
```

### Custom Schemas (`schemas.py`)

Extend base schemas for project-specific functionality:

```python
# projects/experimance/schemas.py
from experimance_common.schemas_base import RenderRequest as _BaseRenderRequest
from .constants import Era, Biome

class RenderRequest(_BaseRenderRequest):
    """Project-specific render request."""
    era: Era
    biome: Biome
    interaction_strength: float = 0.0
    
    # Project-specific fields
    sand_displacement: float = 0.0
    audience_count: int = 0
```

## Service Configuration

### Core Service (`core.toml`)

Controls the main orchestration service:

```toml
# projects/experimance/core.toml

[camera]
# Depth camera settings
device_id = 0
output_resolution = [640, 480]
fps = 30
enable_color = false
depth_range_mm = [200, 1500]

[depth_processing]
# Interaction detection settings
hand_detection_enabled = true
change_detection_enabled = true
interaction_threshold = 0.3
min_hand_area = 100
blur_kernel_size = 5
fps = 15

[state_machine]
# Experience state management
era_min_duration = 30.0
era_max_duration = 300.0
biome_change_probability = 0.1
idle_decay_rate = 0.02
interaction_boost = 0.1

[presence]
# Audience presence detection
timeout_seconds = 30
detection_methods = ["depth_camera", "agent_vision"]
always_present = false  # For testing

[zmq]
# Service-specific ZMQ settings
publisher_port = 5555
worker_ports = [5561, 5562, 5563]
```

### Display Service (`display.toml`)

Controls visual output and rendering:

```toml
# projects/experimance/display.toml

[display]
# Display hardware settings
fullscreen = true
monitor_index = 0
resolution = [1920, 1080]
vsync = true
multisampling = 4

[rendering]
# Rendering settings
background_color = [0.0, 0.0, 0.0, 1.0]
image_blend_mode = "normal"
transition_duration = 2.0
fade_duration = 0.5

[layers]
# Display layer configuration
background_enabled = true
image_enabled = true
overlay_enabled = true
text_enabled = true
debug_enabled = false

[effects]
# Visual effects
shader_effects_enabled = true
particle_effects_enabled = false
bloom_enabled = false
color_correction_enabled = true

[text]
# Text overlay settings
font_family = "Arial"
font_size = 24
color = [1.0, 1.0, 1.0, 1.0]
position = [0.05, 0.95]  # Bottom left
```

### Image Server (`image_server.toml`)

Controls AI image generation:

```toml
# projects/experimance/image_server.toml

[generation]
# Generation settings
default_generator = "flux"
fallback_generator = "stable_diffusion"
batch_size = 1
steps = 20
guidance_scale = 7.5
seed = -1  # Random seed

[models]
# Model configuration
flux_model = "black-forest-labs/FLUX.1-schnell"
sd_model = "runwayml/stable-diffusion-v1-5"
lora_models = []
controlnet_models = []

[cache]
# Caching settings
enabled = true
max_size_gb = 10
cleanup_threshold_gb = 8
cache_directory = "media/images/generated"

[remote]
# Remote generation services
vastai_enabled = false
fal_enabled = false
runware_enabled = false
timeout_seconds = 60

[prompts]
# Prompt engineering
base_prompt = "aerial satellite view"
negative_prompt = "people, text, watermark"
style_strength = 0.8
```

### Audio Service (`audio.toml`)

Controls spatial audio and sound design:

```toml
# projects/experimance/audio.toml

[supercollider]
# SuperCollider engine settings
server_options = "-u 57110 -a 1024 -i 2 -o 8"
boot_timeout = 10
sample_rate = 48000
block_size = 512

[audio]
# Audio hardware settings
input_device = "default"
output_device = "default"
channels = 8
buffer_size = 1024
latency_ms = 20

[spatial]
# Spatial audio settings
enabled = true
room_size = [10.0, 8.0, 3.0]  # meters
listener_position = [5.0, 4.0, 1.5]
speaker_positions = [
    [-3.0, -2.0, 2.0],  # Front left
    [3.0, -2.0, 2.0],   # Front right
    [-3.0, 2.0, 2.0],   # Rear left
    [3.0, 2.0, 2.0],    # Rear right
]

[soundscapes]
# Environmental audio
ambient_volume = 0.7
interaction_volume = 0.8
music_volume = 0.6
crossfade_duration = 3.0

[effects]
# Audio effects
reverb_enabled = true
delay_enabled = false
compression_enabled = true
eq_enabled = true
```

### Agent Service (`agent.toml`)

Controls conversational AI and vision:

```toml
# projects/experimance/agent.toml

[conversation]
# Conversation AI settings
llm_provider = "openai"
model = "gpt-4"
temperature = 0.7
max_tokens = 150
system_prompt = "You are a wise spirit of the earth..."

[voice]
# Text-to-speech settings
tts_provider = "elevenlabs"
voice_id = "21m00Tcm4TlvDq8ikWAM"
stability = 0.5
similarity_boost = 0.5
speed = 1.0

[speech]
# Speech-to-text settings
stt_provider = "deepgram"
language = "en-US"
model = "nova-2"
smart_format = true

[vision]
# Vision processing settings
webcam_enabled = true
webcam_device_id = 0
webcam_resolution = [640, 480]
webcam_fps = 15

audience_detection_enabled = true
face_detection_enabled = true
object_detection_enabled = false

vlm_provider = "openai"
vlm_model = "gpt-4-vision-preview"

[tools]
# Available tools for the agent
biome_control_enabled = true
system_control_enabled = false
information_access_enabled = true
```

### Health Monitor (`health.toml`)

Controls system monitoring and alerts:

```toml
# projects/experimance/health.toml

[monitoring]
# Health check settings
check_interval_seconds = 60
service_timeout_seconds = 30
resource_check_enabled = true
zmq_check_enabled = true

[alerts]
# Alert configuration
email_enabled = true
ntfy_enabled = false
webhook_enabled = false

[thresholds]
# Alert thresholds
cpu_percent = 80.0
memory_percent = 85.0
disk_percent = 90.0
service_restart_count = 3

[dashboard]
# Web dashboard settings
enabled = true
port = 8080
refresh_interval_seconds = 5
```

## Environment Variables

### Core Environment Variables (`.env`)

```env
# projects/experimance/.env

# Project identification
PROJECT_ENV=experimance

# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
ELEVENLABS_API_KEY=your_elevenlabs_key
DEEPGRAM_API_KEY=your_deepgram_key

# Remote services
VASTAI_API_KEY=your_vastai_key
FAL_API_KEY=your_fal_key
RUNWARE_API_KEY=your_runware_key

# Hardware settings
REALSENSE_DEVICE_ID=0
WEBCAM_DEVICE_ID=0
AUDIO_DEVICE_NAME=default

# Development settings
DEBUG=false
LOG_LEVEL=INFO
MOCK_HARDWARE=false

# Security
SECRET_KEY=your_secret_key
DASHBOARD_PASSWORD=your_dashboard_password

# Monitoring
ALERT_EMAIL=your-email@example.com
ALERT_EMAIL_TO=your-email@example.com
ALERT_EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Network
ZMQ_EVENTS_PORT=5555
DASHBOARD_PORT=8080
```

### Environment Variable Precedence

1. **System environment variables** (highest priority)
2. **Project .env file**
3. **Default values in configuration**

```bash
# Override for single command
PROJECT_ENV=fire uv run -m experimance_core

# Override for session
export PROJECT_ENV=fire
uv run -m experimance_core

# Override in .env file
echo "PROJECT_ENV=fire" >> projects/experimance/.env
```

## Hardware Configuration

### Camera Configuration

#### Intel RealSense Setup

```toml
[camera]
device_id = 0
output_resolution = [640, 480]
fps = 30
enable_color = false
depth_range_mm = [200, 1500]

# Advanced settings
auto_exposure = true
laser_power = 150  # 0-360
accuracy = "medium"  # high, medium, low
motion_range = "medium"  # high, medium, low
```

#### Webcam Setup

```toml
[vision]
webcam_enabled = true
webcam_device_id = 0
webcam_resolution = [640, 480]
webcam_fps = 15

# Camera settings
auto_focus = true
exposure = -1  # Auto
brightness = 0
contrast = 0
saturation = 0
```

### Audio Hardware Configuration

#### Multi-Channel Audio Setup

```toml
[audio]
input_device = "USB Audio Interface"
output_device = "USB Audio Interface"
channels = 8
buffer_size = 1024
sample_rate = 48000

[spatial]
speaker_positions = [
    [-3.0, -2.0, 2.0],  # Front left
    [3.0, -2.0, 2.0],   # Front right
    [-3.0, 2.0, 2.0],   # Rear left
    [3.0, 2.0, 2.0],    # Rear right
    [0.0, -3.0, 2.0],   # Center front
    [0.0, 3.0, 2.0],    # Center rear
    [-1.5, 0.0, 2.0],   # Side left
    [1.5, 0.0, 2.0],    # Side right
]
```

### Display Hardware Configuration

#### Multi-Monitor Setup

```toml
[display]
fullscreen = true
monitor_index = 1  # Secondary monitor
resolution = [3840, 2160]  # 4K
vsync = true
multisampling = 4

# Projector settings
gamma_correction = 2.2
brightness = 1.0
contrast = 1.0
color_temperature = 6500
```

## Performance Tuning

### CPU Optimization

```toml
[depth_processing]
fps = 15  # Reduce from 30 for lower CPU usage
blur_kernel_size = 3  # Reduce from 5
thread_count = 4  # Match CPU cores

[generation]
batch_size = 1  # Reduce for lower memory usage
steps = 15  # Reduce from 20 for faster generation
```

### Memory Optimization

```toml
[cache]
max_size_gb = 5  # Reduce cache size
cleanup_threshold_gb = 4

[models]
# Use smaller models
flux_model = "black-forest-labs/FLUX.1-schnell"  # Faster model
precision = "fp16"  # Half precision for lower memory
```

### GPU Optimization

```toml
[generation]
device = "cuda"  # or "mps" for Apple Silicon
mixed_precision = true
compile_model = true  # PyTorch 2.0 compilation
memory_efficient_attention = true
```

### Network Optimization

```toml
[zmq]
high_water_mark = 100  # Reduce for lower memory
timeout_ms = 3000  # Reduce for faster failure detection
tcp_keepalive = true
tcp_keepalive_idle = 600
```

## Security Configuration

### API Key Management

```env
# Use environment-specific keys
OPENAI_API_KEY_DEV=dev_key_here
OPENAI_API_KEY_PROD=prod_key_here

# Rotate keys regularly
ELEVENLABS_API_KEY=new_key_$(date +%Y%m)
```

### Network Security

```toml
[zmq]
# Bind to localhost only for security
bind_address = "127.0.0.1"
# Use encryption for remote connections
encryption_enabled = true
encryption_key = "your_encryption_key"

[dashboard]
# Secure dashboard access
password_protected = true
ssl_enabled = true
ssl_cert_path = "/path/to/cert.pem"
ssl_key_path = "/path/to/key.pem"
```

### File Permissions

```bash
# Secure configuration files
chmod 600 projects/*/env
chmod 644 projects/*/*.toml

# Secure log files
chmod 640 logs/*.log
chown experimance:experimance logs/
```

## Configuration Examples

### Development Configuration

```toml
# projects/experimance/config.toml (development)
[project]
name = "experimance_dev"

[experience]
era_progression_speed = 2.0  # Faster for testing
interaction_sensitivity = 0.3  # More sensitive

[logging]
level = "DEBUG"
```

```env
# projects/experimance/.env (development)
DEBUG=true
LOG_LEVEL=DEBUG
MOCK_HARDWARE=true
```

### Production Configuration

```toml
# projects/experimance/config.toml (production)
[project]
name = "experimance_gallery"

[experience]
era_progression_speed = 1.0
interaction_sensitivity = 0.7

[logging]
level = "INFO"
file_rotation_mb = 100
```

```env
# projects/experimance/.env (production)
DEBUG=false
LOG_LEVEL=INFO
MOCK_HARDWARE=false
ALERT_EMAIL=gallery@example.com
```

### High-Performance Configuration

```toml
# For powerful hardware
[generation]
batch_size = 4
steps = 50
guidance_scale = 7.5

[depth_processing]
fps = 30
output_resolution = [1280, 720]

[audio]
channels = 16
sample_rate = 96000
```

### Low-Resource Configuration

```toml
# For limited hardware
[generation]
batch_size = 1
steps = 10
guidance_scale = 5.0

[depth_processing]
fps = 10
output_resolution = [320, 240]

[cache]
max_size_gb = 1
```

### Testing Configuration

```toml
# For automated testing
[presence]
always_present = true

[depth_processing]
mock_depth_images_path = "media/images/mocks/depth"

[generation]
mock_generation = true
```

This configuration system provides flexibility while maintaining consistency across different deployment scenarios. Start with the default configurations and customize based on your specific installation requirements.