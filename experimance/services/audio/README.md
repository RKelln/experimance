# Experimance Audio Service

The Experimance Audio Service integrates SuperCollider with the main Experimance installation system, providing:
- Environmental audio layers based on the current biome and era
- Sound effects for transitions and interactions 
- Background music tailored to each era

## Features

- ZeroMQ subscription to system events (`EraChanged`, `IdleStatus`, agent interactions)
- OSC communication with SuperCollider for audio control
- Tag-based audio layer management
- JSON configuration for audio layers, triggers, and music loops
- Complete SuperCollider lifecycle management:
  - Automatic startup with configurable script path
  - Graceful shutdown when service stops
  - Clean process termination

## Getting Started

### Prerequisites

- Python 3.11+
- SuperCollider 3.12+ (with sc3-plugins)
- Standard Experimance common libraries

### Installation

```bash
# From the experimance directory
cd services/audio
uv pip install -e .
```

### Running the Audio Service

```bash
# Basic usage
uv run -m experimance_audio.audio_service

# With custom configuration directory
uv run  -m experimance_audio.audio_service --config-dir /path/to/config

# SuperCollider control options
uv run  -m experimance_audio.audio_service --no-sc  # Don't auto-start SuperCollider
uv run  -m experimance_audio.audio_service --sc-script /path/to/custom_script.scd  # Use custom script
uv run  -m experimance_audio.audio_service --sclang-path /path/to/sclang  # Custom SuperCollider executable
```

### Command Line Arguments

- `--config-dir`: Directory containing audio configuration files
- `--osc-host`: SuperCollider host address (default: localhost)
- `--osc-port`: SuperCollider OSC port (default: 57120)
- `--debug`: Enable debug logging
- `--no-sc`: Don't automatically start SuperCollider
- `--sc-script`: Path to SuperCollider script (defaults to experimance_audio.scd in sc_scripts dir)
- `--sclang-path`: Path to SuperCollider language interpreter executable (defaults to 'sclang' in PATH)

## Audio Configuration

Audio configuration is loaded from JSON files in the `config` directory:

- `layers.json`: Environmental audio layers
- `triggers.json`: Sound effect triggers
- `music_loops.json`: Era-based music loops

See the technical design document for configuration schema details.

## Architecture

The audio service consists of:

1. **ZMQ Subscriber Service**: Listens for system events
2. **OSC Bridge**: Communicates with SuperCollider
3. **Config Loader**: Manages audio configuration files
4. **SuperCollider Script**: Handles audio playback and mixing

## Development and Testing

For development and manual testing, use the CLI tool:

```bash
uv run -m experimance_audio.cli
```

This provides an interactive interface for sending OSC commands to SuperCollider.
