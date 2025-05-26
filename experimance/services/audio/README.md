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
- Placeholder music generation for missing audio files:
  - Era-specific musical keys (circle of fifths progression)
  - Unique synthesizer timbres for each era
  - Different musical patterns by slot (drones, arpeggios, melodies)

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

1. **ZMQ Subscriber Service**: Listens for system events via the star topology
2. **OSC Bridge**: Communicates with SuperCollider
3. **Config Loader**: Manages audio configuration files
4. **SuperCollider Script**: Handles audio playback and mixing

### ZeroMQ Architecture

The audio service follows a simplified star topology with the core service as the central hub:

```
                      ┌─────────────┐
                      │             │
                      │    Core     │      PUSH/PULL
                      │  Publisher  │<─────────────┐
                      │             │              │
                      └──────┬──────┘              │
                             │                     │
                          PUB/SUB                  │
                             │                     │
           ┌─────────────────┼─────────────────┐   │ 
           │                 │                 │   │
           ▼                 ▼                 ▼   │
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │   Display   │   │    Audio    │   │   Agent     │
    │   Service   │   │   Service   │   │   Service   │
    └─────────────┘   └─────────────┘   └─────────────┘
                           │  ^              
                           │  │              
                      OSC  │  │              
                           │  │              
                           ▼  │              
                      ┌─────────────┐        
                      │             │        
                      │SuperCollider│
                      │             │
                      └─────────────┘
```

All events, including agent events, are relayed through the core service, eliminating the need for multiple ZMQ connections and simplifying shutdown procedures.

## ZeroMQ Architecture

The audio service now follows a simplified star topology. 

- **Core Service**: Central coordinator that publishes all system events including agent events
- **Audio Service**: Subscribes to a single channel (coordinator's PUB socket) for ALL events
- **Agent Events**: Relayed through core service rather than requiring separate connections


## Development and Testing

For development and manual testing, use the CLI tool:

```bash
uv run -m experimance_audio.cli
```

This provides an interactive interface for sending OSC commands to SuperCollider.

### Testing OSC Communication

To test OSC communication between Python and SuperCollider:

```bash
# Run the test script with help to see all options
./scripts/test_osc.sh help

# Send single test messages (manual mode)
./scripts/test_osc.sh manual --message /spacetime --args forest ancient
./scripts/test_osc.sh manual --message /listening --args true

# Run integrated testing with SuperCollider
./scripts/test_osc.sh integrated

# Run automated unit tests
./scripts/test_osc.sh unittest
```

This testing framework verifies that:
1. OSC messages are properly formatted and sent from Python
2. Messages can be received by OSC clients (verified with oscdump)
3. SuperCollider can receive and respond to the messages
4. Resources are properly cleaned up when processes terminate

See the [test documentation](tests/README.md) for more details.
