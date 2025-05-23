# Audio Service Implementation TODO

This document outlines the steps needed to implement the Experimance Audio Service according to the technical design documents.

## Implementation Phases

### Phase 1: Setup and Basic Communication
- [ ] Define package structure following project conventions
- [ ] Create `pyproject.toml` with proper dependencies:
  - [ ] python-osc (for SuperCollider communication)
  - [ ] pyzmq (for integration with Experimance system)
  - [ ] pydantic (for data validation)
- [ ] Setup directories for audio files, config and SC scripts
- [ ] Implement basic ZMQ subscriber to listen for system events
  - [ ] Connect to `events` channel (`tcp://localhost:5555`)
  - [ ] Listen for `EraChanged`, `RenderRequest`, and `Idle` topics
  - [ ] Listen for `agent_ctrl` channel for audience interaction cues

### Phase 2: SuperCollider OSC Bridge
- [ ] Implement Python OSC client for sending commands to SuperCollider
- [ ] Add command handling for all OSC patterns:
  - [ ] `/spacetime <biome> <era>` - Set main context
  - [ ] `/include <tag>` - Add sound tag to active set
  - [ ] `/exclude <tag>` - Remove tag from active set
  - [ ] `/listening <start|stop>` - Trigger UI/interaction SFX
  - [ ] `/speaking <start|stop>` - Trigger UI/interaction SFX
  - [ ] `/transition <start|stop>` - Scene/era/biome transition cue
  - [ ] `/reload` - Reload configs in SuperCollider
- [ ] Create config loaders for JSON configuration files
- [ ] Add state machine for context management (current era, biome, active tags)

### Phase 3: SuperCollider Script Development
- [ ] Implement basic SuperCollider script that accepts OSC commands
- [ ] Add environment audio layer management
  - [ ] JSON config loading and parsing
  - [ ] Tag-based filtering mechanism
  - [ ] Crossfading between audio environments
- [ ] Add music loop system
  - [ ] Era-based loop loading
  - [ ] Slot-ordered crossfades between era transitions
- [ ] Add triggered sound effect handlers
  - [ ] Ducking system for listening/speaking modes
- [ ] Implement hot-reload functionality

### Phase 4: Service Integration
- [ ] Implement full ZMQ service based on `experimance_common.service.BaseZmqService`
- [ ] Add proper service lifecycle (start, stop, run)
- [ ] Implement signal handling for clean shutdown
- [ ] Add configuration loading from TOML
- [ ] Add logging with configurable levels

### Phase 5: Testing and Reliability
- [ ] Create unit tests for Python components
- [ ] Create manual test script for OSC commands
- [ ] Add error handling and fallback mechanisms
- [ ] Test with sample audio content
- [ ] Add reconnection logic for both ZMQ and OSC
- [ ] Implement recording/playback of OSC sessions for testing
- [ ] Add graceful degradation for missing audio files

### Phase 6: Documentation and Optimization
- [ ] Document API and configuration options
- [ ] Add inline comments and docstrings
- [ ] Optimize audio loading and memory usage
- [ ] Profile and optimize performance
- [ ] Document testing procedures

## Project Structure

```
experimance_audio/
├── __init__.py
├── audio_service.py         # Main service class that connects to ZMQ
├── osc_bridge.py            # Handles OSC communication with SuperCollider  
├── config_loader.py         # Loads and validates JSON config files
├── state_machine.py         # Manages audio state context
├── sc_scripts/
│    └── experimance_audio.scd   # SuperCollider script
├── audio/                   # Sample audio files for testing
│    ├── music/
│    ├── sfx/
│    └── environments/
├── config/
│    ├── layers.json         # Environmental audio configuration
│    ├── triggers.json       # Sound effect triggers
│    └── music_loops.json    # Era-based music loop definitions
└── logs/
     └── osc_recordings/     # Recorded OSC sessions for testing/playback
```

## Important Considerations

1. **Cross-platform Compatibility**: Ensure paths work on both Linux and macOS for development
2. **Fallback Mechanisms**: Have fallbacks for when audio files are missing
3. **Hot-reload Support**: Allow updating configurations without restarting
4. **Resource Management**: Handle large audio files efficiently
5. **Latency**: Minimize delay between receiving ZMQ events and playing audio
6. **Debugging**: Add verbose logging modes for troubleshooting
7. **Clean Shutdown**: Ensure proper cleanup of ZMQ sockets and SC processes
