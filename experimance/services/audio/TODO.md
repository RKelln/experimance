# Audio Service Implementation TODO

This document outlines the steps needed to implement the Experimance Audio Service according to the technical design documents.

## Implementation Phases

### Phase 1: Setup and Basic Communication
- [X] Define package structure following project conventions
- [X] Create `pyproject.toml` with proper dependencies:
  - [X] python-osc (for SuperCollider communication)
  - [X] pyzmq (for integration with Experimance system)
  - [X] pydantic (for data validation)
- [X] Setup directories for audio files, config and SC scripts
- [X] Implement basic ZMQ subscriber to listen for system events
  - [X] Connect to `events` channel (`tcp://localhost:5555`)
  - [X] Listen for `EraChanged`, `RenderRequest`, and `Idle` topics
  - [X] Listen for `agent_ctrl` channel for audience interaction cues

### Phase 2: SuperCollider OSC Bridge
- [X] Implement Python OSC client for sending commands to SuperCollider
- [X] Add command handling for all OSC patterns:
  - [X] `/spacetime <biome> <era>` - Set main audio context
  - [X] `/include <tag>` - Add sound tag to active set
  - [X] `/exclude <tag>` - Remove tag from active set
  - [X] `/listening <start|stop>` - Trigger UI interaction SFX
  - [X] `/speaking <start|stop>` - Trigger UI interaction SFX
  - [X] `/transition <start|stop>` - Trigger transition cues
  - [X] `/reload` - Reload audio configs
- [X] Add SuperCollider lifecycle management
  - [X] Start SuperCollider script during service initialization
  - [X] Command-line arguments to control SC startup behavior
  - [X] Graceful shutdown of SuperCollider when service stops

### Phase 3: SuperCollider Script Development
- [X] Implement basic SuperCollider script that accepts OSC commands
- [X] Add environment audio layer management
  - [X] JSON config loading and parsing
  - [X] Tag-based filtering mechanism
  - [X] Crossfading between audio environments
- [X] Add music loop system
  - [X] Era-based loop loading
  - [X] Slot-ordered crossfades between era transitions
- [X] Add triggered sound effect handlers
  - [X] Ducking system for listening/speaking modes
- [X] Implement hot-reload functionality

### Phase 4: Service Integration
- [X] Implement full ZMQ service based on `experimance_common.service.BaseZmqService`
- [X] Add proper service lifecycle (start, stop, run)
- [X] Implement signal handling for clean shutdown
- [ ] Add configuration loading from TOML
- [X] Add logging with configurable levels

### Phase 5: Testing and Reliability
- [ ] Create unit tests for Python components
- [X] Create manual test script for OSC commands
- [X] Add error handling and fallback mechanisms
- [ ] Test with sample audio content
- [ ] Add reconnection logic for both ZMQ and OSC
- [ ] Implement recording/playback of OSC sessions for testing
- [X] Add graceful degradation for missing audio files

### Phase 6: Documentation and Optimization
- [X] Document API and configuration options
- [X] Add inline comments and docstrings
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
