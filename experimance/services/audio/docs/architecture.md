# Audio Service Architecture

## Overview

The audio service subscribes to system events and translates them into OSC commands
for SuperCollider, which plays audio layers, music loops, and trigger sounds.

## What this service does

- Receives context updates (biome, era, tags) and interaction state
- Sends OSC commands to SuperCollider for playback and mixing
- Manages SuperCollider lifecycle (startup/shutdown) and JACK configuration
- Loads audio configuration from JSON files and applies defaults

## Environment assumptions

- OS: Linux (production)
- Python: 3.11+
- SuperCollider: 3.12+ with sc3-plugins
- JACK: jackdbus available for multi-channel routing
- Audio hardware: 5.1-capable USB device for surround output

## Event Flow

```
Core Service (ZMQ publish)
        |
        v
Audio Service (ZMQ subscribe) -> OscBridge -> SuperCollider (OSC)
```

## Subscribed events

The audio service subscribes to the unified events channel and responds to:

- `SpaceTimeUpdate` - biome/era changes and tag updates
- `PresenceStatus` - audience presence state
- `SpeechDetected` - agent/human speech for ducking
- `ChangeMap` - interaction visualization cues

## OSC responsibilities

The audio service sends OSC to SuperCollider on port `5570` by default.
SuperCollider responds by updating the audio state and playback.

Primary OSC message patterns:

- `/spacetime <biome> <era>`
- `/include <tag>`
- `/exclude <tag>`
- `/listening <start|stop>`
- `/speaking <start|stop>`
- `/transition <start|stop>`
- `/trigger <name>`
- `/volume/master <value>`
- `/volume/environment <value>`
- `/volume/music <value>`
- `/volume/sfx <value>`
- `/reload`

See `services/audio/docs/supercollider.md` for command details and examples.

## SuperCollider lifecycle

- The service auto-starts SuperCollider when `supercollider.auto_start = true`.
- The script path defaults to `services/audio/sc_scripts/experimance_audio.scd`.
- Logs are written via the audio service log path configuration.

## Files touched

- `services/audio/src/experimance_audio/audio_service.py`
- `services/audio/src/experimance_audio/osc_bridge.py`
- `services/audio/src/experimance_audio/config.py`
- `services/audio/sc_scripts/experimance_audio.scd`
- `services/audio/config/*.json`
