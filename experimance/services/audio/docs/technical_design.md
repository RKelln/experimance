# Audio System Technical Design

## Overview

The audio system provides responsive, layered soundscapes driven by the
installation context (biome, era, tags) and interaction cues.

## Environment assumptions

- Python 3.11+
- SuperCollider 3.12+
- Linux with JACK for 5.1 routing

## Architecture summary

- **Python service** subscribes to system events over ZMQ and sends OSC.
- **SuperCollider** loads configuration and performs audio playback/mixing.

See `services/audio/docs/architecture.md` for the runtime flow.

## Sound configuration model

### Environmental layers

Each layer includes:

- `path` - relative to `media/audio`
- `tags` - metadata and selectors
- `prompt` - human-readable description
- `volume` - optional

The layer becomes eligible when its tags intersect the active tag set and any
`requires`/`requires_any`/`requires_none` constraints are satisfied.

### Triggered effects

Each trigger includes:

- `trigger` - event name
- `path` - relative to `media/audio`
- `volume` - optional

### Music loops

Loops are defined per era with ordered slots. The slot order is consistent
across eras to preserve crossfade alignment.

```
{
  "era_loops": {
    "pre_industrial": [
      { "path": "audio/music/pre_industrial_loop1.wav", "prompt": "..." }
    ]
  }
}
```

## OSC command patterns

The audio service sends OSC on port `5570` by default. Key commands:

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

See `services/audio/docs/supercollider.md` for details.

## Workflow

- Sound designers edit JSON configs and drop audio assets in `media/audio`.
- SuperCollider can hot-reload configs via `/reload`.
- The audio service drives context changes from system events.

## Files touched

- `services/audio/config/*.json`
- `services/audio/src/experimance_audio/audio_service.py`
- `services/audio/src/experimance_audio/osc_bridge.py`
- `services/audio/sc_scripts/experimance_audio.scd`
