# SuperCollider Integration

## Overview

SuperCollider is the audio engine for the service. The audio service sends OSC
commands to SuperCollider, which handles playback, mixing, and routing.

## Environment assumptions

- SuperCollider 3.12+ installed (with sc3-plugins)
- Default OSC port is `5570`
- Scripts are under `services/audio/sc_scripts/`

## Main scripts

- `services/audio/sc_scripts/experimance_audio.scd` - production playback script
- `services/audio/sc_scripts/experimance_audio_gui.scd` - GUI for manual testing
- `services/audio/sc_scripts/surround_sound.scd` - surround test and diagnostics

## OSC commands

SuperCollider listens on port `5570` by default (`~oscRecvPort` in the script).

Commands:

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
- `/synth_info` (debug)
- `/quit`

## GUI workflow

See `services/audio/docs/musician_guide.md` for the GUI workflow, tips, and
troubleshooting.

## Files touched

- `services/audio/sc_scripts/experimance_audio.scd`
- `services/audio/sc_scripts/experimance_audio_gui.scd`
- `services/audio/src/experimance_audio/osc_bridge.py`
