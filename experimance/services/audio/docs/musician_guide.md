# Musician Guide (SuperCollider GUI)

## Overview

The GUI provides a quick way to test and control the audio system without
manually sending OSC commands.

## Environment assumptions

- SuperCollider installed
- Audio service scripts available under `services/audio/sc_scripts/`
- OSC port `5570` reachable (default)

## Quick start

1. Open these scripts in SuperCollider:
   - `services/audio/sc_scripts/experimance_audio.scd`
   - `services/audio/sc_scripts/experimance_audio_gui.scd`
2. In each file: **Language > Evaluate File**.
3. Wait for the audio system to start, then use the GUI.

If startup fails the first time, re-evaluate the main script and try again.

## Interface sections

### Spacetime context

- Select biome and era
- Send `/spacetime` to update the audio context
- Quick presets for common combinations

### Tag management

- Add tags to include extra layers
- Exclude tags to remove layers

### State controls

- Listening, speaking, and transition states
- Sends `/listening`, `/speaking`, `/transition`

### Volume controls

- Master, environment, music, and SFX sliders
- Sends `/volume/*` commands

### Triggers

- Choose a trigger and fire one-shot SFX
- Sends `/trigger <name>`

### Quick tests

- Preset journeys for biome or era transitions
- Useful for auditioning transitions and mix behavior

## Creative usage tips

### Composition testing

1. Set Music Only volume preset.
2. Run an era journey to hear the evolution across eras.
3. Try multiple biomes with the same era.

### Soundscape design

1. Pick a biome and era.
2. Add contextual tags (for example: birds, water).
3. Toggle listening/speaking to check ducking behavior.
4. Adjust volume balance to match the scene.

### Transition testing

1. Use journey buttons for automated transitions.
2. Try manual spacetime changes while audio is playing.
3. Toggle listening/speaking during transitions to check ducking.

## Audio file structure

Audio files live under `media/audio/`:

- `media/audio/environment/`
- `media/audio/music/`
- `media/audio/sfx/`

Missing files will trigger placeholder audio for music and warnings for other
categories.

## Troubleshooting

### GUI does not appear

- Confirm SuperCollider is running
- Verify the script paths are correct
- Check the SuperCollider post window for errors

### OSC messages not working

- Confirm `experimance_audio.scd` is running and listening on port `5570`
- Ensure the GUI `NetAddr` points to `127.0.0.1:5570`

### No audio changes

- Verify master volume > 0
- Run `/reload` via the GUI
- Check for config load errors in the post window

## Advanced OSC usage

In the SuperCollider post window:

```supercollider
m = NetAddr("127.0.0.1", 5570);
m.sendMsg("/synth_info");
m.sendMsg("/spacetime", "forest", "wilderness");
m.sendMsg("/volume/music", 0.0);
m.sendMsg("/include", "stadium");
m.sendMsg("/exclude", "stadium");
```

## Files touched

- `services/audio/sc_scripts/experimance_audio.scd`
- `services/audio/sc_scripts/experimance_audio_gui.scd`
