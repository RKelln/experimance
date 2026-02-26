# Surround Sound Setup

## Overview

The audio service uses a jackdbus-based setup for 5.1 output. The service
configures JACK at startup and routes environment and music to separate channel
pairs.

## Environment assumptions

- Linux with jackdbus (`jack_control`) available
- USB 5.1 audio interface
- SuperCollider configured for 6 output channels

## Channel layout (5.1)

- Channel 0: Left Front
- Channel 1: Right Front
- Channel 2: Center
- Channel 3: LFE
- Channel 4: Left Surround (Rear Left)
- Channel 5: Right Surround (Rear Right)

## Audio routing strategy

- Environmental audio: channels 0,1
- Music: channels 4,5
- SFX: channels 0,1 (default)
- Center/LFE: available for special effects

## JACK and device configuration

The service configures JACK based on `projects/<project>/audio.toml`:

```
[supercollider]
auto_start_jack = true
device = "ICUSBAUDIO7D"
output_channels = 6
jack_output_channels = 6
```

The audio service resolves `device` to a hardware address (e.g., `hw:4,0`)
and configures JACK with `jack_control`.

## SuperCollider channel routing

Use direct channel routing in SynthDefs:

```supercollider
SynthDef(\mysynth, {
    arg channel = 0, freq = 440, amp = 0.5;
    var sig = SinOsc.ar(freq) * amp;
    Out.ar(channel, sig);
}).add;
```

Avoid array outputs and boolean-multiplication routing patterns because they
produce runtime errors.

## When to use / When not to use surround

When to use:

- Installations with a 5.1-capable USB device and full speaker layout

When not to use:

- Laptop-only development or stereo-only systems
- Environments without JACK

## Files touched

- `projects/<project>/audio.toml`
- `services/audio/src/experimance_audio/audio_service.py`
- `services/audio/sc_scripts/experimance_audio.scd`
- `services/audio/sc_scripts/surround_sound.scd`
