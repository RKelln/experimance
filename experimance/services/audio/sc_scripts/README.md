# Experimance Audio SuperCollider Scripts

This folder contains the SuperCollider (version 3.13) scripts that power the audio system for the Experimance installation.

## Main Script: `experimance_audio.scd`

The main script (`experimance_audio.scd`) handles all audio playback, including environmental sounds, music, and transition effects. It receives OSC messages from the Python service and manages the audio output accordingly.

## Placeholder Music Generation System

The script includes a placeholder music generation system that automatically creates appropriate background music when audio files are missing. This ensures continuous audio even during development or when specific audio assets haven't been created yet.

## Usage

This script is automatically loaded by the Experimance Audio Service when it starts. It can also be opened directly in SuperCollider for development and testing.

The script responds to OSC messages sent to port 57120 (SuperCollider's default port):

- `/spacetime <biome> <era>`: Set the current biome and era
- `/include <tag>`: Include an audio tag
- `/exclude <tag>`: Exclude an audio tag
- `/listening <start|stop>`: Signal when the agent is listening
- `/speaking <start|stop>`: Signal when the agent is speaking
- `/transition <start|stop>`: Signal a transition effect
- `/volume/master <value>`: Set master volume (0.0 to 1.0)
- `/volume/environment <value>`: Set environmental sounds volume (0.0 to 1.0)
- `/volume/music <value>`: Set music volume (0.0 to 1.0)
- `/volume/sfx <value>`: Set sound effects volume (0.0 to 1.0)
- `/reload`: Reload audio configurations
- `/quit`: Shutdown gracefully
  
Also for debugging it supports:

- `synth_info`: Display the current active synths and their information
- `test_placeholders`: Test all placeholder sounds
- `test_placeholder <era> <slot>`: Test specific placeholder 

## Volume Control System

The audio system features a layered volume control system with four independent volume controls:

1. **Master Volume**: Controls the overall output level of the entire audio system
2. **Environment Volume**: Controls the volume of environmental ambient sounds (birds, water, wind, etc.)
3. **Music Volume**: Controls the volume of music loops and placeholder music
4. **SFX Volume**: Controls the volume of sound effects such as UI sounds and triggers

Each volume can be set independently via OSC commands, allowing fine-grained control over the audio mix. The actual output volume for each audio component is calculated as its specific volume multiplied by the master volume.

## Development Notes

- All synthesizer definitions are at the end of the script
- Audio configurations are loaded from JSON files in the `../config` directory
- The placeholder system creates music programmatically when audio files are missing
- The system automatically ducks environmental sound during agent interactions

To test the placeholder system, simply reference non-existent audio files in the configuration, and the system will generate appropriate music for each era.

