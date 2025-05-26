# Experimance Audio Configuration

This directory contains the configuration files for the Experimance audio system. These JSON files define the audio layers, trigger sounds, and music loops used by the SuperCollider script.

## Configuration Files

### 1. `layers.json`

Contains environmental audio layers that play continuously based on the current biome, era, or other active tags.

Format:
```json
[
  {
    "path": "relative/path/to/audio.wav",
    "tags": ["tag1", "tag2", ...],
    "volume": 0.8
  },
  ...
]
```

- `path`: Relative path to the audio file from the audio directory
- `tags`: List of tags that will trigger this layer to play
- `volume`: Playback volume (0.0 to 1.0)

### 2. `triggers.json`

Contains one-shot sound effects that play in response to specific events.

Format:
```json
[
  {
    "trigger": "event_name",
    "path": "relative/path/to/audio.wav",
    "volume": 1.0
  },
  ...
]
```

- `trigger`: Event name that triggers this sound (e.g., "transition", "listening", "speaking")
- `path`: Relative path to the audio file from the audio directory
- `volume`: Playback volume (0.0 to 1.0)

### 3. `music_loops.json`

Contains era-specific music loops that provide background music for each era.

Format:
```json
{
  "era_loops": {
    "era_name": [
      {
        "path": "relative/path/to/audio.wav",
        "prompt": "Description of this audio layer",
        "volume": 0.7
      },
      ...
    ],
    ...
  }
}
```

- `era_name`: Name of the era (e.g., "wilderness", "pre_industrial")
- `path`: Relative path to the audio file from the audio directory
- `prompt`: Description of the audio for reference/documentation
- `volume`: Playback volume (0.0 to 1.0)

## Missing Audio Files

If an audio file specified in these configurations is not found, the system will automatically generate placeholder sounds:

- For music loops: Era-specific procedurally generated music
- For environmental layers and triggers: A warning will be printed

## Updating Configurations

Changes to these configuration files can be loaded at runtime by sending the `/reload` OSC message to SuperCollider:

```bash
uv run -m experimance_audio.cli reload
```
