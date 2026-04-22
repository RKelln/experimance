# Audio Configuration

## Overview

The audio service reads configuration from a project TOML file and a set of JSON
audio content files. The TOML file controls service behavior, while JSON files
define playable audio content.

## Environment assumptions

- OS: Linux (production)
- Configs live under `projects/<project>/audio.toml`
- Audio JSON files live under `services/audio/config/`

## Configuration files

### Project TOML

- `projects/<project>/audio.toml`
- Example: `projects/experimance/audio.toml`

Key sections:

- `[supercollider]` runtime control, device selection, and JACK settings
- `[audio]` default volume levels and config locations

### Audio JSON

Located in `services/audio/config/`:

- `layers.json` - environmental audio layers
- `triggers.json` - one-shot sound effects
- `music_loops.json` - era-based loop sets

## Audio schema

`services/audio/config/audio_schema.json` documents expected fields, ranges,
OSC command patterns, and naming recommendations.

## SuperCollider runtime notes

- `supercollider.device` can be `hw:X,Y`, `plughw:X,Y`, or a partial name.
- The service resolves device names to hardware addresses before generating the
  temporary `.scd` launch script.
- If `auto_start_jack = false`, SuperCollider may still try JACK depending on
  script/runtime defaults, so explicit device pinning is recommended when
  debugging startup issues.
- `output_channels` and `input_channels` are injected into the runtime script.
- If JACK is already running with the correct parameters, the audio service
  should reuse it rather than stopping and reconfiguring it.

## General project configuration

`data/experimance_config.json` defines shared lists used by audio configuration
(biomes, eras, common tags, and trigger types).

## Schema notes

### `layers.json`

Required fields:

- `path` (string) - relative to `media/audio`
- `prompt` (string)
- `tags` (array of strings)
- `interval` (string): `loop`, `frequent`, `occasional`, or `rare`

Optional fields:

- `volume` (number, 0.0 to 1.0)
- `crossfade_time` (number, seconds, for `loop` only)
- `requires`, `requires_any`, `requires_none` (arrays of strings)
- `weight` (number)

Eligibility logic:

1. At least one tag in `tags` is active.
2. All `requires` tags are present (if provided).
3. At least one `requires_any` tag is present (if provided).
4. No `requires_none` tag is present (if provided).

### `triggers.json`

- `trigger` (string) - name of event
- `path` (string) - relative to `media/audio`
- `volume` (number, optional)

### `music_loops.json`

```
{
  "era_loops": {
    "era_name": [
      { "path": "...", "prompt": "...", "volume": 0.7 }
    ]
  }
}
```

## Missing audio files

- Missing music loops generate placeholder audio in SuperCollider.
- Missing environment layers or triggers log warnings and skip playback.

## Loading behavior

The audio service loads JSON configs at startup and on `/reload`.

Files are loaded by `AudioConfigLoader`:

- `services/audio/src/experimance_audio/config_loader.py`

## Validation

- Run `uv run scripts/validate_schemas.py` from the repo root to check schema compatibility.

## Device selection quick reference

List ALSA devices:

```bash
aplay -l
arecord -l
```

If startup fails with `Cannot open PCM device` or JACK errors, set an explicit
`device` in your project TOML and re-test using manual OSC commands against the
running audio service.

## Files touched

- `services/audio/config/*.json`
- `services/audio/config/audio_schema.json`
- `projects/<project>/audio.toml`
- `services/audio/src/experimance_audio/config_loader.py`
