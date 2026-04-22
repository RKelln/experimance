# Experimance Audio Service

## Overview

### What this service does

The Experimance Audio Service integrates SuperCollider with the installation to provide:

- Environmental audio layers based on the current biome and era
- Sound effects for transitions and interactions
- Background music tailored to each era
- 6-channel surround sound support via jackdbus

### Environment assumptions

- OS: Linux (production) with JACK available
- Python: 3.11+
- SuperCollider: 3.12+ (with sc3-plugins)
- Audio hardware: USB audio interface for 5.1 output
- Required services: core service publishing events on the unified ZMQ channel

## Quick start

```bash
# Set the active project (do this once)
uv run set-project experimance

# Start the service
uv run -m experimance_audio
```

For local development:

```bash
./scripts/dev audio
```

## Setup

```bash
# From the repo root
cd services/audio
uv add -e .
```

If your user needs audio device access:

```bash
sudo usermod -a -G audio $USER
```

## Configuration

Configuration is loaded from the project TOML file and JSON audio configs.

- Project config: `projects/<project>/audio.toml`
- Audio configs: `services/audio/config/*.json`

See `services/audio/docs/configuration.md` for configuration details, schema notes, and file locations.

## Usage

```bash
# With a specific config file
uv run -m experimance_audio --config projects/experimance/audio.toml

# Override config fields via CLI (auto-generated from config models)
uv run -m experimance_audio --supercollider-script-path "services/audio/sc_scripts/experimance_audio.scd"
```

Interactive CLI for manual testing:

```bash
uv run -m experimance_audio.cli
```

## Testing

```bash
# Test OSC messaging (see options)
./services/audio/scripts/test_osc.sh help

# Manual OSC test
./services/audio/scripts/test_osc.sh manual --message /spacetime --args forest ancient

# Explicit send-only mode (same as default)
./services/audio/scripts/test_osc.sh manual --send-only --message /spacetime --args forest ancient

# Optional: verify reception locally with oscdump
./services/audio/scripts/test_osc.sh manual --oscdump --message /spacetime --args forest ancient

# Integrated OSC test (SuperCollider + test messages)
./services/audio/scripts/test_osc.sh integrated

# Override the auto-detected config when needed
./services/audio/scripts/test_osc.sh manual --config projects/experimance/nochat_demo/audio.toml --message /reload

# Unit tests
./services/audio/scripts/test_osc.sh unittest
```

See `services/audio/docs/testing.md` for test details and troubleshooting tips.

Recommended troubleshooting order:

1. Start audio service with the intended config.
2. Send manual OSC messages via `test_osc.sh manual` or `experimance_audio.cli`.
3. Use integrated test only when validating standalone SuperCollider startup.

## Troubleshooting

- If SuperCollider fails to start, verify `sclang` is in PATH and `services/audio/sc_scripts/experimance_audio.scd` exists.
- OSC test tools now read the active audio config by default, including the configured OSC send port. If OSC messages do not arrive, confirm the active project/config is the one you expect and that SuperCollider is listening on that configured port (see `services/audio/docs/supercollider.md`).
- If JACK does not start, check `jack_control status` and audio device access; see `services/audio/docs/surround_sound.md`.
- If integrated testing boots SC on the wrong ALSA device (`hw:0`) and crashes, use service-managed startup with explicit `device` in project TOML (for example `hw:1,0` on this machine).
- `WebEngineContext` warnings in headless mode are usually non-fatal; focus on `Cannot open PCM device` / JACK errors for root cause.

## Integrations

- **Core service** publishes `SpaceTimeUpdate`, `PresenceStatus`, `SpeechDetected`, and `ChangeMap` events over ZMQ.
- **SuperCollider** receives OSC commands (default port 5570) and plays audio based on JSON config files.

## Additional Docs

- `services/audio/docs/index.md` - Index of audio service documentation.
- `services/audio/docs/architecture.md` - Service architecture, ZMQ flow, and OSC responsibilities.
- `services/audio/docs/configuration.md` - Config files, schema, and config loading behavior.
- `services/audio/docs/supercollider.md` - SuperCollider scripts, OSC commands, and GUI workflow.
- `services/audio/docs/musician_guide.md` - GUI instructions and creative testing tips.
- `services/audio/docs/surround_sound.md` - Multi-channel routing and JACK/jackdbus setup.
- `services/audio/docs/testing.md` - OSC tests, scripts, and verification steps.
- `services/audio/docs/technical_design.md` - Technical design summary and OSC patterns.
- `services/audio/docs/credits.md` - Audio asset credits and licensing notes.
