# pipewire_multi_sink.py

Creates a PipeWire virtual multi-channel sink that routes different stereo pairs to different physical audio devices. Used to set up multi-speaker output for the agent service.

See `scripts/pipewire_multi_sink.py`.  
Related: [`test_multi_channel_audio.md`](test_multi_channel_audio.md)

## Quick Start

```bash
# Interactive mode — lists available sinks and prompts for selection
uv run python scripts/pipewire_multi_sink.py

# Non-interactive — combine sinks 0 and 1 into "Virtual-4ch" and set as default
uv run python scripts/pipewire_multi_sink.py \
    --name "Virtual-4ch" --select "0,1" --non-interactive --make-default

# Remove an existing virtual sink
uv run python scripts/pipewire_multi_sink.py --destroy "Virtual-4ch"
uv run python scripts/pipewire_multi_sink.py --destroy "123"  # by node ID
```

## Options

| Flag | Description |
|---|---|
| `--name NAME` | Name for the virtual sink (default: `Virtual-Multi`) |
| `--select "0,1,2"` | Comma-separated indices of hardware sinks to combine |
| `--non-interactive` | Skip prompts; requires `--select` |
| `--make-default` | Set the created virtual sink as system default output |
| `--unlink-existing` | Remove existing port connections before linking (prevents audio doubling) |
| `--destroy NAME\|ID` | Remove an existing virtual sink by name or PipeWire node ID |

## Channel Mapping

Selected hardware sinks contribute their channels in order. The virtual sink is named with standard audio positions:

| Total channels | Layout |
|---|---|
| 1 | MONO |
| 2 | FL, FR |
| 4 | FL, FR, RL, RR |
| 6 | FL, FR, RL, RR, SL, SR |
| 8 | FL, FR, FC, LFE, RL, RR, SL, SR |

Example with two stereo sinks selected:
- Sink 0 (laptop speakers, 2ch) → virtual channels 0, 1 (FL, FR)
- Sink 1 (Bluetooth speaker, 2ch) → virtual channels 2, 3 (RL, RR)

## After Creating a Virtual Sink

Test and calibrate delays using `test_multi_channel_audio.py`:

```bash
uv run python scripts/test_multi_channel_audio.py --list-devices
uv run python scripts/test_multi_channel_audio.py
```

Then configure the agent service in `projects/<project>/agent.toml`:
```toml
multi_channel_output = true
output_channels = 4
audio_output_device_name = "Virtual-Multi"
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Audio doubles when virtual sink is created | `--unlink-existing` |
| Sink creation fails | Check for existing sink with same name first |
| No sinks found | Ensure PipeWire is running: `systemctl --user status pipewire` |
| Test script ignores virtual sink | Verify it's the default: `wpctl status` |

## Requirements

- PipeWire with CLI tools: `pw-dump`, `pw-cli`, `pw-link`, `wpctl`
- Python 3.7+ (standard library only — no pip deps)

## Technical Details

1. Uses `pw-dump` for JSON-based sink discovery (more reliable than `pw-cli ls`)
2. Creates null-audio-sink via `pw-cli` with proper channel position strings
3. Links virtual sink monitor ports to hardware sink playback ports via `pw-link`
4. Handles PipeWire's async node creation with retries
5. Parses `pw-link` output to manage existing connections before linking
