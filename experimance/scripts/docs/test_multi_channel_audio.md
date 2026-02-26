# test_multi_channel_audio.py

Interactive calibration tool for multi-channel audio delay. When using multiple speaker outputs (e.g. laptop speakers + Bluetooth speaker through a PipeWire virtual sink), this tool helps you find the per-channel delays that make echo cancellation work correctly on a conference mic/speaker combo.

See `scripts/test_multi_channel_audio.py`.  
Related: [`pipewire_multi_sink.md`](pipewire_multi_sink.md)

## When to Use This

The agent voice is routed through multiple speakers to get more volume. But conference-style USB devices (e.g. Yealink) do echo cancellation on what they hear — so timing differences between speaker outputs cause the echo cancellation to fail and the agent hears itself. Adding per-channel delays aligns the audio so the echo canceller works.

## Quick Start

```bash
# List available audio devices
uv run python scripts/test_multi_channel_audio.py --list-devices

# Basic calibration session
uv run python scripts/test_multi_channel_audio.py

# Load voice audio for more realistic testing
uv run python scripts/test_multi_channel_audio.py --file media/audio/cartesia_sophie.wav

# Start from an existing agent config
uv run python scripts/test_multi_channel_audio.py --config projects/fire/agent.toml
```

## Setup: Creating the Multi-Channel Device First

### Linux (PipeWire)

```bash
uv run python scripts/pipewire_multi_sink.py
```

This creates a `Virtual-Multi` sink routing 4 channels to different speakers.

### macOS (Audio MIDI Setup)

1. Open **Audio MIDI Setup** (Applications → Utilities → Audio MIDI Setup)
2. Click **+** → **Create Multi-Output Device**, name it `Virtual-Multi`
3. Check your output devices (e.g. Built-in Output + AirPods Pro)
4. Set one as **Master Device** (the fastest/most reliable one)
5. Enable **Drift Correction** on all non-master devices
6. Right-click the device → **Use this device for sound output**

### Agent Config

In `projects/<project>/agent.toml`:
```toml
multi_channel_output = true
output_channels = 4
audio_output_device_name = "Virtual-Multi"

# After calibration, add these:
[backend_config.pipecat.channel_delays]
0 = 0.120    # Laptop speakers (FL)
1 = 0.120    # Laptop speakers (FR)
2 = 0.000    # Bluetooth speaker (reference)
3 = 0.000    # Bluetooth speaker
```

## Interactive Commands

| Command | Description |
|---|---|
| `p <bpm> <duration>` | Play click track (default: `p 120 5`) |
| `t <channel>` | Play test tone on a single channel (`t 0` for channel 0) |
| `c` | Play click track on all channels simultaneously |
| `d <channel> <ms>` | Set delay in milliseconds (`d 0 120`) |
| `v <channel> <vol>` | Set volume 0.0–1.0 (`v 0 0.8`) |
| `s` | Show current delay and volume settings |
| `voice <filepath>` | Load a voice audio file for testing |
| `play voice` | Play loaded voice audio through all channels |
| `i [channels]` | Interactive live-adjustment mode (`i 0,1`) |
| `w <filename>` | Write current settings to a TOML file |
| `q` | Quit |

## Calibration Workflow

1. **Play all channels** — `c` — listen for echo or chorus effect
2. **Identify the problem speakers** — usually laptop speakers lag behind Bluetooth
3. **Set reference** — fastest speakers (usually Bluetooth) at 0ms
4. **Add delays** — start with 100–150ms on laptop speakers: `d 0 120`, `d 1 120`
5. **Fine-tune** — load a voice file (`voice media/audio/cartesia_sophie.wav`) and adjust until echo disappears
6. **Tweak interactively** — `i 0,1` for real-time slider-style adjustment
7. **Save** — `w fire_delays.toml`
8. **Update agent config** — copy values to `projects/<project>/agent.toml`

## Typical Delay Values

| Device type | Typical delay |
|---|---|
| Laptop speakers | 100–150ms |
| USB speakers | 50–100ms |
| Bluetooth speakers | 0–50ms (use as reference) |
| Conference devices | 20–80ms |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Still hearing echo | Increase delays on the problematic speaker |
| Delays seem too high | Check you're using the right reference speaker |
| No audio output | Verify multi-channel device with `--list-devices` |
| Choppy audio | Try lower BPM or shorter duration |

## Linux Audio Troubleshooting

### ALSA permission issues

```bash
sudo usermod -a -G audio $USER
# Log out and back in for the group to take effect
```

### PipeWire status and sinks

```bash
systemctl --user status pipewire pipewire-pulse
pactl list short sinks
pactl list short sources
pw-dump > /tmp/pipewire_dump.json   # inspect nodes/ports/links
```

### PipeWire playback/record tests

```bash
pw-play some_audio_file.wav
pw-record test_capture.wav
```

### USB device not found

```bash
sudo alsa force-reload
lsusb | grep -i audio
dmesg | tail -n 50
```

## Requirements

- Multi-channel audio device (created with `pipewire_multi_sink.py`)
- `numpy`, `scipy`, `pyaudio`, `toml` (installed with agent service dependencies)
- Conference microphone with echo cancellation for agent usage
- Optional: `--auto-calibrate` requires a connected microphone

## CLI Flags

| Flag | Description |
|---|---|
| `--list-devices` | Print all PyAudio devices and exit |
| `--file PATH` | Load a WAV file as voice audio on startup |
| `--config PATH` | Load initial delays/settings from agent TOML |
| `--output-device NAME` | Specify output device name or index |
| `--input-device NAME` | Specify input device name or index |
| `--auto-calibrate` | Experimental: auto-detect delays via microphone recording |
