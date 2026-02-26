# audio_recovery.py

Diagnoses and recovers from audio device issues, particularly with USB audio devices (Yealink conference mic, ICUSBAUDIO7D 5.1 card, ReSpeaker mic array).

See `scripts/audio_recovery.py`.

## Quick Start

```bash
# List all devices
uv run python scripts/audio_recovery.py list-devices

# Run full diagnosis
uv run python scripts/audio_recovery.py diagnose

# Test a specific device type
uv run python scripts/audio_recovery.py test-yealink
uv run python scripts/audio_recovery.py test-icusbaudio7d
uv run python scripts/audio_recovery.py test-respeaker
```

## Commands

| Command | Description |
|---|---|
| `list-devices` | List all PyAudio input/output devices with channel and rate info |
| `diagnose` | Run full audio system diagnosis |
| `test-yealink` | Test Yealink conference device |
| `test-icusbaudio7d` | Test ICUSBAUDIO7D 5.1 surround card (expects 6+ output channels) |
| `test-respeaker` | Test ReSpeaker XVF3800 mic array |
| `reset-yealink` | Comprehensive Yealink reset (stop services → USB reset → restart) |
| `reset-icusbaudio7d` | Comprehensive ICUSBAUDIO7D reset |
| `fix-respeaker` | Fix ReSpeaker when PipeWire has exclusive control |

## What a "Comprehensive Reset" Does

1. Stop PipeWire, PulseAudio, and JACK user services
2. USB reset of the target device (via sysfs)
3. Wait 3 seconds for device reinitialization
4. Restart PipeWire/JACK services
5. Wait 2 more seconds, then re-test the device

## Device Notes

### Yealink Conference Mic/Speaker
- Detected by name containing `yealink`
- Needs both input and output channels to function correctly
- Used for agent voice I/O with built-in echo cancellation

### ICUSBAUDIO7D (5.1 USB Audio Card)
- Detected by name containing `icusbaudio7d`
- Expects at minimum 6 output channels for 5.1 surround
- Fewer channels indicates the device is stuck in a limited mode
- Manual recovery if script fails: unplug/replug, try a USB 3.0 port, or reboot

### ReSpeaker XVF3800 4-Mic Array
- Detected by name containing `respeaker`, `array`, or `seeed`
- Native parameters: 16kHz, 16-bit, 2-channel
- If not visible to PyAudio but visible in PipeWire, run `fix-respeaker`
- Adjust PipeWire volume if input is too quiet:
  ```bash
  pactl set-source-volume alsa_input.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_* 150%
  ```

## Troubleshooting

If reset commands have limited success:
1. Unplug and replug the USB device
2. Try a different USB port (preferably USB 3.0)
3. Disable USB power management:
   ```bash
   sudo sh -c 'echo on > /sys/bus/usb/devices/*/power/control'
   ```
4. Restart the entire system
5. Check for firmware updates

## Dependencies

- `experimance_common.audio_utils` (from `libs/common/src/`)
- PyAudio
- PipeWire / PulseAudio tools (`systemctl --user`, `pactl`)
- JACK tools (optional: `jack_control`)
