# Multi-Channel Audio Quick Reference

## Fire Agent Configuration

### Enable Multi-Channel Output

Add to `projects/fire/agent.toml`:

```toml
# Multi-channel audio output
multi_channel_output = true
output_channels = 4
audio_output_device_name = "pipewire"

# Channel delays (seconds) - laptop speakers need delay to sync with Bluetooth
[backend_config.pipecat.channel_delays]
0 = 0.120    # Laptop left
1 = 0.120    # Laptop right  
2 = 0.000    # Bluetooth left
3 = 0.000    # Bluetooth right

# Channel volumes (0.0-1.0)
[backend_config.pipecat.channel_volumes]
0 = 1.0      # Laptop left
1 = 1.0      # Laptop right
2 = 0.8      # Bluetooth left
3 = 0.8      # Bluetooth right
```

## Calibration Workflow

1. **Test current setup**:
   ```bash
   PROJECT_ENV=fire uv run -m fire_agent
   ```

2. **If echo/chorus detected, calibrate delays**:
   ```bash
   uv run python scripts/test_multi_channel_audio.py --file media/audio/cartesia_sophie.wav
   > d 0 120    # Set laptop speakers to 120ms delay
   > d 1 120
   > i 0,1      # Interactive mode for fine-tuning
   ```

3. **Interactive controls**:
   - `,` / `.` - Adjust ±1ms
   - `<` / `>` - Adjust ±5ms  
   - `v` - Toggle voice/clicks
   - `ESC` - Exit

4. **Update config with calibrated values**

5. **Test with Fire agent again**

## Typical Delay Values

- **Laptop speakers**: 100-140ms delay (to compensate for faster response)
- **Bluetooth speakers**: 0ms delay (reference, slower inherent latency)
- **USB speakers**: 20-50ms delay  
- **Wired speakers**: 0-10ms delay

## Troubleshooting

### No Audio
- Check PipeWire Virtual-Multi sink: `pw-link -l | grep Virtual-Multi`
- Verify device recognition: Test with `--list-devices`

### Still Echo
- Use interactive calibration tool with voice audio
- Fine-tune in 1ms increments
- Consider room acoustics

### Performance Issues  
- Reduce delay buffer size if needed
- Check system audio latency
