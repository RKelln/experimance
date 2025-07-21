#!/usr/bin/env python3
"""
Audio Device Lister for Experimance Agent Service

This script lists all available PyAudio input and output devices to help with
configuring the pipecat backend audio device selection.

Usage:
    uv run python list_audio_devices.py

This will list all audio devices with their indices, which can be used in the
audio_input_device_index and audio_output_device_index configuration options.

Other sound debugging tools:
    - `pactl list short sources` for PulseAudio input devices
    - `pactl list short sinks` for PulseAudio output devices
    - `jack_lsp` for JACK audio connections
    - `arecord -l` for ALSA input devices
    - `aplay -l` for ALSA output devices
    - `cat /proc/asound/cards` for a list of ALSA sound cards
    - `cat /proc/asound/card4/stream0` for detailed ALSA device info
    - `jack_control dp` for JACK device info
    - `jack_control status` for JACK server status
    - `jack_control list` for all JACK connections

For example if jackdbus is running then you can set the device index and ALSA use:
```bash
jack_control stop
jack_control ds alsa
jack_control dps device hw:4,0
jack_control dps outchannels 6
jack_control start
```
"""

import pyaudio


def list_audio_devices():
    """List all available audio devices with their indices and capabilities."""
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    print("Available Audio Devices:")
    print("=" * 80)
    print(f"{'Index':<5} {'Name':<40} {'Inputs':<7} {'Outputs':<8} {'Default Rate':<12}")
    print("-" * 80)
    
    try:
        # Get default devices
        try:
            default_input = p.get_default_input_device_info()['index']
        except:
            default_input = None
            
        try:
            default_output = p.get_default_output_device_info()['index']
        except:
            default_output = None
        
        # List all devices
        for i in range(p.get_device_count()):
            try:
                device_info : pyaudio._PaDeviceInfo = p.get_device_info_by_index(i)
                
                name = device_info['name'][:39]  # Truncate long names
                max_inputs = device_info['maxInputChannels']
                max_outputs = device_info['maxOutputChannels']
                default_rate = int(device_info['defaultSampleRate'])
                
                # Add indicators for default devices
                indicators = ""
                if i == default_input:
                    indicators += " [DEFAULT INPUT]"
                if i == default_output:
                    indicators += " [DEFAULT OUTPUT]"
                
                print(f"{i:<5} {name:<40} {max_inputs:<7} {max_outputs:<8} {default_rate:<12}{indicators}")
                
            except Exception as e:
                print(f"{i:<5} Error reading device info: {e}")
                
    finally:
        p.terminate()
    
    print("-" * 80)
    print("\nNotes:")
    print("- Inputs: Number of input channels (microphone capability)")
    print("- Outputs: Number of output channels (speaker capability)")
    print("- Use device index numbers in your config.toml file")
    print("- Set audio_input_device_index for microphone device")
    print("- Set audio_output_device_index for speaker device")
    print("- Leave as null/None to use system defaults")
    
    if default_input is not None:
        print(f"\nDefault input device: {default_input}")
    if default_output is not None:
        print(f"Default output device: {default_output}")


if __name__ == "__main__":
    try:
        list_audio_devices()
    except ImportError:
        print("Error: PyAudio is not installed.")
        print("This should be available since pipecat requires it.")
        print("If you see this error, check your environment setup.")
    except Exception as e:
        print(f"Error listing audio devices: {e}")
