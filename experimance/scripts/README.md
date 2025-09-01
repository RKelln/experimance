# Scripts Directory

This directory contains utility scripts for managing the Experimance project.

## Available Scripts

### `audio_cache_manager.py`

Comprehensive audio generation cache management tool for inspection, cleanup, and maintenance.

**Usage:**
```bash
# Show cache statistics
uv run python scripts/audio_cache_manager.py stats

# List cache items (newest first)
uv run python scripts/audio_cache_manager.py list --limit 10

# Find and remove duplicates
uv run python scripts/audio_cache_manager.py duplicates --remove-duplicates --confirm

# Clean old items (30+ days old)
uv run python scripts/audio_cache_manager.py clean --days 30

# Remove items matching pattern
uv run python scripts/audio_cache_manager.py remove-pattern "test.*" --confirm

# Clear entire cache
uv run python scripts/audio_cache_manager.py clear --confirm
```

**Features:**
- **Cache Statistics**: Total items, size, age range, CLAP similarity scores
- **Item Listing**: Sort by timestamp, quality, duration, or prompt
- **Duplicate Detection**: Find and remove redundant cache entries
- **Pattern Matching**: Remove items by regex patterns
- **Safety Features**: Dry-run mode, confirmation prompts, detailed reporting
- **Space Management**: Track and optimize storage usage

See [README_AUDIO_CACHE.md](README_AUDIO_CACHE.md) for detailed documentation.

### `list_cameras.py`

Comprehensive Reolink camera discovery tool using intelligent progressive detection.

**Usage:**
```bash
# Comprehensive discovery (default - smartest)
uv run python scripts/list_cameras.py

# Fast port scan only (shows all HTTPS devices)
uv run python scripts/list_cameras.py --fast

# Signature-based detection (precise Reolink identification)  
uv run python scripts/list_cameras.py --signature

# Test specific IP first (fastest for known cameras)
uv run python scripts/list_cameras.py --known-ip 192.168.2.229

# Test credentials on discovered cameras
uv run python scripts/list_cameras.py --test-creds admin your_password

# Verbose output for debugging
uv run python scripts/list_cameras.py --verbose
```

**Discovery Methods:**
- **Comprehensive (Default)**: Smart progressive discovery - tests known IP first, falls back to network scan, then signature verification
- **Fast**: Ultra-fast port scanning for HTTPS devices (seconds)
- **Signature**: Credential-free detection using Reolink API signatures
- **Known IP**: Direct testing of specific IP address

**Features:**
- **No Credential Broadcast**: Never sends passwords during network discovery
- **Progressive Fallback**: Intelligently escalates from fastest to most thorough methods
- **Security Conscious**: Uses port scanning and signature detection before credentials
- **Comprehensive Output**: Shows camera details and provides configuration guidance

### `test_reolink_camera.py`

Comprehensive Reolink camera testing and control tool for debugging and exploration.

**Usage:**
```bash
# Continuous presence monitoring
uv run python scripts/test_reolink_camera.py --host 192.168.2.229 --user admin --password your_password

# Check camera status and capabilities  
uv run python scripts/test_reolink_camera.py --host 192.168.2.229 --user admin --password your_password --status

# Explore all supported API commands
uv run python scripts/test_reolink_camera.py --host 192.168.2.229 --user admin --password your_password --explore

# Camera control (stealth mode)
uv run python scripts/test_reolink_camera.py --host 192.168.2.229 --user admin --password your_password --camera-off
uv run python scripts/test_reolink_camera.py --host 192.168.2.229 --user admin --password your_password --camera-on

# Individual feature control
uv run python scripts/test_reolink_camera.py --host 192.168.2.229 --user admin --password your_password --ir-lights off
uv run python scripts/test_reolink_camera.py --host 192.168.2.229 --user admin --password your_password --power-led off

# Debug mode with raw AI data
uv run python scripts/test_reolink_camera.py --host 192.168.2.229 --user admin --password your_password --debug
```

**Features:**
- **Presence Detection**: Real-time person, vehicle, and pet detection monitoring
- **Camera Control**: Stealth mode, IR lights, power LED control
- **API Exploration**: Test all camera capabilities and supported features  
- **Debug Mode**: Raw AI state data and comprehensive status information
- **Flexible Options**: Configurable polling intervals, different protocols
- **Production Integration**: Used as reference for ReolinkDetector development

**Camera Models Tested:**
- Reolink RLC-820A (firmware v3.1.0.2368_23062508)


### `test_multi_channel_audio.py`

In cases where you want to have multiple speakers with the same output but separate audio devices you
will need to sync the delay between the devices. This is especially important for the voice agent
when trying to make the agent's voice louder than what is possible with a single conference-style
mic+speaker combo device. When adding additional speaker outputs you need to be careful to set 
delays so that the echo cancellation on the device continues to work so the AI agent doesn't
hear itself.

On Linux with pipewire support you can use `pipewire_multi_sink.py` to create a multi-channel 
virtual audio output device that mimics the functionality of MacOS Multi audio device 
(without the automatic clock drift sync). 

**Quick Start:**
```bash
# Basic usage with default settings
uv run python scripts/test_multi_channel_audio.py

# List available audio devices
uv run python scripts/test_multi_channel_audio.py --list-devices

# Test with voice audio file for realistic calibration
uv run python scripts/test_multi_channel_audio.py --file media/audio/cartesia_sophie.wav

# Load existing agent config and calibrate
uv run python scripts/test_multi_channel_audio.py --config projects/fire/agent.toml
```

**Setup Steps:**

1. **Create Multi-Channel Audio Device** 
   1. **Linux with PipeWire:**
      ```bash
      uv run python scripts/pipewire_multi_sink.py
      ```
      This creates a Virtual-Multi sink that routes 4 channels to different speakers.
   
   2. **macOS:**
      1. **Open Audio MIDI Setup:**
         - Applications → Utilities → Audio MIDI Setup
         - Or press `Cmd+Space` and search for "Audio MIDI Setup"
      
      2. **Create Multi-Output Device:**
         - Click the "+" button in the bottom-left corner
         - Select "Create Multi-Output Device"
         - Name it "Virtual-Multi"
      
      3. **Configure Output Devices:**
         - Check the boxes for your output devices (e.g., "Built-in Output" for laptop speakers)
         - Check additional devices (e.g., "AirPods Pro" or external USB speakers)
         - **Important:** Set one device as "Master Device" (usually the fastest/most reliable one)
         - Adjust individual device volume if needed
      
      4. **Configure Drift Correction:**
         - Check "Drift Correction" for any devices that aren't the master
         - This helps keep devices synchronized (though not perfect)
      
      5. **Set as System Output:**
         - In Audio MIDI Setup, right-click your Virtual-Multi device
         - Select "Use this device for sound output" 
         - Or go to System Preferences → Sound → Output and select your Virtual-Multi device

2. **Configure Agent** - Enable multi-channel output in `projects/fire/agent.toml`:
   ```toml
   multi_channel_output = true
   output_channels = 4
   audio_output_device_name = "Virtual-Multi"  # Use the pipewire device
   
   # Uncomment and set delays (in seconds) after calibration:
   [backend_config.pipecat.channel_delays]
   0 = 0.120    # Laptop speakers (typical: 100-150ms)
   1 = 0.120    # Laptop speakers 
   2 = 0.000    # Bluetooth speakers (reference)
   3 = 0.000    # Bluetooth speakers
   ```

3. **Calibrate Delays** - Run the calibration tool and follow the interactive workflow.

**Interactive Commands:**

- **`p <bpm> <duration>`** - Play click track (default: `p 120 5` for 120 BPM, 5 seconds)
- **`t <channel>`** - Play test tone on specific channel (e.g., `t 0` for left laptop speaker)
- **`c`** - Play click track simultaneously on all channels (great for hearing timing differences)
- **`d <channel> <ms>`** - Set delay for channel in milliseconds (e.g., `d 0 120` sets 120ms delay)
- **`v <channel> <vol>`** - Set volume for channel (e.g., `v 0 0.8` for 80% volume)
- **`s`** - Show current delay and volume settings
- **`voice <filepath>`** - Load voice audio file (more realistic than click tracks)
- **`play voice`** - Play loaded voice audio through all channels
- **`i [channels]`** - Interactive mode with live adjustment (e.g., `i 0,1` for laptop speakers)
- **`w <filename>`** - Write current config to TOML file
- **`q`** - Quit

**Calibration Workflow:**

1. **Listen for Echo/Chorus:** Start with `c` to play clicks on all channels
2. **Identify Problem Speakers:** If you hear echo/chorus, it's usually from the laptop speakers
3. **Set Reference:** Use fastest speakers (usually Bluetooth) as reference (0ms delay)
4. **Add Delays:** Start with 100-150ms delays on laptop speakers: `d 0 120`, `d 1 120`
5. **Fine-Tune:** Use `voice` files and adjust delays until echo disappears
6. **Test Interactively:** Use `i 0,1` for real-time adjustment of laptop speakers
7. **Save Config:** Use `w fire_delays.toml` to save calibrated settings
8. **Update Agent:** Copy delay values to your `projects/fire/agent.toml`

**Advanced Features:**

- **Voice Audio Testing:** More realistic than click tracks for final calibration
- **Auto-Calibration:** `--auto-calibrate` with microphone feedback (experimental)
- **Device Selection:** `--output-device` and `--input-device` for specific hardware
- **Config Loading:** `--config` to start with existing agent settings
- **Real-Time Adjustment:** Interactive mode for live delay tweaking

**Typical Delay Values:**
- **Laptop speakers:** 100-150ms (due to audio processing latency)
- **USB speakers:** 50-100ms  
- **Bluetooth speakers:** 0-50ms (often used as reference)
- **Conference devices:** 20-80ms

**Troubleshooting:**
- **Still hearing echo?** Increase delays on problematic speakers
- **Delays too high?** Check if you're using the right reference speakers
- **No audio output?** Verify your multi-channel device is working with `--list-devices`
- **Choppy audio?** Try lower BPM or shorter duration tests

**🛠 Linux troubleshooting**

#### ALSA permission issues
If your user cannot access ALSA devices, add yourself to the `audio` group and re-login:

```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Log out and back in (or reboot) for the group to take effect
```

#### PulseAudio compatibility / quick checks
On many Linux setups PipeWire provides a PulseAudio compatibility layer. The following commands are useful for quick checks and restarts:

```bash
# Restart PulseAudio (if using PulseAudio)
pulseaudio --kill && pulseaudio --start

# List available sinks (outputs) and sources (inputs)
pactl list short sinks
pactl list short sources
```

#### PipeWire & pw tools (useful when running PipeWire)
If you use PipeWire (common on modern Linux distributions), these commands help inspect and debug the audio graph and devices:

```bash
# Check PipeWire service status (user service)
systemctl --user status pipewire pipewire-pulse

# Dump PipeWire state to JSON for inspection
pw-dump > /tmp/pipewire_dump.json

# Query PipeWire objects interactively
pw-cli info

# Play / record quick test using pipewire-utils (pw-play / pw-record / pw-cat)
# pw-play --help  # see available targets on your machine
pw-play some_audio_file.wav   # simple playback test
pw-record test_capture.wav    # simple recording test

# Use pactl (PulseAudio client) with PipeWire's PulseAudio compatibility
pactl list short sinks
pactl list short sources
```

Notes:
- Use `pw-dump` to inspect nodes, ports and links when troubleshooting routing or device availability.
- If your distribution exposes PipeWire tools as `pipewire-*` or `pw-*`, consult `--help` for exact flags; the examples above are intentionally minimal and widely available.

#### Device not found / USB device issues
If a device is not visible:

```bash
# Reload ALSA modules (Debian/Ubuntu)
sudo alsa force-reload

# Check USB devices for audio
lsusb | grep -i audio

# Check kernel messages for device attach/detach
dmesg | tail -n 50
```

**Requirements:**
- Multi-channel audio device (created with `pipewire_multi_sink.py`)
- PyAudio, NumPy, SciPy (installed with agent dependencies)
- Conference microphone with echo cancellation for agent usage


### `tune_detector.py`

Interactive tool for tuning audience detection parameters with live webcam feedback.

**Usage:**
```bash
# List available detector profiles
uv run python scripts/tune_detector.py --list-profiles

# Start tuning with default profile (indoor_office)
uv run python scripts/tune_detector.py

# Start with a specific profile
uv run python scripts/tune_detector.py --profile gallery_dim

# Use a different camera (if multiple cameras available)
uv run python scripts/tune_detector.py --camera 1

# Enable verbose logging
uv run python scripts/tune_detector.py --verbose
```

**Features:**
- Real-time visualization of HOG person detection (green boxes)
- Motion detection overlay (red contours)
- Interactive parameter adjustment via trackbars
- Live confidence and presence detection feedback
- Save tuned parameters to new detector profiles

**Controls:**
- **q/ESC**: Quit the application
- **s**: Save current settings to a new profile (`{profile_name}_tuned.toml`)
- **r**: Reset to original profile values
- **h**: Toggle HOG detection visualization
- **m**: Toggle motion detection visualization  
- **i**: Toggle parameter info display

**Parameters you can tune:**
- **Detection Scale Factor**: How much to downscale frames (smaller = faster, less accurate)
- **Min Person Height**: Minimum height in pixels for person detection
- **Motion Threshold**: Minimum contour area for motion detection
- **Motion Intensity**: Sensitivity threshold for motion detection
- **Stability**: Majority vote threshold for temporal smoothing
- **HOG Threshold**: Person detection confidence threshold
- **HOG Scale**: Pyramid scale factor for multi-scale detection
- **Win Stride**: HOG window stride (larger = faster, less accurate)
- **MOG2 Var Threshold**: Background subtraction variance threshold
- **MOG2 History**: Number of frames for background model
- **Person Base Confidence**: Base confidence score for person detection
- **Motion Weight**: Weight of motion confidence in final score

**Tips:**
- Start with a default profile that matches your environment
- Adjust parameters gradually and observe the effect on detection
- Test with different lighting conditions and audience scenarios
- Save multiple profiles for different environments
- Use the verbose flag to see detailed detection information

**Requirements:**
- Webcam connected to the system
- OpenCV installed (included in agent dependencies)
- No pipecat dependencies needed (loads only core detection modules)

### `create_new_project.py`
Interactive script to create a new project configuration.

**Usage:**
```bash
uv run python scripts/create_new_project.py
```

**What it does:**
- Prompts for a new project name
- Lets you select which services to include
- Copies service configuration files from existing projects or defaults
- Creates project-specific `constants.py`, `schemas.py`, and `.env` files
- Creates type stubs for the new project
- Updates the global type stubs to include the new project

**Features:**
- Interactive service selection with descriptions
- Choice to copy configs from existing projects or use service defaults
- Creates minimal template files when no source is available
- Automatically updates type stubs for proper IDE support
- Validates project names and prevents conflicts

**Example workflow:**
1. Run the script: `uv run python scripts/create_new_project.py`
2. Enter project name: `my_art_project`
3. Select services: `1,2,5` (core, display, image_server)
4. Choose config source: existing project or defaults
5. Confirm and let it create all the files
6. Customize the generated configs for your project

### `update_pyi_stubs.py`
Updates type stub files for dynamic module loading. Run this after adding schemas or constants to the shared or per project code.

**Usage:**
```bash
uv run python scripts/update_pyi_stubs.py          # Update files
uv run python scripts/update_pyi_stubs.py --diff   # Show diffs first
uv run python scripts/update_pyi_stubs.py --dry-run # Preview only
```

**What it does:**
- Regenerates `libs/common/src/experimance_common/schemas.pyi`
- Regenerates `libs/common/src/experimance_common/constants.pyi`
- Ensures proper type checking for dynamically loaded project-specific modules

## Adding New Scripts

When adding new utility scripts:

1. **Make them executable:** `chmod +x scripts/your_script.py`
2. **Add a shebang:** `#!/usr/bin/env python3`
3. **Add to this README:** Document what it does and how to use it
4. **Use `uv run`:** Scripts should be runnable with `uv run python scripts/your_script.py`
5. **Include error handling:** Use try/catch and provide helpful error messages

## Project Structure Integration

These scripts work with the multi-project architecture:

- **Project configs** go in `projects/{project_name}/{service}.toml`
- **Project schemas** extend base schemas in `projects/{project_name}/schemas.py`
- **Project constants** override base constants in `projects/{project_name}/constants.py`
- **Environment files** set PROJECT_ENV in `projects/{project_name}/.env`

The scripts help maintain this structure automatically.
