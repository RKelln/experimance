# Experimance Software Installation Guide

This guide will help you set up and run the Experimance installation software.

## Prerequisites

- Python 3.11 (currently, `pyrealsense2` highest Python version is 3.11)
- [uv](https://github.com/astral-sh/uv) package manager (Installation: `curl -sSf https://astral.sh/uv/install.sh | sh`)
- System dependencies (these will be checked and installed by the setup script on Debian/Ubuntu):

> **Note**: We use `uv` exclusively for package management. By default, it has less verbose output than pip, which is a good thing! If you want to see more details during installation, add `-v` or `--verbose` flags.
  - `libssl-dev` - OpenSSL development libraries
  - `libusb-1.0-0-dev` - USB development libraries for RealSense
  - `libsdl2-dev` - SDL2 development libraries
  - `ffmpeg` - For video processing
  - `libasound-dev` and `portaudio19-dev` - For audio processing

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/RKelln/experimance.git
   cd experimance/installation/software
   ```

2. Run the setup script:
   ```bash
   cd experimance
   ./setup.sh
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

4. Start the services:
   ```bash
   # In separate terminals
   cd experimance
   
   # Option 1: With activated virtual environment
   source .venv/bin/activate
   python -m services.experimance.experimance
   
   # Option 2: Using uv run (recommended)
   uv run python -m services.experimance.experimance
   uv run python -m services.image_server.image_server
   uv run python -m services.display.display
   ```

### Troubleshooting Installation

If you encounter issues with specific dependencies:

1. **PyRealSense2** - If the setup script fails to install pyrealsense2:
   ```bash
   # Try installing directly
   uv pip install --ignore-requires-python pyrealsense2
   
   # Or install the RealSense SDK first
   # See: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
   ```

2. **SDL2** - If PySDL2 installation fails:
   ```bash
   # Install SDL2 development libraries first
   sudo apt-get install libsdl2-dev
   
   # Then install the Python packages
   uv pip install pysdl2 pysdl2-dll
   ```

## Project Structure

- `experimance/`: Python-based micro-services
  - `services/`: Individual services
    - `experimance/`: Core service managing state machine
    - `image_server/`: Image generation service
    - `display/`: Display service for projector
    - `agent/`: Voice interaction agent
    - `audio/`: Audio service for sounds and music
    - `transition/`: Service for managing transitions
  - `libs/`: Shared libraries
    - `common/`: Common utilities and schemas

- `godot_display/`: Godot game engine based display (alternative)

- `transition_rs/`: Rust-based image transition generator

## Configuration

Each service has its own configuration file in TOML format. Default configurations are included, but you can modify them as needed:

- `experimance/services/experimance/config.toml`: Main service config
- `experimance/services/image_server/config.toml`: Image generation config
- `experimance/services/display/config.toml`: Display service config

## Development Environment

For development, you can use the provided setup with uv:

```bash
# Install a specific service's dependencies
uv pip install -r experimance/services/image_server/requirements.txt

# Install the common library in development mode
uv pip install -e experimance/libs/common

# Install the experimance package in development mode
uv pip install -e experimance
```

## Running the Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific service
uv run pytest experimance/services/experimance/tests/

# Run with coverage
uv run pytest --cov=experimance
```

## Integration with Hardware

- **Depth Camera**: Connect Intel RealSense D415 via USB 3.0
- **Projector**: Connect via HDMI
- **Vibe Sensors**: Connect via Arduino, which sends OSC messages
- **Microphone/Speakers**: Standard USB audio setup

## Troubleshooting

- **ZeroMQ Errors**: Check that ports are not in use by other applications
- **Depth Camera Issues**: Make sure librealsense is properly installed
- **Display Problems**: Check OpenGL drivers and monitor configuration

### Installation and Import Testing Utilities

We provide several utilities to help troubleshoot installation and import issues:

```bash
cd experimance

# Basic import test - checks if the basic modules can be imported
uv run python utils/tests/simple_test.py

# Environment check - verifies Python version and system dependencies
uv run python utils/tests/check_env.py

# Comprehensive import test - tests all package components
uv run python utils/tests/test_imports.py

# Fix import paths - creates .pth file in site-packages
uv run python fix_imports.py
```

See `experimance/utils/tests/README.md` for detailed information about these utilities and common troubleshooting steps.

For more detailed information, refer to the [technical design document](experimance/technical_design.md).
