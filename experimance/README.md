# Experimance

Interactive sand-table art installation with AI-generated satellite imagery.

## Overview

This project consists of multiple services that communicate via ZeroMQ:
- Coordinator service - Central orchestration service
- Display service - Handles visualization on the sand table
- Image generation service - Creates AI-generated satellite imagery
- Transition service - Handles smooth transitions between images
- Audio service - Provides audio feedback and soundscapes
- Agent service - AI agents that interact with the installation

## Installation

We use [uv](https://github.com/astral-sh/uv) as our package manager for faster, more reliable dependency management. Make sure it's installed before proceeding:

```bash
curl -sSf https://astral.sh/uv/install.sh | sh
```

### Quick Install

```bash
./setup.sh
```

This will:
1. Create a Python virtual environment using uv
2. Install all required dependencies
3. Install the experimance-common library
4. Install the main experimance package in development mode

### Manual Installation

If you prefer to install manually:

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install libssl-dev libusb-1.0-0-dev libsdl2-dev ffmpeg libasound2-dev portaudio19-dev

# Create and activate virtual environment
uv venv --python=3.11 .venv
source .venv/bin/activate

# Install common library
cd libs/common
uv pip install -e .
cd ../..

# Install main package (without optional extras)
uv pip install -e . --no-deps

# Install core dependencies
uv pip install numpy pyzmq pydantic toml python-dotenv asyncio aiohttp uuid pillow opencv-python

# For display functionality
uv pip install pyglet PyOpenGL PyOpenGL-accelerate pysdl2 pysdl2-dll ffmpegcv
```

**Note about SDL2**: The system package `libsdl2-dev` is required for PySDL2 to work properly. The Python packages `pysdl2` and `pysdl2-dll` provide Python bindings to SDL2.

### Why uv?

We use `uv` instead of `pip` for several reasons:
- Much faster installation times
- More reliable dependency resolution
- Better caching of packages
- Improved handling of binary dependencies
- Compatible with standard pip commands (just replace `pip` with `uv pip`)

## Project Structure

- `libs/` - Shared libraries used by services
  - `common/` - Common utilities, constants, and ZMQ communication
- `infra/` - Infrastructure code for deployment
- `services/` - Individual microservices
  - Each service folder contains a standalone service
- `scripts/` - Utility scripts

See `technical_design.md` for more details on the architecture.

## Importing the Package

When developing with the experimance package, you can import it in your Python code:

```python
import experimance
```

If you encounter import errors, use one of these solutions:

1. Import via bootstrap (recommended):
```python
import bootstrap  # This will set up the import paths
import experimance
```

2. Manually add to PYTHONPATH:
```python
import sys
import os
sys.path.insert(0, os.path.abspath('/path/to/experimance'))
import experimance
```

3. Activate the virtual environment before running your script:
```bash
source .venv/bin/activate
python your_script.py
```

4. Use uv run to execute scripts with the right environment:
```bash
uv run python your_script.py
```

## Troubleshooting

If you encounter issues with installation or imports, we provide several testing utilities:

```bash
# Basic import test
uv run python utils/tests/simple_test.py

# Check environment setup
uv run python utils/tests/check_env.py

# Verify comprehensive imports
uv run python utils/tests/test_imports.py
```

See `utils/tests/README.md` for detailed information about these utilities and common troubleshooting steps.
