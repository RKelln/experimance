# Experimance

Interactive sand-table art installation with AI-generated satellite imagery.

## Overview

This project consists of multiple services that communicate via ZeroMQ:
- Core service (`experimance_core`) - Central orchestration service
- Display service (`experimance_display`) - Handles visualization on the sand table
- Image generation service (`experimance_image_server`) - Creates AI-generated satellite imagery
- Transition service (`experimance_transition`) - Handles smooth transitions between images
- Audio service (`experimance_audio`) - Provides audio feedback and soundscapes
- Agent service (`experimance_agent`) - AI agents that interact with the installation

## Project Structure

The project uses a modern Python package structure with src layout:

```
experimance/
├── libs/
│   └── common/                # Common utilities shared by all services
│       ├── pyproject.toml
│       └── src/
│           └── experimance_common/
│
├── services/                  # Independent service packages
│   ├── core/                  # Core service
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_core/
│   │
│   ├── display/
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_display/
│   │
│   ├── audio/
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_audio/
│   │
│   ├── agent/
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_agent/
│   │
│   ├── image_server/
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── experimance_image_server/
│   │
│   └── transition/            # Image transition service
│       ├── pyproject.toml
│       └── src/
│           └── experimance_transition/
│
├── infra/                     # Infrastructure code for deployment
├── scripts/                   # Utility scripts
└── utils/                     # Testing and utility modules
```

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
3. Install all services in development mode

### Manual Installation

If you prefer to install manually:

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install libssl-dev libusb-1.0-0-dev libsdl2-dev ffmpeg libasound2-dev portaudio19-dev

# Create and activate virtual environment
uv sync
```

**Note about SDL2**: The system package `libsdl2-dev` is required for PySDL2 to work properly. The Python packages `pysdl2` and `pysdl2-dll` provide Python bindings to SDL2.

## Running Services

Each service can be run independently:

# TODO:
```bash
uv run experimance
```

Or run services separately:
```bash
uv run python -m experimance_[service_name] --arg
```

## Development

Each service is a complete Python package with its own:
- Dependencies in pyproject.toml
- Source code in src/package_name/
- Tests in its own tests/ directory

This structure makes development cleaner by:
- Avoiding complex import hacks
- Supporting proper editable installs
- Making packages independently testable

### Install tools

```bash
uv tool install ruff
uv tool install pytest
uv add --dev pytest pytest-asyncio pytest-mock pytest-cov 
```

## Package management

Use `uv` and see: https://docs.astral.sh/uv/concepts/projects/workspaces/ for reference.

To update all packages to latest versions:
```
uv sync
```

To update a particular service (e.g. core):
```
uv sync --package experimance-core
```

To update a particular dependency:
```
uv lock --upgrade-package opencv-python
```

### Troubleshooting packages

If you encounter issues with installation or imports, we provide several testing utilities:

```bash
# Basic import test
uv run python utils/tests/simple_test.py

# Check environment setup
uv run python utils/tests/check_env.py
```

See `utils/tests/README.md` for detailed information about these utilities and common troubleshooting steps.


## Testing

Run tests with uv and pytest:

```bash
# Install development dependencies
uv sync --only-dev

# Run all tests
uv run -m pytest

# Run tests with coverage
uv run -m pytest --cov=experimance_common

# Run specific tests
uv run -m pytest -v utils/tests/test_zmq_utils.py -k test_name
```

See [`utils/tests/README.md`](utils/tests/README.md) and [`utils/tests/README_ZMQ_TESTS.md](utils/tests/README_ZMQ_TESTS.md) for more details.

