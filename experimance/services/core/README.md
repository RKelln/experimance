# Core Service

The **Core Service** is the central coordinator for Experimance interactive art installations. It owns the experience state machine, processes depth-camera data for interaction detection, and coordinates audio, display, and image generation services through a ZMQ event-driven architecture.

This package contains two separate core implementations:

| Package             | Project              | Input                        |
|---------------------|----------------------|------------------------------|
| `experimance_core`  | Experimance          | Intel RealSense depth camera |
| `fire_core`         | Feed the Fires       | Audience stories + transcripts via agent service |

---

## What this service does

**Experimance core** interprets depth-camera input as hand gestures and presence, drives an era-based narrative (Wilderness → AI/Future), and dispatches image generation, audio, and display updates at every state transition.

**Fire core** receives audience stories and live conversation from the agent service, uses an LLM to infer environmental settings, and orchestrates a panoramic image pipeline (base image → tiles → display) with smart interruption so new stories supersede old ones gracefully.

---

## Environment assumptions

- Linux (Ubuntu 20.04+)
- Python 3.11 (maximum while `pyrealsense2` lags upstream)
- `uv` package manager
- Intel RealSense D-series camera (experimance) or none (fire)
- All other services reachable on localhost (ZMQ on standard ports)

---

## Quick start

```bash
# Set the active project (once)
uv run set-project experimance   # or: fire

# Install dependencies
cd services/core
uv sync

# Run (experimance)
uv run -m experimance_core

# Run (fire)
uv run -m fire_core
```

### Common flags

```bash
# Experimance – mock depth camera (no hardware required)
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth

# Experimance – bypass presence detection for testing
uv run -m experimance_core --presence-always-present

# Experimance – visualize depth processing in real-time
uv run -m experimance_core --visualize

# Fire – mock LLM (no API key required)
uv run -m fire_core --llm-provider mock

# Both – verbose debug logging
uv run -m experimance_core --log-level DEBUG
uv run -m fire_core --log-level DEBUG

# Both – see all options
uv run -m experimance_core --help
uv run -m fire_core --help
```

All configuration fields are accessible as CLI flags in `--section-field-name` format.

---

## Configuration

| File                              | Purpose                             |
|-----------------------------------|-------------------------------------|
| `services/core/config.toml`       | Default/example values              |
| `projects/experimance/core.toml`  | Production overrides (experimance)  |
| `projects/fire/core.toml`         | Production overrides (fire)         |

The active project is stored in `projects/.project`. Switch with `uv run set-project <name>` or by editing the file.

---

## Running tests

```bash
cd services/core

# All tests
uv run -m pytest

# Specific test file
uv run -m pytest tests/test_presence_manager.py

# Tests matching a pattern
uv run -m pytest -k "presence"

# Verbose output
uv run -m pytest -v
```

Test files:

| File | Coverage |
|------|----------|
| `test_config.py` | Configuration loading and validation |
| `test_core_service.py` | Core service lifecycle |
| `test_presence_manager.py` | Presence detection and hysteresis |
| `test_experimance_core_state_management.py` | Era progression and state machine |
| `test_core_image_ready_handling.py` | Image-ready message handling |
| `test_depth_integration.py` | Depth processing integration |
| `test_queue_smoothing.py` | Change-score smoothing |
| `test_pipeline_mocked.py` | Full mocked pipeline (mock camera + mock ZMQ) |
| `test_realsense_camera_mocked.py` | Camera error and recovery paths |
| `test_tiler_new.py` | Fire core tiler unit tests |
| `test_llm_integration.py` | Fire core LLM integration |

---

## Project structure

```
services/core/
├── README.md                    # This file
├── config.toml                  # Default config (project overrides in projects/)
├── pyproject.toml               # Package manifest and entry points
├── pytest.ini                   # Pytest config
├── docs/                        # Additional documentation (see below)
├── scripts/                     # Camera recovery and debugging tools
├── src/
│   ├── experimance_core/        # Experimance project core
│   └── fire_core/               # Feed the Fires core
├── tests/                       # Unit and integration tests
└── typings/                     # Type stubs
```

---

## Deployment (experimance)

```bash
# Camera permissions
sudo usermod -a -G dialout $USER   # re-login required

# RealSense drivers
sudo apt install librealsense2-*
rs-enumerate-devices               # verify camera detected

# Systemd service (if configured)
sudo systemctl start experimance-core
journalctl -u experimance-core -f
```

Hardware requirements: Intel RealSense D415/D435/D455, USB 3.0, 2–4 GB RAM.

---

## Additional docs

| Doc | Description |
|-----|-------------|
| [docs/index.md](docs/index.md) | Full documentation index |
| [docs/architecture.md](docs/architecture.md) | Service composition, ZMQ port map, message schemas, state machine, module map |
| [docs/depth-camera.md](docs/depth-camera.md) | Camera setup, error recovery config, mock processor, test scripts, troubleshooting |
| [docs/fire-core.md](docs/fire-core.md) | Feed the Fires pipeline: request state machine, tiling, LLM, CLI tool |
| [docs/new-service-guide.md](docs/new-service-guide.md) | How to create a new project-specific core service |
| [docs/roadmap.md](docs/roadmap.md) | Near-term goals and known gaps |
