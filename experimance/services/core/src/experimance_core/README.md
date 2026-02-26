# experimance_core

Core orchestration service for the **Experimance** interactive art installation.

Processes Intel RealSense depth-camera input to detect hand gestures and audience presence, drives an era-based narrative (Wilderness → AI/Future), and coordinates image generation, audio, and display services via ZMQ.

## Quick start

```bash
uv run set-project experimance
uv run -m experimance_core

# Mock depth camera (no hardware required)
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth

# Bypass presence detection for testing
uv run -m experimance_core --presence-always-present

# See all options
uv run -m experimance_core --help
```

## Documentation

Full documentation is in [`services/core/docs/`](../../docs/):

- [`architecture.md`](../../docs/architecture.md) – state machine, ZMQ events, module map
- [`depth-camera.md`](../../docs/depth-camera.md) – camera setup, error recovery, mock processor
