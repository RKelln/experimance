# Experimance Display Service

The Display Service renders the visual output of the Experimance installation: satellite landscape images, panoramic backgrounds, masked video overlays, text from the AI agent, and shader effects. It subscribes to a ZMQ events channel and composites everything in real time using OpenGL/pyglet.

## Environment

- Python 3.11+
- OpenGL 3.3+ compatible GPU (dedicated GPU recommended)
- Linux (primary target); macOS and Windows untested
- 4 GB RAM minimum, 8 GB recommended for 4K

## Quick Start

```bash
# 1. Install dependencies (from project root)
uv sync

# 2. Set the active project (once)
uv run set-project experimance

# 3. Start in windowed mode for development
uv run -m experimance_display --windowed

# 4. In another terminal, send a test image
uv run experimance-display-cli image /path/to/image.png
```

## Running the Service

```bash
# Windowed mode (development)
uv run -m experimance_display --windowed

# With debug logging and overlay
uv run -m experimance_display --windowed --debug --log-level DEBUG

# Production (fullscreen)
uv run -m experimance_display

# All options
uv run -m experimance_display --help
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config`, `-c` | project config | Path to `display.toml` |
| `--name`, `-n` | `display-service` | Service instance name |
| `--log-level`, `-l` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--windowed`, `-w` | — | Force windowed mode (overrides config) |
| `--debug` | — | Enable debug overlay |

### Keyboard Controls

| Key | Action |
|-----|--------|
| ESC or Q | Exit gracefully |
| F11 | Toggle fullscreen |
| F1 | Toggle debug overlay (FPS, layer info) |
| Ctrl+C | Graceful shutdown from terminal |

## Configuration

The service reads a TOML config file. The default path is `projects/<active_project>/display.toml`; the service root `config.toml` is a fallback.

Minimal example:

```toml
[display]
fullscreen = false
resolution = [1920, 1080]
fps_limit = 30

[title_screen]
enabled = true
text = "Experimance"
duration = 5.0

[video_overlay]
enabled = true
default_video_path = "video_overlay.mp4"
start_mask_path = "media/images/display/ring_map.png"
```

> All keys must be under their correct section header. For example, `fullscreen` must be inside `[display]`, not at the root level.

See [docs/configuration.md](docs/configuration.md) for the complete reference.

## Architecture

Rendering layers (back to front):
1. Background image (satellite landscape, crossfade transitions)
2. Panorama (wide-aspect base image + positioned tiles, optional)
3. Video overlay (masked, sand-interaction-driven)
4. Text overlays (multiple concurrent, speaker-styled)
5. Shader effects (vignette, sparks, etc.)
6. Debug overlay

The service subscribes to port 5555 (`events` channel) for all incoming messages. See [docs/architecture.md](docs/architecture.md) for the full design.

## Message Types

The service handles these ZMQ message types on the `events` channel:

| Type | Description |
|------|-------------|
| `DisplayMedia` | Primary content: images, sequences, videos, or clear |
| `DisplayText` | Show a text overlay (speaker-styled) |
| `RemoveText` | Remove a text overlay by ID |
| `ChangeMap` | Update the video overlay mask |
| `TransitionReady` | Custom transition video (not yet implemented) |
| `LoopReady` | Animated loop for a still image (not yet implemented) |

## Development

### Running Tests

```bash
# Headless tests (no display required)
cd services/display
uv run pytest tests/

# From the project root
uv run pytest services/display/tests/
```

See [docs/testing.md](docs/testing.md) for window tests and writing new tests.

### Adding New Features

1. **New message type**: add to `experimance_common.zmq.config.MessageType` and subscribe in `DisplayServiceConfig`
2. **New renderer**: create in `src/experimance_display/renderers/`, register with `LayerManager`
3. **New shader**: add `.frag` to `shaders/`, configure under `[shader_effects]`
4. **New config key**: update `config.py` and `docs/configuration.md`

### Debugging

```bash
# Debug overlay + verbose logs
uv run -m experimance_display --windowed --debug --log-level DEBUG

# Press F1 at runtime to toggle the overlay
```

Key log messages:

```
INFO  - Window initialized: 1920x1080       # Display ready
INFO  - DisplayService started              # Service accepting messages
ERROR - Error handling DisplayMedia         # Message processing failure
DEBUG - FPS: 29.8                           # Performance monitor
INFO  - Exit key pressed, shutting down     # Clean exit
```

## Troubleshooting

**Window does not appear** — try `--windowed`; verify OpenGL support with `glxinfo | grep OpenGL`.

**Low FPS** — disable vsync (`vsync = false`), lower resolution, or check GPU memory usage. Enable the debug overlay (F1) to monitor FPS.

**ZMQ connection issues** — verify port 5555 is not in use (`ss -tlnp | grep 5555`) and that other services are running.

**Config not loading** — ensure all keys are under the correct section headers; use `--log-level DEBUG` to see loading details.

**Panorama issues** — see [docs/panorama.md](docs/panorama.md).

## Additional Docs

| Doc | Description |
|-----|-------------|
| [docs/index.md](docs/index.md) | Full documentation index |
| [docs/architecture.md](docs/architecture.md) | Component design, message types, rendering pipeline |
| [docs/configuration.md](docs/configuration.md) | Complete config key reference |
| [docs/cli.md](docs/cli.md) | CLI testing tool commands and workflows |
| [docs/panorama.md](docs/panorama.md) | Panorama renderer — tiles, blur, coordinate system |
| [docs/shaders.md](docs/shaders.md) | Shader effects — active shaders, development workflow |
| [docs/testing.md](docs/testing.md) | Running tests, writing new tests |
| [docs/loop-animation.md](docs/loop-animation.md) | Loop animation design (future feature) |
| [docs/magic-effects.md](docs/magic-effects.md) | Fog and fairy lights projection system (experimental) |
| [docs/roadmap.md](docs/roadmap.md) | Near-term goals and known gaps |
