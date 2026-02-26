# Display Service CLI Tool

The CLI tool sends ZMQ messages to a running display service for manual testing and development. You do not need the full Experimance system running — just the display service.

## Environment

- Run from the **project root** or the **display service directory**
- Requires a live display service listening on port 5555 (default)
- Python 3.11+, `uv` installed

## Quick Start

```bash
# From project root — start the display service
uv run -m experimance_display --windowed

# In another terminal — send a test image
uv run experimance-display-cli image /path/to/image.png

# Or invoke the module directly
uv run python -m experimance_display.cli image /path/to/image.png
```

## Commands

### `list` — List available test resources

```bash
uv run experimance-display-cli list
```

Shows available generated images, mock files, and videos.

---

### `image` — Display an image

```bash
uv run experimance-display-cli image [path]
uv run experimance-display-cli image /path/to/image.webp
```

Sends a `DisplayMedia` message. If `path` is omitted, a random image from `media/images/generated/` is used.

---

### `text` — Show a text overlay

```bash
uv run experimance-display-cli text [content] [options]
uv run experimance-display-cli text "System ready" --speaker system --duration 10.0 --position top_center
```

Options:

| Flag | Default | Values |
|------|---------|--------|
| `--id` | auto-generated | any string |
| `--speaker` | `system` | `agent`, `system`, `debug` |
| `--duration` | infinite | seconds (float) |
| `--position` | `bottom_center` | `top_left`, `top_center`, `top_right`, `center_left`, `center`, `center_right`, `bottom_left`, `bottom_center`, `bottom_right` |

---

### `remove-text` — Remove a text overlay

```bash
uv run experimance-display-cli remove-text <text_id>
```

---

### `video-mask` — Send a video mask update

```bash
uv run experimance-display-cli video-mask [path]
uv run experimance-display-cli video-mask /path/to/mask.png --fade-in 0.5 --fade-out 2.0
```

Sends a `ChangeMap` message. If `path` is omitted, a random mask from `media/images/mocks/mask/` is used.

Options: `--fade-in` (default `0.2`), `--fade-out` (default `1.0`)

---

### `era-change` — Send a space/time update event

```bash
uv run experimance-display-cli era-change [era] [biome]
uv run experimance-display-cli era-change wilderness forest
uv run experimance-display-cli era-change anthropocene urban
```

If omitted, a random era/biome pair is chosen. Known pairs:

- `wilderness` + `forest` / `grassland` / `wetland`
- `anthropocene` + `urban` / `suburban` / `industrial`
- `rewilded` + `forest` / `grassland` / `wetland`

---

### `transition` — Send a transition-ready event

```bash
uv run experimance-display-cli transition [path] --from-image prev --to-image next
```

`path` defaults to `media/videos/video_overlay.mp4` if available.

---

### `loop` — Send a loop-ready event

```bash
uv run experimance-display-cli loop [path] [still_uri] --type idle_animation
```

---

### `cycle-images` — Cycle through images continuously

```bash
uv run experimance-display-cli cycle-images
uv run experimance-display-cli cycle-images /custom/directory --interval 5.0
```

Cycles until `Ctrl+C`.

---

### `demo` — Run an interactive feature demo

```bash
uv run experimance-display-cli demo
```

Runs text overlays, era change, image cycling, video mask, text removal, and a transition in sequence.

---

### `panorama` — Test panorama display mode

```bash
uv run experimance-display-cli panorama --config /path/to/display.toml [options]
```

Requires `--config` pointing to a config file with panorama enabled. Generates and sends a base image followed by tiled images.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--base-width` | 5760 | Base image width |
| `--base-height` | 1080 | Base image height |
| `--tile-count` | 3 | Number of tiles to generate |
| `--tile-width` | 800 | Generated tile image width |
| `--tile-height` | 600 | Generated tile image height |

---

### `scaling` — Test panorama scaling modes

```bash
uv run experimance-display-cli scaling --config /path/to/display.toml --mode width --show-info
```

Sends images of various panoramic dimensions to verify the configured `rescale` mode.

---

### `stress` — Stress test with random messages

```bash
uv run experimance-display-cli stress 0.5
```

Continuously sends random text, image, and mask messages at the given interval (seconds). Use `Ctrl+C` to stop.

---

## Global Flags

| Flag | Description |
|------|-------------|
| `-v` / `--verbose` | Enable INFO logging |
| `-vv` / `--debug` | Enable DEBUG logging |
| `--quiet` | Suppress all output except errors |

## ZMQ Port

All messages are published to the unified events channel: **port 5555** (default).

## Example Workflows

### Test text overlays

```bash
uv run experimance-display-cli text "Welcome to Experimance" --speaker agent --duration 5.0
uv run experimance-display-cli text "Status: Running" --speaker system --position top_right
uv run experimance-display-cli text "FPS: 60" --speaker debug --position top_left --id fps_debug
uv run experimance-display-cli remove-text fps_debug
```

### Test image cycling

```bash
uv run experimance-display-cli list
uv run experimance-display-cli cycle-images --interval 4.0
```

### Test video mask

```bash
uv run experimance-display-cli video-mask --fade-in 1.0 --fade-out 3.0
```

## Troubleshooting

**No images listed** — Check that `media/images/generated/` and `media/images/mocks/` exist at the project root.

**ZMQ errors** — Ensure nothing else is already bound to port 5555, and that ZMQ is installed (`uv sync`).

**File not found** — Use absolute paths, or verify files exist relative to the project root.

**Panorama mode fails** — Pass `--config` with a path to a `display.toml` that has `[panorama] enabled = true`.
