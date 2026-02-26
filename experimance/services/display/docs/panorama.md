# Panorama Renderer

The panorama renderer is designed for wide-aspect or multi-projector displays. It composites a large background image with independently-fading positioned tiles, applies a blur-to-sharp transition on load, and optionally mirrors the image horizontally for symmetric 360° layouts.

**Files touched:**
- `src/experimance_display/renderers/panorama_renderer.py`
- `src/experimance_display/config.py` — `PanoramaConfig`, `PanoramaTilesConfig`

## When to use / When not to use

**Use panorama mode** when:
- The display spans multiple projectors or a very wide physical canvas
- You need to place satellite images at specific horizontal positions (tiles)
- You want a blur-reveal effect when new content arrives

**Do not use panorama mode** when:
- A single-screen landscape image is sufficient (use the default `ImageRenderer` instead)
- Performance is constrained — blur processing is GPU-intensive

## Configuration

Enable in your project's `display.toml`:

```toml
[panorama]
enabled = true
output_width = 6000        # Total logical width (includes mirrored half)
output_height = 1080
rescale = "shortest"       # "width" | "height" | "shortest"
mirror = true
blur = true
start_blur = 5.0
end_blur = 0.0
blur_duration = 4.0
disappear_duration = 0     # 0 = disabled

[panorama.tiles]
width = 300
height = 400
fade_duration = 3.0
rescale = "shortest"       # Override tile-specific rescaling
```

See [configuration.md](configuration.md) for all key descriptions.

## Sending Messages

All content arrives via `DisplayMedia` messages on the events channel (port 5555).

### Base image

Send an image **without** a `position` field to set the panoramic background:

```python
{
    "type": "DisplayMedia",
    "content_type": "image",
    "uri": "file:///path/to/panorama.jpg",
    "fade_in": 2.0
}
```

Base images will:
- Scale to fit `output_width × output_height` (respecting `rescale`)
- Blur-in from `start_blur` → `end_blur` over `blur_duration`
- Fade in over `fade_in` seconds (independent of blur timing)
- Mirror automatically if `mirror = true`

### Positioned tile

Send an image **with** a `position` field to place a tile:

```python
{
    "type": "DisplayMedia",
    "content_type": "image",
    "uri": "file:///path/to/tile.png",
    "position": [1200, 300],   # [x, y] in panorama space (top-left of tile)
    "fade_in": 1.5
}
```

Tiles will:
- Position at `[x, y]` in the logical panorama space
- Scale to the configured `tiles.width × tiles.height`
- Fade in individually over `fade_in` (or config `tiles.fade_duration`)
- Gain a mirrored copy automatically if `mirror = true`

### Clear

```python
{
    "type": "DisplayMedia",
    "content_type": "clear",
    "fade_in": 3.0    # fade_in reused as fade-out duration for clears
}
```

Fades out all panorama content. In-progress tile fades are cancelled gracefully.

## Coordinate System

- **Panorama space**: logical dimensions `output_width × output_height` (e.g. 6000×1080)
- **`position`**: `[x, y]` is the top-left corner of the tile in panorama space
- **Mirroring**: when enabled, the left half (`0` to `output_width/2`) is mirrored to the right
- **Screen mapping**: panorama coordinates are scaled automatically to fit the display window

### Example layout (output_width = 6000, mirror = true)

```
Panorama space  [0 ──────────────── 3000 ─────────── 6000]
                  left half (real)       right half (mirror)
```

Place tiles at x = 0, 300, 600 … 2700 to fill the visible left half; mirrored copies appear automatically.

## CLI Testing

Use the `panorama` CLI command to generate and send test images:

```bash
# Requires --config pointing to a file with panorama enabled
uv run experimance-display-cli panorama --config projects/experimance/display.toml \
    --base-width 5760 --base-height 1080 --tile-count 4

# Test scaling modes
uv run experimance-display-cli scaling --config projects/experimance/display.toml \
    --mode shortest --show-info
```

## Performance Considerations

- **Blur**: real-time Gaussian blur is GPU-intensive; disable with `blur = false` if needed
- **Mirroring**: doubles sprite count; monitor with debug overlay (F1)
- **GPU memory**: large panoramic textures consume significant VRAM
- **Development**: use `rescale = "shortest"` to fit large panoramas in a smaller dev window
