# Display Service Configuration Reference

Configuration is loaded from a TOML file. The default path is resolved via `get_project_config_path("display")`, which looks in `projects/<active_project>/display.toml`. A fallback `config.toml` lives at the service root.

Pass an explicit path with `--config /path/to/display.toml`.

> **Important**: All keys must be under the correct TOML section header. For example, `fullscreen` must be inside `[display]`, not at the root level.

---

## `[display]`

Core window and rendering settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fullscreen` | bool | `true` | Run in fullscreen mode |
| `monitor` | int | `0` | Monitor index (0 = primary) |
| `resolution` | `[int, int]` | `[1920, 1080]` | Window size when not fullscreen |
| `fps_limit` | int | `60` | Frame rate cap |
| `vsync` | bool | `true` | Enable vertical sync |
| `debug_overlay` | bool | `false` | Show FPS and layer debug overlay |
| `debug_text` | bool | `false` | Show debug text in all screen positions |
| `debug_image` | path or null | `null` | Image to show in debug overlay |
| `background_color` | `[R,G,B,A]` | `[0,0,0,255]` | Window background color (RGBA, for layer debugging) |
| `profile` | bool | `false` | Enable performance profiling |
| `headless` | bool | `false` | Disable window creation (for testing) |
| `mask` | string or null | `null` | Window mask: `"circle"`, a file path, or `null` |

---

## `[transitions]`

Controls image crossfade and preloading.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_crossfade_duration` | float | `1.0` | Duration of default image crossfade (seconds) |
| `video_fade_in_duration` | float | `0.2` | Video overlay fade-in duration |
| `video_fade_out_duration` | float | `1.0` | Video overlay fade-out duration |
| `text_fade_duration` | float | `0.3` | Text overlay fade duration |
| `preload_frames` | bool | `true` | Preload transition frames into memory |
| `max_preload_mb` | int | `500` | Memory cap for preloading (MB) |

> Note: `config.toml` uses the legacy key `crossfade_duration` which maps to `default_crossfade_duration`.

---

## `[title_screen]`

Startup title screen shown before any content arrives.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Show title screen on startup |
| `text` | string | `"Experimance"` | Text to display |
| `duration` | float | `3.0` | How long the title screen stays visible (seconds) |
| `fade_duration` | float or `[float, float]` | `[2.0, 5.0]` | Fade-in duration, or `[fade_in, fade_out]` |

---

## `[video_overlay]`

Masked video layer that sits above the background image.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable the video overlay |
| `default_video_path` | string or null | `null` | Path to video file (relative to `media/videos/` or absolute) |
| `loop_video` | bool | `true` | Loop video playback |
| `fallback_mask_enabled` | bool | `true` | Create a fallback mask if none is provided |
| `start_mask_path` | string or null | `null` | Mask image to load on startup |
| `size` | `[int, int]` | `[1024, 1024]` | Video overlay size in pixels |

---

## `[panorama]`

Wide-aspect panoramic display mode. See [panorama.md](panorama.md) for full usage.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable panorama renderer |
| `rescale` | `"width"` / `"height"` / `"shortest"` | `"width"` | How to scale images to fit panorama dimensions |
| `output_width` | int | `1920` | Total panorama width (including any mirrored portion) |
| `output_height` | int | `1080` | Panorama height |
| `blur` | bool | `true` | Apply blur-to-sharp transition when a new image loads |
| `start_blur` | float | `5.0` | Initial Gaussian blur sigma |
| `end_blur` | float | `0.0` | Final blur sigma (0 = sharp) |
| `blur_duration` | float | `10.0` | Duration of the blur transition (seconds) |
| `mirror` | bool | `true` | Mirror image horizontally |
| `disappear_duration` | float | `0` | Global fade-to-transparent duration after last tile arrives; `0` = disabled |

### `[panorama.tiles]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `width` | int | `1920` | Target tile width in panorama space |
| `height` | int | `1080` | Target tile height in panorama space |
| `fade_duration` | float | `3.0` | Default tile fade-in duration (seconds) |
| `rescale` | `"width"` / `"height"` / `"shortest"` / null | `"width"` | Tile-specific rescale override; inherits from `[panorama]` if `null` |

---

## `[rendering]`

Rendering system settings. Currently informational; most rendering configuration is handled internally.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_texture_cache_mb` | int | `512` | Texture cache size cap (MB) |
| `shader_path` | path | `"services/display/shaders/"` | Directory for GLSL shader files |
| `font_path` | path | `"fonts/"` | Directory for font files |
| `preload_common_resources` | bool | `true` | Preload common resources on startup |
| `backend` | `"opengl"` | `"opengl"` | Rendering backend (only `opengl` is supported) |

---

## `[text_styles.<speaker>]`

Style for a named speaker. Built-in speakers: `agent`, `system`, `debug`, `title`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `font_size` | int | `28` | Font size in pixels |
| `color` | `[R,G,B,A]` | `[255,255,255,255]` | Text color (RGBA) |
| `anchor` | string | `"baseline_center"` | pyglet text anchor point |
| `position` | string | `"bottom_center"` | Screen position (`top_left` … `bottom_right`) |
| `background` | bool | `true` | Show background rectangle behind text |
| `background_color` | `[R,G,B,A]` | `[0,0,0,128]` | Background color (RGBA) |
| `padding` | int | `10` | Padding around text (pixels) |
| `max_width` | int or null | `null` | Maximum text width before wrapping |
| `align` | `"left"` / `"center"` / `"right"` | `"left"` | Text alignment within the label |

Valid `position` values: `top_left`, `top_center`, `top_right`, `center_left`, `center`, `center_right`, `bottom_left`, `bottom_center`, `bottom_right`.

---

## `[shader_effects]`

Optional full-screen GLSL effects rendered after the main layers. See [shaders.md](shaders.md).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable the shader effects system |
| `auto_reload_shaders` | bool | `false` | Reload shaders on file change |

### `[shader_effects.effects.<name>]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable this specific effect |
| `shader_file` | string | — | Path to the `.frag` shader file |
| `order` | int | `10` | Render order (higher = rendered on top) |
| `blend_mode` | `"alpha"` / `"additive"` | `"alpha"` | Blending mode |
| `uniforms` | dict[str, float] | `{}` | Uniform values passed to the shader |

---

## `[zmq]`

ZMQ subscriber configuration. Defaults are set in code and typically do not need to be changed.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `subscriber.address` | string | `"tcp://localhost"` | Address to subscribe to |
| `subscriber.port` | int | `5555` | Events channel port |
| `subscriber.topics` | list | see below | Message types to subscribe to |

Default subscribed topics: `DISPLAY_MEDIA`, `DISPLAY_TEXT`, `REMOVE_TEXT`, `CHANGE_MAP`.

---

## Minimal Example

```toml
[display]
fullscreen = false
resolution = [1920, 1080]
fps_limit = 30

[title_screen]
enabled = true
text = "Experimance"
duration = 5.0
fade_duration = [2.0, 3.0]

[video_overlay]
enabled = true
default_video_path = "video_overlay.mp4"
start_mask_path = "media/images/display/ring_map.png"
```
