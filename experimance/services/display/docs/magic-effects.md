# Fog and Fairy Lights Projection System (Experimental)

A real-time visual system built with Python + pyglet + OpenGL that transforms static panoramic images into dynamic, dream-like scenes with screen-space fog and fairy light particle effects.

This is a **design document for an experimental / future feature**. It is not yet integrated into the main display service. See `src/experimance_display/magic_demo.py` and `magic_demo_simple.py` for prototype implementations.

## What this does

- **Screen-space fog**: animated, depth-aware, mask-modulated
- **Fairy light particles**: depth-tested, twinkling, audio/OSC-reactive
- **OSC controls**: live parameter tuning and mood switching
- **Projection stretch compensation**: e.g. 6× horizontal expansion across multiple projectors

## Rendering Pipeline

### Prepass (per new image)
- Run depth estimation (Depth Anything v2, MiDaS, etc.)
- Run segmentation (SAM2, SegFormer, etc.)
- Save: depth map (`R16F`/`R32F`), segmentation masks (`trees.png`, `sky.png`, `ground.png`)

### Pass A — Base
- Render the panorama (or gradient placeholder)
- Apply horizontal mirroring and stretch compensation

### Pass B — Fog
- Screen-space raymarch with 8 steps and blue-noise dithering
- Exponential height fog (denser near ground, fades with height)
- Modulated by 3D noise (billowing motion)
- Weighted by segmentation masks (more on ground/water, less in canopy/sky)
- Fog colour interpolates between low/high tints

### Pass C — Fairy Lights
- Point-sprite particles with pseudo-depth
- Depth-tested against scene depth, softened by tree masks
- Animated drift using noise-based flow fields
- Twinkle and audio-reactive brightness
- Bloom/glow composited additively

## OSC Controls

| Address | Type | Description |
|---------|------|-------------|
| `/view/x_stretch` | float | Horizontal stretch factor (e.g. `6.0`) |
| `/view/mirror` | int | Toggle mirroring (`0`/`1`) |
| `/fog/density` | float | Fog density |
| `/fog/falloff` | float | Height falloff rate |
| `/fog/noise_scale` | float | Spatial noise frequency |
| `/fog/wind` | float | Wind speed for fog animation |
| `/lights/count` | int | Number of fairy light particles |
| `/lights/size` | float | Particle size |
| `/lights/speed` | float | Particle drift speed |
| `/lights/twinkle` | float | Twinkle intensity |
| `/audio/energy` | float | Audio energy input (drives glow intensity) |
| `/mood/set` | string | Switch to named preset (`"deep_forest"`, `"moonlit"`, `"dawn"`) |

## Mood Presets

| Preset | Fog colour | Wind | Particle count |
|--------|-----------|------|---------------|
| `deep_forest` | Dark green | Low | ~480 |
| `moonlit` | Blue/cool | Gentle | ~420 |
| `dawn` | Warm golden | Moderate | ~500+ |

## Performance Targets

| Hardware | Fog steps | Particles |
|----------|-----------|-----------|
| RTX A4000 @ 1080p 60Hz | 8 | 500–800 |
| Apple M-chip @ 1080p 60Hz | 6–8 | 300–600 |

Blue-noise dithering maintains quality at low step counts.

## Projection Handling

The venue sends an HD frame that is stretched 6× horizontally across multiple projectors. All procedural domains (noise, particle motion, fog sampling) use stretch-aware coordinates so visuals appear undistorted after expansion.

## Integration Path

1. Connect depth estimation output to a shared media directory the display service reads
2. Expose fog/light parameters through the existing `shader_effects` config system
3. Add OSC listener to `DisplayService` for live tuning
4. Replace prototype demo files with production renderer module under `renderers/`
