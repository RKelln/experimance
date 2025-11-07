# Fog + Fairy Lights Projection System (Design Summary)

## Overview
A real-time visual system built with **Python + pyglet + OpenGL** that transforms static images into dynamic, dream-like panoramas with:
- **Screen-space fog** (animated, depth-aware, mask-modulated)
- **Fairy light particles** (depth-tested, twinkling, audio/OSC-reactive)
- **OSC controls** for live parameter tuning and mood switching
- Support for **venue-specific projection stretching** (e.g., 6× horizontal expansion)

---

## Rendering Pipeline
1. **Prepass (per new image)**
   - Run depth estimation (e.g., Depth Anything v2, MiDaS).
   - Run segmentation (e.g., SAM2, SegFormer).
   - Save outputs:  
     - Depth map (`R16F/R32F`)  
     - Masks (`trees.png`, `sky.png`, `ground.png`)

2. **Pass A – Base**
   - Render the panorama (or gradient placeholder).
   - Apply horizontal **mirroring** and **stretch compensation**.

3. **Pass B – Fog**
   - Screen-space raymarch with **8 steps** and **blue-noise dithering**.
   - **Exponential height fog** (denser near ground, fades with height).
   - Modulated by **3D noise** (billowing motion).
   - Weighted by **masks** (more on ground/water, less in canopy/sky).
   - Fog color interpolates between low/high tints.

4. **Pass C – Fairy Lights**
   - Point-sprite particles with pseudo-depth.
   - Depth-tested against scene depth + softened by tree masks.
   - Animated drift using noise-based flow fields.
   - **Twinkle** and **audio-reactive brightness**.
   - Bloom/glow composited additively.

---

## OSC Controls
- **View / Geometry**
  - `/view/x_stretch f` – horizontal stretch factor (e.g., `6.0`)
  - `/view/mirror i` – toggle mirroring

- **Fog**
  - `/fog/density f`
  - `/fog/falloff f`
  - `/fog/noise_scale f`
  - `/fog/wind f`

- **Lights**
  - `/lights/count i`
  - `/lights/size f`
  - `/lights/speed f`
  - `/lights/twinkle f`

- **Audio Reactivity**
  - `/audio/energy f` – drives light glow intensity

- **Moods**
  - `/mood/set s` – switch preset (e.g., `"deep_forest"`, `"moonlit"`, `"dawn"`)

---

## Mood Presets (Examples)
- **Deep Forest**
  - Dark green fog, low wind, ~480 lights
- **Moonlit**
  - Blue/cool fog, gentle drift, ~420 lights
- **Dawn**
  - Warm golden fog, more lights, higher density

---

## Projection Handling
- Venue sends **HD frame** stretched 6× horizontally across multiple projectors.
- All **procedural domains** (noise, particle motion, fog sampling) use **stretch-aware coordinates** so visuals appear correct after expansion.
- Mirroring can be toggled for symmetry-based layouts.

---

## Performance Targets
- **RTX A4000 @ 1080p 60Hz**
  - Fog: 8 steps
  - Particles: 500–800
- **Apple M-chip @ 1080p 60Hz**
  - Fog: 6–8 steps
  - Particles: 300–600
- Blue-noise dithering ensures stability with low raymarch step counts.

---

## Extensibility
- Replace gradient with generated panoramas.
- Add **god-rays** post-process for lights.
- Add **bloom/blur** pass for stronger glow.
- Future: connect audio engine directly or add OSC-driven video overlays.

