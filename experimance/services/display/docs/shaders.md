# Shader Effects

The shader effects system renders configurable full-screen GLSL fragment shaders on top of all other layers. Effects are composited additively or with alpha blending and can be individually ordered and parameterised.

**Files touched:**
- `src/experimance_display/renderers/shader_renderer.py`
- `shaders/` — GLSL source files
- `shaders/shader_test_harness.py` — standalone shader development tool
- `config.py` — `ShaderEffectsConfig`, `ShaderEffectConfig`

## When to use / When not to use

**Use shader effects** for atmospheric post-processing: vignettes, particle effects, fog, glow.

**Do not enable** when targeting lower-end hardware or when frame time is already tight — each effect adds a full-screen render pass.

## Active Shaders

### `basic.vert`
Standard vertex shader for full-screen quad rendering. Required by all fragment shaders.

### `passthrough.frag`
No-op fragment shader; passes the input texture through unchanged. Used for testing.

### `turbulent_vignette_improved.frag`
Animated vignette with turbulent edges.
- Creates soft, smoky boundaries around content
- Configurable mask zones and turbulence parameters
- Good for atmospheric edge effects

Uniforms: `vignette_strength`, `turbulence_amount`, `time`, `resolution`

### `rising_sparks_improved.frag`
Fire-like particle effects.
- Converts bright pixels into animated rising sparks
- Multi-projector support with horizontal compression
- Additive blending for realistic glow

Uniforms: `spark_intensity`, `time`, `resolution`

## Configuration

```toml
[shader_effects]
enabled = true
auto_reload_shaders = false   # Set true during development

[shader_effects.effects.turbulent_vignette]
enabled = true
shader_file = "services/display/shaders/turbulent_vignette_improved.frag"
order = 10
blend_mode = "alpha"
uniforms = { vignette_strength = 0.7, turbulence_amount = 0.25 }

[shader_effects.effects.rising_sparks]
enabled = true
shader_file = "services/display/shaders/rising_sparks_improved.frag"
order = 20
blend_mode = "additive"
uniforms = { spark_intensity = 0.5 }
```

- `order`: higher numbers render on top of lower numbers
- `blend_mode`: `"alpha"` (standard transparency) or `"additive"` (glow/bloom)
- Shader paths are relative to the **project root**

## Shader Requirements

All fragment shaders must accept these uniforms:

| Uniform | Type | Description |
|---------|------|-------------|
| `time` | `float` | Elapsed time in seconds |
| `resolution` | `vec2` | Output resolution in pixels |

Effect-specific uniforms are passed via the `uniforms` dict in config.

Vertex shader input: position and texture coordinates.  
OpenGL 3.3+ required.

## Development Workflow

Use the standalone test harness for rapid iteration without running the full service:

```bash
# From the project root
uv run services/display/shaders/shader_test_harness.py
```

Only production-ready shaders belong in `services/display/shaders/`. Development and experimental shaders should live in `utils/examples/shaders/` or a scratch directory.

## File Organisation

```
services/display/shaders/
├── basic.vert                          # Vertex shader (required)
├── passthrough.frag                    # No-op fragment shader
├── turbulent_vignette_improved.frag    # Vignette effect
├── rising_sparks_improved.frag         # Particle sparks effect
└── shader_test_harness.py              # Standalone development tool
```
