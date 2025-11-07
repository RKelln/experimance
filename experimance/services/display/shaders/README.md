# Display Service Shaders

This directory contains fragment shaders used by the Experimance display service for real-time visual effects.

## Active Shaders

### Core Shaders
- **basic.vert** - Standard vertex shader for full-screen quad rendering
- **passthrough.frag** - Simple passthrough fragment shader for testing
- **test_gradient.frag** - Test gradient shader for development

### Production Effects
- **turbulent_vignette_improved.frag** - Animated vignette with turbulent edges
  - Creates soft, smoky boundaries around content
  - Configurable mask zones and turbulence parameters
  - Perfect for atmospheric edge effects

- **rising_sparks_improved.frag** - Fire-like particle effects
  - Converts bright pixels into animated rising sparks
  - Multi-projector support with horizontal compression
  - Additive blending for realistic glow effects

## Shader Development

### Testing
Use the shader test harness for rapid development:
```bash
uv run utils/examples/shader_test_harness.py
```

### Integration
Shaders are loaded automatically based on the project configuration in `projects/[project]/display.toml`.

### Requirements
- OpenGL 3.3+ compatible
- Fragment shader uniforms: `time`, `resolution`, plus effect-specific parameters
- Vertex shader input: position and texture coordinates

## Configuration

Shaders are configured in the project's `display.toml` file under the `[shader_effects]` section. See the main shader effects documentation for complete configuration details.

## File Organization

Only production-ready shaders should be kept in this directory. Development and experimental shaders should be placed in `utils/examples/shaders/` for testing.
