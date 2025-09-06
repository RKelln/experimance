# Display Service Shaders

This directory contains fragment shaders used by the display service for visual effects.

## Available Shaders

### basic.vert
The standard vertex shader used by all fragment shaders. Handles full-screen quad rendering.

### passthrough.frag
A simple passthrough shader that renders the scene texture without modifications. Useful for testing and as a template.

### turbulent_vignette.frag
Creates animated cloudy effects at the top and bottom edges of the image with:
- Multi-octave fractal noise for detailed turbulence
- Progressive transparency gradients
- Upward-flowing movement for natural look
- Configurable vignette strength and turbulence amount

**Uniforms:**
- `vignette_strength` (float): Controls the intensity of the vignette effect (0.0 to 1.0)
- `turbulence_amount` (float): Controls the amount of turbulence distortion (0.0 to 1.0)
- `time` (float): Animation time (automatically provided)

### rising_sparks.frag
Generates realistic fire-like sparks with individual properties:
- Individual spark velocities and lifetimes
- Twinkling effects with brightness variation
- Color sampling from bright areas of the scene
- Wind effects and thermal patterns
- Lower-half spawning for realistic fire behavior

**Uniforms:**
- `spark_intensity` (float): Controls the overall intensity of the spark effect (0.0 to 1.0)
- `time` (float): Animation time (automatically provided)

## Using Shaders

Shaders are automatically loaded by the display service when shader effects are enabled in the configuration. See the main display service configuration for setup details.

### toml config

```toml
# Example configuration for enabling shader effects in the display service
#
# This configuration enables two shader effects:
# 1. Turbulent vignette - creates animated cloudy edges
# 2. Rising sparks - generates fire-like sparks
#
# To use this configuration:
# 1. Copy this to your project configuration
# 2. Adjust the shader file paths if needed
# 3. Tune the uniform values for desired visual effects

[shader_effects]
enabled = true
auto_reload_shaders = false  # Set to true for development/hot-reload

[shader_effects.effects.turbulent_vignette]
enabled = true
shader_file = "services/display/shaders/turbulent_vignette.frag"
order = 10  # Lower order = renders first (background effect)

[shader_effects.effects.turbulent_vignette.uniforms]
vignette_strength = 0.7   # 0.0-1.0, how strong the vignette effect is
turbulence_amount = 0.25  # 0.0-1.0, how much turbulence in the edges

[shader_effects.effects.rising_sparks]
enabled = true
shader_file = "services/display/shaders/rising_sparks.frag"
order = 20  # Higher order = renders on top

[shader_effects.effects.rising_sparks.uniforms]
spark_intensity = 0.5  # 0.0-1.0, intensity of spark effects

```

### Development

For rapid shader development, use the shader test harness:
```bash
uv run utils/examples/shader_test_harness.py
```

Controls:
- **Space**: Switch between shaders
- **T**: Switch between test images
- **R**: Reload shaders from files
- **Escape**: Exit

## Technical Notes

- All fragment shaders use OpenGL 1.5 core profile
- Shaders receive a `scene_texture` uniform containing the current frame
- The `time` uniform is automatically updated for animations
- Custom uniforms can be configured in the display service config
