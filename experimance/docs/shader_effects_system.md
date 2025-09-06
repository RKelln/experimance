# Shader Effects System

This document describes the shader effects system for the Experimance display service, providing advanced visual effects for interactive art installations.

## Overview

The shader effects system adds cinematic post-processing effects to the display pipeline:

1. **Turbulent Vignette**: Creates an animated soft-edge vignette with turbulent boundaries
2. **Rising Sparks**: Particle effect that makes bright pixels "float up" as sparks
3. **Configurable Parameters**: Real-time adjustment of effect parameters
4. **Multi-Projector Support**: Horizontal compression for panoramic projection systems

## Architecture

### Components

- **ShaderRenderer**: Main renderer class that integrates with the layer system
- **Multi-Effect Support**: Multiple shader renderers working in sequence
- **Configuration-Driven**: All effects configured through TOML files
- **Hot-Reload Support**: Automatic shader reloading during development

### File Structure

```
services/display/
├── src/experimance_display/renderers/
│   └── shader_renderer.py           # Main shader renderer
├── shaders/
│   ├── basic.vert                   # Standard vertex shader
│   ├── turbulent_vignette_improved.frag
│   ├── rising_sparks_improved.frag
│   ├── test_gradient.frag           # Development/testing
│   ├── passthrough.frag             # Development/testing
│   ├── shader_test_harness.py       # Interactive shader testing
│   └── README.md                    # Shader documentation
└── ...

projects/[project_name]/
└── display.toml                     # Project-specific shader configuration

utils/examples/
├── shader_integration_example.py   # Integration example
└── shader_effects_config_example.toml  # Configuration example
```

## Shader Effects

### Turbulent Vignette

Creates a vignette effect (darkened edges) with animated turbulent boundaries that suggest the visible center area has soft, smoky edges.

**Features:**
- Adjustable vignette strength (0.0 to 1.0)
- Animated turbulent noise at vignette boundaries
- Configurable turbulence amount and zone sizes
- Time-based animation for organic movement
- Configurable mask zones for fine-tuning edge behavior

**Configuration:**
```toml
[shader_effects.effects.turbulent_vignette]
enabled = true
shader_file = "turbulent_vignette_improved.frag"
order = 10

[shader_effects.effects.turbulent_vignette.uniforms]
vignette_strength = 1.0      # 0.0-1.0, overall vignette intensity
turbulence_amount = 0.5      # 0.0-1.0, turbulence at edges
full_mask_zone = 0.08        # Size of fully masked edge area
gradient_zone = 0.25         # Size of gradient transition zone
```

**Use Cases:**
- Creating atmospheric boundaries around content
- Simulating smoke or heat distortion effects
- Focusing attention on center content area

### Rising Sparks

Converts bright pixels in the image into animated sparks that rise and fade, creating a particle effect from the existing image content.

**Features:**
- Automatically detects bright pixels as spark sources
- Animated rising motion with random drift and wind effects
- Color-reactive: uses colors from source pixels
- Configurable intensity and particle behavior
- **Multi-projector support**: Horizontal compression for panoramic displays
- Additive blending for realistic glow effects

**Configuration:**
```toml
[shader_effects.effects.sparks]
enabled = true
shader_file = "rising_sparks_improved.frag"
order = 20
blend_mode = "additive"  # Essential for proper glow effects

[shader_effects.effects.sparks.uniforms]
spark_intensity = 0.5           # 0.0-1.0, intensity of spark effects
horizontal_compression = 6.0    # Compression factor for multi-projector setups
```

**Multi-Projector Support:**
For panoramic projection systems where a single 1920x1080 image is stretched across multiple projectors, the `horizontal_compression` parameter prevents sparks from appearing stretched. Set this to the horizontal stretch factor (e.g., 6.0 for 6 projectors).

**Use Cases:**
- Fire and ember effects
- Magical sparkle effects
- Converting static bright spots into dynamic particles
- 360° panoramic installations with proper aspect ratios

## Configuration

### Display Service Configuration

Add to your project's `display.toml` file:

```toml
[shader_effects]
enabled = true
auto_reload_shaders = false  # Set to true for development/hot-reload

# Example: Turbulent vignette effect
[shader_effects.effects.turbulent_vignette]
enabled = true
shader_file = "turbulent_vignette_improved.frag"
order = 10  # Lower order = renders first (background effect)

[shader_effects.effects.turbulent_vignette.uniforms]
vignette_strength = 1.0      # 0.0-1.0, overall vignette intensity
turbulence_amount = 0.5      # 0.0-1.0, turbulence at edges
full_mask_zone = 0.08        # Size of fully masked edge area
gradient_zone = 0.25         # Size of gradient transition zone

# Example: Rising sparks effect
[shader_effects.effects.sparks]
enabled = true
shader_file = "rising_sparks_improved.frag"
order = 20  # Higher order = renders on top
blend_mode = "additive"  # Essential for proper glow effects

[shader_effects.effects.sparks.uniforms]
spark_intensity = 0.5           # 0.0-1.0, intensity of spark effects
horizontal_compression = 1.0    # Compression factor for multi-projector setups
```

### Configuration Options

- **enabled**: Enable/disable the entire shader system
- **auto_reload_shaders**: Hot-reload shaders when files change (development)
- **effects.[name].enabled**: Enable/disable individual effects
- **effects.[name].shader_file**: Path to fragment shader (relative to `services/display/shaders/`)
- **effects.[name].order**: Render order (lower numbers render first)
- **effects.[name].blend_mode**: "alpha" (default) or "additive" for glow effects
- **effects.[name].uniforms**: Effect-specific parameters

### Runtime Parameter Adjustment

Parameters can be overridden via environment variables:
```bash
# Override any shader uniform
export EXPERIMANCE_SHADER_EFFECTS_EFFECTS_SPARKS_UNIFORMS_SPARK_INTENSITY=0.8
export EXPERIMANCE_SHADER_EFFECTS_EFFECTS_TURBULENT_VIGNETTE_UNIFORMS_VIGNETTE_STRENGTH=0.9
```

## Development Workflow

### 1. Rapid Prototyping

Use the shader test harness for quick iteration:

```bash
cd services/display/shaders
uv run python shader_test_harness.py
```

This automatically loads all compatible shaders and provides:
- Interactive shader switching (Space/Backspace, 1-9 keys)
- Real-time uniform parameter adjustment (Q/W/E/T/Y/U keys)
- Automatic hot-reload support (A key)
- Test background with bright spots for spark effects
- Press H for complete controls list

### 2. Enhanced Testing

Test with realistic content:

```bash
uv run utils/examples/test_shader_effects.py
```

**Controls:**
- Space: Switch test modes
- 1-3: Adjust vignette strength
- 4-6: Adjust turbulence amount
- 7-9: Adjust spark intensity
- A: Toggle vignette animation
- S: Toggle turbulence animation

### 3. Integration Testing

Test integration with display service:

```bash
uv run utils/examples/shader_integration_example.py
```

### 4. Shader Development

Shaders are embedded in the renderer code for easy deployment. To modify effects:

1. Edit shader source in `shader_renderer.py`
2. Test with shader test harness
3. Refine with enhanced testing
4. Validate integration

## Technical Details

### Shader Pipeline

1. **Vertex Shader**: Standard full-screen quad processing
2. **Fragment Shader**: Per-pixel effects processing
3. **Uniforms**: Time, resolution, and effect parameters
4. **Textures**: Scene content as input texture

### Performance Considerations

- **Noise Sampling**: Optimized fractal noise with limited octaves
- **Spark Sampling**: Limited sample points to maintain frame rate
- **Single-Pass Rendering**: Combined effects use one shader pass
- **LOD**: Automatic quality scaling based on effect parameters

### OpenGL Requirements

- OpenGL 3.2+ (for shader support)
- Fragment shader support
- Texture sampling capabilities

## Integration with Display Service

### Layer System Integration

The ShaderRenderer integrates as a high-order layer in the layer management system:

```python
# Register shader renderer
shader_renderer = ShaderRenderer(config, window, batch, order=10)
layer_manager.register_renderer('shader_effects', shader_renderer)
```

### Rendering Order

1. Background layers (panorama, images)
2. Video overlays
3. Text overlays
4. **Shader effects** (post-processing)
5. Debug overlays

### Message Handling

The shader renderer can respond to configuration messages to adjust effects in real-time based on system state or user interaction.

## Best Practices

### Effect Design

1. **Subtlety**: Start with lower intensities and increase as needed
2. **Performance**: Monitor frame rates, especially with high turbulence
3. **Content Awareness**: Spark effects work best with content that has bright spots
4. **Animation**: Use time-based animations for organic, living effects

### Configuration

1. **Project-Specific**: Different projects may need different default intensities
2. **Hardware Scaling**: Consider reducing parameters on lower-end hardware
3. **Content Adaptation**: Adjust parameters based on content type

### Development

1. **Test Early**: Use the test harness throughout development
2. **Incremental**: Develop effects incrementally and test each change
3. **Documentation**: Comment shader code thoroughly for future maintenance

## Troubleshooting

### Common Issues

**Shader Compilation Errors:**
- Check OpenGL version compatibility
- Verify shader syntax with test harness
- Review uniform variable names and types

**Performance Issues:**
- Reduce turbulence sampling (fewer noise octaves)
- Lower spark sample count
- Decrease effect intensities

**Visual Artifacts:**
- Check texture coordinate handling
- Verify blend modes and alpha handling
- Review noise function implementations

**Integration Issues:**
- Verify layer order configuration
- Check render target setup
- Confirm uniform parameter passing

### Debug Tools

1. **Shader Test Harness**: Isolated shader testing
2. **Parameter Controls**: Real-time adjustment capabilities
3. **Performance Monitoring**: Frame rate and render time tracking
4. **Visual Debug**: Parameter visualization options

## Future Enhancements

### Potential Additions

1. **Heat Distortion**: Refraction-based heat shimmer effects
2. **Particle Systems**: More sophisticated particle behaviors
3. **Color Grading**: Post-processing color adjustment
4. **Bloom Effects**: Bright pixel enhancement
5. **Dynamic Parameters**: Content-reactive parameter adjustment

### Technical Improvements

1. **Framebuffer Pipeline**: Off-screen rendering for complex effects
2. **Multi-Pass Rendering**: Separate passes for different effects
3. **Compute Shaders**: GPU-accelerated particle simulation
4. **Temporal Effects**: Frame-to-frame coherent animations

## Examples

See the `utils/examples/` directory for complete working examples of shader development and integration.
