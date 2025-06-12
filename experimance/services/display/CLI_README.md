# Display Service CLI Tool

This CLI tool allows you to manually test the display service by sending various ZMQ messages. It's useful for development, debugging, and testing display functionality without running the full Experimance system.

## Usage

All commands should be run using `uv run` from the display service directory:

```bash
cd /path/to/experimance/services/display
uv run cli.py <command> [options]
```

## Available Commands

### Basic Commands

#### List Available Resources
```bash
uv run cli.py list
```
Shows all available test images, mock files, and videos that can be used for testing.

#### Display an Image
```bash
uv run cli.py image <image_path>
uv run cli.py image /path/to/image.webp --type satellite_landscape
```
Sends an `ImageReady` message to display the specified image.

#### Show Text Overlay
```bash
uv run cli.py text "Hello World!"
uv run cli.py text "System ready" --speaker system --duration 10.0 --position top_center
```
Sends a `TextOverlay` message. Options:
- `--id`: Custom text ID (auto-generated if not provided)
- `--speaker`: agent, system, or debug (default: system)
- `--duration`: Duration in seconds (default: infinite)
- `--position`: Text position (default: bottom_center)

#### Remove Text Overlay
```bash
uv run cli.py remove-text <text_id>
```
Removes a specific text overlay by ID.

#### Send Video Mask
```bash
uv run cli.py video-mask /path/to/mask.png
uv run cli.py video-mask /path/to/mask.png --fade-in 0.5 --fade-out 2.0
```
Sends a `VideoMask` message for overlay masking.

#### Send Era Change Event
```bash
uv run cli.py era-change wilderness forest
uv run cli.py era-change anthropocene urban
```
Sends an `EraChanged` event with era and biome parameters.

### Advanced Commands

#### Send Transition Ready
```bash
uv run cli.py transition /path/to/transition.mp4 --from-image prev --to-image next
```
Sends a `TransitionReady` message for custom transitions between images.

#### Send Loop Ready
```bash
uv run cli.py loop /path/to/loop.mp4 "file:///path/to/still.webp" --type idle_animation
```
Sends a `LoopReady` message for animated loops of still images.

#### Cycle Through Images
```bash
uv run cli.py cycle-images
uv run cli.py cycle-images /custom/directory --interval 3.0
```
Continuously cycles through available images. Use Ctrl+C to stop.

#### Run Interactive Demo
```bash
uv run cli.py demo
```
Runs a comprehensive demo showing various display features:
1. Text overlays with different speakers
2. Era change events
3. Image cycling
4. Video mask demonstration
5. Text removal
6. Transition effects

## Example Workflows

### Testing Text Overlays
```bash
# Show agent message
uv run cli.py text "Welcome to Experimance" --speaker agent --duration 5.0

# Show system status
uv run cli.py text "System Status: Running" --speaker system --position top_right

# Show debug info (stays visible)
uv run cli.py text "FPS: 60.0 | GPU: 45%" --speaker debug --position top_left --id debug_overlay

# Remove debug info
uv run cli.py remove-text debug_overlay
```

### Testing Image Display
```bash
# List available images
uv run cli.py list

# Display a specific image
uv run cli.py image /path/to/generated/image.webp

# Cycle through all images slowly
uv run cli.py cycle-images --interval 5.0
```

### Testing Video Masks
```bash
# Apply video mask
uv run cli.py video-mask /path/to/mock_video_mask.png

# Apply with custom fade timing
uv run cli.py video-mask /path/to/mask.png --fade-in 1.0 --fade-out 3.0
```

### Testing Era Changes
```bash
# Test different eras and biomes
uv run cli.py era-change wilderness forest
uv run cli.py era-change anthropocene urban
uv run cli.py era-change rewilded grassland
```

## ZMQ Ports Used

The CLI publishes to these default ports:
- **Images**: 5558 (ImageReady, TransitionReady, LoopReady)
- **Events**: 5555 (EraChanged)
- **Display Control**: 5560 (TextOverlay, RemoveText, VideoMask)

## Tips

1. **Run with display service**: Start the actual display service in another terminal to see the visual results
2. **Use absolute paths**: The CLI converts relative paths to absolute paths and file URIs automatically
3. **Check available resources**: Use `uv run cli.py list` to see what test files are available
4. **Monitor logs**: The CLI shows info-level logs to confirm messages are sent
5. **Interrupt long commands**: Use Ctrl+C to stop cycling or demo commands

## Troubleshooting

### No images listed
- Make sure you're in the correct directory structure
- Check that `media/images/generated/` and `media/images/mocks/` directories exist
- Verify image files have supported extensions (.webp, .png, .jpg)

### ZMQ binding errors
- Ensure no other services are already bound to the same ports
- Check that ZMQ is properly installed
- Try running individual commands rather than long-running ones

### File not found errors
- Use absolute paths or ensure files exist relative to project root
- Check file permissions
- Verify file extensions match what the CLI expects
