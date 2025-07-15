# Vision Processing for Experimance Agent Service

This document describes the vision processing capabilities of the Experimance Agent Service, including webcam capture, audience detection, and Vision Language Model (VLM) analysis.

## Overview

The vision system consists of three main components:

1. **WebcamManager**: Handles webcam capture and basic image preprocessing
2. **VLMProcessor**: Provides scene understanding using Vision Language Models (Moondream)
3. **AudienceDetector**: Combines motion detection and VLM analysis for robust audience detection

## Installation

### Basic Requirements

```bash
# Install with vision support
uv sync --extra vision-models

# For GPU support (recommended)
uv sync --extra vision-gpu --extra vision-models
```

### System Dependencies

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3-opencv libopencv-dev
```

On macOS:
```bash
brew install opencv
```

## Configuration

Enable vision processing in your agent configuration:

```toml
[agent.vision]
# Webcam settings
webcam_enabled = true
webcam_device_id = 0
webcam_width = 640
webcam_height = 480

# Audience detection
audience_detection_enabled = true
audience_detection_interval = 2.0
audience_detection_threshold = 0.5

# VLM settings
vlm_enabled = true
vlm_model = "moondream"
vlm_analysis_interval = 10.0
vlm_device = "cuda"  # or "cpu"
```

## Features

### Webcam Capture

- Asynchronous frame capture
- Configurable resolution and frame rate
- Automatic preprocessing for different use cases
- Frame saving for debugging

### Audience Detection

The audience detection system uses a multi-modal approach:

1. **Motion Detection**: Uses background subtraction to detect movement
2. **VLM Analysis**: Uses Moondream to identify people in the scene
3. **Stability Filtering**: Combines results over time to reduce false positives

Detection confidence is calculated based on:
- Motion intensity and consistency
- VLM confidence in people detection
- Historical stability of detections

### Vision Language Model Analysis

- Scene description generation
- Audience presence detection
- Interaction context analysis
- Configurable analysis prompts

Currently supported models:
- **Moondream2**: Lightweight, efficient VLM for real-time analysis

## Testing

Test the vision components independently:

```bash
# From the agent service directory
uv run python tests/test_vision.py
```

This will test:
- Webcam capture functionality
- VLM model loading and inference
- Integrated audience detection

### Additional Test Scripts

```bash
# Test CPU-optimized detection performance
uv run python tests/test_cpu_detection.py

# Test complete vision
uv run python tests/test_vision.py

# Live detection test with colorful output (great for testing from different locations)
uv run python tests/test_cpu_live_detection.py --mode accurate
uv run python tests/test_cpu_live_detection.py --mode fast
uv run python tests/test_cpu_live_detection.py --mode balanced
```

### Live Detection Testing

The `test_cpu_live_detection.py` script provides a continuous detection test with large, colorful output that's visible from across the room:

**Features:**
- Large ASCII art status display (AUDIENCE DETECTED / NO AUDIENCE)
- Color-coded confidence levels and detection bars
- Real-time performance metrics (detection time, FPS)
- Detailed breakdowns of person and motion detection
- Three performance modes: `fast`, `balanced`, `accurate`

**Usage:**
```bash
# Default accurate mode
uv run python tests/test_cpu_live_detection.py

# Fast mode for high frame rates
uv run python tests/test_cpu_live_detection.py --mode fast

# Balanced mode for good speed/accuracy
uv run python tests/test_cpu_live_detection.py --mode balanced
```

**Testing Tips:**
- Move around in front of the camera to test motion detection
- Stand still to test static person detection  
- Leave the frame to test absence detection
- Use Ctrl+C to stop and view final statistics

### Interactive Parameter Tuning

For real-time parameter adjustment with visual feedback, use the interactive tuning tool:

```bash
# Run interactive tuner with default profile
uv run python tune_detector.py

# Start with specific profile and camera
uv run python tune_detector.py --profile gallery_dim --camera 0

# List available profiles
uv run python tune_detector.py --list-profiles
```

The interactive tuner provides:
- Live webcam feed with detection overlays
- Real-time parameter adjustment via trackbars
- HOG detection bounding boxes and motion contours
- Save tuned parameters to new detector profiles

See [scripts/README.md](../../scripts/README.md) for more development tools.

## Performance Considerations

### CPU vs GPU

- **CPU**: Works for basic testing, but slower VLM inference (2-5 seconds per analysis)
- **GPU**: Recommended for real-time operation (0.1-0.5 seconds per analysis)

### Memory Usage

- Base system: ~500MB
- With Moondream loaded: ~2-3GB
- GPU VRAM: ~1-2GB (when using CUDA)

### Optimization Tips

1. **Reduce VLM analysis frequency** for better performance
2. **Use motion detection primarily** and VLM for verification
3. **Resize images** before VLM processing (configured via `vlm_max_image_size`)
4. **Adjust detection thresholds** based on your installation environment

## Troubleshooting

### Common Issues

1. **Webcam not detected**:
   - Check `webcam_device_id` in configuration
   - Ensure camera is not in use by other applications
   - Try different device IDs (0, 1, 2, etc.)

2. **VLM loading fails**:
   - Ensure sufficient RAM/VRAM available
   - Try CPU mode if GPU issues
   - Check internet connection for model downloads

3. **Poor detection accuracy**:
   - Adjust `audience_detection_threshold`
   - Increase `motion_threshold` in code for less sensitive motion detection
   - Ensure good lighting conditions

### Debug Information

Enable detailed logging:

```python
import logging
logging.getLogger("experimance_agent.vision").setLevel(logging.DEBUG)
```

Check vision status via agent debug API:

```python
status = await agent.get_debug_status()
print(status["vision"])
```

## Integration with Agent Backend

The vision system integrates with the conversation AI by:

1. **Audience Detection Events**: Published via ZMQ when audience presence changes
2. **Scene Context**: VLM analysis results sent to conversation backend as system messages
3. **Transcript Display**: Coordinates with conversation state for appropriate text display

## Future Enhancements

- Support for additional VLM models (LLaVA, CLIP, etc.)
- Face recognition and tracking
- Gesture detection
- Multi-camera support
- Real-time pose estimation
- Integration with depth cameras

## API Reference

### WebcamManager

```python
# Initialize and start
webcam = WebcamManager(config)
await webcam.start()

# Capture frames
frame = await webcam.capture_frame()
rgb_frame = webcam.preprocess_for_vlm(frame)

# Get status
info = webcam.get_capture_info()
```

### VLMProcessor

```python
# Initialize with model
vlm = VLMProcessor(config)
await vlm.start()

# Analyze scenes
result = await vlm.analyze_scene(rgb_frame, "scene_description")
audience_detected = await vlm.detect_audience(rgb_frame)

# Get analysis history
last_analysis = vlm.get_last_analysis()
```

### AudienceDetector

```python
# Initialize detector
detector = AudienceDetector(config)
await detector.start()

# Detect audience
result = await detector.detect_audience(frame, webcam, vlm)
stats = detector.get_detection_stats()
```
