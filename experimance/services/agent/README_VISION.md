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

## Configuration

Enable vision processing in your agent configuration:

```toml
[agent.vision]
# Webcam settings
webcam_enabled = true
webcam_device_id = 0
webcam_width = 640
webcam_height = 480

# Audience detection with detector profiles
audience_detection_enabled = true
audience_detection_interval = 2.0
audience_detection_threshold = 0.5
detector_profile = "face_detection"  # Use optimized detector profile

# VLM settings
vlm_enabled = true
vlm_model = "moondream"
vlm_analysis_interval = 10.0
vlm_device = "cuda"  # or "cpu"
```

## Camera Optimization

### Camera Setup and Analysis

The vision system includes comprehensive camera analysis and optimization tools. Use these to get the best performance from your webcam:

```bash
# Analyze your camera capabilities
uv run python scripts/tune_detector.py --camera-name "EMEET" --camera-info

# This will show:
# - Device identification and supported formats
# - Current camera settings and available controls
# - OpenCV compatibility test
# - WebcamManager integration test
```

### Camera Profiles

Camera settings are automatically configured via detector profiles. Each profile includes optimized camera settings for different environments:

#### Face Detection Profile (Recommended)
- **Format**: YUYV 640x480 @ 30fps (optimal balance)
- **Exposure**: Aperture Priority Mode with dynamic framerate
- **Environment**: Mixed indoor lighting
- **Best for**: Gallery installations, seated audiences

#### Gallery Dim Profile
- **Format**: MJPG 1280x720 @ 30fps (better low-light)
- **Exposure**: Enhanced low-light settings
- **Environment**: Dim gallery/museum lighting
- **Best for**: Dark exhibition spaces

#### Indoor Office Profile
- **Format**: YUYV 640x480 @ 30fps (stable performance)
- **Exposure**: Fixed framerate for stable lighting
- **Environment**: Bright office/studio lighting
- **Best for**: Development and testing

### Camera Format Selection

**YUYV vs MJPG**: 
- **YUYV 640x480**: Lower CPU usage, faster processing, excellent for detection
- **MJPG 1280x720**: Higher quality, better for low light, more CPU intensive
- **Recommendation**: Use YUYV 640x480 for most installations

### Manual Camera Settings

If automatic camera profile application fails, you can manually configure camera settings:

```bash
# List your camera's available controls
v4l2-ctl --device=/dev/video0 --list-ctrls

# Apply manual settings for gallery environment
v4l2-ctl --device=/dev/video0 --set-ctrl=auto_exposure=3
v4l2-ctl --device=/dev/video0 --set-ctrl=backlight_compensation=50
v4l2-ctl --device=/dev/video0 --set-ctrl=gain=15
v4l2-ctl --device=/dev/video0 --set-ctrl=brightness=5
v4l2-ctl --device=/dev/video0 --set-ctrl=contrast=35

# Set format and resolution
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=YUYV
```

### Camera Testing and Troubleshooting

```bash
# Test with specific camera and profile
uv run python scripts/tune_detector.py --camera-name "EMEET" --profile face_detection

# Test different cameras by name pattern
uv run python scripts/tune_detector.py --camera-name "Logitech"
uv run python scripts/tune_detector.py --camera-name "HD.*Camera"

# List available cameras
v4l2-ctl --list-devices

# Test camera formats and resolutions
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### Environment-Specific Optimization

**Gallery/Museum Settings:**
- Use face_detection or gallery_dim profile
- Higher backlight compensation (40-70)
- Moderate gain (15-25)

**Office/Studio Settings:**
- Use indoor_office profile
- Lower gain (5-15) in bright environments
- Fixed framerate for stable conditions

**Outdoor/Bright Settings:**
- Lower gain and brightness
- Higher contrast for sun conditions
- Enable shadow detection
- Consider higher resolution for distant subjects

### Camera Profile Customization

You can create custom camera profiles by modifying existing ones in `services/agent/profiles/`:

```toml
[camera]
name = "Custom Gallery"
description = "Custom camera settings for my gallery"
environment = "gallery"
auto_apply = true

[camera.settings]
# Format settings
preferred_format = "YUYV"
preferred_width = 640
preferred_height = 480
preferred_fps = 30

# Exposure and lighting
auto_exposure = 3
exposure_dynamic_framerate = 1
backlight_compensation = 50
gain = 20

# Image quality
brightness = 10
contrast = 40
saturation = 45
sharpness = 3
gamma = 115
power_line_frequency = 2  # 60Hz (use 1 for 50Hz regions)
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
2. **VLM Analysis**: Uses Moondream to identify people in the scene (requires good/discrete GPU)
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
uv run python scripts/tune_detector.py

# Start with specific profile and camera
uv run python scripts/tune_detector.py --profile gallery_dim --camera 0

# List available profiles
uv run python scripts/tune_detector.py --list-profiles
```

The interactive tuner provides:
- Live webcam feed with detection overlays
- Real-time parameter adjustment via trackbars
- HOG detection bounding boxes and motion contours
- Save tuned parameters to new detector profiles

## Camera Performance Tips

### Getting the Best Detection Results

1. **Lighting Conditions**:
   - Avoid backlighting (windows behind subjects)
   - Ensure even lighting across the detection area
   - Use the camera profile's backlight compensation settings
   - Consider additional lighting for dim environments

2. **Camera Positioning**:
   - Mount camera at head height or slightly above
   - Ensure clear view of the audience area
   - Avoid obstructions in the camera's field of view
   - Position to minimize camera shake/vibration

3. **Format and Resolution**:
   - Use YUYV 640x480 for best detection performance
   - Higher resolutions don't always improve detection accuracy
   - Maintain consistent 30 FPS for smooth detection

4. **Real-time Tuning**:
   ```bash
   # Use interactive tuner for live optimization
   uv run python scripts/tune_detector.py --camera-name "EMEET" --profile face_detection
   
   # Adjust these key parameters:
   # - Face detection threshold (0.3-0.7)
   # - Motion sensitivity (500-2000)
   # - Detection confidence weights
   ```

### Troubleshooting Detection Issues

**Face Detection Not Working:**
```bash
# Check if face detection is enabled in profile
uv run python scripts/tune_detector.py --profile face_detection

# Verify YuNet model is available
ls models/face_detection_yunet_2023mar.onnx

# Test with higher sensitivity
# Lower face_score_threshold in profile (0.3-0.5)
```

**Motion Detection Too Sensitive:**
```bash
# Increase motion threshold in profile
# Typical values: 600-2000 (higher = less sensitive)

# Adjust camera gain and brightness to reduce noise
v4l2-ctl --device=/dev/video0 --set-ctrl=gain=10
v4l2-ctl --device=/dev/video0 --set-ctrl=brightness=0
```

**Camera Not Found:**
```bash
# List all video devices
v4l2-ctl --list-devices

# Test different device IDs
uv run python scripts/tune_detector.py --camera 0
uv run python scripts/tune_detector.py --camera 1

# Check camera permissions
sudo usermod -a -G video $USER
# (logout and login required)
```

**Poor Image Quality:**
```bash
# Check supported formats and choose best one
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Manual quality optimization
v4l2-ctl --device=/dev/video0 --set-ctrl=contrast=35
v4l2-ctl --device=/dev/video0 --set-ctrl=sharpness=3
v4l2-ctl --device=/dev/video0 --set-ctrl=saturation=45
```

## Performance Considerations

### CPU vs GPU

- **CPU**: Works for basic testing, but slower VLM inference (2-5 seconds per analysis)
- **GPU**: Recommended for real-time operation (0.1-0.5 seconds per analysis)

### Memory Usage

- Base system: ~500MB
- With Moondream loaded: ~2-3GB
- GPU VRAM: ~1-2GB (when using CUDA)

### Optimization Tips

1. **Use appropriate detector profiles** for your environment:
   - `face_detection`: Best for seated audiences, fast and accurate
   - `gallery_dim`: Optimized for low-light museum/gallery settings
   - `indoor_office`: Stable performance in bright, controlled lighting
   
2. **Camera format selection**:
   - YUYV 640x480: Fastest processing, recommended for most installations
   - MJPG 1280x720: Better quality for challenging lighting conditions
   
3. **Performance optimization**:
   - Reduce VLM analysis frequency for better performance
   - Use face detection instead of HOG for seated audiences
   - Apply camera profiles automatically for consistent settings
   - Monitor detection times via interactive tuner
   
4. **Environmental tuning**:
   - Use interactive tuner for real-time parameter adjustment
   - Save optimized settings as custom detector profiles
   - Test in actual installation lighting conditions

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
