# Depth Camera Integration

## Overview

The Core Service uses Intel RealSense depth cameras for real-time interaction detection. Two modules handle this:

- **`realsense_camera.py`** – Low-level hardware wrapper: frame acquisition, automatic error recovery, USB reset.
- **`depth_processor.py`** – High-level async pipeline: hand detection, change detection, `DepthFrame` output.

For development without hardware, `mock_depth_processor.py` reads grayscale image files and produces the same `DepthFrame` interface.

### When to use the mock processor

Use the mock processor whenever:
- Running in CI or on a machine without a RealSense camera.
- Developing new features that depend on depth input.
- Running unit/integration tests deterministically.

Use the real camera in production and for final calibration.

---

## Environment Assumptions

- Linux (Ubuntu 20.04+)
- USB 3.0 port with stable power
- `librealsense2` drivers installed (`sudo apt install librealsense2-*`)
- Python 3.11 (maximum compatible version while `pyrealsense2` lags upstream Python)

---

## Quick Start

```bash
# Run with real camera
uv run -m experimance_core

# Run with mock depth images (no hardware required)
mkdir -p media/images/mocks/depth
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth

# Test camera connection directly
uv run python services/core/tests/test_camera.py --info

# Test camera with visualization
uv run python services/core/tests/test_camera.py --real --visualize

# Test mock camera
uv run python services/core/tests/test_camera.py --mock --duration 30
```

---

## Configuration

Camera and processing settings live under the `[camera]` section in `projects/experimance/core.toml` (defaults in `services/core/config.toml`).

```toml
[camera]
resolution = [1280, 720]     # Depth camera resolution (width, height)
fps = 30                     # Camera frames per second
min_depth = 0.49             # Minimum depth in metres (sand surface bottom)
max_depth = 0.56             # Maximum depth in metres (sand surface top)
align_frames = true          # Align depth to colour frame
colorizer_scheme = 2         # 2 = white to black

flip_horizontal = true
flip_vertical = true
circular_crop = true
blur_depth = true

output_resolution = [1024, 1024]      # Processed output size
change_threshold = 10                 # Per-pixel depth difference threshold
significant_change_threshold = 0.006  # Whole-frame change threshold
detect_hands = true
crop_to_content = true
lightweight_mode = false     # Skip some steps for higher FPS
verbose_performance = false  # Show frame-level timing
debug_mode = false           # Include intermediate images

[depth_processing]
mock_depth_images_path = ""  # Non-empty → use MockDepthProcessor
```

### Camera error-recovery settings

| Key                         | Type  | Default | Description                                                 |
|-----------------------------|-------|---------|-------------------------------------------------------------|
| `camera.max_retries`        | int   | 3       | Attempts before giving up and setting `CameraState.ERROR`   |
| `camera.retry_delay`        | float | 2.0     | Initial backoff delay in seconds                            |
| `camera.max_retry_delay`    | float | 30.0    | Maximum backoff delay in seconds                            |
| `camera.aggressive_reset`   | bool  | false   | Use USB hardware reset on failure                           |
| `camera.skip_advanced_config` | bool | false | Skip advanced JSON config on re-init (faster recovery)      |

Override at runtime: `--camera-max-retries 5 --camera-aggressive-reset`.

---

## Data Structures

### `DepthFrame`

The unit of work flowing from the depth pipeline to the core service.

```python
@dataclass
class DepthFrame:
    depth_image: np.ndarray           # Processed, resized depth image
    color_image: Optional[np.ndarray] # Colour image (if available)
    hand_detected: Optional[bool]     # Hand/obstruction detected
    change_score: float               # Motion score [0–1]
    frame_number: int                 # Sequential counter
    timestamp: float                  # Capture timestamp (monotonic)

    @property
    def has_interaction(self) -> bool:
        return self.hand_detected or self.change_score > 0.1
```

### `CameraConfig` (Pydantic model in `config.py`)

```python
class CameraConfig(BaseModel):
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    align_frames: bool = True
    min_depth: float = 0.0
    max_depth: float = 10.0
    detect_hands: bool = True
    crop_to_content: bool = True
    change_threshold: int = 60
    output_resolution: Tuple[int, int] = (1024, 1024)
    lightweight_mode: bool = False
    verbose_performance: bool = False
    max_retries: int = 3
    retry_delay: float = 2.0
    max_retry_delay: float = 30.0
    aggressive_reset: bool = False
    skip_advanced_config: bool = False
```

---

## Usage

### Streaming frames in a service

```python
from experimance_core.config import CameraConfig
from experimance_core.depth_processor import DepthProcessor

async def _depth_processing_task(self):
    processor = DepthProcessor(self.config.camera)
    if not await processor.initialize():
        logger.error("Camera initialization failed")
        return
    async for frame in processor.stream_frames():
        await self._process_depth_frame(frame)
```

### Mock processor in tests

```python
from experimance_core.mock_depth_processor import MockDepthProcessor

processor = MockDepthProcessor(camera_config)
await processor.initialize()

# Control hand detection rate for deterministic tests
processor.set_hand_detection_rate(0.5)  # 50% of frames report a hand

async for frame in processor.stream_frames():
    assert isinstance(frame, DepthFrame)
    if frame.frame_number >= 10:
        break

processor.stop()
```

The mock processor reads 8-bit grayscale PNG/JPG files from `mock_depth_images_path`. Brighter pixels = closer depth. Images are resized to `camera.output_resolution` automatically.

### Patching the processor in service integration tests

```python
from unittest.mock import patch
from experimance_core.mock_depth_processor import MockDepthProcessor

with patch('experimance_core.experimance_core.create_depth_processor') as mock_factory:
    mock_factory.return_value = MockDepthProcessor(camera_config)
    service = ExperimanceCoreService(config=test_config)
    # run test
```

---

## Test Scripts

Located in `services/core/tests/` and `services/core/scripts/`.

### `tests/test_camera.py` – interactive camera test

```bash
# Show connected camera information
uv run python services/core/tests/test_camera.py --info

# Real camera, run until Ctrl+C
uv run python services/core/tests/test_camera.py --real

# Mock camera, 30-second timed run
uv run python services/core/tests/test_camera.py --mock --duration 30

# Visual debug window (6-panel composite; press 'q' to quit, 's' to save)
uv run python services/core/tests/test_camera.py --real --visualize

# Verbose performance logging
uv run python services/core/tests/test_camera.py --real --verbose
```

All options: `--mock | --real | --visualize | --info | --functions | --interactive | --duration N | --verbose`

### `scripts/recover_camera.py` – hardware recovery

Use when the camera is stuck and cannot be initialised:

```bash
uv run python services/core/scripts/recover_camera.py
```

Runs diagnostics, attempts an aggressive hardware reset, and verifies recovery.

### `scripts/break_camera.py` – intentional breakage for testing

```bash
uv run python services/core/scripts/break_camera.py
# Choose a breakage method, then run recover_camera.py to test recovery
```

### `scripts/test_camera_async.py` – async cancellation test

```bash
uv run python services/core/scripts/test_camera_async.py
# Press Ctrl+C at any point to test cancellation
```

### `scripts/debug_camera.py` – low-level diagnostics

```bash
uv run python services/core/scripts/debug_camera.py
```

---

## Error Handling

### Automatic recovery sequence

1. Camera operation fails (any exception).
2. Error is logged with type and attempt number.
3. If `aggressive_reset` is set, USB hardware reset is attempted.
4. Exponential backoff delay (`retry_delay` × 2^n, capped at `max_retry_delay`).
5. Retry up to `max_retries` times.
6. On final failure: `CameraState.ERROR` is set and the service health report reflects this.

### Common error messages

| Error message                       | Likely cause                                | Fix                                         |
|-------------------------------------|---------------------------------------------|---------------------------------------------|
| `Couldn't resolve requests`         | Camera configuration mismatch               | Check `[camera]` config, try lower FPS      |
| `Device or resource busy`           | Another process holds the camera            | `lsof \| grep uvcvideo`; kill the process   |
| `get_xu(ctrl=1) failed`             | Advanced mode config error on reconnect     | Set `camera.skip_advanced_config = true`    |
| Frame capture timeout               | USB bandwidth saturated                     | Reduce resolution, disconnect other USB devices |
| `Failed to initialize camera`       | Camera not detected                         | Check USB 3.0 connection; run `rs-enumerate-devices` |

---

## Troubleshooting

**Camera not detected**

```bash
rs-enumerate-devices            # List connected RealSense devices
lsof | grep uvcvideo            # Check for processes using the camera
sudo usermod -a -G dialout $USER  # Add user to camera permissions group (then re-login)
```

**Low frame rate**

```toml
[camera]
resolution = [640, 480]
lightweight_mode = true
```

Or pass `--camera-lightweight-mode` at runtime.

**Camera won't recover**

```bash
# Try manual recovery script
uv run python services/core/scripts/recover_camera.py

# If that fails, physically reconnect USB and run:
uv run python services/core/tests/test_camera.py --info
```

**Visualization window doesn't appear**

Check `DISPLAY` env var is set (required for OpenCV GUI). On a headless system, use `--mock` mode without `--visualize`.

---

## Performance

| Mode       | Resolution | Typical FPS | Notes                         |
|------------|-----------|-------------|-------------------------------|
| Real camera | 1280×720 | ~30         | Production setting            |
| Real camera | 640×480  | ~30         | Faster processing             |
| Mock       | any       | ~30         | CPU limited, no hardware wait |

Enable `verbose_performance = true` (or `--camera-verbose-performance`) to see per-frame timing in logs.

---

## Files Touched

| File                                       | Role                              |
|--------------------------------------------|-----------------------------------|
| `src/experimance_core/realsense_camera.py` | Hardware wrapper, retry logic     |
| `src/experimance_core/depth_processor.py`  | Async pipeline, `DepthFrame`      |
| `src/experimance_core/mock_depth_processor.py` | File-based mock               |
| `src/experimance_core/depth_factory.py`    | Selects real vs mock processor    |
| `src/experimance_core/depth_visualizer.py` | OpenCV debug visualisation        |
| `src/experimance_core/depth_utils.py`      | Image utility functions           |
| `src/experimance_core/camera_utils.py`     | USB / hardware utilities          |
| `src/experimance_core/config.py`           | `CameraConfig` Pydantic model     |
| `tests/test_camera.py`                     | Interactive camera test script    |
| `tests/test_depth_integration.py`          | Integration tests                 |
| `tests/test_realsense_camera_mocked.py`    | Mocked camera unit tests          |
| `scripts/recover_camera.py`               | Hardware recovery tool            |
| `scripts/break_camera.py`                 | Intentional-breakage test tool    |
| `scripts/debug_camera.py`                 | Low-level diagnostics             |
| `scripts/test_camera_async.py`            | Async cancellation tests          |
