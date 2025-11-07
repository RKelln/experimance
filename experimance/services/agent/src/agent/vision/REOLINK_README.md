# Reolink Camera Integration for Experimance Agent

This module provides Reolink IP camera integration for the Experimance Agent service, offering reliable audience detection using the camera's built-in AI capabilities.

## Features

- **Simple & Reliable**: Uses camera's hardware AI detection (much more reliable than computer vision)
- **Binary Interface**: Clean boolean presence detection
- **Fast Response**: ~50-100ms detection time vs seconds for computer vision
- **State Management**: Includes stability checking and state change detection
- **Comprehensive Discovery**: Intelligent camera discovery with progressive fallback
- **Full Integration**: Matches existing audience detector interface

## Implementation Files

1. **`reolink_detector.py`** - Modern async Reolink detector class
2. **`reolink_discovery.py`** - Comprehensive camera discovery system
3. **`../../scripts/list_cameras.py`** - Camera discovery CLI tool
4. **`../../scripts/test_reolink_camera.py`** - Comprehensive testing and control script
5. **Updated `__init__.py`** - Reolink exports

## Quick Start

### 1. Discovery

Find your camera automatically:

```bash
# Quick discovery across network
uv run python scripts/list_cameras.py

# Comprehensive discovery with signature verification
uv run python scripts/list_cameras.py --comprehensive

# Test specific camera
uv run python scripts/test_reolink_camera.py --host 192.168.2.229 --user admin --password your_password --status
```

### 2. Configuration

Add to your `config.toml`:

```toml
[vision]
detection_method = "reolink"
reolink_enabled = true
reolink_host = "192.168.2.229"  # Your camera IP (optional - will auto-discover)
reolink_user = "admin"
# put password in .env
reolink_https = true
audience_detection_interval = 1.0
```

### 3. Basic Usage

```python
from agent.vision import ReolinkDetector
from agent.config import VisionConfig

config = VisionConfig(detection_method="reolink", reolink_host="192.168.2.229", ...)

# Auto-discovery method (recommended)
detector = await ReolinkDetector.create_with_discovery(
    user="admin", 
    password="your_password",
    known_host=config.reolink_host  # optional
)

# Direct creation method
detector = ReolinkDetector(host="192.168.2.229", user="admin", password="your_password")
await detector.initialize()

# Detect audience
result = await detector.detect_audience()
print(f"Person present: {result['audience_detected']}")
```

## Integration with Existing System

The Reolink detector implements the same interface as the existing CPU detector:

- **Same method signatures**: `detect_audience()`, `get_detection_stats()`, etc.
- **Same result format**: Returns identical data structure
- **Same state management**: Includes stability checking and change detection
- **Auto-Discovery Integration**: Seamless camera discovery with intelligent fallback
- **Drop-in replacement**: Can replace CPU detector with minimal code changes

## Discovery System

The comprehensive discovery system provides three strategies:

### 1. Fast Discovery (`discover_reolink_cameras_fast`)
- Quick port scan across network subnet
- Tests port 80 and 443 only
- Returns potential cameras in ~5-10 seconds
- Good for known networks

### 2. Signature Discovery (`discover_reolink_cameras_signature`)
- Credential-free camera verification
- Tests HTTP response patterns specific to Reolink
- More reliable than port scanning alone
- Takes longer but more accurate

### 3. Comprehensive Discovery (`discover_reolink_cameras_comprehensive`)
- **Intelligent Progressive Strategy**: 
  1. If known host provided → test that first
  2. If not found → fast port scan
  3. Then signature verification on candidates
- **No Credential Broadcasting**: Never sends passwords during discovery
- **Security Conscious**: Escalates from fastest to most thorough
- **Recommended Method**: Use `ReolinkDetector.create_with_discovery()`

## Benefits vs CPU Detection

| Feature           | Reolink Camera               | CPU Computer Vision            |
| ----------------- | ---------------------------- | ------------------------------ |
| **Reliability**   | ✅ Hardware AI, very accurate | ⚠️ Variable, lighting dependent |
| **Speed**         | ✅ ~50ms response time        | ❌ 200-1000ms processing        |
| **CPU Usage**     | ✅ Minimal                    | ❌ High CPU load                |
| **Lighting**      | ✅ Works in all conditions    | ❌ Struggles in poor light      |
| **Setup**         | ✅ Auto-discovery available   | ❌ Complex tuning required      |
| **Seated People** | ✅ Detects well               | ⚠️ Can miss seated people       |

## Configuration Options

The following options are available in `VisionConfig`:

- `reolink_enabled: bool` - Enable Reolink detection
- `reolink_host: str` - Camera IP address (optional for auto-discovery)
- `reolink_user: str` - Camera username (default: "admin")
- `reolink_https: bool` - Use HTTPS (default: True)
- `reolink_channel: int` - Camera channel (default: 0)
- `reolink_timeout: int` - Request timeout (default: 10)
- put password in `.env` file in the project path

## Detection Methods

The `detection_method` config supports:
- `"cpu"` - OpenCV computer vision only
- `"vlm"` - Vision Language Model only  
- `"hybrid"` - Both CPU and VLM
- `"reolink"` - Reolink camera AI only

## Camera Compatibility

**Tested Models:**
- **Reolink RLC-820A** (firmware v3.1.0.2368_23062508)

**Requirements:**
Should work with most Reolink cameras that support:
- AI person detection (`people` feature)
- HTTP API access
- Modern firmware (2023+)

## Testing & Development Tools

### Scripts Available

1. **`scripts/list_cameras.py`** - Comprehensive discovery tool
   ```bash
   # Quick scan
   uv run python scripts/list_cameras.py
   
   # Full comprehensive scan with signature verification
   uv run python scripts/list_cameras.py --comprehensive
   ```

2. **`scripts/test_reolink_camera.py`** - Full camera testing suite
   ```bash
   # Basic presence monitoring
   uv run python scripts/test_reolink_camera.py --host IP --user admin --password PASS
   
   # Camera status and capabilities
   uv run python scripts/test_reolink_camera.py --host IP --user admin --password PASS --status
   
   # Camera control (stealth mode, IR lights, etc.)
   uv run python scripts/test_reolink_camera.py --host IP --user admin --password PASS --camera-off
   ```

## Advanced Usage

### Hybrid Setup Example

For maximum reliability with CPU fallback:

```python
# Custom hybrid implementation combining Reolink + CPU
try:
    detector = await ReolinkDetector.create_with_discovery(user="admin", password="pass")
    print("Using Reolink camera detection")
except Exception as e:
    print(f"Camera unavailable, falling back to CPU: {e}")
    detector = CPUAudienceDetector(config)
    await detector.initialize()
```

### Production Integration

```python
from agent.vision import ReolinkDetector
import asyncio

async def main():
    # Let auto-discovery find the camera
    detector = await ReolinkDetector.create_with_discovery(
        user="admin", 
        password="your_password"
    )
    
    # Monitor continuously
    while True:
        result = await detector.detect_audience()
        if result['audience_detected']:
            print(f"Person detected!")
        
        await asyncio.sleep(1.0)
```

## Security Notes

- **Create dedicated camera user** instead of using admin account
- **Use HTTPS** when possible (enabled by default)
- **Camera uses self-signed SSL** certificates (warnings automatically disabled)
- **Consider network segmentation** for camera access
- **Discovery is credential-free** - no passwords sent during network scanning

## Next Steps

1. **Discover cameras**: Use `scripts/list_cameras.py` to find available cameras
2. **Test setup**: Use `scripts/test_reolink_camera.py` to validate camera functionality
3. **Update config**: Add Reolink settings to your project's `config.toml`
4. **Integration**: Use `ReolinkDetector.create_with_discovery()` in your agent code
5. **Monitoring**: Monitor performance with `get_detection_stats()`

The integration provides a much more reliable and efficient audience detection solution compared to computer vision methods, with intelligent auto-discovery and comprehensive testing tools!
