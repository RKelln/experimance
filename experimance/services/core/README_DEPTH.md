# Depth Camera Integration for Experimance Core Service

## Overview

The Experimance Core Service uses Intel RealSense depth cameras for real-time user interaction detection. The `robust_camera.py` module provides a modern, async/await-based interface that replaces the legacy `depth_finder.py` implementation with robust error handling, retry logic, and comprehensive testing support.

## Architecture

### Core Components

1. **`robust_camera.py`** - Modern camera interface with:
   - Async/await support for seamless service integration
   - Comprehensive error handling with automatic retry and hardware reset
   - Mock support for development and testing
   - Type hints and dataclasses for clean code structure

2. **Configuration System** - Centralized camera and processing settings
3. **Processing Pipeline** - Real-time depth frame analysis and interaction detection
4. **Integration Layer** - ZMQ event publishing for service coordination

### Data Flow

```
RealSense Camera → robust_camera.py → DepthProcessor → Core Service →
  ↓
Interaction Detection → State Updates → Era Progression → 
  ↓
ZMQ Events → Audio/Display/Image Services
```

## Key Features

### 🚀 Modern Design
- **Type hints** throughout for better IDE support and code clarity
- **Dataclasses** for clean configuration and data structures  
- **Async/await** for seamless integration with the core service
- **Enum-based state management** for clear operational states

### 🛡️ Robust Error Handling
- **Automatic retry** with exponential backoff on camera errors
- **Hardware reset** functionality when operations fail
- **Graceful degradation** - continues operating even after errors
- **Comprehensive logging** for debugging and monitoring

### 🧪 Testing & Development
- **Mock camera support** for development without hardware
- **Configurable test parameters** for various scenarios
- **Isolated testing** - tests don't affect real camera operation
- **Performance validation** - FPS and processing benchmarks

## Configuration

### Camera Settings

```python
@dataclass
class CameraConfig:
    # Camera hardware settings
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    align_frames: bool = True
    min_depth: float = 0.0
    max_depth: float = 10.0
    
    # Processing settings
    detect_hands: bool = True
    crop_to_content: bool = True
    change_threshold: int = 60
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Performance tuning
    lightweight_mode: bool = False
    verbose_performance: bool = False
```

### Service Configuration

Configure camera settings in `services/core/config.toml`:

```toml
[depth_processing]
camera_config_path = ""     # Optional advanced mode JSON
resolution = [640, 480]     # Validated working resolution
fps = 30                   # Target frame rate
change_threshold = 50      # Interaction detection threshold
min_depth = 0.49          # Minimum depth (meters)
max_depth = 0.56          # Maximum depth (meters)
output_size = [1024, 1024] # Processed output size
```

## Usage

### Basic Camera Operation

```python
import asyncio
from experimance_core.robust_camera import CameraConfig, DepthProcessor

async def main():
    # Create configuration
    config = CameraConfig(
        resolution=(640, 480),
        fps=30,
        detect_hands=True,
        max_retries=3
    )
    
    # Create processor
    processor = DepthProcessor(config)
    
    # Initialize camera
    if await processor.initialize():
        print("Camera ready!")
        
        # Process frames
        async for frame in processor.stream_frames():
            print(f"Frame {frame.frame_number}: hands={frame.hand_detected}")
            print(f"Interaction score: {frame.change_score:.2f}")
            
            if frame.frame_number >= 100:
                break
    else:
        print("Failed to initialize camera")
    
    # Cleanup
    processor.stop()

asyncio.run(main())
```

### Integration with Core Service

```python
class ExperimanceCoreService:
    def __init__(self):
        # Create camera config from service config
        self.camera_config = CameraConfig(
            resolution=tuple(self.config.depth_processing.resolution),
            fps=self.config.depth_processing.fps,
            change_threshold=self.config.depth_processing.change_threshold,
            min_depth=self.config.depth_processing.min_depth,
            max_depth=self.config.depth_processing.max_depth
        )
        
        self.depth_processor = DepthProcessor(self.camera_config)
    
    async def _depth_processing_task(self):
        """Background task for continuous depth processing."""
        if not await self.depth_processor.initialize():
            logger.error("Failed to initialize camera")
            return
        
        async for frame in self.depth_processor.stream_frames():
            await self._process_depth_frame(frame)
    
    async def _process_depth_frame(self, frame: DepthFrame):
        """Process a depth frame and update service state."""
        # Update interaction state
        self.hand_detected = frame.hand_detected
        self.interaction_score = frame.change_score
        
        # Publish events based on interaction
        if frame.has_interaction:
            await self._publish_interaction_events(frame)
```

### Using Mock Camera for Development

```python
from experimance_core.robust_camera import MockDepthProcessor

# Create mock processor for testing
config = CameraConfig(fps=10)  # Faster for testing
mock_processor = MockDepthProcessor(config)

await mock_processor.initialize()

async for frame in mock_processor.stream_frames():
    assert isinstance(frame, DepthFrame)
    assert frame.depth_image.shape == config.output_resolution
    
    if frame.frame_number >= 10:
        break

mock_processor.stop()
```

## Data Structures

### DepthFrame

```python
@dataclass
class DepthFrame:
    depth_image: np.ndarray              # Processed depth image
    color_image: Optional[np.ndarray]    # Optional color image
    hand_detected: Optional[bool]        # Hand presence detection
    change_score: float                  # Interaction change score [0-1]
    frame_number: int                    # Sequential frame counter
    timestamp: float                     # Frame capture timestamp
    
    @property
    def has_interaction(self) -> bool:
        """True if frame indicates user interaction."""
        return self.hand_detected or self.change_score > 0.1
```

## Event Publishing

The depth processor integrates with the service's ZMQ event system:

### AudioCommand Events
```json
{
  "type": "AudioCommand",
  "trigger": "interaction_start",
  "hand_detected": true,
  "timestamp": "2025-06-14T21:25:35.925Z"
}
```

### VideoMask Events
```json
{
  "type": "VideoMask", 
  "interaction_score": 0.75,
  "depth_difference_score": 0.45,
  "hand_detected": true,
  "timestamp": "2025-06-14T21:25:35.925Z"
}
```

### RenderRequest Events
```json
{
  "type": "RenderRequest",
  "current_era": "modern",
  "current_biome": "temperate_forest", 
  "interaction_score": 0.8,
  "seed": 1623456789,
  "timestamp": "2025-06-14T21:25:35.925Z"
}
```

## Testing

### Test Scripts

#### Comprehensive Testing
```bash
# Run full camera test suite
cd services/core
python tests/test_camera.py

# Test with mock camera (no hardware required)
python tests/test_camera.py --mock

# Test with verbose performance logging
python tests/test_camera.py --verbose

# Test specific duration
python tests/test_camera.py --duration 30
```

#### Quick Validation
```bash
# Test camera import and basic functionality
python -c "
import asyncio
from experimance_core.robust_camera import CameraConfig, DepthProcessor

async def test():
    config = CameraConfig(resolution=(640, 480))
    processor = DepthProcessor(config)
    
    if await processor.initialize():
        print('✅ Camera initialized successfully')
        frame = await processor.get_processed_frame()
        if frame:
            print(f'✅ Frame captured: {frame.depth_image.shape}')
        else:
            print('❌ Failed to capture frame')
    else:
        print('❌ Failed to initialize camera')
    
    processor.stop()

asyncio.run(test())
"
```

### Test Configuration

The test script supports several options:

- `--mock`: Use mock camera instead of real hardware
- `--verbose`: Enable verbose performance logging  
- `--duration N`: Run test for N seconds (default: 10)
- `--fps N`: Target FPS for testing (default: 30)

### Performance Validation

The test script validates:
- **FPS Performance**: Achieves target frame rate
- **Processing Latency**: Frame processing time < 33ms
- **Memory Usage**: Stable memory usage over time
- **Error Recovery**: Handles simulated failures correctly

## Error Handling

### Automatic Recovery

The robust camera handles these common scenarios:

1. **"Couldn't resolve requests"** - Camera configuration issues
2. **"Device or resource busy"** - Camera in use by another process
3. **"get_xu(ctrl=1) failed"** - Advanced mode configuration errors
4. **Timeout errors** - Camera not responding
5. **Frame processing errors** - Corrupted data

### Recovery Process

1. **Detect Error** - Any camera operation failure
2. **Log Details** - Record error type and attempt number  
3. **Hardware Reset** - Reset camera via USB
4. **Wait** - Exponential backoff delay
5. **Retry** - Attempt operation again
6. **Escalate** - Report failure after max retries

### Retry Configuration

```python
config = CameraConfig(
    max_retries=5,        # Try up to 5 times
    retry_delay=2.0,      # Start with 2 second delays
    max_retry_delay=30.0  # Cap delays at 30 seconds
)
```

## Performance Optimization

### Production Settings
```python
production_config = CameraConfig(
    resolution=(640, 480),      # Validated working resolution
    fps=30,                     # Smooth operation
    max_retries=5,              # Allow multiple attempts
    retry_delay=2.0,            # Reasonable delay
    detect_hands=True,          # Enable interaction detection
    crop_to_content=True,       # Optimize output
    lightweight_mode=False,     # Full feature set
    verbose_performance=False   # Minimal logging
)
```

### Development Settings
```python
dev_config = CameraConfig(
    resolution=(320, 240),      # Faster processing
    fps=15,                     # Lower CPU usage
    max_retries=2,              # Fail fast for debugging
    retry_delay=1.0,            # Quick iteration
    lightweight_mode=True,      # Skip expensive operations
    verbose_performance=True    # Detailed performance logs
)
```

### Performance Characteristics

- **Real Camera**: ~30 FPS with (640, 480) resolution
- **Mock Camera**: ~60+ FPS (limited by processing)
- **Memory Usage**: Optimized with efficient frame processing
- **CPU Usage**: Configurable based on resolution and features
- **Error Recovery**: Minimal overhead during normal operation

## Troubleshooting

### Common Issues

**"Failed to initialize camera"**
- Check camera connection and USB 3.0 port
- Verify camera not in use by another process
- Try different USB port or cable
- Check camera permissions and drivers

**"Camera reset failed"**  
- Disconnect and reconnect camera physically
- Restart the application
- Check system USB drivers
- Verify camera power supply

**"Frame capture timeout"**
- Reduce resolution or frame rate
- Check USB bandwidth (disconnect other devices)
- Verify stable camera power
- Check for USB power management settings

**"Mock camera not working"**
- Verify mock configuration is valid
- Check test script permissions
- Review logs for mock-specific errors

### Debug Mode

```python
# Enable detailed logging
import logging
logging.getLogger('robust_camera').setLevel(logging.DEBUG)

# Use debug configuration
debug_config = CameraConfig(
    resolution=(320, 240),  # Small for debugging
    fps=5,                  # Slow for analysis
    max_retries=1,          # Fail fast
    verbose_performance=True # Show all timing info
)
```

### Performance Monitoring

```python
import time

frame_times = []
async for frame in processor.stream_frames():
    start_time = time.time()
    await process_frame(frame)
    process_time = time.time() - start_time
    frame_times.append(process_time)
    
    if process_time > 0.1:  # 100ms threshold
        logger.warning(f"Slow frame processing: {process_time:.3f}s")
    
    # Report average every 100 frames
    if len(frame_times) >= 100:
        avg_time = sum(frame_times) / len(frame_times)
        logger.info(f"Average processing time: {avg_time:.3f}s")
        frame_times.clear()
```

## Migration from Legacy Code

### From depth_finder.py

**Old Code:**
```python
from experimance_core.depth_finder import depth_generator

def _create_depth_generator_factory(self):
    def depth_factory():
        return depth_generator(
            json_config=config.camera_config_path,
            size=tuple(config.resolution),
            # ... many parameters
        )
    return depth_factory

self.depth_generator = self._async_depth_wrapper(depth_factory())
```

**New Code:**
```python
from experimance_core.robust_camera import DepthProcessor, CameraConfig

self.camera_config = CameraConfig(
    json_config_path=config.camera_config_path,
    resolution=tuple(config.resolution),
    fps=config.fps,
    detect_hands=True,
    max_retries=3
)

self.depth_processor = DepthProcessor(self.camera_config)
await self.depth_processor.initialize()
```

### Processing Loop Migration

**Old Code:**
```python
async for depth_image, hand_detected in self.depth_generator:
    # Manual processing and error handling
    change_score = calculate_change_score(depth_image, previous_image)
    await self._process_depth_frame(depth_image, hand_detected, change_score)
```

**New Code:**
```python
async for frame in self.depth_processor.stream_frames():
    # Rich frame with all data pre-processed and automatic error handling
    await self._process_robust_depth_frame(frame)
```

## Integration Points

### With Audio Service
- **Interaction Triggers**: Hand detection events trigger interaction sounds
- **State Coordination**: Audio state changes based on interaction levels
- **Environmental Audio**: Biome-based soundscape coordination

### With Display Service  
- **Video Masks**: Real-time interaction visualization
- **Era Transitions**: Visual coordination during state changes
- **Idle Indicators**: Visual feedback for wilderness reset states

### With Image Server
- **Render Requests**: Image generation triggered by interaction levels
- **Context Passing**: Era/biome information for relevant image generation
- **Timing Coordination**: Smooth transitions between generated images

### With Agent Service
- **Presence Detection**: Audience detection for conversation initiation
- **Biome Suggestions**: User preference learning and suggestion handling
- **Conversation State**: Interaction context for AI responses

## Best Practices

### Development
- Use mock camera for initial development and testing
- Enable verbose logging during debugging
- Test with various resolutions and frame rates
- Validate error recovery with simulated failures

### Production
- Use validated camera settings (640x480, 30fps)
- Configure appropriate retry limits (3-5 attempts)
- Monitor logs for recurring error patterns  
- Ensure stable USB 3.0 connections

### Testing
- Run comprehensive tests before deployment
- Test both real and mock camera modes
- Validate performance under continuous operation
- Check resource cleanup after errors

This robust camera system provides reliable depth sensing for the Experimance installation with comprehensive error handling, modern async design, and thorough testing support.
    lightweight_mode=True,          # Minimal processing for higher FPS
    verbose_performance=True
)
```

## Usage

### Basic Usage

```python
import asyncio
from experimance_core.robust_camera import CameraConfig, DepthProcessor

async def main():
    # Create configuration
    config = CameraConfig(
        resolution=(640, 480),
        fps=30,
        detect_hands=True
    )
    
    # Create processor
    processor = DepthProcessor(config)
    
    # Initialize camera
    if await processor.initialize():
        print("Camera ready!")
        
        # Process single frame
        frame = await processor.get_processed_frame()
        if frame:
            print(f"Hand detected: {frame.hand_detected}")
            print(f"Change score: {frame.change_score:.2f}")
        
        # Stream frames
        async for frame in processor.stream_frames():
            print(f"Frame {frame.frame_number}: interaction={frame.has_interaction}")
            
            if frame.frame_number >= 100:
                break
    
    # Cleanup
    processor.stop()

asyncio.run(main())
```

### Mock Mode for Development

```python
from experimance_core.robust_camera import MockDepthProcessor

async def test_without_camera():
    config = CameraConfig(fps=10)  # Faster for testing
    processor = MockDepthProcessor(config)
    
    await processor.initialize()
    
    async for frame in processor.stream_frames():
        # Test your processing logic with deterministic mock data
        assert isinstance(frame.depth_image, np.ndarray)
        assert frame.depth_image.shape == config.output_resolution
        
        if frame.frame_number >= 10:
            break
    
    processor.stop()
```

### Integration with Core Service

```python
class ExperimanceCoreService:
    def __init__(self):
        # Create camera config from service config
        self.camera_config = CameraConfig(
            resolution=tuple(self.config.depth_processing.resolution),
            fps=self.config.depth_processing.fps,
            max_depth=self.config.depth_processing.max_depth,
            min_depth=self.config.depth_processing.min_depth,
            change_threshold=self.config.depth_processing.change_threshold
        )
        
        self.depth_processor = DepthProcessor(self.camera_config)
    
    async def _depth_processing_task(self):
        """Background task for continuous depth processing."""
        if not await self.depth_processor.initialize():
            logger.error("Failed to initialize camera")
            return
        
        async for frame in self.depth_processor.stream_frames():
            await self._process_depth_frame(frame)
    
    async def _process_depth_frame(self, frame):
        """Process a depth frame and update service state."""
        # Update service state
        self.hand_detected = frame.hand_detected
        self.interaction_score = frame.change_score
        
        # Publish events based on frame data
        if frame.has_interaction:
            await self._publish_interaction_event()
        
        # Update state machine
        self._update_interaction_scoring(frame.change_score, frame.hand_detected)
```

## Testing

### Running Tests

The system includes comprehensive tests for both real and mock camera modes:

```bash
# Navigate to the core service directory
cd services/core

# Test with real camera (if available)
uv run python tests/test_camera.py

# Test with mock camera
uv run python tests/test_camera.py --mock

# Test with verbose performance output
uv run python tests/test_camera.py --verbose

# Test specific scenarios
uv run python tests/test_camera.py --mock --frames 50 --fps 15
```

### Test Script Options

The `tests/test_camera.py` script supports various options:

- `--mock` - Use mock camera instead of real hardware
- `--verbose` - Enable detailed performance logging
- `--frames N` - Process N frames (default: 30)
- `--fps N` - Target frame rate (default: 30)

### Expected Performance

- **Real Camera**: ~29-30 FPS with 640x480 resolution
- **Mock Camera**: ~28-30 FPS (depends on system performance)
- **Memory Usage**: Stable, no memory leaks
- **Error Recovery**: Automatic retry on camera failures

### Test Output Example

```
Testing robust camera module...
Config: CameraConfig(resolution=(640, 480), fps=30, ...)

Successfully initialized camera
Frame 0001: hands=False, change=0.00, time=33.2ms
Frame 0002: hands=False, change=0.02, time=32.8ms
Frame 0030: hands=True, change=0.45, time=33.1ms

Summary:
- Processed 30 frames in 1.02 seconds
- Average FPS: 29.4
- Average processing time: 33.0ms
- Hands detected: 12 frames
- Significant changes: 8 frames
```

## Error Handling

### Automatic Recovery

The system automatically handles common camera errors:

1. **"Couldn't resolve requests"** - Camera configuration issues
2. **"Device or resource busy"** - Camera in use by another process
3. **"get_xu(ctrl=1) failed"** - Advanced mode configuration errors
4. **Timeout errors** - Camera not responding
5. **Frame processing errors** - Corrupted data or processing failures

### Recovery Process

1. **Detect Error** - Any camera operation failure
2. **Log Details** - Record error type and attempt number
3. **Hardware Reset** - Reset the camera via USB
4. **Wait** - Exponential backoff delay (2s, 4s, 8s, ...)
5. **Retry** - Attempt the operation again
6. **Escalate** - After max retries, report failure

### Manual Error Recovery

```python
# Check if camera is working
processor = DepthProcessor(config)
if not await processor.initialize():
    logger.error("Camera initialization failed - check connection")
    
    # Try mock mode as fallback
    processor = MockDepthProcessor(config)
    await processor.initialize()
```

## Data Structures

### DepthFrame

Rich frame data with processing results and metadata:

```python
@dataclass
class DepthFrame:
    depth_image: np.ndarray              # Processed depth image
    color_image: Optional[np.ndarray]    # Color image (if available)
    hand_detected: Optional[bool]        # Hand/obstruction detection
    change_score: float                  # Motion detection score [0-1]
    frame_number: int                    # Sequential frame number
    timestamp: float                     # Frame capture timestamp
    
    @property
    def has_interaction(self) -> bool:
        """True if hands detected or significant change occurred."""
        return self.hand_detected or self.change_score > 0.1
```

### CameraConfig

Comprehensive configuration with type safety:

```python
@dataclass
class CameraConfig:
    # Camera settings
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    align_frames: bool = True
    min_depth: float = 0.0
    max_depth: float = 10.0
    
    # Processing settings
    detect_hands: bool = True
    crop_to_content: bool = True
    change_threshold: int = 60
    output_resolution: Tuple[int, int] = (1024, 1024)
    
    # Advanced settings
    json_config_path: Optional[str] = None
    lightweight_mode: bool = False
    verbose_performance: bool = False
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0
    max_retry_delay: float = 30.0
```

## Performance Optimization

### CPU Usage
- **Optimized processing** - Uses existing optimized OpenCV functions
- **Configurable frame rate** - Adjust FPS based on requirements
- **Lightweight mode** - Minimal processing for higher performance
- **Efficient change detection** - Only processes changed regions

### Memory Usage
- **Efficient frame processing** - Frames processed in-place where possible
- **Configurable resolution** - Use appropriate resolution for needs
- **Automatic cleanup** - Frames are garbage collected automatically
- **No memory leaks** - Proper resource management

### Monitoring Performance

```python
# Enable performance logging
config = CameraConfig(verbose_performance=True)
processor = DepthProcessor(config)

# Monitor in production
async for frame in processor.stream_frames():
    if frame.frame_number % 100 == 0:  # Log every 100 frames
        logger.info(f"Processed {frame.frame_number} frames")
```

## Troubleshooting

### Common Issues

**Camera not detected**
- Check USB 3.0 connection
- Verify camera permissions: `sudo usermod -a -G dialout $USER`
- Try different USB port
- Check if camera is in use: `lsof | grep uvcvideo`

**Low frame rate**
- Reduce resolution: `CameraConfig(resolution=(320, 240))`
- Enable lightweight mode: `CameraConfig(lightweight_mode=True)`
- Check USB bandwidth (disconnect other USB devices)
- Verify system performance

**Initialization failures**
- Check camera connection and drivers
- Try running tests: `uv run python tests/test_camera.py`
- Use mock mode for development: `--mock` flag
- Check logs for specific error messages

**Frame processing errors**
- Verify camera configuration
- Check depth range settings
- Use verbose mode for debugging: `--verbose` flag

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('robust_camera').setLevel(logging.DEBUG)

# Use test configuration
config = CameraConfig(
    resolution=(320, 240),    # Smaller for debugging
    fps=10,                   # Slower for analysis
    max_retries=1,           # Fail fast
    verbose_performance=True  # Show timing details
)
```

## Migration from Legacy Code

### From depth_finder.py

The robust camera module provides a compatibility function for easy migration:

```python
# Old code
from experimance_core.depth_finder import depth_generator
async for depth_image, hand_detected in depth_generator(...):
    process_frame(depth_image, hand_detected)

# New code - compatibility mode
from experimance_core.robust_camera import robust_depth_generator
async for depth_image, hand_detected in robust_depth_generator(...):
    process_frame(depth_image, hand_detected)

# New code - modern interface (recommended)
from experimance_core.robust_camera import DepthProcessor, CameraConfig
processor = DepthProcessor(CameraConfig(...))
async for frame in processor.stream_frames():
    process_frame(frame)
```

### Benefits of Migration

- **Improved reliability** - Automatic error recovery and retry logic
- **Better performance** - Optimized frame processing and memory usage
- **Enhanced debugging** - Comprehensive logging and verbose mode
- **Future-proof** - Modern async/await interface and type safety

## Service Integration

### Core Service Configuration

Configure in `services/core/config.toml`:

```toml
[depth_processing]
resolution = [640, 480]        # Camera resolution
fps = 30                       # Target frame rate
change_threshold = 60          # Motion detection sensitivity
min_depth = 0.0               # Minimum depth (meters)
max_depth = 10.0              # Maximum depth (meters)
output_size = [1024, 1024]    # Processed output size
```

### Event Publishing

The depth processor integrates with the service's event system:

```python
# AudioCommand - interaction sounds
{
  "type": "AudioCommand",
  "trigger": "interaction_start" | "interaction_stop",
  "hand_detected": true/false,
  "timestamp": "2025-06-14T..."
}

# VideoMask - visual feedback
{
  "type": "VideoMask", 
  "interaction_score": 0.75,
  "depth_difference_score": 0.45,
  "hand_detected": true,
  "timestamp": "2025-06-14T..."
}

# RenderRequest - image generation
{
  "type": "RenderRequest",
  "current_era": "modern",
  "interaction_score": 0.8,
  "timestamp": "2025-06-14T..."
}
```

## Future Enhancements

### Planned Features
- **Multiple camera support** - Handle multiple RealSense cameras
- **Hot-swapping** - Handle camera disconnect/reconnect gracefully
- **Advanced metrics** - Frame rate, error rate, performance statistics
- **Configuration validation** - Validate camera capabilities vs. config

### Extension Points
- **Custom processors** - Inherit from DepthProcessor for custom logic
- **Custom error handlers** - Override error handling behavior
- **Custom mock generators** - Create domain-specific mock data
- **Plugin system** - Add custom processing steps

This robust camera system provides a solid foundation for reliable depth sensing in the Experimance installation, with comprehensive testing support and room for future enhancements.
