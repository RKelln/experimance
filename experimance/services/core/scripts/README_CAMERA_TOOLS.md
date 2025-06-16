# Camera Recovery and Testing Tools

This directory contains tools for managing RealSense camera issues and testing camera recovery mechanisms.

## Scripts

### 🔧 `recover_camera.py`
**Purpose**: Recover stuck RealSense cameras using aggressive reset strategies.

**Usage**:
```bash
uv run python services/core/scripts/recover_camera.py
```

**What it does**:
- Diagnoses current camera state
- Attempts aggressive hardware reset
- Verifies recovery was successful
- Fully async and cancellable with Ctrl+C

### 💥 `break_camera.py`
**Purpose**: Intentionally break the camera for testing recovery mechanisms.

**Usage**:
```bash
uv run python services/core/scripts/break_camera.py
```

**Breaking methods**:
1. **Pipeline leak**: Start camera pipeline and exit without stopping
2. **Context leak**: Create multiple RealSense contexts without cleanup
3. **Mid-stream termination**: Exit during active frame capture
4. **Maximum chaos**: All methods combined

**What it does**:
- Shows current camera state
- Lets you choose breaking method
- Intentionally exits without cleanup
- Leaves camera in problematic state for testing

### 🧪 `test_camera_async.py`
**Purpose**: Test that async camera functions work correctly and are cancellable.

**Usage**:
```bash
uv run python services/core/scripts/test_camera_async.py
```

**What it tests**:
- Camera diagnostics with timeout
- Process killing functionality
- Camera reset with cancellation support
- Proper Ctrl+C handling

## Typical Testing Workflow

1. **Break the camera**:
   ```bash
   uv run python services/core/scripts/break_camera.py
   # Choose method 1, 2, 3, or 4
   ```

2. **Verify it's broken**:
   ```bash
   uv run -m experimance_core
   # Should show camera errors and recovery attempts
   ```

3. **Test manual recovery**:
   ```bash
   uv run python services/core/scripts/recover_camera.py
   ```

4. **Test cancellation**:
   ```bash
   uv run python services/core/scripts/test_camera_async.py
   # Try pressing Ctrl+C during reset operations
   ```

## Key Improvements

### ✅ Async and Cancellable Operations
- All camera operations now use `async`/`await`
- Proper `asyncio.CancelledError` handling
- Timeout protection on all blocking operations
- Can be interrupted with Ctrl+C

### ✅ Better Error Recovery
- Separate hardware reset from retry logic
- Multiple reset strategies (hardware reset, USB reset, process killing)
- Graceful degradation when reset fails
- Better state management

### ✅ Testing Support
- Dedicated tools to create and fix broken camera states
- Consistent breaking patterns for testing
- Verification that recovery actually works

## Troubleshooting

### Camera Won't Reset
1. Try the aggressive recovery: `recover_camera.py`
2. Check for other processes using camera: `lsusb` and `ps aux | grep realsense`
3. Physically disconnect and reconnect the USB cable
4. Restart the system if all else fails

### Scripts Won't Run
1. Make sure you're in the experimance project root
2. Check that uv is working: `uv --version`
3. Verify pyrealsense2 is installed: `uv run python -c "import pyrealsense2; print('OK')"`

### Recovery Seems to Work But Camera Still Fails
1. The camera hardware might be genuinely faulty
2. Try a different USB port (preferably USB 3.0)
3. Check USB power management settings
4. Test with RealSense Viewer: `realsense-viewer`

## Implementation Notes

- **Timeouts**: All operations have reasonable timeouts to prevent hanging
- **Cancellation**: Uses proper `asyncio.CancelledError` handling throughout
- **Cleanup**: Services now properly clean up camera resources on shutdown
- **Testing**: Breaking tools help verify recovery mechanisms work correctly

The async rewrite makes the camera operations much more robust and testable while maintaining compatibility with the existing service architecture.
