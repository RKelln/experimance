# Cartesia TTS Shutdown Fix

This directory contains a patched version of the Pipecat Cartesia TTS service that fixes WebSocket shutdown issues that can cause applications to hang during exit.

## Problem

The original Cartesia TTS service has similar issues to the AssemblyAI STT service where WebSocket connections can hang during shutdown, particularly:

1. **WebSocket cleanup hangs** - The WebSocket close() operation can block indefinitely
2. **Context cancellation delays** - Graceful context cleanup can be slow during forced shutdowns  
3. **Task cancellation issues** - Background receive tasks may not cancel properly
4. **ThreadPoolExecutor hangs** - Background threads from Cartesia audio processing can prevent clean shutdown

These issues were identified through diagnostic output showing `ThreadPoolExecutor-0_0` threads and WebSocket connections as primary hang sources.

## Solution

The patch addresses these issues by:

1. **Adding `force` parameter to `_disconnect()`** - Allows skipping graceful cleanup during forced shutdowns
2. **Enhanced task cancellation** - Better handling of receive task cancellation with fallback to direct cancellation
3. **Timeout protection** - Adds timeouts to context cancellation messages to prevent infinite waits
4. **Improved error handling** - Better exception handling during WebSocket cleanup operations
5. **State tracking** - Added `_connected` flag to track connection state properly

## Changes Made

### Key Method Changes

- **`_disconnect(force=False)`** - Enhanced with force parameter and timeout protection
- **`stop()`** - Calls `_disconnect(force=False)` for graceful shutdown
- **`cancel()`** - Calls `_disconnect(force=True)` for fast shutdown
- **`_handle_interruption()`** - Added error handling for context cancellation
- **`flush_audio()`** - Added error handling for audio flushing
- **`_receive_messages()`** - Better exception handling and cleanup
- **`_receive_task_handler()`** - Added proper error reporting

### Other Improvements

- Added `asyncio` import for timeout functionality
- Enhanced WebSocket close error handling
- Improved HTTP client cleanup in `CartesiaHttpTTSService`
- Added `_connected` state tracking throughout

## Installation

Run the apply script to patch your Pipecat installation:

```bash
cd patches/pipecat/services/cartesia/
chmod +x apply_patch.sh
./apply_patch.sh
```

## Testing

Three test scripts are provided:

### 1. Basic Shutdown Test
```bash
uv run python patches/pipecat/services/cartesia/test_cartesia_shutdown.py
```
Tests basic shutdown behavior with mocked connections.

### 2. Realistic Shutdown Test  
```bash
uv run python patches/pipecat/services/cartesia/test_realistic_cartesia_shutdown.py
```
Tests with realistic WebSocket behavior simulation and various edge cases.

### 3. Real API Test
```bash
export CARTESIA_API_KEY="your_api_key_here"
uv run python patches/pipecat/services/cartesia/test_real_cartesia_shutdown.py
```
Tests with real Cartesia API connections (requires API key).

## Expected Results

With the patch applied:

- **Graceful shutdowns** should complete context cleanup properly
- **Forced shutdowns** should skip context cleanup and complete in <1 second
- **WebSocket hangs** should be eliminated through timeout protection
- **Task cancellation** should work reliably with fallback mechanisms

## Verification

The patch can be verified by checking for the presence of the force parameter:

```python
from pipecat.services.cartesia.tts import CartesiaTTSService
service = CartesiaTTSService(api_key="test", voice_id="test")
if 'force' in service._disconnect.__code__.co_varnames:
    print("✅ Patch applied successfully")
else:
    print("❌ Original version detected")
```

## Compatibility

This patch is designed for:
- Pipecat version with Cartesia TTS support
- Python 3.11+
- Compatible with existing Cartesia TTS usage patterns

The patch maintains full backward compatibility with existing code while adding the enhanced shutdown behavior.

## Integration with Experimance

This patch directly addresses the hanging issues identified in the Experimance agent service diagnostics, specifically targeting:
- ThreadPoolExecutor threads from Pipecat audio services
- WebSocket connection cleanup hangs
- Background task cleanup during aggressive shutdown procedures

When integrated with the agent service's aggressive cleanup procedures, this patch should eliminate the Cartesia TTS-related components from the hang sources.
