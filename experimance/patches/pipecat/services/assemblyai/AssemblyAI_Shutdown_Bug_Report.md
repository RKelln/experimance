# AssemblyAI STT Service Shutdown Bug Report

## Issue Description

The AssemblyAI STT service in Pipecat has a shutdown timing issue that causes ~10 second delays when the pipeline is forcefully cancelled (e.g., SIGINT) rather than gracefully stopped.

## Root Cause

The `_disconnect()` method in `AssemblyAISTTService` has several issues:

1. **Receive task cancelled too late**: The WebSocket receive task is only cancelled after attempting graceful shutdown, which can hang for up to 5 seconds
2. **No connection state checking**: The code attempts to send termination messages even when the WebSocket is already closed
3. **Long timeout**: 5-second timeout for termination handshake is too long for forced shutdowns
4. **Poor error handling**: Doesn't properly handle already-closed WebSocket connections

## Symptoms

- 10+ second delay during application shutdown when using SIGINT
- Warning messages: `"Error during termination handshake: no close frame received or sent"`
- Log message: `"Pipeline CancelFrame received without EndFrame (forced shutdown)"`

## Solution

### Key Changes Made

1. **Cancel receive task first**: Cancel the WebSocket receive task before attempting graceful shutdown
2. **Robust task cancellation**: Added fallback direct task cancellation when TaskManager isn't initialized
3. **Distinguish forced vs graceful shutdown**: `cancel()` now does forced shutdown, `stop()` does graceful shutdown
4. **Skip handshake for forced shutdowns**: Forced shutdowns bypass the termination handshake entirely for instant shutdown
5. **Reduce graceful timeout**: Reduced graceful termination handshake timeout from 5s to 1s
6. **Better error handling**: Added specific handling for `ConnectionClosed` exceptions
7. **Improved receive task**: Added proper `CancelledError` handling in the receive task

### Modified Files

- `pipecat/services/assemblyai/stt.py`

### Patch Applied

The patch has been applied to the local installation at:
`/home/experimance/Documents/art/experimance/experimance/.venv/lib/python3.11/site-packages/pipecat/services/assemblyai/stt.py`

### Testing

Test the fix by:
1. Starting a Pipecat pipeline with AssemblyAI STT
2. Sending SIGINT (Ctrl+C) to force shutdown
3. Verify shutdown completes within 2-3 seconds instead of 10+ seconds

**Test Results**: 
- Before fix: 10+ second shutdown delay
- After fix: 
  - Forced shutdown (SIGINT/CancelFrame): ~0.01 seconds (nearly instant)
  - Graceful shutdown (EndFrame): ~1 second 
- Test confirmed working with `test_assemblyai_shutdown.py`

## Upstream Bug Report

This issue should be reported to the Pipecat project with the following details:

### Title
`AssemblyAI STT service causes 10+ second delays during forced shutdown`

### Description
The AssemblyAI STT service's `_disconnect()` method has timing issues that cause significant delays when the pipeline is forcefully cancelled. The receive task is cancelled too late, and the termination handshake timeout is too long for forced shutdowns.

### Proposed Fix
See the changes made in this patch - primarily reordering the cleanup sequence and reducing timeouts for better responsiveness during forced shutdowns.

### Reproduction
1. Create a pipeline with AssemblyAI STT service
2. Send SIGINT to force shutdown
3. Observe 10+ second delay before shutdown completes

### Expected Behavior
Shutdown should complete within 2-3 seconds even during forced termination.

### Environment
- Pipecat version: [current version]
- Python version: 3.11+
- OS: Linux/macOS/Windows

## Files Changed

- `libs/common/README_SERVICE_TESTING.md` - If any testing documentation needs updates
- Local pipecat installation patched directly
