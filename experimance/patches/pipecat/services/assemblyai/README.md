# Pipecat AssemblyAI STT Shutdown Fix

This directory contains a patched version of the Pipecat AssemblyAI STT service that fixes a critical shutdown timing issue.

## Issue Fixed

The original AssemblyAI STT service had a shutdown bug that caused 10+ second delays during forced shutdowns (SIGINT). This was particularly problematic for the experimance agent which needs to shutdown quickly.

## Fix Details

- **File**: `pipecat/services/assemblyai/stt.py`
- **Issue**: Long shutdown delays due to improper task cancellation order and lengthy termination timeouts
- **Solution**: 
  - Cancel receive task first before attempting graceful shutdown
  - Distinguish between forced (`cancel()`) and graceful (`stop()`) shutdowns
  - Skip termination handshake entirely for forced shutdowns  
  - Reduce timeouts for graceful shutdowns
  - Add robust fallback task cancellation

## Performance Results

- **Before fix**: 10+ second shutdown delay
- **After fix**: 
  - Forced shutdown (SIGINT): ~0.06 seconds
  - Graceful shutdown: ~1 second

## Files

- `stt.py` - Patched AssemblyAI STT service with shutdown fix
- `models.py` - Supporting models (reference copy)
- `apply_patch.sh` - Script to reapply the patch after package updates
- `README.md` - This documentation file
- `AssemblyAI_Shutdown_Bug_Report.md` - Detailed bug report for upstream submission
- `test_assemblyai_shutdown.py` - Mock test (fast, no API key needed)
- `test_real_assemblyai_shutdown.py` - Real connection test (requires API key)
- `test_realistic_assemblyai_shutdown.py` - Alternative test implementation

## Usage

If pipecat gets updated and overwrites the fix, run:

```bash
./apply_patch.sh
```

This will restore the patched version from this directory.

## Upstream Status

This fix should be submitted as a bug report to the Pipecat project. See `AssemblyAI_Shutdown_Bug_Report.md` in this directory for details.

## Test Files

The following test files can verify the fix works:
- `test_assemblyai_shutdown.py` - Mock test (fast, no API key needed)  
- `test_real_assemblyai_shutdown.py` - Real connection test (requires API key)
- `test_realistic_assemblyai_shutdown.py` - Alternative test implementation

Run tests from the project root:
```bash
# Mock test (fast)
uv run python patches/pipecat/services/assemblyai/test_assemblyai_shutdown.py

# Real connection test (requires ASSEMBLYAI_API_KEY)
uv run python patches/pipecat/services/assemblyai/test_real_assemblyai_shutdown.py
```

## Patch Applied On

Date: 2025-07-26
Pipecat Version: 0.0.76
Python Version: 3.11.13
