# Project Patches

This directory contains patches for third-party libraries used in the experimance project.

## Current Patches

### Pipecat AssemblyAI STT Shutdown Fix

**Location**: `pipecat/services/assemblyai/`
**Issue**: AssemblyAI STT service had 10+ second shutdown delays during forced shutdowns
**Fix**: Optimized task cancellation and termination handshake for instant shutdown
**Status**: ✅ Applied and tested

**Reapply after updates**:
```bash
cd patches/pipecat/services/assemblyai/
./apply_patch.sh
```

**Test the fix**:
```bash
# Mock test (fast, no API key needed)
uv run python patches/pipecat/services/assemblyai/test_assemblyai_shutdown.py

# Real connection test (requires API key)
uv run python patches/pipecat/services/assemblyai/test_real_assemblyai_shutdown.py
```

## Adding New Patches

1. Create a directory structure matching the library path
2. Copy the patched files
3. Add a README explaining the fix
4. Create an `apply_patch.sh` script for easy reapplication
5. Update this README

## Testing Patches

After applying patches, always test to ensure they work correctly:

- Use the project's test files
- Verify the original issue is resolved
- Check for any regressions

## Upstream Status

Patches should be submitted upstream when possible. See individual patch READMEs for upstream status and bug report details.
