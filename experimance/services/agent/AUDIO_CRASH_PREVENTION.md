# Audio Crash Prevention - Emergency Instructions

## Problem
Sudden agent crashes during TTS initialization (right before audio output starts).

## Quick Fix
If crashes occur again:
```bash
uv run scripts/audio_recovery.py reset
```

## Emergency Debug Mode
If crashes persist, enable comprehensive audio monitoring:

1. Add `audio_health_monitoring = true` to `agent.toml`
4. Restart agent

This enables:
- Pre-startup health checks
- Automatic recovery attempts  
- Ongoing audio device monitoring
- Detailed crash diagnostics

## Symptoms of Audio Issues
- Sudden process termination during startup
- Crashes right after "Generating TTS" messages
- No error messages in logs (process just stops)

## Root Cause
- USB audio device conflicts (Yealink device state issues)
- Audio driver problems during first TTS generation
- Device locks or kernel driver state corruption

## Files Modified
- `pipecat_backend.py` - Added emergency audio monitoring (disabled by default)
- `scripts/debug_crashes.sh` - Enhanced crash detection
- `scripts/test_audio_health.py` - Audio health testing tool

## Production Safety
- All monitoring code is disabled by default (`audio_health_monitoring = False`)
- Minimal performance impact in normal operation
- Only basic error logging is active unless debug mode is enabled
