# Gallery Hour Scheduling for macOS LaunchAgents

The `launchd_scheduler.sh` script provides gallery hour automation for Experimance projects running on macOS.

## Overview

This script enhances existing LaunchAgent services with time-based scheduling while preserving auto-restart capabilities. Perfect for gallery installations that need automatic startup/shutdown but also require manual override capabilities.

## Key Features

- ✅ **Preserves existing services** - No modification of main service plist files
- ✅ **Keeps RunAtLoad=true** - Services auto-start after machine reboot  
- ✅ **Gallery hour automation** - Start/stop services during operating hours
- ✅ **Manual override** - Gallery staff can immediately start/stop services
- ✅ **Staged startup/shutdown** - TouchDesigner and Python services coordinate properly
- ✅ **Auto-restart preserved** - Services restart if they crash during gallery hours
- ✅ **Multiple schedules** - Gallery, daily, or custom timing
- ✅ **Complete stop option** - manual-unload prevents auto-restart for maintenance

## Quick Start

```bash
# Current working directory: /Users/fireproject/Documents/experimance/experimance

# Setup gallery hours (Tuesday-Saturday, 11AM-6PM)
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery

# Check status and see schedule
./infra/scripts/launchd_scheduler.sh fire show-schedule

# Manual override for gallery staff
./infra/scripts/launchd_scheduler.sh fire manual-stop     # Stop everything now (may auto-restart)
./infra/scripts/launchd_scheduler.sh fire manual-unload   # Stop everything now (no auto-restart)
./infra/scripts/launchd_scheduler.sh fire manual-start    # Start everything now

# Remove scheduling (return to always-on)
./infra/scripts/launchd_scheduler.sh fire remove-schedule
```

## How It Works

### Architecture

1. **Existing Services** (unchanged):
   - `com.experimance.fire.agent` - AI interaction service
   - `com.experimance.fire.health` - Health monitoring  
   - `com.experimance.touchdesigner.fire.*` - TouchDesigner visualization
   - All keep `RunAtLoad=true` for auto-startup after reboot

2. **Gallery Scheduler Services** (added):
   - `com.experimance.fire.gallery-starter` - Starts services during gallery hours
   - `com.experimance.fire.gallery-stopper` - Stops services after gallery hours
   - Use `StartCalendarInterval` for precise timing

### Gallery Hours Schedule

**Default Gallery Schedule:**
- **Days**: Tuesday through Saturday
- **Start**: 10:55 AM (5 minutes before 11 AM opening)
- **Stop**: 6:05 PM (5 minutes after 6 PM closing)
- **Closed**: Sunday and Monday

**What Happens:**
- Machine stays on 24/7
- Services auto-start after reboot with staging delays (TouchDesigner 10s, Python 30s)
- Gallery scheduler starts services 5 minutes before opening
- Gallery scheduler stops services 5 minutes after closing
- Manual override available for special events

## Staged Service Management

The scheduler implements intelligent staging for reliable TouchDesigner and Python service coordination:

### Startup Sequence
1. **TouchDesigner services** start first (10 seconds after boot via plist)
2. **20-second delay** for TouchDesigner to fully initialize
3. **Python services** start second (30 seconds after boot via plist)

### Shutdown Sequence  
1. **Python services** stopped first (clean API disconnection)
2. **10-second grace period** for graceful shutdown
3. **TouchDesigner services** stopped last

### Manual Operations
- `manual-start`: Follows staging (TD first → 20s delay → Python)
- `manual-stop`: Follows staging (Python first → 10s delay → TD) + may auto-restart
- `manual-unload`: Follows staging + prevents auto-restart (maintenance mode)

## Usage

### Setup Gallery Scheduling

```bash
# Gallery hours (Tuesday-Saturday, 11AM-6PM)
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery

# Daily schedule (Every day, 9AM-10PM)  
./infra/scripts/launchd_scheduler.sh fire setup-schedule daily

# Custom schedule (prompts for times)
./infra/scripts/launchd_scheduler.sh fire setup-schedule custom
```

### Manual Controls (Gallery Staff)

```bash
# Emergency stop (services may auto-restart)
./infra/scripts/launchd_scheduler.sh fire manual-stop

# Complete stop (no auto-restart)
./infra/scripts/launchd_scheduler.sh fire manual-unload

# Emergency start (staged: TouchDesigner first, then Python services)
./infra/scripts/launchd_scheduler.sh fire manual-start

# Check what's running
./infra/scripts/launchd_scheduler.sh fire show-schedule
```

### Remove Scheduling

```bash
# Return to always-on mode
./infra/scripts/launchd_scheduler.sh fire remove-schedule
```

## Logs

### Gallery Automation Logs
- `~/Library/Logs/experimance/fire_gallery_starter.log` - Gallery opening
- `~/Library/Logs/experimance/fire_gallery_starter_error.log` - Startup errors
- `~/Library/Logs/experimance/fire_gallery_stopper.log` - Gallery closing
- `~/Library/Logs/experimance/fire_gallery_stopper_error.log` - Shutdown errors

### Monitor Logs
```bash
# Watch gallery automation
tail -f ~/Library/Logs/experimance/fire_gallery_*.log

# Watch main services (unchanged)
tail -f ~/Library/Logs/experimance/fire_agent_launchd_error.log
tail -f ~/Library/Logs/experimance/fire_touchdesigner_*_error.log
```

## Schedule Types

### Gallery Schedule
- **Days**: Tuesday, Wednesday, Thursday, Friday, Saturday
- **Start**: 10:55 AM
- **Stop**: 6:05 PM
- **Closed**: Sunday, Monday

### Daily Schedule  
- **Days**: Every day
- **Start**: 9:00 AM
- **Stop**: 10:00 PM

### Custom Schedule
- Prompts for custom start/stop times
- Applied daily

## Troubleshooting

### Services Don't Start at Gallery Hours
1. **Check scheduler status**: `./infra/scripts/launchd_scheduler.sh fire show-schedule`
2. **Check logs**: `tail -f ~/Library/Logs/experimance/fire_gallery_starter_error.log`
3. **Manual test**: `./infra/scripts/launchd_scheduler.sh fire manual-start`

### Services Don't Stop After Hours
1. **Check stopper logs**: `tail -f ~/Library/Logs/experimance/fire_gallery_stopper_error.log`
2. **Manual test**: `./infra/scripts/launchd_scheduler.sh fire manual-stop`
3. **Complete stop**: `./infra/scripts/launchd_scheduler.sh fire manual-unload` (no auto-restart)
4. **Note**: Services with `KeepAlive` will restart after `manual-stop` - this is expected behavior

### Gallery Staff Emergency Override
```bash
# Emergency procedures for gallery staff:

# 1. Stop everything immediately (services may restart automatically)
./infra/scripts/launchd_scheduler.sh fire manual-stop

# 2. Stop everything completely (for maintenance - no auto-restart)
./infra/scripts/launchd_scheduler.sh fire manual-unload

# 3. Start everything immediately (staged startup with delays)
./infra/scripts/launchd_scheduler.sh fire manual-start

# 4. Check what's supposed to be running
./infra/scripts/launchd_scheduler.sh fire show-schedule

# 5. Disable automation temporarily (maintenance)
./infra/scripts/launchd_scheduler.sh fire remove-schedule

# 6. Re-enable automation after maintenance
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery
```

## Integration with TouchDesigner

The scheduler works seamlessly with TouchDesigner LaunchAgents created by `touchdesigner_agent.sh`:

```bash
# Install TouchDesigner LaunchAgent
./infra/scripts/touchdesigner_agent.sh /path/to/file.toe install --project=fire

# Add gallery scheduling to all services (including TouchDesigner)
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery

# Both fire_agent and TouchDesigner now follow gallery hours
```

## Implementation Details

### LaunchAgent Properties Used

**Gallery Starter/Stopper Services:**
- `StartCalendarInterval` - Defines when to run
- `RunAtLoad=false` - Only run on schedule, not at boot
- `KeepAlive=false` - Run once then exit

**Existing Services (preserved):**
- `RunAtLoad=true` - Auto-start after reboot  
- `KeepAlive=true` - Auto-restart if crashed
- `StartInterval` - TouchDesigner: 10s delay, Python services: 30s delay
- No scheduling added - controlled by starter/stopper

### File Locations

- **Script**: `./infra/scripts/launchd_scheduler.sh`
- **Main Services**: `~/Library/LaunchAgents/com.experimance.fire.*.plist`
- **Scheduler Services**: `~/Library/LaunchAgents/com.experimance.fire.gallery-*.plist`
- **Logs**: `~/Library/Logs/experimance/`

## Troubleshooting

### Recent Bug Fixes (September 2025)

**Issue**: Gallery scheduler services not loading properly
- **Symptoms**: Scheduler shows "Not loaded" or XML parsing errors
- **Cause**: Self-referencing services and XML encoding problems  
- **Fix**: Updated to filter scheduler services from their own commands and use semicolon separators instead of `&&`

**Issue**: TouchDesigner not included in scheduling
- **Symptoms**: TouchDesigner service not starting during gallery hours
- **Cause**: This was actually a false alarm - TouchDesigner was included but scheduler bugs prevented it from working
- **Fix**: Fixed scheduler bugs resolved TouchDesigner scheduling

### Common Issues

**Gallery Scheduler Shows "Not loaded"**
```bash
# Check for XML parsing errors
plutil ~/Library/LaunchAgents/com.experimance.fire.gallery-*.plist

# If errors found, regenerate:
./infra/scripts/launchd_scheduler.sh fire remove-schedule
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery
```

**Services Not Starting During Gallery Hours**
```bash
# Check scheduler logs
tail -f ~/Library/Logs/experimance/fire_gallery_starter.log
tail -f ~/Library/Logs/experimance/fire_gallery_starter_error.log

# Test manual start to verify service health
./infra/scripts/launchd_scheduler.sh fire manual-start
```

**TouchDesigner Not Responding to Scheduling**
```bash
# Verify TouchDesigner service exists and is properly configured
launchctl list | grep touchdesigner
plutil ~/Library/LaunchAgents/com.experimance.touchdesigner.*.plist

# Check for multiple TouchDesigner processes
ps aux | grep -i touchdesigner
```

## Comparison with Linux cron

| Feature | Linux cron | macOS cron | macOS LaunchAgent Scheduler |
|---------|------------|------------|----------------------------|
| Runs without login | ✅ | ❌ | ✅ |
| GUI app support | ❌ | ❌ | ✅ |
| Automatic restart | ❌ | ❌ | ✅ |
| Manual override | ❌ | ❌ | ✅ |
| Precise timing | ✅ | ✅ | ✅ |
| Apple recommended | N/A | ❌ | ✅ |

The LaunchAgent scheduler provides the most reliable solution for gallery automation on macOS.
