# Gallery Hour Scheduling for macOS LaunchAgents

The `launchd_scheduler.sh` script provides gallery hour automation for Experimance projects running on macOS.

## Overview

This script enhances existing LaunchAgent services with time-based scheduling while preserving auto-restart capabilities. Perfect for gallery installations that need automatic startup/shutdown but also require manual override capabilities.

## Key Features

- ✅ **Preserves existing services** - No modification of main service plist files
- ✅ **Keeps RunAtLoad=true** - Services auto-start after machine reboot  
- ✅ **Gallery hour automation** - Start/stop services during operating hours
- ✅ **Manual override** - Gallery staff can immediately start/stop services
- ✅ **Auto-restart preserved** - Services restart if they crash during gallery hours
- ✅ **Multiple schedules** - Gallery, daily, or custom timing

## Quick Start

```bash
# Current working directory: /Users/fireproject/Documents/experimance/experimance

# Setup gallery hours (Tuesday-Saturday, 11AM-6PM)
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery

# Check status and see schedule
./infra/scripts/launchd_scheduler.sh fire show-schedule

# Manual override for gallery staff
./infra/scripts/launchd_scheduler.sh fire manual-stop   # Stop everything now
./infra/scripts/launchd_scheduler.sh fire manual-start  # Start everything now

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
- Services auto-start after reboot
- Gallery scheduler starts services 5 minutes before opening
- Gallery scheduler stops services 5 minutes after closing
- Manual override available for special events

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
# Emergency stop (immediate)
./infra/scripts/launchd_scheduler.sh fire manual-stop

# Emergency start (immediate)  
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
3. **Note**: Services with `KeepAlive` will restart - this is expected behavior

### Gallery Staff Emergency Override
```bash
# Emergency procedures for gallery staff:

# 1. Stop everything immediately (technical issues)
./infra/scripts/launchd_scheduler.sh fire manual-stop

# 2. Start everything immediately (special event)
./infra/scripts/launchd_scheduler.sh fire manual-start

# 3. Check what's supposed to be running
./infra/scripts/launchd_scheduler.sh fire show-schedule

# 4. Disable automation temporarily (maintenance)
./infra/scripts/launchd_scheduler.sh fire remove-schedule

# 5. Re-enable automation after maintenance
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
- No scheduling added - controlled by starter/stopper

### File Locations

- **Script**: `./infra/scripts/launchd_scheduler.sh`
- **Main Services**: `~/Library/LaunchAgents/com.experimance.fire.*.plist`
- **Scheduler Services**: `~/Library/LaunchAgents/com.experimance.fire.gallery-*.plist`
- **Logs**: `~/Library/Logs/experimance/`

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
