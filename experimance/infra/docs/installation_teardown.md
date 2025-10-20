# Installation Teardown Guide

This guide covers how to properly disable, pause, or completely remove an Experimance installation.

## Quick Reference

| Goal | Command | Services Auto-Restart? | Files Removed? |
|------|---------|------------------------|----------------|
| **Pause installation** | `launchd_scheduler.sh fire manual-unload` | ❌ No | ❌ No |
| **Remove scheduling only** | `launchd_scheduler.sh fire remove-schedule` | ✅ Yes (always-on) | ❌ No |
| **Complete removal** | See "Complete Removal" section | ❌ No | ✅ Yes |

## Scenarios

### 1. Temporarily Disable Services (Pause Installation)

**When to use:** Installation is ending but might return, maintenance needed, or testing.

```bash
# Stop services and prevent auto-restart
./infra/scripts/launchd_scheduler.sh fire manual-unload

# Verify services are stopped
./infra/scripts/deploy.sh fire status
```

**What this does:**
- ✅ Stops all services immediately
- ✅ Prevents auto-restart on reboot
- ✅ Keeps all files in place
- ✅ Can re-enable later with `manual-start` or `setup-schedule`

**To re-enable later:**
```bash
# Start services manually (one-time)
./infra/scripts/launchd_scheduler.sh fire manual-start

# Or re-enable gallery scheduling
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery
```

### 2. Return to Always-On Mode

**When to use:** Remove gallery hour scheduling but keep services running 24/7.

```bash
# Remove scheduling but keep services running
./infra/scripts/launchd_scheduler.sh fire remove-schedule
```

**What this does:**
- ✅ Services run continuously
- ✅ Auto-restart on reboot
- ✅ Auto-restart if crashed
- ❌ No gallery hour automation

### 3. Complete Removal

**When to use:** Installation is permanently ending, machine being repurposed, or clean slate needed.

#### Step 1: Stop Services
```bash
# Stop all services
./infra/scripts/launchd_scheduler.sh fire manual-unload

# Verify nothing is running
./infra/scripts/deploy.sh fire status
```

#### Step 2: Remove LaunchAgent Files
```bash
# Remove Python service LaunchAgents
rm ~/Library/LaunchAgents/com.experimance.fire.*.plist

# Remove TouchDesigner LaunchAgents (if installed)
rm ~/Library/LaunchAgents/com.experimance.touchdesigner.fire.*.plist

# Verify removal
ls -la ~/Library/LaunchAgents/com.experimance.*fire*
```

#### Step 3: Clean Up Logs (Optional)
```bash
# Remove service logs
rm -rf ~/Library/Logs/experimance/fire_*

# Or archive logs for future reference
mkdir -p ~/experimance_archive/$(date +%Y%m%d)
mv ~/Library/Logs/experimance/fire_* ~/experimance_archive/$(date +%Y%m%d)/
```

#### Step 4: Remove Project Files (Optional)
```bash
# If completely done with Experimance on this machine
cd ~/Documents
rm -rf experimance/

# Or archive the project
mv experimance/ experimance_archive_$(date +%Y%m%d)/
```

#### Step 5: Verify Clean State
```bash
# Check for any remaining LaunchAgents
launchctl list | grep experimance
launchctl list | grep fire

# Check for running processes
ps aux | grep -i experimance
ps aux | grep -i touchdesigner

# Should return nothing if fully removed
```

## Troubleshooting

### Services Won't Stop

**Problem:** Services keep restarting after `manual-stop`

**Solution:** Use `manual-unload` instead
```bash
# manual-stop allows auto-restart (by design)
./infra/scripts/launchd_scheduler.sh fire manual-stop  # Services may restart

# manual-unload prevents auto-restart
./infra/scripts/launchd_scheduler.sh fire manual-unload  # Services stay stopped
```

### Can't Remove LaunchAgent Files

**Problem:** Permission denied or "file in use" errors

**Solution:** Unload first, then remove
```bash
# Unload services first
./infra/scripts/launchd_scheduler.sh fire manual-unload

# Wait a moment for processes to fully exit
sleep 5

# Then remove files
rm ~/Library/LaunchAgents/com.experimance.fire.*.plist
```

### Services Still Running After Removal

**Problem:** Processes still active after removing plist files

**Solution:** Force kill remaining processes
```bash
# Find experimance processes
ps aux | grep -i experimance

# Kill by PID (replace <PID> with actual process ID)
kill -9 <PID>

# Or kill all Python services (use with caution!)
pkill -f "uv run -m experimance"

# Kill TouchDesigner (if needed)
pkill -f TouchDesigner
```

## Storage Cleanup

After teardown, you may want to clean up additional storage:

### Images and Media
```bash
# Check disk usage
du -sh ~/Documents/experimance/media/images/

# Remove generated images
rm -rf ~/Documents/experimance/media/images/generated/

# Or archive for historical reference
tar -czf ~/images_archive_$(date +%Y%m%d).tar.gz ~/Documents/experimance/media/images/
```

### Audio Cache
```bash
# Remove cached audio
rm -rf ~/Documents/experimance/media/audio/cache/

# Check cache size first
du -sh ~/Documents/experimance/media/audio/cache/
```

### Logs and Transcripts
```bash
# Archive logs before removal
cd ~/Documents/experimance
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/

# Remove old logs
rm -rf logs/*
```

## Re-enabling After Teardown

### From Paused State (manual-unload)
```bash
# Simply restart services
./infra/scripts/launchd_scheduler.sh fire manual-start

# Or restore gallery scheduling
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery
```

### From Complete Removal
```bash
# Need to redeploy services
./infra/scripts/deploy.sh fire install

# Then optionally add scheduling
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery
```

## Best Practices

### For Temporary Shutdowns
- ✅ Use `manual-unload` - clean and reversible
- ✅ Keep logs for troubleshooting
- ✅ Document why services were disabled
- ❌ Don't delete plist files unless permanent

### For Installation End
- ✅ Archive logs and media before cleanup
- ✅ Document what was removed and why
- ✅ Verify nothing is running after removal
- ✅ Keep one backup of configuration files

### For Machine Repurposing
- ✅ Complete removal (all steps)
- ✅ Verify LaunchAgent cleanup
- ✅ Check for orphaned processes
- ✅ Document storage cleanup decisions

## Related Documentation

- [LaunchAgent Scheduler](../scripts/README_LAUNCHD_SCHEDULER.md) - Service scheduling and manual controls
- [Deployment Guide](deployment.md) - Installation and configuration
- [Emergency Reference](emergency-reference.md) - Quick fixes and recovery
- [New Machine Setup](new_machine_setup.md) - Fresh installation procedures

## Quick Command Reference

```bash
# Status check
./infra/scripts/deploy.sh fire status
./infra/scripts/launchd_scheduler.sh fire show-schedule

# Pause installation (no auto-restart)
./infra/scripts/launchd_scheduler.sh fire manual-unload

# Resume installation
./infra/scripts/launchd_scheduler.sh fire manual-start

# Remove scheduling (return to always-on)
./infra/scripts/launchd_scheduler.sh fire remove-schedule

# Complete removal
./infra/scripts/launchd_scheduler.sh fire manual-unload
rm ~/Library/LaunchAgents/com.experimance.fire.*.plist
rm ~/Library/LaunchAgents/com.experimance.touchdesigner.fire.*.plist

# Verify clean state
launchctl list | grep experimance
ps aux | grep -i experimance
```

## Support

For issues during teardown:
1. Check service status: `./infra/scripts/deploy.sh fire status`
2. Review logs: `tail -f ~/Library/Logs/experimance/fire_*_error.log`
3. Force kill if needed: `pkill -f "uv run -m experimance"`
4. Consult [Emergency Reference](emergency-reference.md) for recovery procedures
