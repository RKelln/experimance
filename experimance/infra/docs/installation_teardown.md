# Installation Teardown Guide

This guide covers how to properly disable, pause, or completely remove an Experimance installation.

## Quick Reference

### Linux (systemd)

| Goal | Command | Services Auto-Restart? | Files Removed? |
|------|---------|------------------------|----------------|
| **Stop services** | `sudo ./infra/scripts/deploy.sh <project> stop` | ✅ Yes (on reboot) | ❌ No |
| **Uninstall (recommended)** | `sudo ./infra/scripts/deploy.sh <project> uninstall` | ❌ No | ❌ No |
| **Complete removal** | See "Complete Removal" section | ❌ No | ✅ Yes |

### macOS (launchd)

| Goal | Command | Services Auto-Restart? | Files Removed? |
|------|---------|------------------------|----------------|
| **Pause installation** | `./infra/scripts/launchd_scheduler.sh fire manual-unload` | ❌ No | ❌ No |
| **Remove scheduling only** | `./infra/scripts/launchd_scheduler.sh fire remove-schedule` | ✅ Yes (always-on) | ❌ No |
| **Complete removal** | See "Complete Removal" section | ❌ No | ✅ Yes |

## Scenarios

### Linux Systems (Ubuntu/Debian with systemd)

#### 1. Uninstall Services (Recommended for Pausing)

**When to use:** Installation is ending temporarily, moving to another machine, or maintenance needed.

```bash
# Stop services, disable auto-restart, and remove schedules
sudo ./infra/scripts/deploy.sh experimance uninstall

# Verify services are stopped and disabled
sudo ./infra/scripts/deploy.sh experimance status
```

**What this does:**
- ✅ Stops all services immediately (core, image_server, display, health)
- ✅ Disables services (prevents auto-start on boot)
- ✅ Removes cron schedules
- ✅ Keeps all files, logs, and Python environment intact
- ✅ Can re-enable later with `start` command

**What is preserved:**
- All files in the repository
- All logs in `/var/log/experimance` and `logs/`
- All cache data
- Python environment and dependencies
- Systemd template files (can be reused)

**To re-enable later:**
```bash
# Restart services (re-enables and starts them)
sudo ./infra/scripts/deploy.sh experimance start
```

#### 2. Stop Services Temporarily

**When to use:** Brief maintenance or testing, services should restart on next boot.

```bash
# Stop services (will restart on reboot)
sudo ./infra/scripts/deploy.sh experimance stop

# Verify services are stopped
sudo ./infra/scripts/deploy.sh experimance status
```

**What this does:**
- ✅ Stops all services immediately
- ⚠️ Services remain enabled (will restart on reboot)
- ❌ Does not remove schedules

#### 3. Complete Removal

**When to use:** Installation is permanently ending, machine being repurposed, or clean slate needed.

##### Step 1: Uninstall Services
```bash
# Stop and disable all services
sudo ./infra/scripts/deploy.sh experimance uninstall

# Verify nothing is running
sudo ./infra/scripts/deploy.sh experimance status
systemctl list-units | grep experimance
```

##### Step 2: Remove Systemd Template Files (Optional)
```bash
# Remove systemd template files (affects all projects)
sudo rm /etc/systemd/system/*@.service
sudo rm /etc/systemd/system/experimance@.target
sudo systemctl daemon-reload

# Verify removal
ls -la /etc/systemd/system/ | grep experimance
```

##### Step 3: Remove Sudoers Configuration
```bash
# Remove sudo permissions
sudo rm /etc/sudoers.d/experimance-experimance

# Verify removal
sudo visudo -c
```

##### Step 4: Clean Up Logs (Optional)
```bash
# Archive logs before removal
sudo tar -czf ~/experimance_logs_$(date +%Y%m%d).tar.gz /var/log/experimance

# Remove logs
sudo rm -rf /var/log/experimance
rm -rf ~/Documents/experimance/experimance/logs/*
```

##### Step 5: Remove Project Files (Optional)
```bash
# Archive the project
cd ~/Documents/experimance
tar -czf ~/experimance_archive_$(date +%Y%m%d).tar.gz experimance/

# Remove project directory
rm -rf ~/Documents/experimance/experimance/
```

##### Step 6: Verify Clean State
```bash
# Check for any remaining systemd units
systemctl list-units --all | grep experimance

# Check for running processes
ps aux | grep -i experimance

# Check cron schedules
crontab -l | grep experimance

# Should return nothing if fully removed
```

### macOS Systems (launchd)

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

### 3. Complete Removal (macOS)

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

### Linux Systems

#### Services Won't Stop

**Problem:** Services keep restarting after `stop` command

**Solution:** Use `uninstall` command instead
```bash
# stop allows auto-restart on reboot (by design)
sudo ./infra/scripts/deploy.sh experimance stop  # Services restart on reboot

# uninstall prevents auto-restart
sudo ./infra/scripts/deploy.sh experimance uninstall  # Services stay stopped
```

#### Can't Verify Service Status

**Problem:** `systemctl` commands fail or show unexpected status

**Solution:** Use the diagnose command
```bash
# Run comprehensive diagnostics
sudo ./infra/scripts/deploy.sh experimance diagnose

# Check individual service
sudo systemctl status core@experimance.service

# Check target
sudo systemctl status experimance@experimance.target
```

#### Cron Jobs Still Running

**Problem:** Scheduled tasks continue even after uninstall

**Solution:** Manually check and remove
```bash
# Check current crontab
crontab -l

# Edit and remove experimance entries
crontab -e

# Or remove all experimance cron jobs
crontab -l | grep -v experimance | crontab -
```

### macOS Systems

#### Services Won't Stop

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

### Linux Systems

#### From Uninstalled State
```bash
# Simply restart services (re-enables and starts them)
sudo ./infra/scripts/deploy.sh experimance start

# Verify services are running
sudo ./infra/scripts/deploy.sh experimance status
```

#### From Complete Removal
```bash
# Need to redeploy services
sudo ./infra/scripts/deploy.sh experimance install prod

# Then start services
sudo ./infra/scripts/deploy.sh experimance start

# Optionally add scheduling
sudo ./infra/scripts/deploy.sh experimance schedule-gallery
```

### macOS Systems

#### From Paused State (manual-unload)
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

### For Temporary Shutdowns (Both Platforms)

#### Linux
- ✅ Use `uninstall` command - clean and reversible
- ✅ Keep logs for troubleshooting
- ✅ Document why services were disabled
- ❌ Don't delete systemd template files unless permanent

#### macOS
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

#### Linux
- ✅ Complete removal (all steps)
- ✅ Remove systemd files and sudoers config
- ✅ Check for orphaned processes
- ✅ Document storage cleanup decisions

#### macOS
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

### Linux Systems (systemd)

```bash
# Status check
sudo ./infra/scripts/deploy.sh experimance status
sudo ./infra/scripts/deploy.sh experimance diagnose

# Stop services (restart on reboot)
sudo ./infra/scripts/deploy.sh experimance stop

# Uninstall services (no auto-restart)
sudo ./infra/scripts/deploy.sh experimance uninstall

# Restart services
sudo ./infra/scripts/deploy.sh experimance start

# Complete removal
sudo ./infra/scripts/deploy.sh experimance uninstall
sudo rm /etc/systemd/system/*@.service
sudo rm /etc/systemd/system/experimance@.target
sudo rm /etc/sudoers.d/experimance-experimance
sudo systemctl daemon-reload

# Verify clean state
systemctl list-units --all | grep experimance
ps aux | grep -i experimance
crontab -l | grep experimance
```

### macOS Systems (launchd)

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
