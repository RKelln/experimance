# Emergency SSH Recovery Guide

This guide helps recover remote access when you cannot SSH into the gallery machine.

## Immediate Actions (Try in Order)

### 1. Basic Connectivity Check
From your local machine:
```bash
# Test basic network connectivity
ping experimance-pc  # or the Tailscale hostname

# Test SSH port specifically  
telnet experimance-pc 22

# Check Tailscale status
tailscale status
tailscale ping experimance-pc
```

### 2. Try Alternative Connection Methods
```bash
# If MagicDNS fails, try direct Tailscale IP
tailscale ip -4 experimance-pc
ssh experimance@100.xxx.xxx.xxx  # Use actual IP

# Try with verbose SSH for debugging
ssh -v experimance@experimance-pc

# Try with different SSH options
ssh -o ConnectTimeout=30 -o ServerAliveInterval=10 experimance@experimance-pc
```

### 3. Physical Access Recovery
If you can physically access the gallery machine:

```bash
# 1. Check if system is responsive
# Press Ctrl+Alt+F1 to get to a text console
# Login as experimance user

# 2. Check system status
sudo systemctl status ssh tailscaled
sudo /home/experimance/experimance/infra/scripts/system_diagnostic.sh quick

# 3. Check network connectivity
ip addr show
ping 8.8.8.8
tailscale status

# 4. Restart networking if needed
sudo systemctl restart NetworkManager
sudo systemctl restart tailscaled

# 5. Check SSH service
sudo systemctl restart ssh
sudo journalctl -u ssh.service -f

# 6. Check system resources
htop  # Look for high CPU/memory usage
df -h  # Check disk space

# 7. Emergency service restart
sudo systemctl restart experimance@experimance.target
```

## Diagnostic Tools (Physical Access Required)

### Full System Diagnostic
```bash
cd /home/experimance/experimance
sudo ./infra/scripts/system_diagnostic.sh full
```

### Remote Access Health Check
```bash
sudo ./infra/scripts/remote_access_monitor.sh check
sudo ./infra/scripts/remote_access_monitor.sh recover
```

### Check Recent Logs
```bash
# System logs
sudo journalctl --since "1 hour ago" | grep -i "error\|fail\|kill"

# SSH logs
sudo journalctl -u ssh.service --since "1 hour ago"

# Tailscale logs
sudo journalctl -u tailscaled.service --since "1 hour ago"

# Experimance service logs
sudo journalctl -u "*@experimance.*" --since "1 hour ago" -o cat
```

## Common Issues and Solutions

### Issue 1: High System Load
**Symptoms**: System is very slow, SSH times out
**Solution**:
```bash
# Check load
uptime

# Find CPU hogs
ps -eo pid,ppid,cmd,%cpu --sort=-%cpu | head -10

# If Experimance core service is using too much CPU:
sudo systemctl restart core@experimance.service

# If system is severely overloaded:
sudo systemctl stop experimance@experimance.target
sudo systemctl start experimance@experimance.target
```

### Issue 2: Memory Exhaustion
**Symptoms**: System becomes unresponsive, OOM killer active
**Solution**:
```bash
# Check memory
free -h

# Check for OOM kills
dmesg | grep -i "killed process"

# Drop caches to free memory
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# Restart services to free memory
sudo systemctl restart experimance@experimance.target
```

### Issue 3: Network/Tailscale Issues
**Symptoms**: Cannot ping, SSH connection refused
**Solution**:
```bash
# Check network interfaces
ip addr show

# Check default route
ip route show default

# Restart networking
sudo systemctl restart NetworkManager

# Restart Tailscale
sudo systemctl restart tailscaled
sudo tailscale up

# Check Tailscale status
tailscale status
tailscale netcheck
```

### Issue 4: Disk Space Full
**Symptoms**: Services fail to start, log errors about disk space
**Solution**:
```bash
# Check disk usage
df -h

# Clean up logs
sudo journalctl --vacuum-time=7d

# Clean up Experimance logs
sudo /home/experimance/experimance/infra/scripts/preventive_maintenance.sh run

# Find large files
find /var/log -size +100M -ls 2>/dev/null
find /home/experimance -size +100M -ls 2>/dev/null
```

### Issue 5: SSH Service Down
**Symptoms**: "Connection refused" error
**Solution**:
```bash
# Check SSH service
sudo systemctl status ssh

# Restart SSH
sudo systemctl restart ssh

# Check SSH configuration
sudo sshd -t

# Check if SSH is listening
sudo ss -tlnp | grep :22
```

## Prevention Setup

### Install Monitoring Tools
```bash
cd /home/experimance/experimance

# Install remote access monitor
sudo ./infra/scripts/remote_access_monitor.sh install
sudo systemctl start remote-access-monitor.service

# Install preventive maintenance
sudo ./infra/scripts/preventive_maintenance.sh install-cron

# View monitoring status
sudo systemctl status remote-access-monitor.service
sudo ./infra/scripts/preventive_maintenance.sh status
```

### Configure Alerts
The health monitoring service can send notifications when issues are detected. Configure it in:
- `projects/experimance/health.toml` for notification settings
- Set up ntfy.sh notifications for instant alerts

### Regular Checks
From your development machine, you can monitor the gallery remotely:
```bash
# Quick health check via SSH
ssh experimance@experimance-pc "sudo /home/experimance/experimance/infra/scripts/system_diagnostic.sh quick"

# Check service status
ssh experimance@experimance-pc "sudo systemctl status experimance@experimance.target"
```

## Emergency Contacts and Resources

- **Physical Access**: Ensure gallery staff know how to access the machine
- **Power Cycle**: Know location of power button/switch
- **Network**: Have backup internet (mobile hotspot) available
- **Support**: Keep this guide accessible from your phone

## Hardware-Level Recovery

If software methods fail:

1. **Power Cycle**: Hold power button for 10 seconds, restart
2. **Safe Mode**: Boot into recovery mode if system won't start normally
3. **Live USB**: Boot from Ubuntu USB to access/repair filesystem
4. **Hardware Reset**: Check hardware connections, reseat components

## Backup Access Methods

Consider setting up:
1. **Secondary User**: Create another user with SSH access
2. **Serial Console**: If hardware supports it
3. **Wake-on-LAN**: For remote power control
4. **VPN**: Alternative to Tailscale
5. **Backup Pi**: Raspberry Pi with SSH access as emergency gateway

## Log Locations

- System diagnostic: `/var/log/experimance/system-diagnostic.log`
- Remote access monitor: `/var/log/experimance/remote-access.log`
- Preventive maintenance: `/var/log/experimance/maintenance.log`
- Health monitoring: `/var/cache/experimance/health/`
- System logs: `journalctl` or `/var/log/syslog`
