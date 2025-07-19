# Experimance Deployment Guide

## Quick Start (2-3 hours)

This guide will get you from zero to a fully monitored Experimance installation.

### Prerequisites

1. **Ubuntu 22.04 LTS** (or compatible)
2. **Root access** for systemd setup
3. **Git repository** cloned to your workspace
4. **Python 3.11+** and `uv` package manager

### Development vs Production

**Development**: Use your current user account and `./scripts/dev <service>` for individual service testing
**Production**: Create dedicated `experimance` user and use systemd services for full deployment

## Development Setup (30 minutes)

### Quick Development Start

```bash
# Clone and setup
git clone https://github.com/RKelln/experimance.git
cd experimance

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Run individual services for development
./scripts/dev core     # Start core service
./scripts/dev display  # Start display service (in another terminal)
./scripts/dev health   # Start health service (in another terminal)

# Available services auto-detected from services/ directory
./scripts/dev          # Shows available services
```

The dev script automatically:
- Sets `EXPERIMANCE_ENV=development`
- Uses `uv run` to manage dependencies
- Auto-detects available services
- Uses local cache directories
- Maps service names to correct modules

## Production Setup (2-3 hours)

### Step 1: Create User and Setup (15 minutes)

```bash
# Create experimance user (production only)
sudo useradd -m -s /bin/bash experimance

# Clone repository
sudo -u experimance git clone https://github.com/RKelln/experimance.git /home/experimance/experimance
cd /home/experimance/experimance

# Install uv for experimance user
sudo -u experimance bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
```

### Step 2: Configure Project (10 minutes)

```bash
# Copy and customize project configuration
sudo -u experimance cp projects/experimance/.env.example projects/experimance/.env

# Edit the .env file with your API keys
sudo -u experimance nano projects/experimance/.env

# The deploy script will create system directories automatically
```

### Step 3: Install Services (30 minutes)

```bash
# Install systemd services (creates directories, installs uv, sets up services)
sudo ./infra/scripts/deploy.sh experimance install

# This automatically:
# - Copies systemd service files configured for uv
# - Sets up /var/cache/experimance and other system directories  
# - Installs uv and project dependencies for experimance user
# - Configures permissions
# - Sets EXPERIMANCE_ENV=production for services
```

### Step 4: Start Services (10 minutes)

```bash
# Start all services
sudo ./infra/scripts/deploy.sh experimance start

# Check status
sudo ./infra/scripts/deploy.sh experimance status

# View detected services for project
./infra/scripts/deploy.sh experimance services
```

### Step 5: Set Up Monitoring (30 minutes)

The health service is now built-in and starts automatically with the other services. Configuration is handled through config files:

#### Configure Health Monitoring

```bash
# Health service configuration
sudo -u experimance nano projects/experimance/health.toml

# Key settings:
# [health_service]
# check_interval_seconds = 300  # 5 minutes
# startup_grace_period_seconds = 60
# notification_level = "warning"  # error, warning, info
# 
# [notifications]
# notify_on_healthy = false
# notify_on_unknown = true
# buffer_time_seconds = 10
```

#### Setup ntfy Push Notifications (Recommended)

```bash
# Configure ntfy in health.toml
sudo -u experimance nano projects/experimance/health.toml

# Add ntfy configuration:
# [[notifications.handlers]]
# type = "ntfy"
# enabled = true
# topic = "experimance-your-installation-name"
# priority = "high"
# server = "https://ntfy.sh"

# Install ntfy app on your phone:
# - Android: https://play.google.com/store/apps/details?id=io.heckel.ntfy
# - iOS: https://apps.apple.com/us/app/ntfy/id1625396347
# - Web: https://ntfy.sh/app

# Subscribe to your topic: experimance-your-installation-name
# Test notifications will be sent automatically
```

#### Alternative: Email Alerts

```bash
# Configure email in health.toml
# [[notifications.handlers]]
# type = "email"
# enabled = true
# smtp_server = "smtp.gmail.com"
# smtp_port = 587
# username = "your-email@gmail.com" 
# password = "your-app-password"
# to_email = "your-email@gmail.com"
```

### Step 6: Configure SSH Access (15 minutes)

```bash
# Enable SSH if not already enabled
sudo systemctl enable ssh
sudo systemctl start ssh

# Add your public key for key-based auth
sudo -u experimance mkdir -p /home/experimance/.ssh
echo "your-public-key-here" | sudo -u experimance tee /home/experimance/.ssh/authorized_keys
sudo -u experimance chmod 600 /home/experimance/.ssh/authorized_keys
sudo -u experimance chmod 700 /home/experimance/.ssh

# Test SSH connection from your machine
ssh experimance@your-installation-ip
```

### Step 7: Test Everything (20 minutes)

```bash
# Check all services are running
sudo ./infra/scripts/deploy.sh experimance status

# Health service will automatically start monitoring and sending notifications
# Check health service logs
sudo journalctl -u experimance-health@experimance -f

# Test service restart
sudo ./infra/scripts/deploy.sh experimance restart

# View health status files
ls -la /var/cache/experimance/health/
cat /var/cache/experimance/health/experimance-core.json
```

## Daily Operations

### Check Status
```bash
# Quick status check  
sudo ./infra/scripts/deploy.sh experimance status

# View specific service
sudo systemctl status experimance-core@experimance

# List all detected services
./infra/scripts/deploy.sh experimance services
```

### Development vs Production Commands

#### Development (Individual Services)
```bash
# Start individual services for development/testing
./scripts/dev core     # Core service
./scripts/dev display  # Display service  
./scripts/dev health   # Health monitoring
./scripts/dev agent    # AI agent service

# Each dev service:
# - Uses current user (no sudo needed)
# - Sets EXPERIMANCE_ENV=development  
# - Uses local cache directories
# - Handles dependencies automatically with uv
```

#### Production (All Services via systemd)
```bash
# Manage all services together (requires sudo for systemd)
sudo ./infra/scripts/deploy.sh experimance start
sudo ./infra/scripts/deploy.sh experimance stop  
sudo ./infra/scripts/deploy.sh experimance restart
sudo ./infra/scripts/deploy.sh experimance status

# Individual service management
sudo systemctl start experimance-core@experimance
sudo systemctl stop experimance-display@experimance
sudo systemctl restart experimance-health@experimance
```

### Restart Services
```bash
# Restart all services (production)
sudo ./infra/scripts/deploy.sh experimance restart

# Restart specific service
sudo systemctl restart experimance-core@experimance

# Development: just stop and restart the ./scripts/dev command
```

### View Logs

#### Service Logs (systemd - Production)
```bash
# View all logs for a service
sudo journalctl -u experimance-core@experimance -f

# View recent logs
sudo journalctl -u experimance-display@experimance -n 50

# View all services for a project
sudo journalctl -u "experimance-*@experimance" -f

# Health service logs (includes notification activity)
sudo journalctl -u experimance-health@experimance -f
```

#### Development Logs
```bash
# Development services log to console by default
# Health status can be checked in local cache
ls -la cache/health/  # Development
cat cache/health/experimance-core.json
```

#### Health Monitoring Logs
Health monitoring uses different locations based on environment:
- **Development**: `cache/health/` (local to project)
- **Production**: `/var/cache/experimance/health/` (system-wide)

```bash
# View health status files
ls -la /var/cache/experimance/health/  # Production
ls -la cache/health/                   # Development

# View specific service health
cat /var/cache/experimance/health/experimance-core.json
```

### Updates
```bash
# Update code (as experimance user)
sudo -u experimance bash -c "cd /home/experimance/experimance && git pull origin main"

# Restart services after update  
sudo ./infra/scripts/deploy.sh experimance restart

# Or use uv to update dependencies
sudo -u experimance bash -c "cd /home/experimance/experimance && uv sync"
```

## Monitoring Setup

### Built-in Health Service

The health service is automatically installed and started with other services. It monitors all services and sends notifications based on configuration.

### Configuration Files

Health monitoring is configured through TOML files:

```bash
# Main health service configuration  
projects/experimance/health.toml

# Project-specific overrides
projects/your-project/health.toml  # Optional overrides
```

### Key Configuration Options

```toml
# projects/experimance/health.toml
[health_service]
check_interval_seconds = 300        # Check every 5 minutes
startup_grace_period_seconds = 60   # Wait 60s after service start
notification_level = "warning"      # error, warning, or info
buffer_time_seconds = 10           # Aggregate notifications for 10s

[notifications]  
notify_on_healthy = false          # Don't spam with healthy notifications
notify_on_unknown = true           # Alert on unknown status
notification_level = "warning"     # Minimum level to send notifications

# ntfy Push Notifications (Recommended)
[[notifications.handlers]]
type = "ntfy"
enabled = true
topic = "experimance-your-installation"  # Make this unique
priority = "high"                       # low, default, high, urgent  
server = "https://ntfy.sh"

# Email Notifications (Alternative/Fallback)
[[notifications.handlers]]
type = "email"
enabled = false                    # Enable if you want email alerts
smtp_server = "smtp.gmail.com"
smtp_port = 587
username = "your-email@gmail.com"
password = "your-app-password"     # Gmail app password
to_email = "your-email@gmail.com"
```

### ntfy Push Notifications Setup

1. **Choose a unique topic**: `experimance-your-installation-name-$(date +%s)`

2. **Install ntfy app**:
   - Android: https://play.google.com/store/apps/details?id=io.heckel.ntfy
   - iOS: https://apps.apple.com/us/app/ntfy/id1625396347  
   - Web: https://ntfy.sh/app

3. **Subscribe to your topic** in the app

4. **Test notifications**: The health service will automatically send test notifications when it starts

### Email Setup (Alternative)

1. **Gmail Setup**:
   - Enable 2-factor authentication
   - Generate an app password: https://support.google.com/accounts/answer/185833
   - Use the app password in `password` field

2. **Other email providers**: Adjust `smtp_server` and `smtp_port` accordingly

### Monitoring Behavior

The health service:
- **Checks all services** every 5 minutes (configurable)
- **Sends notifications** only for status changes or errors (not constantly)
- **Buffers notifications** for 10 seconds to prevent spam  
- **Uses intelligent filtering** based on notification_level
- **Handles startup gracefully** with a grace period for service initialization

### Health Status Files

Health status is stored in JSON files for inter-service communication:

```bash
# Production
ls -la /var/cache/experimance/health/
cat /var/cache/experimance/health/experimance-core.json

# Development  
ls -la cache/health/
cat cache/health/experimance-core.json
```

Each file contains:
```json
{
  "service_name": "experimance-core",
  "status": "healthy",
  "timestamp": "2025-07-18T22:45:00Z",
  "checks": {
    "service_initialization": {"status": "healthy", "message": "Service initialized successfully"},
    "periodic_health_check": {"status": "healthy", "message": "Service is responsive"}
  }
}
```

## Troubleshooting

### Services Won't Start

```bash
# Check service status
sudo systemctl status experimance-core@experimance

# Check logs
sudo journalctl -u experimance-core@experimance -n 50

# Check permissions
ls -la /var/cache/experimance
ls -la /var/log/experimance

# Reset and restart
sudo systemctl reset-failed
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance restart
```

### No Network Communication

```bash
# Check ZMQ ports
netstat -ln | grep ':555'

# Check firewall
sudo ufw status

# Test ZMQ connection
python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect('tcp://localhost:5555')
print('Connected')
"
```

### High Resource Usage

```bash
# Check resource usage
htop

# Check GPU usage (if available)
nvidia-smi

# Check disk space
df -h

# Check service memory usage
sudo systemctl status experimance-core@experimance
```

### Update Issues

```bash
# Check git status
cd /home/experimance/experimance
git status

# Force update
git reset --hard origin/main
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance restart

# Rollback to previous version
git log --oneline -10
git reset --hard <commit-hash>
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance restart
```

## Emergency Procedures

### Complete System Reset

```bash
# Stop all services
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance stop

# Reset git repository
cd /home/experimance/experimance
git reset --hard origin/main
git clean -fd

# Reinstall
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance install
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance start
```

### Remote Recovery

If you can't SSH in:

1. **Physical access**: Connect monitor and keyboard
2. **Serial console**: If available
3. **Remote power cycle**: If you have a smart power strip
4. **Backup system**: Keep a backup USB drive with the installation

### Contact Information

Keep these handy for emergencies:

- **Git repository**: https://github.com/RKelln/experimance
- **Installation IP**: _________________
- **SSH key location**: _________________
- **Email alerts**: _________________
- **Backup location**: _________________

## Security Notes

- **Change default passwords** in dashboard.py
- **Use SSH keys** instead of passwords
- **Keep system updated**: `sudo apt update && sudo apt upgrade`
- **Monitor logs** for suspicious activity
- **Restrict network access** if possible

## Performance Optimization

### For Better Performance

```bash
# Increase systemd service limits
sudo systemctl edit experimance-core@experimance
# Add:
# [Service]
# LimitNOFILE=65536
# LimitNPROC=32768

# Optimize Python
export PYTHONUNBUFFERED=1
export PYTHONOPTIMIZE=1

# GPU optimization
nvidia-smi -pl 300  # Set power limit
nvidia-smi -lgc 1500  # Set graphics clock
```

### For Lower Resource Usage

```bash
# Reduce image generation quality
# Edit projects/experimance/config.toml:
# [image_generation]
# image_size = [512, 512]  # Instead of [1024, 1024]
# guidance_scale = 5.0     # Instead of 7.5

# Reduce logging
# Edit projects/experimance/.env:
# EXPERIMANCE_LOG_LEVEL=WARNING
```

## Backup Strategy

### Automated Backups

```bash
# Daily backup script
cat > /home/experimance/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR="/var/backups/experimance"
mkdir -p $BACKUP_DIR

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz \
  /home/experimance/experimance/projects/ \
  /home/experimance/experimance/data/ \
  /etc/experimance/

# Backup logs (last 7 days)
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz \
  /var/log/experimance/

# Clean old backups (keep 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
EOF

chmod +x /home/experimance/backup.sh

# Add to cron
echo "0 3 * * * /home/experimance/backup.sh" | sudo crontab -
```

This infrastructure setup provides:
- ✅ **Easy service management** with systemd
- ✅ **Auto-recovery** with restart policies
- ✅ **Phone/desktop notifications** via email
- ✅ **Remote SSH access** with key-based auth
- ✅ **Remote updates** with rollback capability
- ✅ **Project switching** for different installations
- ✅ **Web dashboard** for mobile monitoring
- ✅ **Comprehensive logging** and health checks
- ✅ **Backup and recovery** procedures

The total setup time is about 2-3 hours, and it provides enterprise-grade monitoring and management for your art installation.
