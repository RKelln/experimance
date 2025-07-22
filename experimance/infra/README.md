# Experimance Infrastructure Summary

## What We've Built

A comprehensive infrastructure solution for remote monitoring and management of the Experimance installation, designed for minimal maintenance during month-long exhibitions.

## Key Features ✅

### 1. **Easy Service Management**
- **Development setup**: `./infra/scripts/deploy.sh experimance install dev` (no sudo needed)
- **Production deployment**: `sudo ./infra/scripts/deploy.sh experimance install prod`
- **Project switching**: Support for multiple installations (experimance, sohkepayin, etc.)
- **Service lifecycle**: Start, stop, restart, status checking all automated
- **Explicit modes**: Clear separation between development and production environments

### 2. **Auto-Recovery**
- **systemd integration**: Native Ubuntu service management with automatic restart
- **Failure detection**: Services automatically restart on crashes
- **Resource limits**: Prevent runaway processes from consuming all resources

### 3. **Remote Monitoring**
- **Push notifications**: Simple, reliable ntfy.sh notifications to your phone
- **Email alerts**: Configurable SMTP notifications as fallback
- **Web dashboard**: Mobile-friendly interface at `http://installation-ip:8080`
- **Health checks**: Automated monitoring every 5 minutes
- **SSH access**: Secure remote access for troubleshooting

### 4. **Safe Updates**
- **Rollback capability**: Automatic rollback on failed updates
- **Backup system**: Configuration and state backups before changes
- **Git integration**: Simple `git pull` workflow with safety checks

### 5. **Kiosk-Style Display Support**
- **Wayland compatibility**: Automatic detection and configuration for Ubuntu 24.04+ Wayland sessions
- **Desktop session waiting**: Services wait for user login before starting
- **Xwayland integration**: Seamless fallback to X11 applications on Wayland
- **Auto-recovery**: Display service automatically adapts to desktop environment changes

## Files

```
infra/
├── systemd/                    # Service definitions (updated for standardized naming)
│   ├── core@.service          # Core service (service_type@project format)
│   ├── display@.service       # Display service with Wayland support
│   ├── health@.service        # Health monitoring service
│   ├── agent@.service         # Agent service
│   ├── audio@.service         # Audio service
│   ├── image_server@.service  # Image generation service
│   └── experimance@.target    # Service group target
├── scripts/                    # Management automation
│   ├── deploy.sh              # Main deployment script (simplified, no special cases)
│   ├── get_project_services.py # Dynamic service detection
│   ├── setup_display_env.sh   # Wayland/Xwayland display environment detection
│   ├── wait_for_desktop_session.sh # Wait for user login before display service
│   └── (other utility scripts)
└── docs/                      # Documentation
    ├── deployment.md          # Complete deployment guide
    └── README.md              # This summary

scripts/                       # Development tools
└── dev                        # Development service runner

services/health/               # Dedicated health monitoring service
├── src/experimance_health/
│   ├── health_service.py      # Health monitoring implementation
│   └── config.py             # Health service configuration
├── config.toml               # Default health configuration
└── (project overrides in projects/*/health.toml)

libs/common/src/experimance_common/
├── health.py                 # Health system utilities
├── notifications.py          # Notification handlers (ntfy, email, etc.)
└── (other common utilities)
```

## Quick Start Commands

### Production (Full Deployment)
Installs functional systemd system services (that wait for the user session to start).

```bash
# One-time setup (creates experimance user and system services)
sudo useradd -m -s /bin/bash experimance
sudo ./infra/scripts/deploy.sh experimance install prod
sudo ./infra/scripts/deploy.sh experimance start

# View available services
sudo ./infra/scripts/deploy.sh experimance services

# Group service management (the goal!)
sudo systemctl start experimance@experimance.target    # Starts all services
sudo systemctl stop experimance@experimance.target     # Stops all services
sudo systemctl restart experimance@experimance.target  # Restarts all services

# Script-based management (still works)
sudo ./infra/scripts/deploy.sh experimance start
sudo ./infra/scripts/deploy.sh experimance stop
sudo ./infra/scripts/deploy.sh experimance restart

# Status checking
sudo ./infra/scripts/deploy.sh experimance status
sudo ./infra/scripts/deploy.sh experimance diagnose

# Individual service control (new format: service_type@project)
sudo systemctl status core@experimance
sudo systemctl status display@experimance  # Includes Wayland support
sudo systemctl restart agent@experimance

# follow logs
sudo journalctl -u image_server@experimance.service -f

# logs since last started target service (all services at once!)
sudo journalctl --since "2025-07-22 11:13:39" -u "*@experimance.*" -o cat  # clean, human-readable

# other output formats
sudo journalctl --since "2025-07-22 11:13:39" -u "*@experimance.*" --no-hostname              # with timestamps
sudo journalctl --since "2025-07-22 11:13:39" -u "*@experimance.*" --no-hostname -o short-precise  # precise timestamps

# get timestamp of last target start (for --since)  
sudo journalctl -u experimance@experimance.target --no-pager -n 20

# follow all service logs live (clean format)
sudo journalctl -f -u "*@experimance.*" -o cat
```

## Systemd Templates vs Instances (Important!)

Our systemd setup uses **template services** for multi-project support:

### Template Files (What We Install)
- `core@.service` - Template for core service
- `display@.service` - Template for display service  
- `experimance@.target` - Template for project target
- Located in `/etc/systemd/system/`

### Instance Services (What Actually Runs)
- `core@experimance.service` - Running instance for experimance project
- `display@experimance.service` - Running instance for experimance project
- `experimance@experimance.target` - Target instance for experimance project
- Created automatically by systemd when you start services

### How It Works
1. **Template**: `core@.service` contains `%i` placeholder for project name
2. **Instance**: `systemctl start core@experimance.service` creates instance with `%i=experimance`
3. **Multiple Projects**: Same templates can run `core@sohkepayin.service`, etc.

### Common Commands
```bash
# Start/stop instances (what you actually run)
sudo systemctl start core@experimance.service
sudo systemctl stop display@experimance.service  
sudo systemctl status agent@experimance.service

# Check template files exist (troubleshooting)
ls -la /etc/systemd/system/*@.service
ls -la /etc/systemd/system/*@.target

# See all instances for a project
sudo systemctl status "*@experimance"
```

### Development (Testing on Any Machine)
```bash
# Development setup (uses current user, local directories, no systemd)
# Automatically installs pyenv + latest Python 3.11.x + uv if needed
./infra/scripts/deploy.sh experimance install dev

# Run individual services for development/testing  
./scripts/dev core     # Start core service
./scripts/dev display  # Start display service
./scripts/dev health   # Start health monitoring
./scripts/dev          # Show available services

# Development automatically:
# - Uses current user (no sudo needed)
# - Installs pyenv if not present
# - Installs latest Python 3.11.x (required for pyrealsense2)
# - Installs uv if not present
# - Uses local cache directories (./cache/)
# - Skips systemd service installation
```

## Emergency Quick Reference

### Check Status
```bash
# Development
./scripts/dev          # Show available services
ps aux | grep experimance  # Check running dev services

# Production (updated service names) 
sudo ./infra/scripts/deploy.sh experimance status
sudo systemctl status "*@experimance"  # All project services
sudo systemctl status core@experimance display@experimance  # Specific services
```

### Restart Everything
```bash
# Development: Stop dev services (Ctrl+C) and restart them

# Production
sudo ./infra/scripts/deploy.sh experimance restart
```

### Install Issues
```bash
# If install fails, the script will show explicit error messages:
# - Missing dependencies: Install pyenv/uv manually or check PATH
# - Missing project: Verify project exists in projects/ directory
# - Permission errors: Check sudo usage for production mode
# - Service detection failed: Verify get_project_services.py exists and works

# Python version issues (pyrealsense2 requires Python 3.11.x):
# The script automatically installs latest Python 3.11.x via pyenv
# If you get pyenv PATH issues:
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
# Then retry the install

# If uv PATH issues occur during development install:
source ~/.bashrc  # Refresh PATH to include ~/.local/bin
# Then retry the install

# Test service detection manually:
cd /path/to/experimance
uv run python infra/scripts/get_project_services.py experimance
# or if uv not in PATH:
~/.local/bin/uv run python infra/scripts/get_project_services.py experimance
```

### View Logs
```bash
# Development: Logs go to console where you started ./scripts/dev
tail logs/dev/*

# Production (updated service names)
sudo journalctl -u "*@experimance.*" -f -o cat        # All services, follow live (clean, human-readable)
sudo journalctl -u "*@experimance.*" -o cat           # All services, recent logs (clean, human-readable)
sudo journalctl -u core@experimance -f     # Individual service logs
sudo journalctl -u health@experimance -f   # Health notifications
```

### Health Monitoring
```bash
# View health status files
ls -la /var/cache/experimance/health/     # Production
ls -la cache/health/                      # Development

# Check health service (production only)
sudo journalctl -u health@experimance -f
```

## Monitoring Options

1. **Push Notifications**: Built-in ntfy.sh support for instant phone alerts
2. **Email Alerts**: Traditional email notifications as alternative
3. **Health Service**: Dedicated monitoring service with intelligent filtering
4. **File-based Status**: JSON health status files for inter-service communication

### Quick Setup

```bash
# Development setup (test on any machine)
./infra/scripts/deploy.sh experimance install dev
./scripts/dev health  # Start health monitoring in development

# Production setup (deploy to exhibition machine)
sudo ./infra/scripts/deploy.sh experimance install prod
sudo ./infra/scripts/deploy.sh experimance start

# Configure health monitoring
# Edit projects/experimance/health.toml for notification settings

# Install ntfy app on your phone and subscribe to your topic  
# The health service automatically sends test notifications when starting

# View health status
ls -la /var/cache/experimance/health/     # Production  
ls -la cache/health/                      # Development
```

## Install Modes

### Development Mode (`dev`)
- **Purpose**: Testing, development, or temporary setups
- **Requirements**: No sudo needed, works on any machine
- **User**: Current user (whoever runs the script)
- **Python**: Automatically installs latest Python 3.11.x via pyenv (required for pyrealsense2)
- **Dependencies**: Automatically installs pyenv and uv for current user if needed
- **Directories**: Uses local `./cache/` directory
- **Services**: No systemd services installed, use `./scripts/dev <service>`
- **Command**: `./infra/scripts/deploy.sh experimance install dev`

### Production Mode (`prod`)
- **Purpose**: Exhibition deployment with systemd management
- **Requirements**: Sudo needed, experimance user must exist
- **User**: experimance user (created separately)
- **Python**: Installs latest Python 3.11.x via pyenv for experimance user
- **Dependencies**: Installs pyenv and uv for experimance user
- **Directories**: Uses system directories `/var/cache/experimance/`
- **Services**: Full systemd service installation and management
- **Command**: `sudo ./infra/scripts/deploy.sh experimance install prod`

## Error Handling

The deploy script now fails explicitly instead of making assumptions:

- **Missing dependencies**: Clear error with installation instructions
- **Missing files**: Explicit error about what file is missing
- **Failed operations**: No silent fallbacks, all failures are reported
- **Service detection**: Must succeed or installation fails
- **Mode detection**: Must be explicit for install, clear error if ambiguous

## Nice-to-Have Features

- [ ] **Backup to cloud**: Automatic configuration backup to Google Drive/Dropbox
- [ ] **Performance metrics**: CPU, memory, GPU usage trends
- [ ] **Remote terminal**: Web-based terminal access
- [ ] **Log search**: Web interface for searching logs
- [ ] **Metrics dashboard**: Grafana-style performance graphs

## Risk Mitigation

- **Hardware failure**: Keep backup hardware on-site
- **Network loss**: Mobile hotspot backup
- **Power outage**: UPS for critical components
- **Service crashes**: Automatic restart + alerting
- **Configuration errors**: Version control + rollback

## Cost Estimate

- **Infrastructure**: $0 (using existing hardware)
- **Monitoring services**: $0 (ntfy.sh)
- **Time investment**: 2-3 hours initial, 1-2 hours monthly
