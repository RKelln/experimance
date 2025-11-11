# Experimance Infrastructure Summary

## What We've Built

A comprehensive infrastructure solution for remote monitoring and management of the Experimance installation, designed for minimal maintenance during month-long exhibitions.

## Key Features

### 1. **Easy Service Management**
- **Development setup**: `./infra/scripts/deploy.sh experimance install dev` (no sudo needed)
- **Production deployment**: `sudo ./infra/scripts/deploy.sh experimance install prod`
- **Project switching**: Support for multiple installations (experimance, fire, etc.)
- **Service lifecycle**: Start, stop, restart, status checking all automated
- **Explicit modes**: Clear separation between development and production environments

### 2. **Auto-Recovery**
- **systemd integration**: Native Ubuntu/MacOS service management with automatic restart
- **Failure detection**: Services automatically restart on crashes
- **Resource limits**: Prevent runaway processes from consuming all resources

### 3. **Remote Monitoring & Access**
- **Push notifications**: Simple, reliable ntfy.sh notifications to your phone
- **Health checks**: Automated monitoring of service health

### 4. **Safe Updates** (TODO)
- **Git integration**: Simple `git pull` workflow

### 5. **Kiosk Mode & Display Support**
- **Kiosk mode management**: Safe, reversible kiosk mode for Ubuntu 24.04 installations
- **Gallery-ready features**: Disables screen lock, notifications, unattended upgrades, and sleep
- **Wayland compatibility**: Automatic detection and configuration for Ubuntu 24.04+ Wayland sessions
- **Independent service control**: Services can restart individually without affecting others
- **No cascade failures**: Fixed systemd configuration prevents one service failure from stopping all services

### 6. **Multi-Machine Deployment**
- **Distributed deployments**: Deploy services across multiple machines (Ubuntu + macOS, etc.)
- **TOML configuration**: Simple `deployment.toml` files define machine assignments and service distribution
- **Hostname override testing**: Test deployment configurations from any machine without being on the target
- **Cross-platform support**: Works with both systemd (Linux) and launchd (macOS) service management

### 7. **Gallery Hour Automation**
- **Automatic scheduling**: Services start/stop during gallery hours (Tuesday-Saturday, 11AM-6PM)
- **Manual override**: Gallery staff can immediately start/stop services for special events
- **Preserves auto-restart**: Services still auto-start after reboot and restart on failure
- **LaunchAgent integration**: Native macOS scheduling using `StartCalendarInterval`
- **Multiple schedules**: Gallery hours, daily schedule, or custom timing
- **TouchDesigner support**: Works with both Python services and TouchDesigner applications


## Files

```
infra/
├── README.md                  # This summary
├── systemd/                   # Linux service definitions (updated for standardized naming)
│   ├── core@.service          # Core service (service_type@project format)
│   ├── display@.service       # Display service with Wayland support
│   ├── health@.service        # Health monitoring service
│   ├── agent@.service         # Agent service
│   ├── audio@.service         # Audio service
│   ├── image_server@.service  # Image generation service
│   ├── experimance@.target    # Service group target
│   └── reset-on-input.service # Reset on input service
├── launchd/                   # macOS service definitions (launchd .plist files)
│   └── README.md              # macOS launchd setup and management guide
├── scripts/                   # Management automation
│   ├── backup_images.sh      # Backup image files
│   ├── deploy.sh              # Main deployment script (install, start, stop, setup schedules)
│   ├── deployment_utils.py    # Multi-machine deployment utilities
│   ├── get_deployment_services.py # Consolidated service detection with hostname override
│   ├── get_project_services.py # Dynamic service detection (single-machine fallback)
│   ├── get_service_module.py  # Service module detection utilities
│   ├── healthcheck.py         # Health monitoring script
│   ├── ia_gallery.py          # Gallery automation utilities (specific to Feed the Fires @ InterAccess)
│   ├── kiosk_mode.sh          # Enable/disable kiosk mode for art installation
│   ├── launchd_scheduler.sh   # macOS LaunchAgent scheduling
│   ├── matter_scheduler.py    # Matter protocol scheduling
│   ├── matter_scheduler.sh    # Matter protocol scheduling (shell wrapper)
│   ├── preventive_maintenance.sh # Automated maintenance to prevent SSH lockouts
│   ├── remote_access_monitor.sh # Monitor SSH/Tailscale connectivity with auto-recovery
│   ├── reset.sh               # System/service reset functionality
│   ├── reset_on_input.py      # Interactive reset trigger script
│   ├── secure_ssh.sh          # SSH security configuration
│   ├── setup_display_env.sh   # Display environment setup
│   ├── shutdown.sh            # Graceful system shutdown
│   ├── startup.sh             # System startup script (for cron/systemd)
│   ├── status.sh              # Service status checking
│   ├── system_diagnostic.sh   # Diagnose SSH lockout causes and system issues
│   ├── touchdesigner_agent.sh # TouchDesigner integration
│   ├── update.sh              # Safe update script with rollback capability
│   ├── wait_for_display.sh    # Wait for display to be ready
│   ├── README_LAUNCHD_SCHEDULER.md # LaunchAgent scheduling documentation
│   ├── README_TOUCHDESIGNER.md # TouchDesigner integration guide
│   └── README.md              # Scripts documentation
├── docs/                      # Documentation
│   ├── deployment.md          # Detailed deployment instructions
│   ├── installation_teardown.md # How to disable or remove installations
│   ├── emergency-reference.md # Quick emergency procedures
│   ├── emergency-ssh-recovery.md # SSH lockout recovery guide
│   └── new_machine_setup.md   # Fresh machine setup procedures
│   
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

## Multi-Machine Deployment

For complex installations requiring multiple machines (e.g., Ubuntu Linux + macOS Mac mini), you can define deployment configurations that specify which services run on which machines.

### Configuration File

Create `projects/<project>/deployment.toml`:

```toml
[machines.ubuntu]
hostname = "fire-ubuntu.local"
services = ["core", "display", "image_server", "health"]
user = "experimance"

[machines.macos]
hostname = "fire-macos.local"
services = ["agent", "health"]
user = "FireProject"

[services]
core.module_name = "fire_core"
agent.module_name = "fire_agent"
display.module_name = "fire_display"
image_server.module_name = "fire_image_server"
health.module_name = "fire_health"
```

### Multi-Machine Commands

```bash
# Test deployment configuration from any machine
uv run python infra/scripts/get_deployment_services.py fire fire-ubuntu.local
# Output: core@fire, display@fire, image_server@fire, health@fire

uv run python infra/scripts/get_deployment_services.py fire fire-macos.local  
# Output: agent@fire, health@fire

# Deploy on each machine (run these commands on the respective machines)
# On Ubuntu machine:
sudo ./infra/scripts/deploy.sh fire install prod
sudo ./infra/scripts/deploy.sh fire start

# On macOS machine:
sudo ./infra/scripts/deploy.sh fire install prod
sudo ./infra/scripts/deploy.sh fire start
```

### How It Works

1. **Service Detection**: The deployment system automatically detects if a project has `deployment.toml`
2. **Hostname Matching**: Uses `hostname` command to determine which machine configuration to use
3. **Service Filtering**: Only deploys services assigned to the current machine
4. **Module Name Mapping**: Uses custom module names (e.g., `fire_core` instead of `core`)
5. **User Configuration**: Different users can be specified per machine
6. **Fallback Compatibility**: Projects without `deployment.toml` still work with existing single-machine logic

### Testing Deployment Configurations

The hostname override feature allows testing configurations from any machine:

```bash
# Test what services would run on Ubuntu machine (from any machine)
uv run python infra/scripts/get_deployment_services.py fire fire-ubuntu.local

# Test what services would run on macOS machine (from any machine)  
uv run python infra/scripts/get_deployment_services.py fire fire-macos.local

# See what services would run on current machine (auto-detect hostname)
uv run python infra/scripts/get_deployment_services.py fire
```

This is especially useful for:
- **Development**: Test multi-machine configurations on your dev machine
- **Configuration Validation**: Verify deployment settings before deploying
- **Debugging**: Troubleshoot service assignment issues
- **CI/CD**: Automated testing of deployment configurations

## Quick Start

For detailed deployment instructions, see [`infra/docs/deployment.md`](docs/deployment.md).

### Development Setup
```bash
# Quick development setup (no sudo needed)
./infra/scripts/deploy.sh experimance install dev
./scripts/dev core     # Start core service
./scripts/dev display  # Start display service
./scripts/dev          # Show available services
```

### Production Setup
```bash
# Full production deployment
sudo useradd -m -s /bin/bash experimance
sudo ./infra/scripts/deploy.sh experimance install prod
sudo ./infra/scripts/deploy.sh experimance start

# Service management
sudo systemctl start experimance@experimance.target    # Start all services
sudo systemctl stop experimance@experimance.target     # Stop all services
sudo ./infra/scripts/deploy.sh experimance status      # Check status
```

## Advanced Features

### Scheduled Tasks
Services can be automatically started/stopped on schedules. See [`infra/docs/deployment.md`](docs/deployment.md) for scheduling setup.

### Kiosk Mode
Safe, reversible kiosk mode for gallery installations. See [`infra/docs/deployment.md`](docs/deployment.md) for kiosk mode management.


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
3. **Multiple Projects**: Same templates can run `core@fire.service`, etc.

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

## Systemd Configuration Improvements

### Independent Service Management ✅
Recent improvements to the systemd configuration provide better service isolation:

- **No cascade failures**: Services use `Wants` + `PartOf` instead of `BindsTo` to prevent one service failure from stopping all others
- **Independent restarts**: Individual services can restart without affecting the target or other services
- **Desktop session dependency**: Uses systemd's native `graphical-session.target` instead of custom scripts per service
- **Clean startup/shutdown**: Services start when target starts, stop when target stops, but can operate independently

### Service Dependencies
```ini
# Each service now uses this pattern:
[Unit]
After=experimance@%i.target graphical-session.target
Wants=experimance@%i.target graphical-session.target  
PartOf=experimance@%i.target

# Target depends on desktop readiness:
[Unit]  
After=network.target user@1000.service graphical-session.target
Wants=network.target user@1000.service graphical-session.target
```

### Benefits
- ✅ Start all services: `sudo systemctl start experimance@experimance.target`
- ✅ Stop all services: `sudo systemctl stop experimance@experimance.target`  
- ✅ Start individual service: `sudo systemctl start core@experimance.service` (activates target if needed)
- ✅ Restart individual service: `sudo systemctl restart display@experimance.service` (if target active)
- ✅ Service failures don't cascade to other services
- ✅ No rapid cycling or race conditions

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

# Remote access monitoring
sudo systemctl status remote-access-monitor.service
sudo ./infra/scripts/remote_access_monitor.sh status
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

### Systemd Issues
```bash
# Check for old/duplicate systemd files
sudo find /etc/systemd/system -name "*experimance*" | sort

# Clean state should show:
# /etc/systemd/system/experimance@.target
# /etc/systemd/system/experimance@.target.wants/
# /etc/systemd/system/experimance@.target.wants/[service]@experimance.service
# /etc/systemd/system/multi-user.target.wants/experimance@experimance.target
# /etc/systemd/system/[service]@.service (template files)

# Remove old duplicate files if found:
sudo rm /etc/systemd/system/multi-user.target.wants/*-health@*.service  # Old naming
sudo systemctl daemon-reload

# Check target status for cycling issues:
sudo journalctl -u experimance@experimance.target --since "10 minutes ago"
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


## Remote Access Monitoring and Troubleshooting

To prevent and diagnose SSH lockout issues, we've added comprehensive monitoring and diagnostic tools:

### Remote Access Monitor
Monitors SSH, Tailscale, and system health with automatic recovery every 60 seconds:
```bash
# Run one-time health check
sudo ./infra/scripts/remote_access_monitor.sh check

# Run health check with automatic recovery
sudo ./infra/scripts/remote_access_monitor.sh recover

# Install as systemd service for continuous monitoring (RECOMMENDED)
sudo ./infra/scripts/remote_access_monitor.sh install
sudo systemctl start remote-access-monitor.service

# View monitoring status and recent logs
sudo ./infra/scripts/remote_access_monitor.sh status

# Check service status
sudo systemctl status remote-access-monitor.service

# Follow live monitoring logs
sudo journalctl -u remote-access-monitor.service -f

# View health state files
sudo cat /var/cache/experimance/remote-access-state.json
sudo cat /var/cache/experimance/health/remote_access_health.json
```

**Features:**
- **Conservative recovery**: Only attempts recovery after 2 consecutive failures (2+ minutes)
- **SSH monitoring**: Service status, port listening, connection counting
- **Tailscale monitoring**: IP assignment, connectivity, DERP health
- **Network monitoring**: Internet, DNS, gateway connectivity
- **System monitoring**: Memory, disk, CPU load with critical thresholds
- **Auto-restart capability**: Restarts SSH and Tailscale services if needed
- **Health logging**: Detailed logs and JSON state files for debugging

### System Diagnostics
Identify potential causes of SSH lockouts:
```bash
# Quick health check
sudo ./infra/scripts/system_diagnostic.sh quick

# Comprehensive system diagnostic
sudo ./infra/scripts/system_diagnostic.sh full
```

### Preventive Maintenance
Automated maintenance to prevent issues:
```bash
# Run maintenance tasks manually
sudo ./infra/scripts/preventive_maintenance.sh run

# Install automatic maintenance (every 6 hours)
sudo ./infra/scripts/preventive_maintenance.sh install-cron

# Check maintenance status
sudo ./infra/scripts/preventive_maintenance.sh status
```

### Common Causes of SSH Lockouts
Based on analysis of your system, here are common issues that can prevent SSH access:

1. **High System Load**: Core service consuming excessive CPU (161% in logs)
2. **Memory Pressure**: Services using high memory causing system slowdown
3. **Network Issues**: Tailscale connectivity problems or DNS failures
4. **Service Failures**: Critical services (SSH, systemd) becoming unresponsive
5. **Disk Space**: Full disks preventing log writes and service operation

The monitoring tools will detect and automatically recover from many of these issues.

---

## Remote access with Tailscale

Assuming you have already set up local passwordless access over ssh for a user `experimance`. 
Tailscale allows use of your existing key‑only OpenSSH over a private WireGuard network. 
No port‑forwarding, no public exposure. MagicDNS lets you `ssh` by hostname.

---

### 1) Install Tailscale (Ubuntu 24.04) — on **both** machines
```bash
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | sudo tee /etc/apt/sources.list.d/tailscale.list

sudo apt-get update && sudo apt-get install -y tailscale
```

### 2) Join the tailnet

Local dev machine (interactive login):
```bash
sudo tailscale up --accept-dns=true
# Follow the browser link to authenticate
```

Gallery machine (headless): create a pre‑auth key in the Tailscale admin, then:
```bash
sudo tailscale up --accept-dns=true --hostname=experimance-pc --advertise-tags=tag:experimance
```
Settings persist. You can re-run tailscale up ... later to change flags.

You may need to use the Tailscale admin to turn off key expiry on the tagged machine, although tagging is supposed to do that automatically.

### 3) Enable and use MagicDNS

In the Tailscale admin: *Admin → DNS*, ensure MagicDNS is enabled (usually on by default).

Verify on a node:
```bash
tailscale dns status             # shows MagicDNS suffix & resolvers
tailscale dns query experimance
```

SSH with MagicDNS short name:
```bash
ssh experimance@experimance-pc
```

### 1) Keep OpenSSH locked down (recommended)

Your existing sshd continues to listen on port 22 locally, but it's only reachable over Tailscale. 
Keep key-only auth and disable passwords:

**Safe automated method (recommended):**
```bash
# Check current SSH configuration
sudo ./infra/scripts/secure_ssh.sh status

# Verify SSH keys are properly set up first
sudo ./infra/scripts/secure_ssh.sh test-keys

# Safely apply SSH hardening (creates backup, tests config, uses reload not restart)
sudo ./infra/scripts/secure_ssh.sh secure

# If needed, restore from backup
sudo ./infra/scripts/secure_ssh.sh restore /var/backups/experimance/ssh/sshd_config_TIMESTAMP.backup
```

**Manual method (advanced users only):**
```bash
# /etc/ssh/sshd_config (essentials)
PasswordAuthentication no
ChallengeResponseAuthentication no
PermitRootLogin no
PubkeyAuthentication yes

sudo systemctl restart ssh
```

The `secure_ssh.sh` script safely hardens SSH by:
- Creating backups before changes
- Testing configuration syntax
- Testing on alternate port first (if connected remotely)
- Using `systemctl reload` instead of `restart` for safer application
- Adding security limits (MaxAuthTries, LoginGraceTime, MaxSessions)

### 5) (Optional) Restrict tailnet access to just you and just SSH

Use tags + a minimal policy (new “grants” style).

*Admin → Access controls → Edit* policy (replace placeholders):
```json
{
  "tagOwners": { "tag:gallery": ["you@example.com"] },
  "grants": [
    { "src": ["you@example.com"], "dst": ["tag:experimance"], "ip": ["tcp:22"] }
  ]
}
```
Remove the default “allow all” grant if it’s present. Only tag owners can apply tag:gallery.

### 6) Sanity checks & useful commands

tailscale status                    # see nodes, names, IPs, tags
tailscale ping experimance-pc       # connectivity; shows the 100.x address
tailscale ip -4 experimance-pc      # print its Tailscale IPv4
ssh -v experimance@experimance-pc   # verbose SSH if troubleshooting
systemctl status ssh                # confirm OpenSSH is running (gallery)

### 7) Notes

No router changes: NAT traversal is automatic.
End‑to‑end encryption: traffic rides an authenticated WireGuard tunnel.
No change to SSH workflow: same keys, same users—just a private path.
Autostart: tailscaled is enabled; tailscale up settings persist across reboots.