# Infrastructure Scripts

This directory contains various scripts for managing and deploying the Experimance project infrastructure.

## Core Deployment Scripts

### `deploy.sh`
Main deployment script for Experimance services. Supports multiple projects, platforms (Linux/macOS), and deployment modes.

```bash
./deploy.sh [project_name] [action] [mode] [--hostname=<hostname>]
```

**Actions**: install, start, stop, restart, status, services, diagnose
**Modes**: dev, prod
**Examples**:
- `./deploy.sh fire install dev`
- `./deploy.sh fire start`
- `./deploy.sh experimance restart prod`

### `touchdesigner_agent.sh`
**macOS only** - Creates and manages LaunchAgents for TouchDesigner files.

```bash
./touchdesigner_agent.sh <touchdesigner_file> [action] [--project=<project>]
```

**Actions**: install, start, stop, restart, status, uninstall
**Examples**:
- `./touchdesigner_agent.sh /path/to/fire.toe install`
- `./touchdesigner_agent.sh /path/to/fire.toe status --project=fire`

See [`README_TOUCHDESIGNER.md`](README_TOUCHDESIGNER.md) for detailed documentation.

### `launchd_scheduler.sh`
**macOS only** - Adds gallery hour scheduling to existing LaunchAgent services. Perfect for gallery installations that need automatic startup/shutdown with manual override capabilities.

```bash
./launchd_scheduler.sh <project> <action> [schedule_type]
```

**Actions**: setup-schedule, remove-schedule, show-schedule, manual-start, manual-stop, manual-unload
**Schedule Types**: gallery, daily, custom
**Examples**:
- `./launchd_scheduler.sh fire setup-schedule gallery`
- `./launchd_scheduler.sh fire manual-stop`
- `./launchd_scheduler.sh fire show-schedule`

See [`README_LAUNCHD_SCHEDULER.md`](README_LAUNCHD_SCHEDULER.md) for detailed documentation.

### `ia_gallery.py`
**Multi-machine gallery control** - Controls Fire project services across Ubuntu (ia360) and macOS (iamini) machines via SSH. 
Designed for InterAccess "Feed the Fires" gallery installation, but can be used as an example for other installations.

```bash
python infra/scripts/ia_gallery.py                  # Interactive menu
python infra/scripts/ia_gallery.py --start          # Start all services
python infra/scripts/ia_gallery.py --stop           # Stop all services
python infra/scripts/ia_gallery.py --status         # Show service status
python infra/scripts/ia_gallery.py --install        # Install as systemd service (Ubuntu only)
```

**Machine configuration**:
- **ia360 (Ubuntu)**: Core, Image Server, Display, Health services
- **iamini (macOS)**: Agent, Health services + TouchDesigner
- **SSH shortcuts**: Uses `ia360` and `iamini` hostnames from ~/.ssh/config

Works with `launchd_scheduler.sh` for coordinated gallery hour automation.

## System Management Scripts

### `startup.sh`
System startup initialization script.

### `shutdown.sh`
Graceful system shutdown script.

### `reset.sh`
System reset and cleanup script.

### `status.sh`
Display system and service status information.

### `update.sh`
Update system and services.

## Environment and Display Scripts

### `setup_display_env.sh`
Configure display environment settings.

### `kiosk_mode.sh`
Set up kiosk mode for display systems.

## Monitoring and Maintenance Scripts

### `healthcheck.py`
Health monitoring and status checking for services.

### `preventive_maintenance.sh`
Automated maintenance tasks and system health checks.

### `system_diagnostic.sh`
Comprehensive system diagnostics and troubleshooting.

### `remote_access_monitor.sh`
Monitor and manage remote access to the system.

## Special Purpose Scripts

### `reset_on_input.py`
Monitor for input events and trigger system resets.

### `secure_ssh.sh`
Configure secure SSH access.

## Utility Scripts

### `deployment_utils.py`
Python utilities for deployment configuration parsing.

### `get_deployment_services.py`
Get deployment service configuration.

### `get_project_services.py` 
Get project-specific service configuration.

## TouchDesigner Setup Workflow (macOS)

For gallery installations using TouchDesigner, follow this workflow:

### 1. Install and Test TouchDesigner Service
Use `touchdesigner_agent.sh` to create the TouchDesigner LaunchAgent and verify it works:

```bash
# Install TouchDesigner LaunchAgent
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe install --project=fire

# Test TouchDesigner service individually
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe start
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe status
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe stop
```

### 2. Add Gallery Hour Automation
Use `launchd_scheduler.sh` to coordinate TouchDesigner + Python services with gallery hours:

```bash
# Set up gallery hour scheduling for ALL services (TouchDesigner + Python)
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery

# Verify gallery scheduling is active
./infra/scripts/launchd_scheduler.sh fire show-schedule

# Test coordinated service management
./infra/scripts/launchd_scheduler.sh fire manual-start    # Start everything
./infra/scripts/launchd_scheduler.sh fire manual-stop     # Stop everything
```

### 3. Why This Two-Step Process?

- **`touchdesigner_agent.sh`**: Creates and tests individual TouchDesigner LaunchAgent
- **`launchd_scheduler.sh`**: Coordinates ALL services (TouchDesigner + Python) for gallery operations

This separation allows you to:
- Install and test TouchDesigner independently
- Add/remove gallery scheduling without affecting individual services
- Use coordinated control for reliable gallery operations

## Usage Patterns

### Multi-Project Support
Most scripts support multiple projects (experimance, fire, etc.) through:
- Project-specific configuration files in `projects/<project>/`
- Environment variable `EXPERIMANCE_PROJECT=<project>`
- Command-line project specification

### Platform Support
- **Linux**: Full systemd service management support
- **macOS**: LaunchAgent support for user-level services
- **Windows**: Limited support through WSL

### Development vs Production
- **Dev mode**: Services run as current user, easier debugging
- **Prod mode**: Services run as system services with proper isolation

## Configuration

Scripts use configuration from:
- `projects/<project>/deployment.toml` - Multi-machine deployment config
- `projects/<project>/<service>.toml` - Service-specific config
- Environment variables `EXPERIMANCE_<SECTION>_<KEY>=value`

## Logging

Scripts create logs in:
- **Linux**: `/var/log/experimance/` (prod) or `logs/` (dev)
- **macOS**: `~/Library/Logs/experimance/`
- **Development**: Local `logs/` directory

## Security Notes

- Scripts handle user permissions automatically
- Production services run with appropriate isolation
- SSH and remote access are secured by default
- TouchDesigner LaunchAgents run as current user for GUI access

## Troubleshooting

1. **Permission errors**: Ensure proper user permissions for the target mode
2. **Service won't start**: Check logs and service configuration
3. **Platform issues**: Verify platform-specific dependencies are installed
4. **Multi-machine deployment**: Ensure deployment.toml is properly configured
