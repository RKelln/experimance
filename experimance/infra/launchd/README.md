# macOS launchd Service Configuration

This directory contains launchd property list (`.plist`) files for running Experimance services on macOS using Apple's native service management system.

## Overview

**launchd** is macOS's equivalent to systemd on Linux. It manages:
- Service startup and shutdown
- Automatic restart on failure
- Environment variables
- Logging
- User and group permissions

## Files

- `com.experimance.agent.fire.plist` - Fire Agent service
- `com.experimance.health.fire.plist` - Health monitoring service
- `com.experimance.agent.template.plist` - Template for agent services (replace PROJECT_NAME)
- `com.experimance.health.template.plist` - Template for health services (replace PROJECT_NAME)

## Installation

### 1. Prerequisites

Ensure the following are set up on the macOS machine:
- Homebrew installed with `uv` package manager
- Experimance project installed in `/opt/experimance`
- User `experimance` exists with appropriate permissions
- Log directories created: `/opt/experimance/logs/`

### 2. Install Service Files

Copy the `.plist` files to the system LaunchDaemons directory:

```bash
# Copy service files (requires sudo)
sudo cp infra/launchd/*.plist /Library/LaunchDaemons/

# Set proper ownership and permissions
sudo chown root:wheel /Library/LaunchDaemons/com.experimance.*.plist
sudo chmod 644 /Library/LaunchDaemons/com.experimance.*.plist
```

### 2a. Creating Services for Other Projects

For projects other than Fire, use the template files:

```bash
# Copy and customize template for your project
cp infra/launchd/com.experimance.agent.template.plist \
   infra/launchd/com.experimance.agent.yourproject.plist

# Replace PROJECT_NAME with your project name
sed -i '' 's/PROJECT_NAME/yourproject/g' \
   infra/launchd/com.experimance.agent.yourproject.plist

# Install the customized service
sudo cp infra/launchd/com.experimance.agent.yourproject.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/com.experimance.agent.yourproject.plist
sudo chmod 644 /Library/LaunchDaemons/com.experimance.agent.yourproject.plist
```

### 3. Load and Start Services

```bash
# Load the services
sudo launchctl load /Library/LaunchDaemons/com.experimance.agent.fire.plist
sudo launchctl load /Library/LaunchDaemons/com.experimance.health.fire.plist

# Start the services
sudo launchctl start com.experimance.agent.fire
sudo launchctl start com.experimance.health.fire
```

## Service Management

### Check Service Status
```bash
# List all loaded services
sudo launchctl list | grep experimance

# Check specific service status
sudo launchctl print system/com.experimance.agent.fire
```

### Start/Stop Services
```bash
# Start services
sudo launchctl start com.experimance.agent.fire
sudo launchctl start com.experimance.health.fire

# Stop services
sudo launchctl stop com.experimance.agent.fire
sudo launchctl stop com.experimance.health.fire
```

### Restart Services
```bash
# Restart (stop then start)
sudo launchctl stop com.experimance.agent.fire
sudo launchctl start com.experimance.agent.fire
```

### Unload Services
```bash
# Unload (disable) services
sudo launchctl unload /Library/LaunchDaemons/com.experimance.agent.fire.plist
sudo launchctl unload /Library/LaunchDaemons/com.experimance.health.fire.plist
```

## Logs

Service logs are written to:
- `/opt/experimance/logs/fire_agent.log` - Agent service stdout
- `/opt/experimance/logs/fire_agent_error.log` - Agent service stderr
- `/opt/experimance/logs/fire_health.log` - Health service stdout
- `/opt/experimance/logs/fire_health_error.log` - Health service stderr

View logs in real-time:
```bash
# Follow agent logs
tail -f /opt/experimance/logs/fire_agent.log

# Follow health logs
tail -f /opt/experimance/logs/fire_health.log
```

## Configuration Details

### Key Properties Explained

- **Label**: Unique identifier for the service
- **ProgramArguments**: Command and arguments to execute
- **WorkingDirectory**: Directory to run the service from
- **EnvironmentVariables**: Environment variables for the service
- **RunAtLoad**: Start service when loaded (boot time)
- **KeepAlive**: Restart policy (restart on unexpected exit)
- **UserName/GroupName**: User/group to run service as
- **ThrottleInterval**: Wait time before restart attempts
- **StandardOutPath/StandardErrorPath**: Log file locations

### Environment Variables

Both services are configured with:
- `PROJECT_ENV=fire` - Specifies the Fire project
- `EXPERIMANCE_ENV=production` - Sets production environment
- `PATH` - Includes Homebrew and system paths

## Troubleshooting

### Service Won't Start
1. Check syntax: `plutil /Library/LaunchDaemons/com.experimance.agent.fire.plist`
2. Verify file permissions: `ls -la /Library/LaunchDaemons/com.experimance.*`
3. Check user exists: `id experimance`
4. Verify paths exist: `ls -la /opt/experimance`

### Service Crashes
1. Check error logs: `cat /opt/experimance/logs/fire_agent_error.log`
2. Verify uv and Python environment: `which uv`
3. Test manual execution: `cd /opt/experimance && uv run -m experimance_agent`

### Permission Issues
1. Ensure `/opt/experimance` is writable by `experimance` user
2. Create log directory: `sudo mkdir -p /opt/experimance/logs && sudo chown experimance /opt/experimance/logs`

## Differences from systemd

| Feature        | systemd                    | launchd                       |
| -------------- | -------------------------- | ----------------------------- |
| Config format  | INI-style `.service` files | XML `.plist` files            |
| Location       | `/etc/systemd/system/`     | `/Library/LaunchDaemons/`     |
| Commands       | `systemctl`                | `launchctl`                   |
| User services  | `--user` flag              | LaunchAgents vs LaunchDaemons |
| Dependencies   | `After=`, `Wants=`         | Limited dependency support    |
| Restart policy | `Restart=`                 | `KeepAlive` dict              |

## Integration with Deployment

These launchd services integrate with the multi-machine deployment described in the main infrastructure documentation. The macOS machine runs:
- `fire_agent` - AI interaction service
- `health` - Health monitoring service

While the Ubuntu machine runs the core, display, and image_server services using systemd.
