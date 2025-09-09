# TouchDesigner LaunchAgent Manager

This script creates and manages macOS LaunchAgents for TouchDesigner files in the Experimance project.

## Prerequisites

- **macOS only** - TouchDesigner and LaunchAgents are macOS-specific
- TouchDesigner installed in one of the standard locations:
  - `/Applications/TouchDesigner.app/`
  - `/Applications/TouchDesigner/TouchDesigner.app/`
  - `/Applications/TouchDesigner099/TouchDesigner099.app/`
  - `/Applications/TouchDesigner088/TouchDesigner088.app/`

## Usage

```bash
./infra/scripts/touchdesigner_agent.sh <touchdesigner_file> [action] [--project=<project>]
```

### Arguments

- `touchdesigner_file`: Path to the TouchDesigner .toe file

### Actions

- `install` - Create and install the LaunchAgent (default)
- `start` - Start the LaunchAgent service  
- `stop` - Stop the LaunchAgent service
- `restart` - Restart the LaunchAgent service
- `status` - Show status of the LaunchAgent service
- `uninstall` - Remove the LaunchAgent service

### Options

- `--project=<project>` - Override project name (default: fire)

## Examples

```bash
# Install LaunchAgent for fire.toe (uses default 'fire' project)
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe

# Install with explicit project name
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe install --project=fire

# Start the service
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe start

# Check status
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe status

# Restart the service
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe restart

# Uninstall the service
./infra/scripts/touchdesigner_agent.sh /path/to/fire.toe uninstall
```

## How It Works

1. **LaunchAgent Creation**: The script creates a `.plist` file in `~/Library/LaunchAgents/` with the format `com.experimance.touchdesigner.<project>.<filename>.plist`

2. **Automatic Restart**: The LaunchAgent is configured to:
   - Start automatically when the user logs in (`RunAtLoad: true`)
   - Restart on failure after 10 seconds (`ThrottleInterval: 10`)
   - Not restart on successful exits (`KeepAlive.SuccessfulExit: false`)

3. **Logging**: Output and errors are logged to:
   - `~/Library/Logs/experimance/<project>_touchdesigner_<filename>.log`
   - `~/Library/Logs/experimance/<project>_touchdesigner_<filename>_error.log`

4. **Environment**: The TouchDesigner process runs with:
   - `PROJECT_ENV=<project>`
   - `EXPERIMANCE_PROJECT=<project>`
   - Working directory set to the Experimance repository root

## Service Management

Once installed, you can also manage the LaunchAgent using macOS's built-in `launchctl`:

```bash
# List all LaunchAgents (look for com.experimance.touchdesigner.*)
launchctl list | grep experimance

# Get detailed status
launchctl list com.experimance.touchdesigner.fire.yourfile

# Manual start/stop
launchctl start com.experimance.touchdesigner.fire.yourfile
launchctl stop com.experimance.touchdesigner.fire.yourfile
```

## Logs

View logs in real-time:

```bash
# Standard output
tail -f ~/Library/Logs/experimance/fire_touchdesigner_yourfile.log

# Error output  
tail -f ~/Library/Logs/experimance/fire_touchdesigner_yourfile_error.log
```

## Troubleshooting

1. **TouchDesigner not found**: Ensure TouchDesigner is installed in a standard location
2. **Permission denied**: The script must be run as a regular user (not root)
3. **File not found**: Ensure the .toe file path is correct and the file exists
4. **Service won't start**: Check the error logs and ensure TouchDesigner can open the file

## File Locations

- **LaunchAgent plist**: `~/Library/LaunchAgents/com.experimance.touchdesigner.<project>.<filename>.plist`
- **Logs**: `~/Library/Logs/experimance/`
- **Script**: `infra/scripts/touchdesigner_agent.sh`

## Security Notes

- The LaunchAgent runs as the current user (not root)
- TouchDesigner may require Full Disk Access permissions to access certain files
- The service will restart automatically if TouchDesigner crashes or exits unexpectedly
