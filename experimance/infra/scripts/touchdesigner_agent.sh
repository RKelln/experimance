#!/bin/bash

# TouchDesigner LaunchAgent Management Script
# Usage: ./touchdesigner_agent.sh <touchdesigner_file> [action] [--project=<project>]
# Actions: install, start, stop, restart, status, uninstall
# Options: --project=<project> - Override project name (defaults to 'fire')
#
# This script creates and manages macOS LaunchAgents for TouchDesigner files.
# LaunchAgents run as the current user and restart automatically on failure.

set -eu

# Trap function to show big error on any script failure
trap_error() {
    local exit_code=$?
    local line_number=$1
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                  CRITICAL ERROR                               ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] SCRIPT FAILED: Unexpected error on line $line_number${NC}"
    echo -e "${RED}Exit code: $exit_code${NC}"
    echo -e "${RED}This could be due to a pipeline failure, command error, or unset variable.${NC}"
    echo -e "${RED}Script execution FAILED and will exit.${NC}"
    echo ""
    exit $exit_code
}

# Set trap to catch any error
trap 'trap_error ${LINENO}' ERR

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging and error helpers
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                  CRITICAL ERROR                                ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    echo -e "${RED}Script execution FAILED and will exit.${NC}"
    echo ""
    exit 1
}

# Platform detection
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM=linux ;;
        Darwin*)    PLATFORM=macos ;;
        CYGWIN*|MINGW*) PLATFORM=windows ;;
        *)          PLATFORM=unknown ;;
    esac
    
    log "Detected platform: $PLATFORM"
}

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Detect platform early
detect_platform

# Validate macOS platform
if [[ "$PLATFORM" != "macos" ]]; then
    error "This script is designed for macOS only. TouchDesigner LaunchAgents require macOS."
fi

# Check if running as root (not allowed for LaunchAgents)
if [[ $EUID -eq 0 ]]; then
    error "Do not run this script as root on macOS. LaunchAgents run as the current user."
fi

# Default values
DEFAULT_PROJECT="fire"
PROJECT_NAME=""
ACTION=""
TD_FILE=""

# Parse command line arguments
parse_arguments() {
    # Handle case with no arguments
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi
    
    # First argument could be either a .toe file or an action
    first_arg="$1"
    shift
    
    # Check if first argument is an action (for non-install commands)
    case "$first_arg" in
        start|stop|restart|status|uninstall)
            ACTION="$first_arg"
            TD_FILE=""  # Will be determined from existing plist files
            ;;
        install)
            ACTION="install"
            if [[ $# -eq 0 ]]; then
                error "Install action requires a TouchDesigner .toe file path"
            fi
            TD_FILE="$1"
            shift
            ;;
        *)
            # First argument is a .toe file path
            TD_FILE="$first_arg"
            ;;
    esac
    
    # Parse remaining arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project=*)
                PROJECT_NAME="${1#*=}"
                shift
                ;;
            install|start|stop|restart|status|uninstall)
                if [[ -z "$ACTION" ]]; then
                    ACTION="$1"
                else
                    error "Action already specified: $ACTION. Found duplicate: $1"
                fi
                shift
                ;;
            *)
                if [[ -z "$TD_FILE" && "$ACTION" == "install" ]]; then
                    TD_FILE="$1"
                else
                    error "Unknown argument: $1"
                fi
                shift
                ;;
        esac
    done
    
    # Set defaults
    if [[ -z "$PROJECT_NAME" ]]; then
        PROJECT_NAME="$DEFAULT_PROJECT"
    fi
    
    if [[ -z "$ACTION" ]]; then
        ACTION="install"
    fi
    
    # Validate arguments based on action
    if [[ "$ACTION" == "install" && -z "$TD_FILE" ]]; then
        error "Install action requires a TouchDesigner .toe file path"
    fi
}

# Show usage information
show_usage() {
    echo ""
    echo -e "${BLUE}TouchDesigner LaunchAgent Management Script${NC}"
    echo ""
    echo -e "${GREEN}Usage:${NC}"
    echo "  $0 <touchdesigner_file> install [--project=<project>]     # Install new LaunchAgent"
    echo "  $0 <action> [--project=<project>]                        # Manage existing LaunchAgent"
    echo ""
    echo -e "${GREEN}Install (requires .toe file):${NC}"
    echo "  touchdesigner_file    Path to the TouchDesigner .toe file"
    echo ""
    echo -e "${GREEN}Actions:${NC}"
    echo "  install      Create and install the LaunchAgent (requires .toe file)"
    echo "  start        Start the LaunchAgent service"
    echo "  stop         Stop the LaunchAgent service"
    echo "  restart      Restart the LaunchAgent service"
    echo "  status       Show status of the LaunchAgent service"
    echo "  uninstall    Remove the LaunchAgent service"
    echo ""
    echo -e "${GREEN}Options:${NC}"
    echo "  --project=<project>    Override project name (default: fire)"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo -e "${BLUE}Install:${NC}"
    echo "  $0 /path/to/fire.toe install --project=fire"
    echo "  $0 /path/to/fire.toe    # install is default action"
    echo ""
    echo -e "${BLUE}Manage existing:${NC}"
    echo "  $0 start                # Start the fire project TouchDesigner"
    echo "  $0 stop                 # Stop the fire project TouchDesigner"
    echo "  $0 status               # Show status"
    echo "  $0 restart --project=fire"
    echo ""
}

# Validate TouchDesigner file
validate_td_file() {
    local td_file="$1"
    
    # Convert to absolute path
    if [[ ! "$td_file" =~ ^/ ]]; then
        td_file="$(pwd)/$td_file"
    fi
    
    # Resolve symbolic links and normalize path (with error handling)
    if command -v realpath >/dev/null 2>&1; then
        td_file="$(realpath "$td_file" 2>/dev/null || echo "$td_file")"
    elif command -v greadlink >/dev/null 2>&1; then
        td_file="$(greadlink -f "$td_file" 2>/dev/null || echo "$td_file")"
    else
        # Fallback for macOS without GNU coreutils (with error handling)
        local dir_path="$(dirname "$td_file")"
        if [[ -d "$dir_path" ]]; then
            td_file="$(cd "$dir_path" 2>/dev/null && pwd || echo "$dir_path")/$(basename "$td_file")"
        fi
    fi
    
    if [[ ! -f "$td_file" ]]; then
        error "TouchDesigner file not found: $td_file"
    fi
    
    if [[ ! "$td_file" =~ \.toe$ ]]; then
        error "File must have .toe extension: $td_file"
    fi
    
    echo "$td_file"
}

# Find existing TouchDesigner LaunchAgent plist files for a project
find_existing_td_plist() {
    local project="$1"
    local launchd_dir="$HOME/Library/LaunchAgents"
    
    # Look for plist files matching the pattern
    local plist_files=($(find "$launchd_dir" -name "com.experimance.touchdesigner.$project.*.plist" 2>/dev/null || true))
    
    if [[ ${#plist_files[@]} -eq 0 ]]; then
        return 1
    elif [[ ${#plist_files[@]} -eq 1 ]]; then
        echo "${plist_files[0]}"
        return 0
    else
        # Multiple files found, show them and let user choose
        echo ""
        warn "Multiple TouchDesigner LaunchAgent plist files found for project '$project':"
        local i=1
        for plist in "${plist_files[@]}"; do
            local basename=$(basename "$plist" .plist)
            echo "  $i) $basename"
            ((i++))
        done
        echo ""
        echo "Please specify the .toe file path to select the specific TouchDesigner service."
        return 1
    fi
}

# Extract .toe file path from a plist file
extract_toe_path_from_plist() {
    local plist_file="$1"
    
    if [[ ! -f "$plist_file" ]]; then
        return 1
    fi
    
    # Extract the second ProgramArgument (the .toe file path)
    local toe_path=$(plutil -extract ProgramArguments.1 raw "$plist_file" 2>/dev/null || echo "")
    
    if [[ -n "$toe_path" && -f "$toe_path" ]]; then
        echo "$toe_path"
        return 0
    else
        return 1
    fi
}

# Auto-detect TouchDesigner file from existing plist
auto_detect_td_file() {
    local project="$1"
    
    # Find existing plist (without logging to avoid pollution)
    local plist_file=$(find_existing_td_plist "$project" 2>/dev/null)
    if [[ $? -ne 0 || -z "$plist_file" ]]; then
        return 1
    fi
    
    local toe_path=$(extract_toe_path_from_plist "$plist_file")
    if [[ $? -ne 0 || -z "$toe_path" ]]; then
        return 1
    fi
    
    echo "$toe_path"
    return 0
}

# Find TouchDesigner application
find_touchdesigner() {
    local td_paths=(
        "/Applications/TouchDesigner.app/Contents/MacOS/TouchDesigner"
        "/Applications/TouchDesigner/TouchDesigner.app/Contents/MacOS/TouchDesigner"
        "/Applications/TouchDesigner099/TouchDesigner099.app/Contents/MacOS/TouchDesigner099"
        "/Applications/TouchDesigner088/TouchDesigner088.app/Contents/MacOS/TouchDesigner088"
    )
    
    for td_path in "${td_paths[@]}"; do
        if [[ -x "$td_path" ]]; then
            echo "$td_path"
            return 0
        fi
    done
    
    error "TouchDesigner application not found in standard locations. Please install TouchDesigner first."
}

# Generate service label from TouchDesigner file
generate_service_label() {
    local td_file="$1"
    local project="$2"
    local filename=$(basename "$td_file" .toe)
    
    # Sanitize filename for use in service label
    local sanitized_name=$(echo "$filename" | sed 's/[^a-zA-Z0-9_-]/_/g' | tr '[:upper:]' '[:lower:]')
    
    echo "com.experimance.touchdesigner.$project.$sanitized_name"
}

# Create LaunchAgent plist for TouchDesigner
create_touchdesigner_launchagent() {
    local td_file="$1"
    local project="$2"
    
    # Validate and get absolute path
    td_file=$(validate_td_file "$td_file")
    
    # Find TouchDesigner application
    local td_app=$(find_touchdesigner)
    
    # Generate service label
    local service_label=$(generate_service_label "$td_file" "$project")
    
    # Create LaunchAgents directory
    local launchd_dir="$HOME/Library/LaunchAgents"
    mkdir -p "$launchd_dir"
    
    # Create log directory
    local log_dir="$HOME/Library/Logs/experimance"
    mkdir -p "$log_dir"
    
    # Generate plist filename
    local plist_file="$launchd_dir/$service_label.plist"
    
    log "Creating TouchDesigner LaunchAgent:"
    log "  TouchDesigner app: $td_app"
    log "  TouchDesigner file: $td_file"
    log "  Project: $project"
    log "  Service label: $service_label"
    log "  Plist file: $plist_file"
    
    # Create the plist file
    cat > "$plist_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$service_label</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>$td_app</string>
        <string>$td_file</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>$REPO_DIR</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PROJECT_ENV</key>
        <string>$project</string>
        <key>EXPERIMANCE_PROJECT</key>
        <string>$project</string>
    </dict>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>StandardErrorPath</key>
    <string>$log_dir/${project}_touchdesigner_$(basename "$td_file" .toe)_error.log</string>
    
    <key>StandardOutPath</key>
    <string>$log_dir/${project}_touchdesigner_$(basename "$td_file" .toe).log</string>
    
    <!-- Restart on failure after 10 seconds -->
    <key>ThrottleInterval</key>
    <integer>10</integer>
    
    <!-- Start after 10 second delay on boot (let other software start first) -->
    <key>StartInterval</key>
    <integer>10</integer>
    
    <!-- Don't restart too frequently -->
    <key>ExitTimeOut</key>
    <integer>30</integer>
</dict>
</plist>
EOF

    # Set proper permissions
    chmod 644 "$plist_file"
    
    log "✓ Created TouchDesigner LaunchAgent: $plist_file"
    
    # Store service label for other functions
    echo "$service_label"
}

# Install LaunchAgent
install_touchdesigner_agent() {
    local td_file="$1"
    local project="$2"
    
    log "Installing TouchDesigner LaunchAgent for project '$project'..."
    
    # Create the LaunchAgent
    local service_label=$(create_touchdesigner_launchagent "$td_file" "$project")
    local plist_file="$HOME/Library/LaunchAgents/$service_label.plist"
    
    # Load the LaunchAgent
    if launchctl load "$plist_file" 2>/dev/null; then
        log "✓ TouchDesigner LaunchAgent installed and loaded successfully"
        
        # Show status
        echo ""
        show_service_status "$service_label"
        
        # Show helpful information
        echo ""
        echo -e "${BLUE}TouchDesigner LaunchAgent Management:${NC}"
        echo -e "${GREEN}Start:${NC}   $0 \"$td_file\" start --project=$project"
        echo -e "${GREEN}Stop:${NC}    $0 \"$td_file\" stop --project=$project"
        echo -e "${GREEN}Status:${NC}  $0 \"$td_file\" status --project=$project"
        echo -e "${GREEN}Logs:${NC}    tail -f $HOME/Library/Logs/experimance/${project}_touchdesigner_$(basename "$td_file" .toe).log"
        echo ""
        
    else
        # Try to check if it's already loaded
        if launchctl list | grep -q "$service_label"; then
            warn "LaunchAgent appears to already be loaded. Current status:"
            show_service_status "$service_label"
        else
            error "Failed to load TouchDesigner LaunchAgent. Check the plist file and try again."
        fi
    fi
}

# Start LaunchAgent service
start_service() {
    local td_file="$1"
    local project="$2"
    
    local service_label=$(generate_service_label "$td_file" "$project")
    
    log "Starting TouchDesigner LaunchAgent: $service_label"
    
    # Check if service is loaded first
    if ! launchctl list | grep -q "$service_label"; then
        warn "LaunchAgent is not loaded. Try installing it first:"
        echo "  $0 \"$td_file\" install --project=$project"
        return 1
    fi
    
    # Try to start the service
    if launchctl start "$service_label" 2>/dev/null; then
        log "✓ TouchDesigner LaunchAgent started successfully"
        sleep 2
        show_service_status "$service_label"
    else
        # Check if it's already running
        local status_output=$(launchctl list "$service_label" 2>/dev/null || echo "")
        if echo "$status_output" | grep -q "PID"; then
            warn "LaunchAgent appears to already be running:"
            show_service_status "$service_label"
        else
            error "Failed to start TouchDesigner LaunchAgent. Check logs for details."
        fi
    fi
}

# Stop LaunchAgent service
stop_service() {
    local td_file="$1"
    local project="$2"
    
    local service_label=$(generate_service_label "$td_file" "$project")
    
    log "Stopping TouchDesigner LaunchAgent: $service_label"
    
    # Check if service is loaded first
    if ! launchctl list | grep -q "$service_label"; then
        warn "LaunchAgent is not loaded or not running"
        return 0
    fi
    
    if launchctl stop "$service_label" 2>/dev/null; then
        log "✓ TouchDesigner LaunchAgent stopped successfully"
        sleep 2
        show_service_status "$service_label"
    else
        warn "TouchDesigner LaunchAgent may not have been running"
        show_service_status "$service_label"
    fi
}

# Restart LaunchAgent service
restart_service() {
    local td_file="$1"
    local project="$2"
    
    log "Restarting TouchDesigner LaunchAgent..."
    stop_service "$td_file" "$project" || true
    sleep 3
    start_service "$td_file" "$project"
}

# Show service status
show_service_status() {
    local service_label="$1"
    
    echo ""
    echo -e "${BLUE}TouchDesigner LaunchAgent Status:${NC}"
    echo "Service: $service_label"
    
    # Check if service is loaded
    if launchctl list | grep -q "$service_label"; then
        echo -e "Status: ${GREEN}Loaded${NC}"
        
        # Get detailed status
        local status_output=$(launchctl list "$service_label" 2>/dev/null || echo "")
        if [[ -n "$status_output" ]]; then
            echo "Details:"
            echo "$status_output" | while IFS= read -r line; do
                echo "  $line"
            done
        fi
    else
        echo -e "Status: ${RED}Not loaded${NC}"
    fi
    echo ""
}

# Get service status for a TouchDesigner file
get_service_status() {
    local td_file="$1"
    local project="$2"
    
    local service_label=$(generate_service_label "$td_file" "$project")
    show_service_status "$service_label"
}

# Uninstall LaunchAgent
uninstall_service() {
    local td_file="$1"
    local project="$2"
    
    local service_label=$(generate_service_label "$td_file" "$project")
    local plist_file="$HOME/Library/LaunchAgents/$service_label.plist"
    
    log "Uninstalling TouchDesigner LaunchAgent: $service_label"
    
    # Stop and unload the service
    if launchctl list | grep -q "$service_label"; then
        log "Stopping and unloading LaunchAgent..."
        launchctl stop "$service_label" 2>/dev/null || true
        launchctl unload "$plist_file" 2>/dev/null || true
    fi
    
    # Remove the plist file
    if [[ -f "$plist_file" ]]; then
        rm -f "$plist_file"
        log "✓ Removed plist file: $plist_file"
    else
        warn "Plist file not found: $plist_file"
    fi
    
    log "✓ TouchDesigner LaunchAgent uninstalled successfully"
}

# Main function
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Auto-detect TouchDesigner file for non-install actions if not provided
    if [[ -z "$TD_FILE" && "$ACTION" != "install" ]]; then
        log "No .toe file specified, attempting to auto-detect from existing LaunchAgent..."
        TD_FILE=$(auto_detect_td_file "$PROJECT_NAME")
        if [[ $? -ne 0 || -z "$TD_FILE" ]]; then
            error "Could not auto-detect TouchDesigner file for project '$PROJECT_NAME'. Use 'install' first or specify the .toe file path."
        fi
        log "Auto-detected TouchDesigner file: $TD_FILE"
    fi
    
    log "TouchDesigner LaunchAgent Manager"
    if [[ -n "$TD_FILE" ]]; then
        log "TouchDesigner file: $TD_FILE"
    fi
    log "Project: $PROJECT_NAME"
    log "Action: $ACTION"
    
    # Execute the requested action
    case "$ACTION" in
        install)
            install_touchdesigner_agent "$TD_FILE" "$PROJECT_NAME"
            ;;
        start)
            start_service "$TD_FILE" "$PROJECT_NAME"
            ;;
        stop)
            stop_service "$TD_FILE" "$PROJECT_NAME"
            ;;
        restart)
            restart_service "$TD_FILE" "$PROJECT_NAME"
            ;;
        status)
            get_service_status "$TD_FILE" "$PROJECT_NAME"
            ;;
        uninstall)
            uninstall_service "$TD_FILE" "$PROJECT_NAME"
            ;;
        *)
            error "Unknown action: $ACTION"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
