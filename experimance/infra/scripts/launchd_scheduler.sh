#!/bin/bash

# Experimance Project LaunchAgent Scheduler
# Usage: ./launchd_scheduler.sh <project> <action> [schedule_type]
# Actions: setup-schedule, remove-schedule, show-schedule, manual-start, manual-stop, manual-unload
# Schedule Types: gallery, custom, daily
#
# This script modifies existing LaunchAgents to add gallery hour scheduling while
# preserving RunAtLoad=true for automatic startup after reboot. Creates additional
# scheduler agents to start/stop services during gallery hours.

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging helpers
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"

# Default values
PROJECT="${1:-fire}"
ACTION="${2:-setup-schedule}"
SCHEDULE_TYPE="${3:-gallery}"

# Find existing plist files for the project
get_existing_plist_files() {
    local project="$1"
    find "$LAUNCHD_DIR" -name "com.experimance.*${project}*.plist" 2>/dev/null || true
}

# Get service labels from existing plist files
get_service_labels() {
    local project="$1"
    local labels=()
    
    while IFS= read -r plist_file; do
        if [[ -f "$plist_file" ]]; then
            local label=$(plutil -extract Label raw "$plist_file" 2>/dev/null || echo "")
            if [[ -n "$label" ]]; then
                labels+=("$label")
            fi
        fi
    done < <(get_existing_plist_files "$project")
    
    printf '%s\n' "${labels[@]}"
}

show_usage() {
    echo ""
    echo -e "${BLUE}Experimance LaunchAgent Scheduler${NC}"
    echo ""
    echo -e "${GREEN}Usage:${NC}"
    echo "  $0 <project> <action> [schedule_type]"
    echo ""
    echo -e "${GREEN}Actions:${NC}"
    echo "  setup-schedule    Add gallery hour scheduling to existing LaunchAgents"
    echo "  remove-schedule   Remove scheduling and return to always-on mode"
    echo "  show-schedule     Show current schedule configuration and service status"
    echo "  manual-start      Manually start all project services"
    echo "  manual-stop       Manually stop all project services (may auto-restart)"
    echo "  manual-restart    Manually restart all project services (stop + auto-restart)"
    echo "  manual-unload     Unload services completely (no auto-restart)"
    echo ""
    echo -e "${GREEN}Schedule Types:${NC}"
    echo "  gallery          Tuesday-Saturday, 11AM-6PM (default)"
    echo "  gallery-extended Tuesday-Saturday, 11AM-6PM (Wed until 9PM)"
    echo "  daily            Every day, 9AM-10PM"
    echo "  custom           Prompts for custom times"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  $0 fire setup-schedule gallery"
    echo "  $0 fire setup-schedule gallery-extended"
    echo "  $0 fire setup-schedule custom"
    echo "  $0 fire show-schedule"
    echo "  $0 fire manual-stop          # Kill services (may auto-restart)"
    echo "  $0 fire manual-restart       # Restart all services"
    echo "  $0 fire manual-unload        # Unload services (no auto-restart)"
    echo "  $0 fire remove-schedule"
    echo ""
    echo -e "${YELLOW}How it works:${NC}"
    echo "• Existing services keep RunAtLoad=true (auto-start after reboot)"
    echo "• Additional scheduler agents start/stop services during gallery hours"  
    echo "• Machine stays on 24/7, services only run during scheduled times"
    echo "• Gallery can manually override with manual-start/manual-stop/manual-unload"
    echo ""
}

# Create scheduler agent that starts services at gallery hours
create_gallery_starter() {
    local schedule_config="$1"
    local project="$2"
    local plist_file="$LAUNCHD_DIR/com.experimance.${project}.gallery-starter.plist"
    
    # Get all service labels to start (excluding gallery scheduler services themselves)
    local service_labels=()
    while IFS= read -r label; do
        # Skip gallery scheduler services to avoid self-referencing
        if [[ "$label" != *"gallery-starter"* && "$label" != *"gallery-stopper"* ]]; then
            service_labels+=("$label")
        fi
    done < <(get_service_labels "$project")
    
    if [[ ${#service_labels[@]} -eq 0 ]]; then
        error "No existing services found for project $project"
    fi
    
    # Create start script content with proper XML escaping
    local start_commands=""
    for label in "${service_labels[@]}"; do
        if [[ -n "$start_commands" ]]; then
            start_commands+="; "
        fi
        start_commands+="launchctl kickstart gui/\$(id -u)/$label"
    done
    
    cat > "$plist_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.experimance.${project}.gallery-starter</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>echo 'Starting $project services for gallery hours'; $start_commands; echo 'All $project services started'</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>$REPO_DIR</string>
    
    <!-- Schedule Configuration -->
    <key>StartCalendarInterval</key>
    $schedule_config
    
    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/experimance/${project}_gallery_starter_error.log</string>
    
    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/experimance/${project}_gallery_starter.log</string>
    
    <!-- Don't keep alive - this is a scheduled task -->
    <key>KeepAlive</key>
    <false/>
    
    <!-- Don't run at load - only run on schedule -->
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

    chmod 644 "$plist_file"
    log "✓ Created gallery starter: $plist_file"
}

# Create scheduler agent that stops services after gallery hours
create_gallery_stopper() {
    local schedule_config="$1"
    local project="$2"
    local plist_file="$LAUNCHD_DIR/com.experimance.${project}.gallery-stopper.plist"
    
    # Get all service labels to stop (excluding gallery scheduler services themselves)
    local service_labels=()
    while IFS= read -r label; do
        # Skip gallery scheduler services to avoid self-referencing
        if [[ "$label" != *"gallery-starter"* && "$label" != *"gallery-stopper"* ]]; then
            service_labels+=("$label")
        fi
    done < <(get_service_labels "$project")
    
    # Create stop script content with proper XML escaping
    local stop_commands=""
    for label in "${service_labels[@]}"; do
        if [[ -n "$stop_commands" ]]; then
            stop_commands+="; "
        fi
        stop_commands+="launchctl kill TERM gui/\$(id -u)/$label"
    done
    
    cat > "$plist_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.experimance.${project}.gallery-stopper</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>echo 'Stopping $project services after gallery hours'; $stop_commands; echo 'All $project services stopped'</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>$REPO_DIR</string>
    
    <!-- Schedule Configuration -->
    <key>StartCalendarInterval</key>
    $schedule_config
    
    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/experimance/${project}_gallery_stopper_error.log</string>
    
    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/experimance/${project}_gallery_stopper.log</string>
    
    <!-- Don't keep alive - this is a scheduled task -->
    <key>KeepAlive</key>
    <false/>
    
    <!-- Don't run at load - only run on schedule -->
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

    chmod 644 "$plist_file"
    log "✓ Created gallery stopper: $plist_file"
}

# Generate schedule configuration for different presets
get_schedule_config() {
    local schedule_type="$1"
    
    case "$schedule_type" in
        gallery)
            # Tuesday-Saturday, 11AM start, 6PM stop
            echo '<array>
        <!-- Start: Tuesday-Saturday at 10:55 AM -->
        <dict>
            <key>Weekday</key>
            <integer>2</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>3</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>4</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>5</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>6</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
    </array>'
            ;;
        gallery-extended)
            # Tuesday-Saturday, 10:55 AM start (same as gallery)
            echo '<array>
        <!-- Start: Tuesday-Saturday at 10:55 AM -->
        <dict>
            <key>Weekday</key>
            <integer>2</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>3</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>4</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>5</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>6</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>55</integer>
        </dict>
    </array>'
            ;;
        daily)
            # Every day at 9AM start
            echo '<dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>'
            ;;
        custom)
            # Prompt user for custom schedule
            echo ""
            read -p "Enter start hour (0-23): " start_hour
            read -p "Enter start minute (0-59): " start_minute
            echo ""
            echo '<dict>
        <key>Hour</key>
        <integer>'$start_hour'</integer>
        <key>Minute</key>
        <integer>'$start_minute'</integer>
    </dict>'
            ;;
        *)
            error "Unknown schedule type: $schedule_type"
            ;;
    esac
}

get_shutdown_schedule_config() {
    local schedule_type="$1"
    
    case "$schedule_type" in
        gallery)
            # Tuesday-Saturday, 6PM stop
            echo '<array>
        <!-- Stop: Tuesday-Saturday at 6:05 PM -->
        <dict>
            <key>Weekday</key>
            <integer>2</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>3</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>4</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>5</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
        <dict>
            <key>Weekday</key>
            <integer>6</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
    </array>'
            ;;
        gallery-extended)
            # Tuesday, Thursday, Friday, Saturday at 6:05 PM; Wednesday at 9:05 PM
            echo '<array>
        <!-- Stop: Tuesday at 6:05 PM -->
        <dict>
            <key>Weekday</key>
            <integer>2</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
        <!-- Stop: Wednesday at 9:05 PM -->
        <dict>
            <key>Weekday</key>
            <integer>3</integer>
            <key>Hour</key>
            <integer>21</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
        <!-- Stop: Thursday at 6:05 PM -->
        <dict>
            <key>Weekday</key>
            <integer>4</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
        <!-- Stop: Friday at 6:05 PM -->
        <dict>
            <key>Weekday</key>
            <integer>5</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
        <!-- Stop: Saturday at 6:05 PM -->
        <dict>
            <key>Weekday</key>
            <integer>6</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>5</integer>
        </dict>
    </array>'
            ;;
        daily)
            # Every day at 10PM stop
            echo '<dict>
        <key>Hour</key>
        <integer>22</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>'
            ;;
        custom)
            # Use the same custom time for now, or prompt separately
            echo ""
            read -p "Enter stop hour (0-23): " stop_hour
            read -p "Enter stop minute (0-59): " stop_minute
            echo ""
            echo '<dict>
        <key>Hour</key>
        <integer>'$stop_hour'</integer>
        <key>Minute</key>
        <integer>'$stop_minute'</integer>
    </dict>'
            ;;
        *)
            error "Unknown schedule type: $schedule_type"
            ;;
    esac
}

# Setup scheduled LaunchAgents
setup_schedule() {
    local schedule_type="$1"
    
    log "Setting up $schedule_type schedule for project $PROJECT..."
    
    # Check that we have existing services
    local existing_plists=$(get_existing_plist_files "$PROJECT")
    if [[ -z "$existing_plists" ]]; then
        error "No existing LaunchAgent services found for project $PROJECT. Run deploy.sh first."
    fi
    
    # Auto-remove existing gallery scheduling if present
    local starter_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-starter.plist"
    local stopper_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-stopper.plist"
    
    if [[ -f "$starter_plist" || -f "$stopper_plist" ]]; then
        log "Removing existing gallery scheduling first..."
        remove_schedule_internal
    fi
    
    log "Found existing services for $PROJECT:"
    while IFS= read -r plist_file; do
        local label=$(plutil -extract Label raw "$plist_file" 2>/dev/null || echo "unknown")
        echo "  - $label"
    done < <(get_existing_plist_files "$PROJECT")
    
    # Create log directory
    mkdir -p "$HOME/Library/Logs/experimance"
    
    # Get schedule configurations
    local startup_config=$(get_schedule_config "$schedule_type")
    local shutdown_config=$(get_shutdown_schedule_config "$schedule_type")
    
    # Create gallery starter and stopper agents
    create_gallery_starter "$startup_config" "$PROJECT"
    create_gallery_stopper "$shutdown_config" "$PROJECT"
    
    # Load the new scheduler agents
    local starter_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-starter.plist"
    local stopper_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-stopper.plist"
    
    launchctl bootstrap gui/$(id -u) "$starter_plist"
    launchctl bootstrap gui/$(id -u) "$stopper_plist"
    
    log "✓ Gallery hour scheduling installed and loaded"
    
    echo ""
    echo -e "${BLUE}Schedule Summary:${NC}"
    case "$schedule_type" in
        gallery)
            echo "Gallery Hours: Tuesday-Saturday"
            echo "  • Start: 10:55 AM (services begin)"
            echo "  • Stop:   6:05 PM (services end)"
            ;;
        gallery-extended)
            echo "Gallery Hours: Tuesday-Saturday"
            echo "  • Start: 10:55 AM (services begin)"
            echo "  • Stop:   6:05 PM (Tues, Thurs, Fri, Sat)"
            echo "  • Stop:   9:05 PM (Wednesday night)"
            ;;
        daily)
            echo "Daily Schedule:"
            echo "  • Start: 9:00 AM (services begin)"
            echo "  • Stop: 10:00 PM (services end)"
            ;;
        custom)
            echo "Custom schedule configured"
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}Key Features:${NC}"
    echo "✓ Existing services keep RunAtLoad=true (start after reboot)"
    echo "✓ Gallery scheduler starts/stops services during hours"
    echo "✓ Machine can stay on 24/7, services only run when needed"
    echo "✓ Manual override available with manual-start/manual-stop"
    
    echo ""
    echo -e "${GREEN}Logs:${NC}"
    echo "Starter: ~/Library/Logs/experimance/${PROJECT}_gallery_starter.log"
    echo "Stopper: ~/Library/Logs/experimance/${PROJECT}_gallery_stopper.log"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Test with: $0 $PROJECT manual-stop"
    echo "2. Then test: $0 $PROJECT manual-start"
    echo "3. Check status: $0 $PROJECT show-schedule"
}

# Remove scheduled LaunchAgents (internal version - minimal output)
remove_schedule_internal() {
    local starter_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-starter.plist"
    local stopper_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-stopper.plist"
    
    # Unload scheduler agents
    launchctl bootout gui/$(id -u) "$starter_plist" 2>/dev/null || true
    launchctl bootout gui/$(id -u) "$stopper_plist" 2>/dev/null || true
    
    # Remove scheduler plist files
    rm -f "$starter_plist"
    rm -f "$stopper_plist"
}

# Remove scheduled LaunchAgents
remove_schedule() {
    log "Removing gallery hour scheduling for project $PROJECT..."
    
    local starter_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-starter.plist"
    local stopper_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-stopper.plist"
    
    # Unload scheduler agents
    launchctl bootout gui/$(id -u) "$starter_plist" 2>/dev/null || true
    launchctl bootout gui/$(id -u) "$stopper_plist" 2>/dev/null || true
    
    # Remove scheduler plist files
    rm -f "$starter_plist"
    rm -f "$stopper_plist"
    
    log "✓ Gallery hour scheduling removed"
    log "Your main services remain available and will auto-start after reboot"
    echo ""
    echo -e "${GREEN}Services are now in always-on mode:${NC}"
    
    # Show current service status
    while IFS= read -r label; do
        if launchctl list | grep -q "$label"; then
            echo -e "  ${GREEN}✓${NC} $label (running)"
        else
            echo -e "  ${RED}✗${NC} $label (stopped)"
        fi
    done < <(get_service_labels "$PROJECT")
}

# Helper function to ensure a service is loaded before starting
ensure_service_loaded() {
    local label="$1"
    
    # Check if service is already loaded
    if launchctl list | grep -q "$label"; then
        return 0  # Already loaded
    fi
    
    # Find the plist file for this label
    local plist_file=""
    while IFS= read -r pf; do
        local file_label=$(plutil -extract Label raw "$pf" 2>/dev/null || echo "")
        if [[ "$file_label" == "$label" ]]; then
            plist_file="$pf"
            break
        fi
    done < <(get_existing_plist_files "$PROJECT")
    
    # Load the service if plist file exists
    if [[ -n "$plist_file" && -f "$plist_file" ]]; then
        launchctl bootstrap gui/$(id -u) "$plist_file" 2>/dev/null
        return $?
    else
        return 1  # Plist file not found
    fi
}

# Manually start all project services
manual_start() {
    log "Manually starting all $PROJECT services..."
    
    local started=0
    local failed=0
    
    # Separate services into Python and TouchDesigner
    local python_services=()
    local td_services=()
    
    while IFS= read -r label; do
        if [[ "$label" == *"touchdesigner"* ]]; then
            td_services+=("$label")
        else
            python_services+=("$label")
        fi
    done < <(get_service_labels "$PROJECT")
    
    # First, start TouchDesigner services
    if [ ${#td_services[@]} -gt 0 ]; then
        echo -e "${BLUE}Starting TouchDesigner services first...${NC}"
        for label in "${td_services[@]}"; do
            echo -n "Starting $label... "
            
            # Ensure service is loaded first
            if ! ensure_service_loaded "$label"; then
                echo -e "${RED}✗ (failed to load)${NC}"
                ((failed++))
                continue
            fi
            
            # Now try to start it
            if launchctl kickstart gui/$(id -u)/"$label" 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
                ((started++))
            else
                echo -e "${RED}✗${NC}"
                ((failed++))
            fi
        done
        
        # Wait for TouchDesigner to initialize before starting Python services
        if [ ${#python_services[@]} -gt 0 ]; then
            echo -e "${YELLOW}Waiting 20 seconds for TouchDesigner to initialize...${NC}"
            sleep 20
        fi
    fi
    
    # Then start Python services
    if [ ${#python_services[@]} -gt 0 ]; then
        echo -e "${BLUE}Starting Python services...${NC}"
        for label in "${python_services[@]}"; do
            echo -n "Starting $label... "
            
            # Ensure service is loaded first
            if ! ensure_service_loaded "$label"; then
                echo -e "${RED}✗ (failed to load)${NC}"
                ((failed++))
                continue
            fi
            
            # Now try to start it
            if launchctl kickstart gui/$(id -u)/"$label" 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
                ((started++))
            else
                echo -e "${RED}✗${NC}"
                ((failed++))
            fi
        done
    fi
    
    echo ""
    log "Manual start complete: $started started, $failed failed"
    
    if [[ $failed -gt 0 ]]; then
        echo -e "${YELLOW}Some services failed to start. Check logs or run:${NC}"
        echo "  $0 $PROJECT show-schedule"
    fi
}

# Manually stop all project services
manual_stop() {
    log "Manually stopping all $PROJECT services..."
    
    local stopped=0
    local failed=0
    
    # Separate services into Python and TouchDesigner
    local python_services=()
    local td_services=()
    
    while IFS= read -r label; do
        if [[ "$label" == *"touchdesigner"* ]]; then
            td_services+=("$label")
        else
            python_services+=("$label")
        fi
    done < <(get_service_labels "$PROJECT")
    
    # First, stop Python services
    if [ ${#python_services[@]} -gt 0 ]; then
        echo -e "${BLUE}Stopping Python services first...${NC}"
        for label in "${python_services[@]}"; do
            echo -n "Stopping $label... "
            if launchctl kill TERM gui/$(id -u)/"$label" 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
                ((stopped++))
            else
                echo -e "${YELLOW}○${NC} (already stopped or failed)"
                ((failed++))
            fi
        done
        
        # Wait for Python services to shut down gracefully
        if [ ${#td_services[@]} -gt 0 ]; then
            echo -e "${YELLOW}Waiting 10 seconds for Python services to shut down...${NC}"
            sleep 10
        fi
    fi
    
    # Then stop TouchDesigner services
    if [ ${#td_services[@]} -gt 0 ]; then
        echo -e "${BLUE}Stopping TouchDesigner services...${NC}"
        for label in "${td_services[@]}"; do
            echo -n "Stopping $label... "
            if launchctl kill TERM gui/$(id -u)/"$label" 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
                ((stopped++))
            else
                echo -e "${YELLOW}○${NC} (already stopped or failed)"
                ((failed++))
            fi
        done
    fi
    
    echo ""
    log "Manual stop complete: $stopped stopped, $failed already stopped/failed"
    
    echo ""
    echo -e "${YELLOW}Note: Services with KeepAlive will restart automatically.${NC}"
    echo -e "${YELLOW}Use 'manual-unload' to truly stop without auto-restart.${NC}"
}

# Manually unload all project services (stops without auto-restart)
manual_unload() {
    log "Manually unloading all $PROJECT services (no auto-restart)..."
    
    local unloaded=0
    local failed=0
    
    # Separate services into Python and TouchDesigner
    local python_services=()
    local td_services=()
    
    while IFS= read -r label; do
        if [[ "$label" == *"touchdesigner"* ]]; then
            td_services+=("$label")
        else
            python_services+=("$label")
        fi
    done < <(get_service_labels "$PROJECT")
    
    # First, stop Python services
    if [ ${#python_services[@]} -gt 0 ]; then
        echo -e "${BLUE}Stopping Python services first...${NC}"
        for label in "${python_services[@]}"; do
            echo -n "Stopping $label... "
            if launchctl kill TERM gui/$(id -u)/"$label" 2>/dev/null; then
                echo -e "${GREEN}✓ killed${NC}"
            else
                echo -e "${YELLOW}○ not running${NC}"
            fi
        done
        
        # Wait for Python services to shut down gracefully
        if [ ${#td_services[@]} -gt 0 ]; then
            echo -e "${YELLOW}Waiting 10 seconds for Python services to shut down...${NC}"
            sleep 10
        fi
    fi
    
    # Then stop TouchDesigner services
    if [ ${#td_services[@]} -gt 0 ]; then
        echo -e "${BLUE}Stopping TouchDesigner services...${NC}"
        for label in "${td_services[@]}"; do
            echo -n "Stopping $label... "
            if launchctl kill TERM gui/$(id -u)/"$label" 2>/dev/null; then
                echo -e "${GREEN}✓ killed${NC}"
            else
                echo -e "${YELLOW}○ not running${NC}"
            fi
        done
    fi
    
    echo ""
    echo -e "${BLUE}Unloading all LaunchAgents to prevent restart...${NC}"
    
    # Then unload all LaunchAgents to prevent restart
    while IFS= read -r label; do
        echo -n "Unloading $label... "
        
        # Find the plist file for this label
        local plist_file=""
        while IFS= read -r pf; do
            local file_label=$(plutil -extract Label raw "$pf" 2>/dev/null || echo "")
            if [[ "$file_label" == "$label" ]]; then
                plist_file="$pf"
                break
            fi
        done < <(get_existing_plist_files "$PROJECT")
        
        # Use bootout instead of unload (more reliable)
        if [[ -n "$plist_file" && -f "$plist_file" ]]; then
            if launchctl bootout gui/$(id -u)/"$label" 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
                ((unloaded++))
            else
                echo -e "${YELLOW}○${NC} (already unloaded or failed)"
                ((failed++))
            fi
        else
            echo -e "${YELLOW}○${NC} (plist not found)"
            ((failed++))
        fi
    done < <(get_service_labels "$PROJECT")
    
    echo ""
    log "Manual unload complete: $unloaded unloaded, $failed already unloaded/failed"
    
    echo ""
    echo -e "${GREEN}Services are now stopped and will NOT restart automatically.${NC}"
    echo -e "${BLUE}Use 'manual-start' to restart them.${NC}"
}

# Helper function to show combined schedule times from both starter and stopper plists
show_schedule_times() {
    local starter_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-starter.plist"
    local stopper_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-stopper.plist"
    
    local weekdays=("Sunday" "Monday" "Tuesday" "Wednesday" "Thursday" "Friday" "Saturday")
    
    # Variables to store times for each weekday (1-7)
    local start_1="" start_2="" start_3="" start_4="" start_5="" start_6="" start_7=""
    local stop_1="" stop_2="" stop_3="" stop_4="" stop_5="" stop_6="" stop_7=""
    
    # Extract start times
    if [[ -f "$starter_plist" ]]; then
        local index=0
        while [[ $index -lt 20 ]]; do
            local hour=$(plutil -extract "StartCalendarInterval.$index.Hour" raw "$starter_plist" 2>/dev/null)
            if [[ $? -ne 0 || "$hour" =~ "Could not extract" ]]; then
                break
            fi
            
            local minute=$(plutil -extract "StartCalendarInterval.$index.Minute" raw "$starter_plist" 2>/dev/null)
            local weekday=$(plutil -extract "StartCalendarInterval.$index.Weekday" raw "$starter_plist" 2>/dev/null)
            
            if [[ -n "$hour" && -n "$minute" && -n "$weekday" && "$weekday" -ge 1 && "$weekday" -le 7 ]]; then
                local time_str=$(printf "%02d:%02d" "$hour" "$minute")
                case $weekday in
                    1) start_1="$time_str" ;;
                    2) start_2="$time_str" ;;
                    3) start_3="$time_str" ;;
                    4) start_4="$time_str" ;;
                    5) start_5="$time_str" ;;
                    6) start_6="$time_str" ;;
                    7) start_7="$time_str" ;;
                esac
            fi
            
            ((index++))
        done
    fi
    
    # Extract stop times
    if [[ -f "$stopper_plist" ]]; then
        local index=0
        while [[ $index -lt 20 ]]; do
            local hour=$(plutil -extract "StartCalendarInterval.$index.Hour" raw "$stopper_plist" 2>/dev/null)
            if [[ $? -ne 0 || "$hour" =~ "Could not extract" ]]; then
                break
            fi
            
            local minute=$(plutil -extract "StartCalendarInterval.$index.Minute" raw "$stopper_plist" 2>/dev/null)
            local weekday=$(plutil -extract "StartCalendarInterval.$index.Weekday" raw "$stopper_plist" 2>/dev/null)
            
            if [[ -n "$hour" && -n "$minute" && -n "$weekday" && "$weekday" -ge 1 && "$weekday" -le 7 ]]; then
                local time_str=$(printf "%02d:%02d" "$hour" "$minute")
                case $weekday in
                    1) stop_1="$time_str" ;;
                    2) stop_2="$time_str" ;;
                    3) stop_3="$time_str" ;;
                    4) stop_4="$time_str" ;;
                    5) stop_5="$time_str" ;;
                    6) stop_6="$time_str" ;;
                    7) stop_7="$time_str" ;;
                esac
            fi
            
            ((index++))
        done
    fi
    
    # Display combined schedule
    local found_any=false
    for weekday in {1..7}; do
        local start_var="start_${weekday}"
        local stop_var="stop_${weekday}"
        local start_time="${!start_var}"
        local stop_time="${!stop_var}"
        
        if [[ -n "$start_time" ]]; then
            local schedule_line="  ${weekdays[$weekday]}: $start_time"
            if [[ -n "$stop_time" ]]; then
                schedule_line="${schedule_line}-${stop_time}"
            fi
            echo "$schedule_line"
            found_any=true
        fi
    done
    
    if [[ "$found_any" != true ]]; then
        echo "  Could not parse schedule times"
    fi
}

# Show current schedule
show_schedule() {
    echo ""
    echo -e "${BLUE}LaunchAgent Status for $PROJECT:${NC}"
    
    # Check for gallery scheduling
    local starter_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-starter.plist"
    local stopper_plist="$LAUNCHD_DIR/com.experimance.${PROJECT}.gallery-stopper.plist"
    
    echo ""
    echo -e "${BLUE}Gallery Hour Scheduling:${NC}"
    if [[ -f "$starter_plist" ]]; then
        echo -e "${GREEN}✓${NC} Gallery starter configured"
        if launchctl list | grep -q "com.experimance.${PROJECT}.gallery-starter"; then
            echo -e "  Status: ${GREEN}Loaded${NC}"
        else
            echo -e "  Status: ${RED}Not loaded${NC}"
        fi
    else
        echo -e "${YELLOW}○${NC} No gallery hour scheduling"
    fi
    
    if [[ -f "$stopper_plist" ]]; then
        echo -e "${GREEN}✓${NC} Gallery stopper configured"
        if launchctl list | grep -q "com.experimance.${PROJECT}.gallery-stopper"; then
            echo -e "  Status: ${GREEN}Loaded${NC}"
        else
            echo -e "  Status: ${RED}Not loaded${NC}"
        fi
    fi
    
    # Show combined schedule times if both exist
    if [[ -f "$starter_plist" || -f "$stopper_plist" ]]; then
        echo ""
        echo -e "${BLUE}Schedule:${NC}"
        show_schedule_times
    fi
    
    echo ""
    echo -e "${BLUE}Main Services (RunAtLoad=true):${NC}"
    
    local running_count=0
    local total_count=0
    
    while IFS= read -r label; do
        ((total_count++))
        local plist_file=""
        
        # Find the plist file for this label
        while IFS= read -r pf; do
            local file_label=$(plutil -extract Label raw "$pf" 2>/dev/null || echo "")
            if [[ "$file_label" == "$label" ]]; then
                plist_file="$pf"
                break
            fi
        done < <(get_existing_plist_files "$PROJECT")
        
        # Check if service is running
        if launchctl list | grep -q "$label"; then
            echo -e "${GREEN}✓${NC} $label (running)"
            ((running_count++))
            
            # Show RunAtLoad status
            if [[ -n "$plist_file" ]]; then
                local run_at_load=$(plutil -extract RunAtLoad raw "$plist_file" 2>/dev/null || echo "false")
                if [[ "$run_at_load" == "true" ]]; then
                    echo -e "  ${GREEN}Auto-start: enabled${NC}"
                else
                    echo -e "  ${YELLOW}Auto-start: disabled${NC}"
                fi
            fi
        else
            echo -e "${RED}✗${NC} $label (stopped)"
            if [[ -n "$plist_file" ]]; then
                local run_at_load=$(plutil -extract RunAtLoad raw "$plist_file" 2>/dev/null || echo "false")
                if [[ "$run_at_load" == "true" ]]; then
                    echo -e "  ${YELLOW}Auto-start: enabled (but not running)${NC}"
                else
                    echo -e "  ${RED}Auto-start: disabled${NC}"
                fi
            fi
        fi
    done < <(get_service_labels "$PROJECT")
    
    echo ""
    echo -e "${BLUE}Summary:${NC}"
    echo "Services: $running_count/$total_count running"
    
    if [[ -f "$starter_plist" ]]; then
        echo "Gallery scheduling: Enabled"
        echo ""
        echo -e "${GREEN}Manual Controls:${NC}"
        echo "Start all:  $0 $PROJECT manual-start"
        echo "Stop all:   $0 $PROJECT manual-stop"
        echo "Remove scheduling: $0 $PROJECT remove-schedule"
    else
        echo "Gallery scheduling: Disabled (always-on mode)"
        echo ""
        echo -e "${GREEN}Available Actions:${NC}"
        echo "Add scheduling: $0 $PROJECT setup-schedule gallery"
        echo "Start all:      $0 $PROJECT manual-start"
        echo "Stop all:       $0 $PROJECT manual-stop"
    fi
    
    echo ""
}

# Restart all project services (services auto-restart due to KeepAlive=true)
manual_restart() {
    log "Manually restarting all $PROJECT services..."
    
    echo ""
    echo -e "${BLUE}Stopping services (they will auto-restart)...${NC}"
    manual_stop
    
    echo ""
    log "Manual restart complete - services will restart automatically due to KeepAlive=true"
}

# Main function
main() {
    case "$ACTION" in
        setup-schedule)
            setup_schedule "$SCHEDULE_TYPE"
            ;;
        remove-schedule)
            remove_schedule
            ;;
        show-schedule)
            show_schedule
            ;;
        manual-start)
            manual_start
            ;;
        manual-stop)
            manual_stop
            ;;
        manual-unload)
            manual_unload
            ;;
        manual-restart)
            manual_restart
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    show_usage
    exit 1
fi

main "$@"
