#!/bin/bash

# Experimance Kiosk Mode Script
# Enables/disables kiosk mode for art installation on Ubuntu 24.04
# Safe settings-based approach that can be easily reversed

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
USER="${EXPERIMANCE_USER:-experimance}"
BACKUP_DIR="/var/backups/experimance/kiosk-settings"
KIOSK_MODE_FILE="/var/lib/experimance/kiosk-mode"

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

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

show_help() {
    cat << EOF
Experimance Kiosk Mode Script

USAGE:
    $0 <command> [options]

COMMANDS:
    enable      Enable kiosk mode for art installation
    disable     Disable kiosk mode (restore normal desktop)
    status      Show current kiosk mode status
    backup      Create backup of current settings
    restore     Restore from backup

OPTIONS:
    --user USER     Target user (default: experimance)
    --help, -h      Show this help message

EXAMPLES:
    $0 enable
    $0 disable
    $0 status
    $0 backup

EOF
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
    fi
    
    # Check if user exists
    if ! id "$USER" &>/dev/null; then
        error "User '$USER' does not exist"
    fi
    
    # Check if GNOME is installed
    if ! command -v gsettings &>/dev/null; then
        error "gsettings not found. This script requires GNOME desktop environment"
    fi
    
    # Check if we're on Ubuntu 24.04
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        if [[ "$ID" != "ubuntu" ]] || [[ "${VERSION_ID}" != "24.04" ]]; then
            warn "This script is designed for Ubuntu 24.04. Current: $ID $VERSION_ID"
        fi
    fi
    
    log "Prerequisites check passed"
}

create_backup() {
    log "Creating backup of current settings..."
    
    # Create backup directory with proper permissions
    mkdir -p "$BACKUP_DIR"
    chown -R "$USER:$USER" "$(dirname "$BACKUP_DIR")"
    chown -R "$USER:$USER" "$BACKUP_DIR"
    chmod 755 "$BACKUP_DIR"
    
    timestamp=$(date '+%Y%m%d_%H%M%S')
    backup_file="$BACKUP_DIR/settings_backup_$timestamp.json"
    
    # Backup current gsettings
    sudo -u "$USER" bash -c "
        export DISPLAY=:0
        export XDG_RUNTIME_DIR=/run/user/$(id -u $USER)
        
        cat > '$backup_file' << 'EOF'
{
    \"timestamp\": \"$(date -Iseconds)\",
    \"user\": \"$USER\",
    \"settings\": {
        \"screensaver_lock\": \"$(gsettings get org.gnome.desktop.screensaver lock-enabled 2>/dev/null || echo 'null')\",
        \"screensaver_idle_delay\": \"$(gsettings get org.gnome.desktop.session idle-delay 2>/dev/null || echo 'null')\",
        \"power_sleep_inactive_ac\": \"$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || echo 'null')\",
        \"power_sleep_inactive_battery\": \"$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 2>/dev/null || echo 'null')\",
        \"notifications_show_banners\": \"$(gsettings get org.gnome.desktop.notifications show-banners 2>/dev/null || echo 'null')\",
        \"notifications_show_in_lock_screen\": \"$(gsettings get org.gnome.desktop.notifications show-in-lock-screen 2>/dev/null || echo 'null')\",
        \"privacy_hide_identity\": \"$(gsettings get org.gnome.desktop.privacy hide-identity 2>/dev/null || echo 'null')\",
        \"privacy_report_technical_problems\": \"$(gsettings get org.gnome.desktop.privacy report-technical-problems 2>/dev/null || echo 'null')\",
        \"wm_focus_mode\": \"$(gsettings get org.gnome.desktop.wm.preferences focus-mode 2>/dev/null || echo 'null')\",
        \"interface_show_battery_percentage\": \"$(gsettings get org.gnome.desktop.interface show-battery-percentage 2>/dev/null || echo 'null')\",
        \"shell_disable_user_extensions\": \"$(gsettings get org.gnome.shell disable-user-extensions 2>/dev/null || echo 'null')\",
        \"media_automount\": \"$(gsettings get org.gnome.desktop.media-handling automount 2>/dev/null || echo 'null')\",
        \"media_automount_open\": \"$(gsettings get org.gnome.desktop.media-handling automount-open 2>/dev/null || echo 'null')\",
        \"media_autorun_never\": \"$(gsettings get org.gnome.desktop.media-handling autorun-never 2>/dev/null || echo 'null')\"
    }
}
EOF
    "
    
    # Also backup system settings
    systemctl is-enabled unattended-upgrades 2>/dev/null > "$BACKUP_DIR/unattended-upgrades.status" || echo "not-found" > "$BACKUP_DIR/unattended-upgrades.status"
    
    # Ensure backup files are owned by the user
    chown "$USER:$USER" "$backup_file"
    chown "$USER:$USER" "$BACKUP_DIR/unattended-upgrades.status"
    
    log "Backup created: $backup_file"
    echo "$backup_file"
}

enable_kiosk_mode() {
    log "Enabling kiosk mode..."
    
    # Create backup first
    backup_file=$(create_backup)
    
    # System-wide changes
    log "Configuring system settings..."
    
    # Disable unattended upgrades
    if systemctl is-enabled unattended-upgrades &>/dev/null; then
        log "Disabling unattended upgrades..."
        systemctl disable unattended-upgrades
        systemctl stop unattended-upgrades
    fi
    
    # User-specific GNOME settings
    log "Configuring GNOME settings for user: $USER"
    
    sudo -u "$USER" bash -c "
        export DISPLAY=:0
        export XDG_RUNTIME_DIR=/run/user/$(id -u $USER)
        
        # Disable screen lock and screensaver
        gsettings set org.gnome.desktop.screensaver lock-enabled false
        gsettings set org.gnome.desktop.session idle-delay 0
        
        # Disable sleep/suspend
        gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
        gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
        
        # Disable notifications and banners
        gsettings set org.gnome.desktop.notifications show-banners false
        gsettings set org.gnome.desktop.notifications show-in-lock-screen false
        
        # Privacy settings
        gsettings set org.gnome.desktop.privacy hide-identity true
        gsettings set org.gnome.desktop.privacy report-technical-problems false
        
        # Window management
        gsettings set org.gnome.desktop.wm.preferences focus-mode 'click'
        
        # Hide system indicators
        gsettings set org.gnome.desktop.interface show-battery-percentage false
        
        # Disable user extensions (keep system ones)
        gsettings set org.gnome.shell disable-user-extensions true
        
        # Disable media automount/autorun
        gsettings set org.gnome.desktop.media-handling automount false
        gsettings set org.gnome.desktop.media-handling automount-open false
        gsettings set org.gnome.desktop.media-handling autorun-never true
    "
    
    # Create kiosk mode indicator file 
    mkdir -p "$(dirname "$KIOSK_MODE_FILE")"
    echo "enabled" > "$KIOSK_MODE_FILE"
    echo "backup_file: $backup_file" >> "$KIOSK_MODE_FILE"
    echo "enabled_at: $(date -Iseconds)" >> "$KIOSK_MODE_FILE"

    log "Kiosk mode enabled successfully!"
    info "Settings backed up to: $backup_file"
    warn "Please restart the user session or reboot for all changes to take effect"
}

disable_kiosk_mode() {
    log "Disabling kiosk mode..."

    if [[ ! -f "$KIOSK_MODE_FILE" ]]; then
        warn "Kiosk mode doesn't appear to be enabled"
        return 0
    fi
    
    # Get the latest backup file
    latest_backup=$(find "$BACKUP_DIR" -name "settings_backup_*.json" -type f | sort -r | head -n 1)
    
    if [[ -z "$latest_backup" ]]; then
        error "No backup file found. Cannot safely restore settings."
    fi
    
    log "Restoring settings from: $latest_backup"
    
    # Restore system settings
    unattended_status=$(cat "$BACKUP_DIR/unattended-upgrades.status" 2>/dev/null || echo "unknown")
    if [[ "$unattended_status" == "enabled" ]]; then
        log "Re-enabling unattended upgrades..."
        systemctl enable unattended-upgrades
        systemctl start unattended-upgrades
    fi
    
    # Restore GNOME settings
    sudo -u "$USER" bash -c "
        export DISPLAY=:0
        export XDG_RUNTIME_DIR=/run/user/$(id -u $USER)
        
        # Parse backup and restore settings
        python3 << 'PYTHON_EOF'
import json
import subprocess
import sys

def run_gsettings(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f\"Error running: {cmd}\", file=sys.stderr)
        return False

with open('$latest_backup', 'r') as f:
    backup = json.load(f)

settings = backup['settings']

# Restore each setting if it was not null in backup
for key, value in settings.items():
    if value != 'null' and value is not None:
        # Map backup keys to gsettings paths
        setting_map = {
            'screensaver_lock': 'org.gnome.desktop.screensaver lock-enabled',
            'screensaver_idle_delay': 'org.gnome.desktop.session idle-delay', 
            'power_sleep_inactive_ac': 'org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type',
            'power_sleep_inactive_battery': 'org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type',
            'notifications_show_banners': 'org.gnome.desktop.notifications show-banners',
            'notifications_show_in_lock_screen': 'org.gnome.desktop.notifications show-in-lock-screen',
            'privacy_hide_identity': 'org.gnome.desktop.privacy hide-identity',
            'privacy_report_technical_problems': 'org.gnome.desktop.privacy report-technical-problems',
            'wm_focus_mode': 'org.gnome.desktop.wm.preferences focus-mode',
            'interface_show_battery_percentage': 'org.gnome.desktop.interface show-battery-percentage',
            'shell_disable_user_extensions': 'org.gnome.shell disable-user-extensions',
            'media_automount': 'org.gnome.desktop.media-handling automount',
            'media_automount_open': 'org.gnome.desktop.media-handling automount-open',
            'media_autorun_never': 'org.gnome.desktop.media-handling autorun-never'
        }
        
        if key in setting_map:
            cmd = f\"gsettings set {setting_map[key]} {value}\"
            if run_gsettings(cmd):
                print(f\"Restored: {key} = {value}\")
            else:
                print(f\"Failed to restore: {key}\", file=sys.stderr)

PYTHON_EOF
    "
    
    # Remove kiosk mode indicator
    rm -f "$KIOSK_MODE_FILE"
    log "Restored settings from backup: $latest_backup"
    log "Kiosk mode disabled successfully!"
    warn "Please restart the user session or reboot for all changes to take effect"
}

show_status() {
    log "Checking kiosk mode status..."

    if [[ -f "$KIOSK_MODE_FILE" ]]; then
        info "Kiosk mode: ENABLED"
        cat "$KIOSK_MODE_FILE"
    else
        info "Kiosk mode: DISABLED"
    fi
    
    # Show some key settings
    echo ""
    info "Current settings for user: $USER"
    
    sudo -u "$USER" bash -c "
        export DISPLAY=:0
        export XDG_RUNTIME_DIR=/run/user/$(id -u $USER)
        
        echo \"  Screen lock: \$(gsettings get org.gnome.desktop.screensaver lock-enabled 2>/dev/null || echo 'unknown')\"
        echo \"  Idle delay: \$(gsettings get org.gnome.desktop.session idle-delay 2>/dev/null || echo 'unknown')\"
        echo \"  Show banners: \$(gsettings get org.gnome.desktop.notifications show-banners 2>/dev/null || echo 'unknown')\"
        echo \"  Sleep on AC: \$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || echo 'unknown')\"
    "
    
    echo ""
    info "System settings:"
    echo "  Unattended upgrades: $(systemctl is-enabled unattended-upgrades 2>/dev/null || echo 'not installed')"
}

restore_from_backup() {
    local backup_file="$1"
    
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file not found: $backup_file"
    fi
    
    log "Restoring from backup: $backup_file"
    
    # Restore system settings
    unattended_status=$(cat "$BACKUP_DIR/unattended-upgrades.status" 2>/dev/null || echo "unknown")
    if [[ "$unattended_status" == "enabled" ]]; then
        log "Re-enabling unattended upgrades..."
        systemctl enable unattended-upgrades
        systemctl start unattended-upgrades
    fi
    
    # Restore GNOME settings using the specified backup file
    sudo -u "$USER" bash -c "
        export DISPLAY=:0
        export XDG_RUNTIME_DIR=/run/user/$(id -u $USER)
        
        # Parse backup and restore settings
        python3 << 'PYTHON_EOF'
import json
import subprocess
import sys

def run_gsettings(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f\"Error running: {cmd}\", file=sys.stderr)
        return False

with open('$backup_file', 'r') as f:
    backup = json.load(f)

settings = backup['settings']

# Restore each setting if it was not null in backup
for key, value in settings.items():
    if value != 'null' and value is not None:
        # Map backup keys to gsettings paths
        setting_map = {
            'screensaver_lock': 'org.gnome.desktop.screensaver lock-enabled',
            'screensaver_idle_delay': 'org.gnome.desktop.session idle-delay', 
            'power_sleep_inactive_ac': 'org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type',
            'power_sleep_inactive_battery': 'org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type',
            'notifications_show_banners': 'org.gnome.desktop.notifications show-banners',
            'notifications_show_in_lock_screen': 'org.gnome.desktop.notifications show-in-lock-screen',
            'privacy_hide_identity': 'org.gnome.desktop.privacy hide-identity',
            'privacy_report_technical_problems': 'org.gnome.desktop.privacy report-technical-problems',
            'wm_focus_mode': 'org.gnome.desktop.wm.preferences focus-mode',
            'interface_show_battery_percentage': 'org.gnome.desktop.interface show-battery-percentage',
            'shell_disable_user_extensions': 'org.gnome.shell disable-user-extensions',
            'media_automount': 'org.gnome.desktop.media-handling automount',
            'media_automount_open': 'org.gnome.desktop.media-handling automount-open',
            'media_autorun_never': 'org.gnome.desktop.media-handling autorun-never'
        }
        
        if key in setting_map:
            cmd = f\"gsettings set {setting_map[key]} {value}\"
            if run_gsettings(cmd):
                print(f\"Restored: {key} = {value}\")
            else:
                print(f\"Failed to restore: {key}\", file=sys.stderr)

PYTHON_EOF
    "
    
    # Remove kiosk mode indicator
    rm -f "$KIOSK_MODE_FILE"
    
    log "Restored settings from backup: $backup_file"
    log "Settings restored successfully!"
    warn "Please restart the user session or reboot for all changes to take effect"
}

list_backups() {
    log "Available backups:"
    
    if [[ -d "$BACKUP_DIR" ]]; then
        find "$BACKUP_DIR" -name "settings_backup_*.json" -type f | sort -r | while read -r backup; do
            timestamp=$(basename "$backup" | sed 's/settings_backup_\(.*\)\.json/\1/')
            readable_date=$(date -d "${timestamp:0:8} ${timestamp:9:2}:${timestamp:11:2}:${timestamp:13:2}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$timestamp")
            echo "  $backup ($readable_date)"
        done
    else
        warn "No backup directory found: $BACKUP_DIR"
    fi
}

main() {
    local command="${1:-}"
    
    case "$command" in
        enable)
            check_prerequisites
            enable_kiosk_mode
            ;;
        disable)
            check_prerequisites
            disable_kiosk_mode
            ;;
        status)
            show_status
            ;;
        backup)
            check_prerequisites
            create_backup
            ;;
        restore)
            if [[ -z "${2:-}" ]]; then
                list_backups
                read -p "Enter backup file path: " backup_file
            else
                backup_file="$2"
            fi
            check_prerequisites
            restore_from_backup "$backup_file"
            ;;
        list-backups)
            list_backups
            ;;
        --help|-h|help)
            show_help
            ;;
        *)
            error "Unknown command: $command. Use --help for usage information."
            ;;
    esac
}

# Handle command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            USER="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Handle interruption
trap 'error "Operation interrupted"' INT TERM

main "$@"
