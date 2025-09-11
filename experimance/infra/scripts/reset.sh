#!/bin/bash
set -e

# Daily reset script for Experimance services
# This script cycles services to prevent audio/memory issues
# and ensures clean state for gallery operations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Parse arguments
PROJECT=""
SKIP_RESET=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --no-reset)
            SKIP_RESET=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --help, -h          Show this help message"
            echo "  --project PROJECT   Set project environment (default: experimance)"
            echo "  --no-reset          Skip audio reset, just do startup"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--project <project_name>] [--no-reset]"
            exit 1
            ;;
    esac
done

# Default to experimance if no project specified
if [[ -z "$PROJECT" ]]; then
    PROJECT="experimance"
fi

# Set PROJECT_ENV for backward compatibility
PROJECT_ENV="$PROJECT"

# Create log directory if it doesn't exist
sudo mkdir -p /var/log/experimance
LOG_FILE="/var/log/experimance/daily-reset.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" | sudo tee -a "$LOG_FILE"
}

log "=== Starting Experimance Daily Reset for project: $PROJECT ==="

# Stop services in reverse dependency order
log "Stopping services..."

# Stop the Experimance target (which stops all services)
sudo systemctl stop "experimance@${PROJECT_ENV}.target"

# Wait for things to settle
sleep 2

# Reset audio devices (the key fix for crashes) - only if not skipped
if [[ "$SKIP_RESET" != "true" ]]; then
    log "Resetting audio devices..."

    # Set up environment for uv (cron has limited PATH)
    export PATH="/home/experimance/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

    if cd "$PROJECT_ROOT" && uv run scripts/audio_recovery.py reset >> "$LOG_FILE" 2>&1; then
        log "Audio reset completed successfully"
    else
        log "WARNING: Audio reset failed, continuing anyway"
    fi

    # Wait for audio system to stabilize
    sleep 3
else
    log "Skipping audio reset (--no-reset flag used)"
fi

# Check system resources before restart
log "System status before restart:"
log "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
log "CPU: $(uptime | awk -F'load average:' '{print $2}')"

# Start the Experimance target (which starts all services)
sudo systemctl start "experimance@${PROJECT_ENV}.target"

# Wait for services to stabilize
sleep 10

# Check system resources after restart
log "System status after restart:"
log "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
log "CPU: $(uptime | awk -F'load average:' '{print $2}')"

exit 0