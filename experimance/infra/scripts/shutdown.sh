#!/bin/bash
# 
# Experimance Shutdown Script
# Simple shutdown script for cron/systemd target management
#

# Default project
PROJECT="experimance"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --help, -h          Show this help message"
            echo "  --project PROJECT   Set project environment (default: experimance)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --project <project_name>"
            exit 1
            ;;
    esac
done

# Set PROJECT_ENV for backward compatibility
PROJECT_ENV="$PROJECT"

# Cron-compatible environment setup
export PATH="/home/experimance/.local/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export HOME="${HOME:-$(eval echo ~$(whoami))}"

# Create log directory if it doesn't exist
sudo mkdir -p /var/log/experimance
LOG_FILE="/var/log/experimance/shutdown.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" | sudo tee -a "$LOG_FILE"
}

log "=== Starting Experimance Shutdown for project: $PROJECT ==="

# Stop the Experimance target (which stops all services)
log "Stopping services..."
sudo systemctl stop "experimance@${PROJECT_ENV}.target"

# Destroy VastAI instances (created by image_server)
log "Destroying VastAI instances (if any)..."
cd "$(dirname "$(dirname "$(dirname "$0")")")" # Go to repo root
if command -v uv &> /dev/null || id "experimance" &>/dev/null; then
    # Run as experimance user since VastAI credentials are under that account
    if id "experimance" &>/dev/null; then
        su - experimance -c "cd $(pwd) && uv run python scripts/vastai_cli.py destroy" >> "$LOG_FILE" 2>&1 || true
    else
        # Fallback: run as current user (development mode)
        uv run python scripts/vastai_cli.py destroy >> "$LOG_FILE" 2>&1 || true
    fi
    log "VastAI destroy operation completed"
else
    log "No uv command or experimance user found, skipping VastAI destroy"
fi

log "=== Experimance Shutdown Complete ==="
exit 0
