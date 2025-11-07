#!/bin/bash
# 
# Experimance Startup Script
# Simple startup script for cron/systemd target management
#

# Cron-compatible environment setup
export PATH="/home/experimance/.local/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export HOME="${HOME:-$(eval echo ~$(whoami))}"

# Parse arguments
PROJECT=""
DO_RESET=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --reset)
            DO_RESET=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --help, -h          Show this help message"
            echo "  --project PROJECT   Set project environment (default: experimance)"
            echo "  --reset             Perform full reset (stop services, reset audio, then start)"
            echo ""
            echo "Environment variables:"
            echo "  PROJECT_ENV         Project environment to start (default: experimance)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--project <project_name>] [--reset]"
            exit 1
            ;;
    esac
done

# Default to experimance if no project specified, or use PROJECT_ENV from environment
if [[ -z "$PROJECT" ]]; then
    PROJECT="${PROJECT_ENV:-experimance}"
fi

# Set PROJECT_ENV for backward compatibility
PROJECT_ENV="$PROJECT"

# If reset flag is set, delegate to reset.sh
if [[ "$DO_RESET" == "true" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    exec "$SCRIPT_DIR/reset.sh" --project "$PROJECT"
fi

# Start the Experimance target (which starts all services)
systemctl start "experimance@${PROJECT_ENV}.target"
