#!/bin/bash
# 
# Experimance Shutdown Script
# Simple shutdown script for cron/systemd target management
#

PROJECT_ENV="${PROJECT_ENV:-experimance}"

# Cron-compatible environment setup
export PATH="/home/experimance/.local/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export HOME="${HOME:-$(eval echo ~$(whoami))}"

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --project PROJECT   Set project environment (default: experimance)"
        echo ""
        echo "Environment variables:"
        echo "  PROJECT_ENV         Project environment to stop (default: experimance)"
        echo ""
        exit 0
        ;;
    --project)
        PROJECT_ENV="$2"
        ;;
esac

# Stop the Experimance target (which stops all services)
systemctl stop "experimance@${PROJECT_ENV}.target"

# Destroy VastAI instances (created by image_server)
cd "$(dirname "$(dirname "$(dirname "$0")")")" # Go to repo root
if command -v uv &> /dev/null || id "experimance" &>/dev/null; then
    # Run as experimance user since VastAI credentials are under that account
    if id "experimance" &>/dev/null; then
        su - experimance -c "cd $(pwd) && uv run python scripts/vastai_cli.py destroy" 2>/dev/null || true
    else
        # Fallback: run as current user (development mode)
        uv run python scripts/vastai_cli.py destroy 2>/dev/null || true
    fi
fi
