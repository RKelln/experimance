#!/bin/bash
# 
# Experimance Startup Script
# Simple startup script for cron/systemd target management
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
        echo "  PROJECT_ENV         Project environment to start (default: experimance)"
        echo ""
        exit 0
        ;;
    --project)
        PROJECT_ENV="$2"
        ;;
esac

# Start the Experimance target (which starts all services)
systemctl start "experimance@${PROJECT_ENV}.target"
