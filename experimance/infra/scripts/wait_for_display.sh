#!/bin/bash
# Simple, working display wait script
set -u

timeout=1800
interval=2
debug=${WAIT_DISPLAY_DEBUG:-false}

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --timeout)
            timeout="$2"
            shift 2
            ;;
        --interval)
            interval="$2"
            shift 2
            ;;
        --debug)
            debug=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--timeout N] [--interval N] [--debug]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

log() {
    echo "[wait_for_display] $(date '+%H:%M:%S') $*"
}

# Check for connected DRM displays
has_connected_display() {
    for f in /sys/class/drm/card*-*/status; do
        [ -f "$f" ] || continue
        if [ "$(cat "$f" 2>/dev/null || echo disconnected)" = "connected" ]; then
            return 0
        fi
    done
    return 1
}

# Check for Wayland socket
has_wayland() {
    [ -n "${WAYLAND_DISPLAY:-}" ] && [ -n "${XDG_RUNTIME_DIR:-}" ] && [ -S "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}" ]
}

log "Waiting for display (timeout=${timeout}s)"
start=$(date +%s)
deadline=$((start + timeout))

while [ $(date +%s) -lt $deadline ]; do
    connected_display=false
    wayland_ready=false
    
    if has_connected_display; then
        connected_display=true
        [ "$debug" = true ] && log "Found connected DRM display"
    fi
    
    if has_wayland; then
        wayland_ready=true
        [ "$debug" = true ] && log "Found Wayland socket"
    fi
    
    if [ "$connected_display" = true ] || [ "$wayland_ready" = true ]; then
        elapsed=$(($(date +%s) - start))
        log "Display ready after ${elapsed}s (drm=$connected_display wayland=$wayland_ready)"
        exit 0
    fi
    
    [ "$debug" = true ] && log "No display detected, waiting..."
    sleep $interval
done

log "Timeout after ${timeout}s - no display found"
exit 1
