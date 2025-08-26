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

# Check for working display (connected AND has valid EDID data)
has_working_display() {
    for f in /sys/class/drm/card*-*/status; do
        [ -f "$f" ] || continue
        if [ "$(cat "$f" 2>/dev/null || echo disconnected)" = "connected" ]; then
            # Check if this connector has EDID data
            local edid_file="$(dirname "$f")/edid"
            if [ -f "$edid_file" ]; then
                local edid_size=$(stat -c%s "$edid_file" 2>/dev/null || echo "0")
                if [ "$edid_size" -gt 0 ]; then
                    [ "$debug" = true ] && log "Found working display: $(basename "$(dirname "$f")") (EDID: ${edid_size} bytes)"
                    return 0
                else
                    [ "$debug" = true ] && log "Display connected but no EDID: $(basename "$(dirname "$f")") (possible 'No Signal' state)"
                fi
            fi
        fi
    done
    return 1
}

# Attempt to fix "No Signal" state by forcing display refresh
fix_no_signal() {
    [ "$debug" = true ] && log "Attempting to fix 'No Signal' state..."
    
    # Method 1: Force DRM reprobe
    if command -v udevadm >/dev/null 2>&1; then
        [ "$debug" = true ] && log "Triggering udev DRM reprobe"
        sudo udevadm trigger --subsystem-match=drm --action=change >/dev/null 2>&1 || true
    fi
    
    # Method 2: Force connector detection
    for connector in /sys/class/drm/card*-*/status; do
        if [ -f "$connector" ]; then
            local detect_file="$(dirname "$connector")/detect"
            if [ -f "$detect_file" ]; then
                [ "$debug" = true ] && log "Forcing detection on $(basename "$(dirname "$connector")")"
                echo detect | sudo tee "$detect_file" >/dev/null 2>&1 || true
            fi
        fi
    done
    
    # Method 3: Try to force card reprobe (newer kernels)
    for card in /sys/class/drm/card*; do
        if [ -d "$card" ]; then
            local reprobe_file="$card/device/rescan"
            if [ -f "$reprobe_file" ]; then
                [ "$debug" = true ] && log "Force reprobing $(basename "$card")"
                echo 1 | sudo tee "$reprobe_file" >/dev/null 2>&1 || true
            fi
        fi
    done
    
    # Give hardware time to respond
    [ "$debug" = true ] && log "Waiting for hardware to respond..."
    sleep 3
}

# Check for Wayland socket
has_wayland() {
    [ -n "${WAYLAND_DISPLAY:-}" ] && [ -n "${XDG_RUNTIME_DIR:-}" ] && [ -S "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}" ]
}

# Setup display environment if missing but Wayland is available
setup_display_env_if_needed() {
    local target_user="${1:-experimance}"
    
    # If we already have display environment, we're good
    if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        [ "$debug" = true ] && log "Display environment already set"
        return 0
    fi
    
    # Manual detection of Wayland socket (most reliable)
    local runtime_dir="/run/user/$(id -u "$target_user" 2>/dev/null || echo 1000)"
    if [ -d "$runtime_dir" ]; then
        for socket in wayland-0 wayland-1; do
            if [ -S "$runtime_dir/$socket" ]; then
                export XDG_RUNTIME_DIR="$runtime_dir"
                export WAYLAND_DISPLAY="$socket"
                [ "$debug" = true ] && log "Setup: XDG_RUNTIME_DIR=$runtime_dir WAYLAND_DISPLAY=$socket"
                return 0
            fi
        done
    fi
    
    [ "$debug" = true ] && log "No Wayland socket found in $runtime_dir"
    return 1
}

log "Waiting for display (timeout=${timeout}s)"
start=$(date +%s)
deadline=$((start + timeout))

while [ $(date +%s) -lt $deadline ]; do
    connected_display=false
    working_display=false
    wayland_ready=false
    
    if has_connected_display; then
        connected_display=true
        [ "$debug" = true ] && log "Found connected DRM display"
    fi
    
    if has_working_display; then
        working_display=true
        [ "$debug" = true ] && log "Found working DRM display (with EDID)"
    fi
    
    if has_wayland; then
        wayland_ready=true
        [ "$debug" = true ] && log "Found Wayland socket"
    fi
    
    # If we have a working display or wayland, we're good
    if [ "$working_display" = true ] || [ "$wayland_ready" = true ]; then
        elapsed=$(($(date +%s) - start))
        log "Display ready after ${elapsed}s (working_display=$working_display wayland=$wayland_ready)"
        exit 0
    fi
    
    # If we have hardware display but it's not working (No Signal state), try to fix it
    if [ "$connected_display" = true ] && [ "$working_display" = false ]; then
        [ "$debug" = true ] && log "Connected display found but not working - attempting fix"
        fix_no_signal
        
        # Check again after fix attempt
        if has_working_display; then
            working_display=true
            [ "$debug" = true ] && log "Fix successful - display now working"
        elif setup_display_env_if_needed; then
            wayland_ready=true
            [ "$debug" = true ] && log "Hardware display not working but Wayland environment setup successful"
        fi
        
        # Try once more after fixes
        if [ "$working_display" = true ] || [ "$wayland_ready" = true ]; then
            elapsed=$(($(date +%s) - start))
            log "Display ready after fix attempts at ${elapsed}s (working_display=$working_display wayland=$wayland_ready)"
            exit 0
        fi
    fi
    
    [ "$debug" = true ] && log "No working display detected, waiting..."
    sleep $interval
done

# After timeout, try to setup Wayland environment as fallback
# (This handles the case where projector is off but we want services to start anyway)
log "Timeout reached, attempting to setup Wayland environment as fallback..."
if setup_display_env_if_needed; then
    log "Display ready via fallback Wayland setup (drm=false wayland=true)"
    exit 0
fi

log "Timeout after ${timeout}s - no display found"
exit 1
