#!/bin/bash
# Setup display environment for systemd services under Wayland
# Based on the display detection logic from scripts/dev

# Function to detect and export display environment variables
setup_display_environment() {
    local target_user="${1:-experimance}"
    
    # Find active desktop session for the target user
    DESKTOP_SESSION=$(loginctl list-sessions --no-legend | grep "$target_user.*seat0.*active" | awk '{print $1}' | head -1)
    
    if [ -z "$DESKTOP_SESSION" ]; then
        # Fallback: try to find any active session for the user
        DESKTOP_SESSION=$(loginctl list-sessions --no-legend | grep "$target_user" | awk '{print $1}' | head -1)
    fi
    
    if [ -n "$DESKTOP_SESSION" ]; then
        echo "Found desktop session: $DESKTOP_SESSION for user $target_user" >&2
        
        # Get display environment from GNOME process
        GNOME_PID=$(pgrep -u "$target_user" gnome-shell | head -1)
        if [ -n "$GNOME_PID" ] && [ -r "/proc/$GNOME_PID/environ" ]; then
            DESKTOP_DISPLAY=$(tr '\0' '\n' < "/proc/$GNOME_PID/environ" | grep '^DISPLAY=' | cut -d= -f2)
            DESKTOP_XDG_SESSION_TYPE=$(tr '\0' '\n' < "/proc/$GNOME_PID/environ" | grep '^XDG_SESSION_TYPE=' | cut -d= -f2)
            DESKTOP_XAUTHORITY=$(tr '\0' '\n' < "/proc/$GNOME_PID/environ" | grep '^XAUTHORITY=' | cut -d= -f2)
            DESKTOP_WAYLAND_DISPLAY=$(tr '\0' '\n' < "/proc/$GNOME_PID/environ" | grep '^WAYLAND_DISPLAY=' | cut -d= -f2)
            DESKTOP_XDG_RUNTIME_DIR=$(tr '\0' '\n' < "/proc/$GNOME_PID/environ" | grep '^XDG_RUNTIME_DIR=' | cut -d= -f2)
            
            # For Wayland sessions, check for Xwayland
            if [ "$DESKTOP_XDG_SESSION_TYPE" = "wayland" ] && pgrep -f "Xwayland.*:0" >/dev/null 2>&1; then
                DESKTOP_DISPLAY=":0"
                # Find the Xwayland auth file for the user
                if [ -n "$DESKTOP_XDG_RUNTIME_DIR" ]; then
                    XWAYLAND_AUTH=$(find "$DESKTOP_XDG_RUNTIME_DIR" -name ".mutter-Xwaylandauth.*" 2>/dev/null | head -1)
                    [ -n "$XWAYLAND_AUTH" ] && DESKTOP_XAUTHORITY="$XWAYLAND_AUTH"
                fi
                echo "Found Xwayland on :0 for Wayland session" >&2
            fi
            
            # Export the detected environment variables
            [ -n "$DESKTOP_DISPLAY" ] && export DISPLAY="$DESKTOP_DISPLAY"
            [ -n "$DESKTOP_XAUTHORITY" ] && export XAUTHORITY="$DESKTOP_XAUTHORITY"
            [ -n "$DESKTOP_WAYLAND_DISPLAY" ] && export WAYLAND_DISPLAY="$DESKTOP_WAYLAND_DISPLAY"
            [ -n "$DESKTOP_XDG_SESSION_TYPE" ] && export XDG_SESSION_TYPE="$DESKTOP_XDG_SESSION_TYPE"
            [ -n "$DESKTOP_XDG_RUNTIME_DIR" ] && export XDG_RUNTIME_DIR="$DESKTOP_XDG_RUNTIME_DIR"
            
            # Print what we found for debugging
            echo "Display environment detected:" >&2
            echo "  DISPLAY=$DISPLAY" >&2
            echo "  XAUTHORITY=$XAUTHORITY" >&2
            echo "  WAYLAND_DISPLAY=$WAYLAND_DISPLAY" >&2
            echo "  XDG_SESSION_TYPE=$XDG_SESSION_TYPE" >&2
            echo "  XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" >&2
            
            return 0
        else
            echo "Warning: Could not read GNOME process environment" >&2
        fi
    else
        echo "Warning: No desktop session found for user $target_user" >&2
    fi
    
    # Fallback: use default values for headless operation
    echo "Using fallback display environment for headless operation" >&2
    export EXPERIMANCE_DISPLAY_HEADLESS=true
    export DISPLAY="${DISPLAY:-:0}"
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u $target_user)}"
    export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-wayland}"
    
    return 1
}

# If called directly, setup environment and run the command
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Setup display environment
    setup_display_environment "experimance"
    
    # Execute the provided command with the environment
    exec "$@"
fi
