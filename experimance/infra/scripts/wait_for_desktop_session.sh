#!/bin/bash
# Wait for a user to have an active desktop session before proceeding
# Usage: wait_for_desktop_session.sh <username>

USERNAME="${1:-experimance}"
MAX_WAIT_TIME=300  # 5 minutes maximum wait
CHECK_INTERVAL=10  # Check every 10 seconds

echo "Waiting for user '$USERNAME' to have an active desktop session..."

wait_start=$(date +%s)

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - wait_start))
    
    if [ $elapsed -ge $MAX_WAIT_TIME ]; then
        echo "Timeout: No desktop session found for user '$USERNAME' after ${MAX_WAIT_TIME}s"
        exit 1
    fi
    
    # Check if user has an active session
    if loginctl list-sessions --no-legend | grep "$USERNAME.*seat0.*active" >/dev/null 2>&1; then
        echo "Found active session for user '$USERNAME'"
        
        # Wait a bit more for desktop environment to fully load
        sleep 5
        
        # Check if gnome-shell is running (indicates desktop is ready)
        if pgrep -u "$USERNAME" gnome-shell >/dev/null 2>&1; then
            echo "Desktop environment ready for user '$USERNAME'"
            exit 0
        fi
        
        echo "User session active but desktop environment not ready yet, continuing to wait..."
    fi
    
    echo "Waiting for desktop session... (${elapsed}s elapsed)"
    sleep $CHECK_INTERVAL
done
