#!/bin/bash
# Image watcher script for Experimance - monitors remote gallery for new images
# Supports both feh and eog viewers with auto-update capabilities

REMOTE_HOST="gallery"
REMOTE_IMAGE_DIR="/home/experimance/experimance/media/images/generated"
REMOTE_LATEST="$REMOTE_IMAGE_DIR/latest.jpg"
LOCAL_TEMP="/tmp/latest_experimance.jpg"
SCP_OPTS="-q -o ConnectTimeout=3 -o ConnectionAttempts=1"

# Default viewer (can be overridden with command line argument)
VIEWER="${1:-auto}"

# Function to detect best available viewer
detect_viewer() {
    if [ "$VIEWER" = "auto" ]; then
        if command -v eog >/dev/null 2>&1; then
            echo "eog"
        elif command -v feh >/dev/null 2>&1; then
            echo "feh"
        else
            echo "none"
        fi
    else
        echo "$VIEWER"
    fi
}

# Function to start the appropriate viewer
start_viewer() {
    local viewer="$1"
    local image_file="$2"
    
    case "$viewer" in
        "eog")
            echo "Starting eog viewer (auto-detects file changes)..."
            eog --single-window "$image_file" &
            VIEWER_PID=$!
            echo "eog started with PID $VIEWER_PID"
            ;;
        "feh")
            echo "Starting feh viewer with auto-reload..."
            feh --reload 2 --scale-down --auto-zoom \
                --title "Experimance Latest Image" \
                --geometry 1024x1024 \
                --borderless \
                "$image_file" &
            VIEWER_PID=$!
            echo "feh started with PID $VIEWER_PID"
            ;;
        *)
            echo "Error: No suitable image viewer found (eog or feh required)"
            echo "Install with: sudo apt install eog  # or feh"
            exit 1
            ;;
    esac
}

# Show usage if help requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [viewer]"
    echo ""
    echo "Viewers:"
    echo "  auto  - Automatically detect best viewer (default)"
    echo "  eog   - Use Eye of GNOME (automatically detects file changes)"
    echo "  feh   - Use feh with 2-second auto-reload"
    echo ""
    echo "Examples:"
    echo "  $0        # Use auto-detection"
    echo "  $0 eog    # Force eog"
    echo "  $0 feh    # Force feh"
    exit 0
fi

SELECTED_VIEWER=$(detect_viewer)

echo "Starting Experimance image watcher with $SELECTED_VIEWER viewer..."

# Initial download of latest image
echo "Downloading current latest image..."
echo "Remote target: $REMOTE_HOST:$REMOTE_LATEST"
scp $SCP_OPTS "$REMOTE_HOST:$REMOTE_LATEST" "$LOCAL_TEMP" 2>/dev/null || {
    echo "Failed to download initial image, creating placeholder..."
    # Quick diagnosis to help troubleshoot SSH and path issues
    if ssh -o BatchMode=yes -o ConnectTimeout=3 "$REMOTE_HOST" "test -f '$REMOTE_LATEST'" 2>/dev/null; then
        echo "Note: Remote file exists but scp failed. Check SSH auth or permissions."
    else
        echo "Note: Remote file not found at: $REMOTE_LATEST"
        echo "Listing remote directory contents (first few entries):"
        ssh -o ConnectTimeout=3 "$REMOTE_HOST" "ls -l \"$REMOTE_IMAGE_DIR\" 2>/dev/null | head -n 10" || true
    fi
    # Create a simple placeholder if no image exists yet
    convert -size 1024x1024 xc:black -fill white -gravity center \
            -pointsize 48 -annotate +0+0 "Waiting for\nfirst image..." "$LOCAL_TEMP" 2>/dev/null || {
        # If imagemagick not available, create empty file
        touch "$LOCAL_TEMP"
    }
}

# Start the selected viewer
start_viewer "$SELECTED_VIEWER" "$LOCAL_TEMP"

# Watch for changes and update the local file
echo "Watching for new images on $REMOTE_HOST..."
echo "Press Ctrl+C to stop watching"

# Function to cleanup on exit
cleanup() {
    echo "Stopping image watcher..."
    if [ -n "$VIEWER_PID" ] && kill -0 $VIEWER_PID 2>/dev/null; then
        kill $VIEWER_PID 2>/dev/null
    fi
    exit 0
}

# Setup cleanup on script exit
trap cleanup INT TERM

# Monitor for changes and update the local file (using polling if inotifywait not available)
if ssh "$REMOTE_HOST" "command -v inotifywait >/dev/null 2>&1"; then
    echo "Using inotifywait for real-time monitoring..."
    # Real-time monitoring with inotifywait
    ssh "$REMOTE_HOST" "inotifywait -m '$REMOTE_IMAGE_DIR' -e close_write -e moved_to --format '%T %f' --timefmt '%H:%M:%S'" | \
    while read time file; do 
        # Check if viewer is still running
        if ! kill -0 $VIEWER_PID 2>/dev/null; then
            echo "Image viewer closed, exiting..."
            break
        fi
        
        if [[ "$file" == "latest.jpg" ]]; then
            echo "$time: New image generated! Downloading..."
            # Download new image to same local path
            # eog will auto-detect the change, feh will auto-reload
            scp $SCP_OPTS "$REMOTE_HOST:$REMOTE_LATEST" "$LOCAL_TEMP" 2>/dev/null && {
                echo "$time: Image updated (viewer will refresh automatically)"
            } || {
                echo "$time: Failed to download new image"
            }
        fi
    done
else
    echo "inotifywait not available, using polling every 3 seconds..."
    LAST_TIMESTAMP=0
    
    while kill -0 $VIEWER_PID 2>/dev/null; do
        # Check timestamp of remote latest.jpg
        CURRENT_TIMESTAMP=$(ssh "$REMOTE_HOST" "stat -c %Y '$REMOTE_LATEST' 2>/dev/null || echo 0")
        
        if [ "$CURRENT_TIMESTAMP" -gt "$LAST_TIMESTAMP" ] && [ "$CURRENT_TIMESTAMP" -gt 0 ]; then
            CURRENT_TIME=$(date '+%H:%M:%S')
            echo "$CURRENT_TIME: New image detected (timestamp changed)! Downloading..."
            
            # Download new image
            scp $SCP_OPTS "$REMOTE_HOST:$REMOTE_LATEST" "$LOCAL_TEMP" 2>/dev/null && {
                echo "$CURRENT_TIME: Image updated (viewer will refresh automatically)"
                LAST_TIMESTAMP=$CURRENT_TIMESTAMP
            } || {
                echo "$CURRENT_TIME: Failed to download new image"
            }
        fi
        
        sleep 3  # Poll every 3 seconds
    done
fi

cleanup