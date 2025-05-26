#!/bin/bash
# SuperCollider wrapper script that ensures proper cleanup

# Default values
SC_SCRIPT=""
LOG_DIR="../logs"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--script) SC_SCRIPT="$2"; shift ;;
        -l|--log-dir) LOG_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Check if script was provided
if [[ -z "$SC_SCRIPT" ]]; then
    echo "Error: No script file provided."
    echo "Usage: $0 -s <script_path> [-l <log_directory>]"
    exit 1
fi

# Check if script exists
if [[ ! -f "$SC_SCRIPT" ]]; then
    echo "Error: Script file '$SC_SCRIPT' not found."
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Generate log filename
LOG_FILE="$LOG_DIR/supercollider_$(date +%Y%m%d_%H%M%S).log"
echo "Starting SuperCollider with script: $SC_SCRIPT"
echo "Logging to: $LOG_FILE"

# Make sure JACK is running
# if ! pgrep -x "jackd" > /dev/null; then
#     echo "JACK audio server is not running. Starting JACK..."
#     # Add your preferred JACK startup command here, for example:
#     # jackd -d alsa -d hw:0 &
#     # Sleep to allow JACK to initialize
#     sleep 2
# fi

# Function to cleanup when script is terminated
cleanup() {
    echo "Received termination signal. Cleaning up..."
    
    # Send OSC quit message to SuperCollider
    echo "Sending quit message to SuperCollider..."
    # You might need to adjust the port number based on your setup
    # This requires oscsend tool from liblo-tools package
    if command -v oscsend &> /dev/null; then
        oscsend localhost 5568 /quit
        sleep 1
    fi
    
    # Force kill SuperCollider process if still running
    if pgrep -f "sclang.*$SC_SCRIPT" > /dev/null; then
        echo "Terminating SuperCollider process..."
        pkill -f "sclang.*$SC_SCRIPT"
    fi
    
    echo "Cleanup complete."
    exit 0
}

# Set up trap for signals
trap cleanup SIGINT SIGTERM SIGHUP

# Run SuperCollider with the script
sclang "$SC_SCRIPT" 2>&1 | tee "$LOG_FILE" &
SC_PID=$!

# Wait for SuperCollider to finish
wait $SC_PID

# Final cleanup
echo "SuperCollider process has exited."
exit 0
