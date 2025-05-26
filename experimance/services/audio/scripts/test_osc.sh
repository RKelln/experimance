#!/bin/bash
# Test script for OSC Bridge communications

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Default values
MODE="help"
PORT=5567
MESSAGE="/spacetime"
ARGS=("temperate_forest" "wilderness")
SC_SCRIPT=""
WAIT_TIME=3

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
show_help() {
  echo -e "${BLUE}Experimance Audio OSC Test Script${NC}"
  echo -e "${YELLOW}Usage:${NC}"
  echo "  $0 [mode] [options]"
  echo ""
  echo -e "${YELLOW}Modes:${NC}"
  echo "  manual     - Send a single OSC message and capture responses"
  echo "  integrated - Start SuperCollider and run a test sequence"
  echo "  unittest   - Run automated unit tests"
  echo ""
  echo -e "${YELLOW}Options for manual mode:${NC}"
  echo "  --port PORT    - OSC port to use (default: $PORT)"
  echo "  --message MSG  - OSC message to send (default: $MESSAGE)"
  echo "  --args A B C   - Arguments for the message"
  echo "  --wait SECS    - Time to wait for responses (default: $WAIT_TIME)"
  echo "  --no-oscdump   - Don't use oscdump to verify message reception"
  echo "  --debug        - Show debug information"
  echo ""
  echo -e "${YELLOW}Options for integrated mode:${NC}"
  echo "  --port PORT    - OSC port to use (default: $PORT)"
  echo "  --script PATH  - Path to SuperCollider script (default: auto-detect)"
  echo "  --sclang PATH  - Path to sclang executable (default: sclang)"
  echo ""
  echo -e "${YELLOW}Examples:${NC}"
  echo "  $0 manual --message /listening --args true"
  echo "  $0 manual --message /spacetime --args desert ancient --port 5567"
  echo "  $0 integrated --script path/to/test_osc.scd"
  echo "  $0 unittest"
}

# Parse arguments
if [ $# -ge 1 ]; then
  MODE="$1"
  shift
fi

case "$MODE" in
  manual)
    CMD="python -m tests.test_osc_bridge --manual"
    while [ $# -gt 0 ]; do
      case "$1" in
        --port)
          PORT="$2"
          CMD="$CMD --port=$PORT"
          shift 2
          ;;
        --message)
          MESSAGE="$2"
          CMD="$CMD --message=$MESSAGE"
          shift 2
          ;;
        --args)
          ARGS=()
          shift
          while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
            ARGS+=("$1")
            shift
          done
          CMD="$CMD --args ${ARGS[*]}"
          ;;
        --wait)
          WAIT_TIME="$2"
          CMD="$CMD --wait=$WAIT_TIME"
          shift 2
          ;;
        --no-oscdump)
          CMD="$CMD --no-oscdump"
          shift
          ;;
        --debug)
          CMD="$CMD --debug"
          shift
          ;;
        *)
          echo -e "${RED}Unknown option: $1${NC}"
          echo "Run '$0 help' for usage information"
          exit 1
          ;;
      esac
    done
    ;;

  integrated)
    CMD="python -m tests.test_osc_bridge --integrated"
    while [ $# -gt 0 ]; do
      case "$1" in
        --port)
          PORT="$2"
          CMD="$CMD --port=$PORT"
          shift 2
          ;;
        --script)
          SC_SCRIPT="$2"
          CMD="$CMD --script=$SC_SCRIPT"
          shift 2
          ;;
        --sclang)
          SCLANG="$2"
          CMD="$CMD --sclang=$SCLANG"
          shift 2
          ;;
        --debug)
          CMD="$CMD --debug"
          shift
          ;;
        *)
          echo -e "${RED}Unknown option: $1${NC}"
          echo "Run '$0 help' for usage information"
          exit 1
          ;;
      esac
    done
    ;;

  unittest)
    CMD="python -m tests.test_osc_bridge"
    ;;

  help)
    show_help
    exit 0
    ;;

  *)
    echo -e "${RED}Unknown mode: $MODE${NC}"
    show_help
    exit 1
    ;;
esac

# Check if oscdump is available for manual testing
if [ "$MODE" = "manual" ] && [[ "$CMD" != *"--no-oscdump"* ]]; then
  if ! command -v oscdump &> /dev/null; then
    echo -e "${YELLOW}WARNING: oscdump command not found${NC}"
    echo -e "Install with: ${GREEN}sudo apt install liblo-tools${NC}"
    echo -e "Continuing without OSC reception verification...\n"
    CMD="$CMD --no-oscdump"
  fi
fi

# Print command and execute
echo -e "${BLUE}Executing:${NC} $CMD"
eval "$CMD"
