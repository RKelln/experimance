#!/bin/bash
#
# Script to reapply the Cartesia TTS shutdown fix after pipecat updates
#
# Usage: ./apply_patch.sh
#

set -e

echo "🔧 Applying Cartesia TTS shutdown fix..."

# Get the project root directory (go up 4 levels from patches/pipecat/services/cartesia/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Find the pipecat installation
PIPECAT_PATH="$VENV_PATH/lib/python3.11/site-packages/pipecat/services/cartesia"

if [ ! -d "$PIPECAT_PATH" ]; then
    echo "❌ Pipecat Cartesia service not found at $PIPECAT_PATH"
    echo "   Make sure pipecat is installed with cartesia support"
    exit 1
fi

# Backup original files
echo "📋 Creating backup of original files..."
cp "$PIPECAT_PATH/tts.py" "$PIPECAT_PATH/tts.py.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

# Apply the patch
echo "⚡ Applying patched tts.py..."
cp "$(dirname "${BASH_SOURCE[0]}")/tts.py" "$PIPECAT_PATH/tts.py"

# Verify the patch was applied
if grep -q "Forced shutdown: skipping graceful context cleanup" "$PIPECAT_PATH/tts.py"; then
    echo "✅ Patch applied successfully!"
    echo "🎯 Cartesia TTS service should now shutdown quickly during forced shutdowns"
else
    echo "❌ Patch verification failed - the fix may not have been applied correctly"
    exit 1
fi

echo ""
echo "📊 To test the fix, run:"
echo "   uv run python patches/pipecat/services/cartesia/test_real_cartesia_shutdown.py"
