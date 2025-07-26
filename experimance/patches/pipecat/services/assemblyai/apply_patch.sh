#!/bin/bash
#
# Script to reapply the AssemblyAI STT shutdown fix after pipecat updates
#
# Usage: ./apply_patch.sh
#

set -e

echo "🔧 Applying AssemblyAI STT shutdown fix..."

# Get the project root directory (go up 4 levels from patches/pipecat/services/assemblyai/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Find the pipecat installation
PIPECAT_PATH="$VENV_PATH/lib/python3.11/site-packages/pipecat/services/assemblyai"

if [ ! -d "$PIPECAT_PATH" ]; then
    echo "❌ Pipecat AssemblyAI service not found at $PIPECAT_PATH"
    echo "   Make sure pipecat is installed with assemblyai support"
    exit 1
fi

# Backup original files
echo "📋 Creating backup of original files..."
cp "$PIPECAT_PATH/stt.py" "$PIPECAT_PATH/stt.py.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

# Apply the patch
echo "⚡ Applying patched stt.py..."
cp "$(dirname "${BASH_SOURCE[0]}")/stt.py" "$PIPECAT_PATH/stt.py"

# Verify the patch was applied
if grep -q "Forced shutdown: skipping graceful termination handshake" "$PIPECAT_PATH/stt.py"; then
    echo "✅ Patch applied successfully!"
    echo "🎯 AssemblyAI STT service should now shutdown quickly during forced shutdowns"
else
    echo "❌ Patch verification failed - the fix may not have been applied correctly"
    exit 1
fi

echo ""
echo "📊 To test the fix, run:"
echo "   uv run python patches/pipecat/services/assemblyai/test_real_assemblyai_shutdown.py"
echo ""
