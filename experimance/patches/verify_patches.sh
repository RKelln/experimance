#!/bin/bash
#
# Script to verify that patches are applied correctly
#
# Usage: ./verify_patches.sh
#

echo "🔍 Verifying applied patches..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

# Check AssemblyAI STT patch
ASSEMBLYAI_STT_PATH="$VENV_PATH/lib/python3.11/site-packages/pipecat/services/assemblyai/stt.py"

if [ -f "$ASSEMBLYAI_STT_PATH" ]; then
    if grep -q "Forced shutdown: skipping graceful termination handshake" "$ASSEMBLYAI_STT_PATH"; then
        echo "✅ AssemblyAI STT shutdown fix is applied"
    else
        echo "❌ AssemblyAI STT shutdown fix is NOT applied"
        echo "   Run: cd patches/pipecat/services/assemblyai && ./apply_patch.sh"
    fi
else
    echo "⚠️  AssemblyAI STT service not found (pipecat not installed?)"
fi

echo ""
echo "🎯 All patches verified!"
