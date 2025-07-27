#!/bin/bash

# Apply OpenAI LLM shutdown fix patch to Pipecat

echo "🔧 Applying OpenAI LLM shutdown fix patch..."

# Find the target file
PATCH_SOURCE="patches/pipecat/services/openai/llm.py"
TARGET_FILE=".venv/lib/python3.11/site-packages/pipecat/services/openai/llm.py"

if [ ! -f "$PATCH_SOURCE" ]; then
    echo "❌ Patch source file not found: $PATCH_SOURCE"
    exit 1
fi

if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ Target file not found: $TARGET_FILE"
    echo "   Make sure Pipecat is installed and the path is correct"
    exit 1
fi

# Create backup
BACKUP_FILE="${TARGET_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$TARGET_FILE" "$BACKUP_FILE"
echo "✅ Created backup: $BACKUP_FILE"

# Apply the patch by copying our enhanced version
cp "$PATCH_SOURCE" "$TARGET_FILE"

echo "✅ OpenAI LLM shutdown fix patch applied successfully!"
echo ""
echo "🔍 Patch adds:"
echo "   - Enhanced _disconnect() method with HTTP client cleanup"
echo "   - Force and graceful shutdown modes"
echo "   - Timeout protection for HTTP connection cleanup"
echo "   - Proper state tracking during shutdown"
echo ""
echo "📋 To verify the patch was applied:"
echo "   grep -n \"🔧 PATCH\" \"$TARGET_FILE\""
