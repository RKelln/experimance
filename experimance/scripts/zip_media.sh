#!/bin/bash

# Script to package the media directory into a zip file for uploading.
# Usage: ./zip_media.sh [output_zip_name]

set -euo pipefail

# Set default name if not provided
ZIP_NAME="${1:-experimance_installation_media_bundle.zip}"

# Determine directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$( dirname "$SCRIPT_DIR" )"
MEDIA_DIR="$REPO_DIR/media"

ZIP_PATH="$REPO_DIR/$ZIP_NAME"

# Check for zip utility
if ! command -v zip &>/dev/null; then
    echo "ERROR: 'zip' utility not found. Please install it (e.g., sudo apt install zip)"
    exit 1
fi

# Create zip archive, excluding generated images and hidden/_ files
echo "Packaging media directory: $MEDIA_DIR -> $ZIP_PATH"
cd "$REPO_DIR"
zip -r "$ZIP_PATH" media \
    -x "media/images/generated/*" \
    "media/**/.*" \
    "media/**/_*"
echo "Media directory zipped to $ZIP_PATH"
