#!/bin/bash

# backup_images.sh: Script to use rsync to backup generated images from remote host to local machine
# Usage: backup_images.sh <backup_folder>

# Check if backup folder parameter is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <backup_folder>"
    exit 1
fi

# Define remote source directory containing generated images
REMOTE_SOURCE="gallery:/home/experimance/experimance/media/images/generated/"

# Backup folder on local machine passed as parameter
BACKUP_FOLDER="$1"

# create backup folder if it doesnt exist
mkdir -p "${BACKUP_FOLDER}"

# Display the backup operation
echo "Backing up images from ${REMOTE_SOURCE} to ${BACKUP_FOLDER} ..."

# Use rsync to preserve modification times only (-t). --crtimes is unreliable in this environment.
echo "Using rsync while preserving modification times (-t)."
rsync -avz -t "${REMOTE_SOURCE}" "${BACKUP_FOLDER}"
RSYNC_EXIT=$?
if [ $RSYNC_EXIT -eq 0 ]; then
    echo "Backup successful (mtimes preserved)."
    exit 0
else
    echo "rsync failed (code $RSYNC_EXIT)."
    exit $RSYNC_EXIT
fi
