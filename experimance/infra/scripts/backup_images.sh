#!/bin/bash

# backup_images.sh: Script to use rsync to backup generated images from remote host to local machine
# Usage: backup_images.sh <remote_host> <backup_folder>

# Check if both parameters are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <remote_host> <backup_folder>"
    echo "  remote_host: can be just hostname or hostname:path"
    echo "  backup_folder: local directory to backup to"
    exit 1
fi

# Parse remote host parameter
REMOTE_HOST="$1"
# Check if remote host already includes a path
if [[ "$REMOTE_HOST" == *":"* ]]; then
    # Remote host includes path, use as-is
    REMOTE_SOURCE="$REMOTE_HOST"
else
    # Remote host is just hostname, append default path
    REMOTE_SOURCE="$REMOTE_HOST:/home/experimance/experimance/media/images/generated/"
fi

# Backup folder on local machine passed as second parameter
BACKUP_FOLDER="$2"

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
