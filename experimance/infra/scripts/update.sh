#!/bin/bash

# Experimance Update Script
# Safely updates the installation with rollback capability

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PROJECT="${1:-experimance}"
BACKUP_DIR="/var/backups/experimance"
USER="experimance"

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

check_git_status() {
    log "Checking git status..."
    
    cd "$REPO_DIR"
    
    if ! git status --porcelain | grep -q .; then
        log "Working directory is clean"
    else
        warn "Working directory has uncommitted changes"
        git status --porcelain
        
        read -p "Continue with update? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Update cancelled"
        fi
    fi
}

create_backup() {
    log "Creating backup..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Create timestamped backup
    timestamp=$(date '+%Y%m%d_%H%M%S')
    backup_path="$BACKUP_DIR/backup_$timestamp"
    
    # Get current git commit
    cd "$REPO_DIR"
    current_commit=$(git rev-parse HEAD)
    
    # Create backup info file
    cat > "$backup_path.info" << EOF
{
    "timestamp": "$timestamp",
    "commit": "$current_commit",
    "project": "$PROJECT",
    "backup_path": "$backup_path"
}
EOF
    
    # Backup configuration and important files
    tar -czf "$backup_path.tar.gz" -C "$REPO_DIR" \
        projects/ \
        logs/ \
        data/ \
        --exclude='logs/*.log' \
        --exclude='**/__pycache__'
    
    log "Backup created: $backup_path.tar.gz"
    echo "$backup_path.tar.gz"
}

update_code() {
    log "Updating code..."
    
    cd "$REPO_DIR"
    
    # Fetch latest changes
    git fetch origin
    
    # Show what will be updated
    if git diff --quiet HEAD origin/main; then
        log "No updates available"
        return 0
    fi
    
    log "Updates available:"
    git log --oneline HEAD..origin/main | head -10
    
    # Update to latest
    git merge origin/main
    
    log "Code updated successfully"
}

update_dependencies() {
    log "Updating dependencies..."
    
    cd "$REPO_DIR"
    
    # Update dependencies as experimance user
    sudo -u "$USER" bash -c "
        source .venv/bin/activate
        uv pip install -e . --upgrade
    "
    
    log "Dependencies updated"
}

restart_services() {
    log "Restarting services..."
    
    # Use the deploy script to restart services
    "$SCRIPT_DIR/deploy.sh" "$PROJECT" restart
    
    log "Services restarted"
}

test_services() {
    log "Testing services..."
    
    # Wait a bit for services to start
    sleep 10
    
    # Check if all services are running
    if "$SCRIPT_DIR/deploy.sh" "$PROJECT" status | grep -q "✗"; then
        error "Some services failed to start after update"
    fi
    
    # Run health check
    if ! python3 "$SCRIPT_DIR/healthcheck.py"; then
        error "Health check failed after update"
    fi
    
    log "Services are healthy"
}

rollback() {
    local backup_file="$1"
    
    warn "Rolling back to previous version..."
    
    # Extract backup info
    backup_info="${backup_file%.tar.gz}.info"
    if [[ -f "$backup_info" ]]; then
        commit=$(python3 -c "
import json
with open('$backup_info') as f:
    data = json.load(f)
    print(data['commit'])
        ")
        
        log "Rolling back to commit: $commit"
        cd "$REPO_DIR"
        git reset --hard "$commit"
        
        # Restore backed up files
        tar -xzf "$backup_file" -C "$REPO_DIR"
        
        # Restart services
        restart_services
        
        log "Rollback completed"
    else
        error "Backup info not found: $backup_info"
    fi
}

main() {
    log "Starting Experimance update process..."
    
    # Pre-checks
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
    
    if [[ ! -d "$REPO_DIR" ]]; then
        error "Repository directory not found: $REPO_DIR"
    fi
    
    # Check git status
    check_git_status
    
    # Create backup
    backup_file=$(create_backup)
    
    # Update process
    if update_code; then
        if update_dependencies; then
            if restart_services; then
                if test_services; then
                    log "Update completed successfully!"
                    
                    # Clean up old backups (keep last 5)
                    find "$BACKUP_DIR" -name "backup_*.tar.gz" -type f | sort -r | tail -n +6 | xargs -r rm -f
                    find "$BACKUP_DIR" -name "backup_*.info" -type f | sort -r | tail -n +6 | xargs -r rm -f
                    
                    exit 0
                else
                    error "Service tests failed"
                fi
            else
                error "Service restart failed"
            fi
        else
            error "Dependency update failed"
        fi
    else
        error "Code update failed"
    fi
    
    # If we get here, something went wrong
    warn "Update failed, attempting rollback..."
    rollback "$backup_file"
    error "Update failed and rollback attempted"
}

# Handle interruption
trap 'error "Update interrupted"' INT TERM

main "$@"
