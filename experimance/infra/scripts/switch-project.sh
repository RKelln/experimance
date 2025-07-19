#!/bin/bash

# Experimance Project Switch Script
# Switches between different projects (experimance, sohkepayin, etc.)

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
CURRENT_PROJECT_FILE="/etc/experimance/current_project"

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

list_projects() {
    log "Available projects:"
    
    for project_dir in "$REPO_DIR/projects"/*; do
        if [[ -d "$project_dir" ]]; then
            project_name=$(basename "$project_dir")
            if [[ -f "$CURRENT_PROJECT_FILE" ]] && [[ "$(cat "$CURRENT_PROJECT_FILE")" == "$project_name" ]]; then
                echo -e "  ${GREEN}* $project_name${NC} (current)"
            else
                echo -e "  $project_name"
            fi
        fi
    done
}

get_current_project() {
    if [[ -f "$CURRENT_PROJECT_FILE" ]]; then
        cat "$CURRENT_PROJECT_FILE"
    else
        echo "experimance"  # default
    fi
}

switch_project() {
    local new_project="$1"
    local current_project
    
    current_project=$(get_current_project)
    
    if [[ "$new_project" == "$current_project" ]]; then
        log "Already using project: $new_project"
        return 0
    fi
    
    # Check if project exists
    if [[ ! -d "$REPO_DIR/projects/$new_project" ]]; then
        error "Project '$new_project' does not exist"
    fi
    
    log "Switching from '$current_project' to '$new_project'..."
    
    # Stop current services
    log "Stopping services for project: $current_project"
    "$SCRIPT_DIR/deploy.sh" "$current_project" stop || warn "Failed to stop some services"
    
    # Update current project file
    mkdir -p "$(dirname "$CURRENT_PROJECT_FILE")"
    echo "$new_project" > "$CURRENT_PROJECT_FILE"
    
    # Update environment
    log "Updating environment for project: $new_project"
    
    # Create/update environment file
    cat > "/etc/experimance/environment" << EOF
export PROJECT_ENV="$new_project"
EOF
    
    # Start new services
    log "Starting services for project: $new_project"
    "$SCRIPT_DIR/deploy.sh" "$new_project" start
    
    log "Project switched successfully to: $new_project"
}

show_project_info() {
    local project="$1"
    local project_dir="$REPO_DIR/projects/$project"
    
    if [[ ! -d "$project_dir" ]]; then
        error "Project '$project' does not exist"
    fi
    
    echo -e "${BLUE}=== Project: $project ===${NC}"
    echo -e "Path: $project_dir"
    
    # Show service config files
    local services=("core" "display" "audio" "agent" "image_server")
    echo -e "\n${BLUE}Service Configs:${NC}"
    
    for service in "${services[@]}"; do
        if [[ -f "$project_dir/${service}.toml" ]]; then
            echo -e "  ${service}: ${GREEN}✓${NC} ${service}.toml"
        else
            echo -e "  ${service}: ${RED}✗${NC} ${service}.toml"
        fi
    done
    
    # Show other project files
    echo -e "\n${BLUE}Project Files:${NC}"
    if [[ -f "$project_dir/.env" ]]; then
        echo -e "  Environment: ${GREEN}✓${NC} .env"
    else
        echo -e "  Environment: ${RED}✗${NC} .env"
    fi
    
    if [[ -f "$project_dir/schemas.py" ]]; then
        echo -e "  Schemas: ${GREEN}✓${NC} schemas.py"
    else
        echo -e "  Schemas: ${RED}✗${NC} schemas.py"
    fi
    
    if [[ -f "$project_dir/constants.py" ]]; then
        echo -e "  Constants: ${GREEN}✓${NC} constants.py"
    else
        echo -e "  Constants: ${RED}✗${NC} constants.py"
    fi
    
    # Show services status if this is the current project
    current_project=$(get_current_project)
    if [[ "$project" == "$current_project" ]]; then
        echo -e "\n${BLUE}Service Status:${NC}"
        "$SCRIPT_DIR/deploy.sh" "$project" status
    fi
}

create_project() {
    # Note: project name parameter is optional since we use the interactive script
    
    log "The comprehensive project creation script provides better templates,"
    log "service selection, and proper setup than this simple script."
    log ""
    log "Please use the interactive project creation script instead:"
    log "  cd $REPO_DIR"
    log "  uv run python scripts/create_new_project.py"
    log ""
    log "Or run it directly from here:"
    read -p "Run the interactive project creation script now? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$REPO_DIR"
        if command -v uv >/dev/null 2>&1; then
            uv run python scripts/create_new_project.py
        else
            python3 scripts/create_new_project.py
        fi
    else
        log "Project creation cancelled. You can run it manually later."
    fi
}

main() {
    case "${1:-}" in
        "list"|"ls")
            list_projects
            ;;
        "current")
            echo "Current project: $(get_current_project)"
            ;;
        "switch")
            if [[ $# -lt 2 ]]; then
                error "Usage: $0 switch <project_name>"
            fi
            if [[ $EUID -ne 0 ]]; then
                error "This script must be run as root for switching projects"
            fi
            switch_project "$2"
            ;;
        "info")
            if [[ $# -lt 2 ]]; then
                error "Usage: $0 info <project_name>"
            fi
            show_project_info "$2"
            ;;
        "create")
            create_project "${2:-}"
            ;;
        *)
            echo "Usage: $0 {list|current|switch|info|create} [project_name]"
            echo ""
            echo "Commands:"
            echo "  list              List all available projects"
            echo "  current           Show current active project"
            echo "  switch <project>  Switch to a different project (requires root)"
            echo "  info <project>    Show project information"
            echo "  create            Create a new project (uses interactive script)"
            echo ""
            echo "Note: For comprehensive project creation with service selection"
            echo "      and proper templates, use: uv run python scripts/create_new_project.py"
            echo ""
            list_projects
            ;;
    esac
}

main "$@"
