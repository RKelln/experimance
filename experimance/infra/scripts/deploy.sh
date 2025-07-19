#!/bin/bash

# Experimance Deployment Script
# Usage: ./deploy.sh [project_name] [action]
# Actions: install, start, stop, restart, status

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging and error helpers
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

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SYSTEMD_DIR="/etc/systemd/system"

# Determine user based on environment
if [[ "${EXPERIMANCE_ENV:-}" == "development" ]] || [[ ! -d "/var/cache" ]]; then
    # Development mode: use current user
    USER="$(whoami)"
    warn "Running in development mode with user: $USER"
elif id experimance &>/dev/null; then
    # Production mode: use experimance user if it exists
USER="experimance"
else
    # Fallback: use current user but warn
    USER="$(whoami)"
    warn "experimance user not found, using current user: $USER"
    warn "For production, create user with: sudo useradd -m -s /bin/bash experimance"
fi

# Default values
PROJECT="${1:-experimance}"
ACTION="${2:-install}"

# Get services dynamically for the project
get_project_services() {
    local project="$1"
    local services_script="$SCRIPT_DIR/get_project_services.py"
    
    if [[ -f "$services_script" ]]; then
        # Use uv run to execute the Python script in the proper environment
        cd "$REPO_DIR"
        if command -v uv >/dev/null 2>&1; then
            uv run python "$services_script" "$project" 2>/dev/null || {
                # Fallback to default services if Python script fails
                echo "Failed to auto-detect services with uv, using defaults" >&2
                echo "experimance-core@${project}"
                echo "experimance-display@${project}"
                echo "image-server@${project}"
                echo "experimance-agent@${project}"
                echo "experimance-audio@${project}"
                echo "experimance-health@${project}"
            }
        else
            # Fallback if uv is not available
            echo "uv not found, using default services" >&2
            echo "experimance-core@${project}"
            echo "experimance-display@${project}"
            echo "image-server@${project}"
            echo "experimance-agent@${project}"
            echo "experimance-audio@${project}"
            echo "experimance-health@${project}"
        fi
    else
        # Fallback if script doesn't exist
        echo "Service detection script not found, using defaults" >&2
        echo "experimance-core@${project}"
        echo "experimance-display@${project}"
        echo "image-server@${project}"
        echo "experimance-agent@${project}"
        echo "experimance-audio@${project}"
        echo "experimance-health@${project}"
    fi
}

# Services to manage (dynamically determined)
readarray -t SERVICES < <(get_project_services "$PROJECT")

check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for systemd operations"
    fi
}

check_user() {
    if [[ "$USER" == "experimance" ]] && ! id "$USER" &>/dev/null; then
        error "User $USER does not exist. Create it with: sudo useradd -m -s /bin/bash experimance"
    fi
    
    log "Using user: $USER"
}

check_project() {
    if [[ ! -d "$REPO_DIR/projects/$PROJECT" ]]; then
        error "Project $PROJECT does not exist in $REPO_DIR/projects/"
    fi
    
    # Show detected services for this project
    log "Detected services for project $PROJECT:"
    for service in "${SERVICES[@]}"; do
        echo "  - $service"
    done
}

install_systemd_files() {
    log "Installing systemd service files..."
    
    # Copy service files
    for service_file in "$SCRIPT_DIR"/../systemd/*.service; do
        if [[ -f "$service_file" ]]; then
            cp "$service_file" "$SYSTEMD_DIR/"
            log "Copied $(basename "$service_file")"
        fi
    done
    
    # Copy target file
    if [[ -f "$SCRIPT_DIR/../systemd/experimance@.target" ]]; then
        cp "$SCRIPT_DIR/../systemd/experimance@.target" "$SYSTEMD_DIR/"
        log "Copied experimance@.target"
    fi
    
    # Reload systemd
    systemctl daemon-reload
    log "Reloaded systemd configuration"
}

setup_directories() {
    log "Setting up directories..."
    
    # Create cache directory
    if [[ "$USER" == "experimance" ]]; then
        # Production: use /var/cache/experimance
        mkdir -p /var/cache/experimance
        chown "$USER:$USER" /var/cache/experimance
    else
        # Development: use local cache directory
        mkdir -p "$REPO_DIR/cache"
        chown "$USER:$USER" "$REPO_DIR/cache"
        warn "Using local cache directory: $REPO_DIR/cache"
    fi
    
    # Create log directory if it doesn't exist
    mkdir -p "$REPO_DIR/logs"
    chown "$USER:$USER" "$REPO_DIR/logs"
    
    # Create images directory if it doesn't exist
    mkdir -p "$REPO_DIR/media/images/generated"
    mkdir -p "$REPO_DIR/media/images/mocks"
    mkdir -p "$REPO_DIR/media/video"
    chown "$USER:$USER" "$REPO_DIR/media/images"
    chown "$USER:$USER" "$REPO_DIR/media/images/generated"
    chown "$USER:$USER" "$REPO_DIR/media/images/mocks"
    chown "$USER:$USER" "$REPO_DIR/media/video"
    
    # create transcripts directory
    mkdir -p "$REPO_DIR/transcripts"
    chown "$USER:$USER" "$REPO_DIR/transcripts"

    log "Directories created and permissions set"
}

install_dependencies() {
    log "Installing Python dependencies..."
    
    # For production, install uv and dependencies as the experimance user
    if [[ "$USER" == "experimance" ]]; then
        log "Installing uv for experimance user..."
        sudo -u experimance bash -c "
            # Install uv if not already installed
            if ! command -v uv >/dev/null 2>&1; then
                curl -LsSf https://astral.sh/uv/install.sh | sh
                source ~/.bashrc
            fi
            
            # Install project dependencies
            cd '$REPO_DIR'
            uv sync
        "
        log "Dependencies installed for production"
    else
        # Development mode - check if uv is available for the current user
        if command -v uv >/dev/null 2>&1; then
            log "Installing dependencies for development..."
            cd "$REPO_DIR"
            uv sync
            log "Dependencies installed for development"
        else
            warn "uv not found for user $USER, skipping dependency installation"
            warn "For development: Use './scripts/dev <service>' which handles dependencies automatically"
            warn "For production: This script will install uv automatically for the experimance user"
        fi
    fi
}

start_services() {
    log "Starting services for project $PROJECT..."
    
    for service in "${SERVICES[@]}"; do
        if systemctl is-enabled "$service" &>/dev/null; then
            systemctl start "$service"
            log "Started $service"
        else
            systemctl enable "$service"
            systemctl start "$service"
            log "Enabled and started $service"
        fi
    done
    
    # Enable and start the target
    systemctl enable "experimance@${PROJECT}.target"
    systemctl start "experimance@${PROJECT}.target"
    log "Started experimance@${PROJECT}.target"
}

stop_services() {
    log "Stopping services for project $PROJECT..."
    
    # Stop the target first
    if systemctl is-active "experimance@${PROJECT}.target" &>/dev/null; then
        systemctl stop "experimance@${PROJECT}.target"
        log "Stopped experimance@${PROJECT}.target"
    fi
    
    for service in "${SERVICES[@]}"; do
        if systemctl is-active "$service" &>/dev/null; then
            systemctl stop "$service"
            log "Stopped $service"
        fi
    done
}

restart_services() {
    log "Restarting services for project $PROJECT..."
    stop_services
    sleep 2
    start_services
}

status_services() {
    log "Checking status of services for project $PROJECT..."
    
    echo -e "\n${BLUE}=== Service Status ===${NC}"
    for service in "${SERVICES[@]}"; do
        if systemctl is-active "$service" &>/dev/null; then
            echo -e "${GREEN}✓${NC} $service: $(systemctl is-active "$service")"
        else
            echo -e "${RED}✗${NC} $service: $(systemctl is-active "$service")"
        fi
    done
    
    echo -e "\n${BLUE}=== Target Status ===${NC}"
    target="experimance@${PROJECT}.target"
    if systemctl is-active "$target" &>/dev/null; then
        echo -e "${GREEN}✓${NC} $target: $(systemctl is-active "$target")"
    else
        echo -e "${RED}✗${NC} $target: $(systemctl is-active "$target")"
    fi
    
    echo -e "\n${BLUE}=== Recent Logs ===${NC}"
    journalctl --no-pager -n 20 -u "experimance-core@${PROJECT}" || true
}

main() {
    case "$ACTION" in
        install)
            check_root
            check_user
            check_project
            install_systemd_files
            setup_directories
            install_dependencies
            log "Installation complete. Use 'sudo ./deploy.sh $PROJECT start' to start services."
            ;;
        start)
            check_root
            check_project
            start_services
            ;;
        stop)
            check_root
            check_project
            stop_services
            ;;
        restart)
            check_root
            check_project
            restart_services
            ;;
        status)
            check_project
            status_services
            ;;
        services)
            # Don't require root for just listing services
            if [[ ! -d "$REPO_DIR/projects/$PROJECT" ]]; then
                error "Project $PROJECT does not exist in $REPO_DIR/projects/"
            fi
            log "Services for project $PROJECT:"
            for service in "${SERVICES[@]}"; do
                echo "  $service"
            done
            ;;
        *)
            error "Unknown action: $ACTION. Use: install, start, stop, restart, status, services"
            ;;
    esac
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [project_name] [action]"
    echo "Projects: $(ls "$REPO_DIR/projects" 2>/dev/null | tr '\n' ' ')"
    echo "Actions: install, start, stop, restart, status, services"
    echo ""
    echo "Environment:"
    echo "  EXPERIMANCE_ENV=development    # Use current user instead of 'experimance'"
    echo ""
    echo "Examples:"
    echo "  $0 experimance services        # Show services for experimance project (no sudo needed)"
    echo "  sudo $0 experimance install    # Install and setup experimance project (sudo needed)"
    echo "  sudo $0 experimance start      # Start all services for experimance project (sudo needed)"
    echo ""
    echo "Development mode:"
    echo "  EXPERIMANCE_ENV=development sudo $0 experimance install"
    echo "  ./scripts/dev <service>        # Run individual service in development (no sudo)"
    echo ""
    echo "Production setup (first time):"
    echo "  sudo useradd -m -s /bin/bash experimance"
    echo "  sudo $0 experimance install    # Installs uv and dependencies for experimance user"
    echo "  sudo $0 experimance start      # systemd requires sudo for service management"
    echo ""
    echo "Note: sudo is only needed for systemd operations and system directory setup."
    echo "      The services themselves run as the experimance user (non-root)."
    exit 1
fi

main "$@"
