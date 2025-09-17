#!/bin/bash
# Matter Device Scheduler Service Management Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_NAME="matter-scheduler"
USERNAME="${USER}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

install_dependencies() {
    log "Installing Python dependencies..."
    cd "$PROJECT_ROOT"
    
    # Check if croniter is installed
    if ! uv run python -c "import croniter" >/dev/null 2>&1; then
        log "Installing croniter dependency..."
        uv add croniter
    else
        success "Dependencies already installed"
    fi
}

install_service() {
    log "Installing systemd service for user $USERNAME..."
    
    # Create user systemd directory
    USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
    mkdir -p "$USER_SYSTEMD_DIR"
    
    # Copy service file
    SERVICE_FILE="$PROJECT_ROOT/infra/systemd/${SERVICE_NAME}@.service"
    if [[ -f "$SERVICE_FILE" ]]; then
        cp "$SERVICE_FILE" "$USER_SYSTEMD_DIR/"
        success "Service file installed to $USER_SYSTEMD_DIR/"
    else
        error "Service file not found: $SERVICE_FILE"
        exit 1
    fi
    
    # Reload systemd
    systemctl --user daemon-reload
    
    success "Service installed successfully"
}

start_service() {
    log "Starting matter scheduler service..."
    
    # Enable and start the service
    systemctl --user enable "${SERVICE_NAME}@${USERNAME}.service"
    systemctl --user start "${SERVICE_NAME}@${USERNAME}.service"
    
    success "Service started and enabled"
}

stop_service() {
    log "Stopping matter scheduler service..."
    
    systemctl --user stop "${SERVICE_NAME}@${USERNAME}.service" || true
    systemctl --user disable "${SERVICE_NAME}@${USERNAME}.service" || true
    
    success "Service stopped and disabled"
}

status_service() {
    log "Checking service status..."
    systemctl --user status "${SERVICE_NAME}@${USERNAME}.service"
}

logs_service() {
    log "Showing service logs..."
    journalctl --user -u "${SERVICE_NAME}@${USERNAME}.service" -f
}

test_config() {
    local config_file="$PROJECT_ROOT/projects/fire/matter_schedule.toml"
    
    if [[ ! -f "$config_file" ]]; then
        error "Configuration file not found: $config_file"
        exit 1
    fi
    
    log "Testing configuration..."
    cd "$PROJECT_ROOT"
    
    if uv run python scripts/matter_scheduler.py "$config_file" --test; then
        success "Configuration test passed"
    else
        error "Configuration test failed"
        exit 1
    fi
}

create_example_config() {
    local config_file="$PROJECT_ROOT/projects/fire/matter_schedule.toml"
    
    if [[ -f "$config_file" ]]; then
        warn "Configuration file already exists: $config_file"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Keeping existing configuration"
            return
        fi
    fi
    
    log "Creating example configuration..."
    # The config file should already exist from previous step
    if [[ -f "$config_file" ]]; then
        success "Example configuration available at: $config_file"
        log "Edit this file to customize your schedules"
    else
        error "Failed to create configuration file"
        exit 1
    fi
}

usage() {
    echo "Usage: $0 {install|start|stop|restart|status|logs|test-config|create-config}"
    echo ""
    echo "Commands:"
    echo "  install      Install dependencies and systemd service"
    echo "  start        Start the scheduler service"
    echo "  stop         Stop the scheduler service"
    echo "  restart      Restart the scheduler service"
    echo "  status       Show service status"
    echo "  logs         Show service logs (follow mode)"
    echo "  test-config  Test the configuration file"
    echo "  create-config Create example configuration file"
    echo ""
    echo "Example usage:"
    echo "  $0 install      # First time setup"
    echo "  $0 create-config # Create config file"
    echo "  $0 test-config  # Test your config"
    echo "  $0 start        # Start scheduling"
}

case "${1:-}" in
    install)
        install_dependencies
        install_service
        log "Installation complete. Next steps:"
        log "1. Run '$0 create-config' to create a configuration file"
        log "2. Edit projects/fire/matter_schedule.toml to set your schedules"
        log "3. Run '$0 test-config' to validate your configuration"
        log "4. Run '$0 start' to begin scheduling"
        ;;
    start)
        test_config
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        sleep 2
        test_config
        start_service
        ;;
    status)
        status_service
        ;;
    logs)
        logs_service
        ;;
    test-config)
        test_config
        ;;
    create-config)
        create_example_config
        ;;
    *)
        usage
        exit 1
        ;;
esac