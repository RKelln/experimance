#!/bin/bash

# Experimance Deployment Script
# Usage: ./deploy.sh [project_name] [action] [mode] [--hostname=<hostname>]
# Actions: install, start, stop, restart, status, services, diagnose
# USB Reset on Input: reset-on-input-start, reset-on-input-stop, reset-on-input-status, reset-on-input-logs, reset-on-input-test
# Schedule Actions: schedule-gallery, schedule-custom, schedule-reset, schedule-shutdown, schedule-remove, schedule-show
# Modes: dev, prod (only for install action)
# Options: --hostname=<hostname> - Override hostname for multi-machine deployment
#
# SYSTEMD TEMPLATE SYSTEM:
# This script uses systemd template services for multi-project support:
# - Template files: core@.service, display@.service, experimance@.target (installed to /etc/systemd/system/)
# - Instance services: core@experimance.service, display@fire.service (created when started)
# - The @ symbol makes it a template, %i gets replaced with project name
# - Multiple projects can share the same templates with different instances
#
# MULTI-MACHINE DEPLOYMENT:
# For distributed deployments, create projects/<project>/deployment.toml to specify which
# services run on which machines. Use --hostname to override auto-detection.

set -euo pipefail

# Trap function to show big error on any script failure
trap_error() {
    local exit_code=$?
    local line_number=$1
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                  CRITICAL ERROR                               ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] SCRIPT FAILED: Unexpected error on line $line_number${NC}"
    echo -e "${RED}Exit code: $exit_code${NC}"
    echo -e "${RED}This could be due to a pipeline failure, command error, or unset variable.${NC}"
    echo -e "${RED}Script execution FAILED and will exit.${NC}"
    echo ""
    exit $exit_code
}

# Set trap to catch any error
trap 'trap_error ${LINENO}' ERR

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
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                  CRITICAL ERROR                                ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    echo -e "${RED}Script execution FAILED and will exit.${NC}"
    echo ""
    exit 1
}

# Platform detection
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM=linux ;;
        Darwin*)    PLATFORM=macos ;;
        CYGWIN*|MINGW*) PLATFORM=windows ;;
        *)          PLATFORM=unknown ;;
    esac
    
    log "Detected platform: $PLATFORM"
}

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SYSTEMD_DIR="/etc/systemd/system"

# Detect platform early
detect_platform

# Helper function to get the correct group for chown operations
get_chown_group() {
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "staff"
    else
        echo "$RUNTIME_USER"
    fi
}

# Helper function to get deployment user from config
get_deployment_user() {
    local project="$1"
    local deployment_utils_script="$SCRIPT_DIR/deployment_utils.py"
    
    if [[ ! -f "$deployment_utils_script" ]]; then
        return 1
    fi
    
    # Try to get user from deployment config
    local deployment_user=""
    
    # Change to repo directory for uv to work properly
    cd "$REPO_DIR" || return 1
    
    # Try uv first (preferred), then fallback to python3
    if command -v uv >/dev/null 2>&1; then
        if [[ -n "$HOSTNAME_OVERRIDE" ]]; then
            deployment_user=$(uv run python "$deployment_utils_script" "$project" user "$HOSTNAME_OVERRIDE" 2>/dev/null || echo "")
        else
            deployment_user=$(uv run python "$deployment_utils_script" "$project" user 2>/dev/null || echo "")
        fi
    elif command -v python3 >/dev/null 2>&1; then
        if [[ -n "$HOSTNAME_OVERRIDE" ]]; then
            deployment_user=$(python3 "$deployment_utils_script" "$project" user "$HOSTNAME_OVERRIDE" 2>/dev/null || echo "")
        else
            deployment_user=$(python3 "$deployment_utils_script" "$project" user 2>/dev/null || echo "")
        fi
    fi
    
    if [[ -n "$deployment_user" ]]; then
        echo "$deployment_user"
        return 0
    else
        return 1
    fi
}

# Helper function to get service module name from deployment config
get_service_module_name() {
    local project="$1"
    local service="$2"
    local deployment_utils_script="$SCRIPT_DIR/deployment_utils.py"
    
    # Extract service name without @project suffix
    local service_name="${service%@*}"
    
    if [[ ! -f "$deployment_utils_script" ]]; then
        # Fall back to default naming
        if [[ "$service_name" == "core" || "$service_name" == "agent" ]]; then
            echo "${project}_${service_name}"
        else
            echo "experimance_${service_name}"
        fi
        return 0
    fi
    
    # Try to get module name from deployment config
    local module_name=""
    
    # Change to repo directory for uv to work properly
    cd "$REPO_DIR" || return 1
    
    # Try uv first (preferred), then fallback to python3
    if command -v uv >/dev/null 2>&1; then
        if [[ -n "$HOSTNAME_OVERRIDE" ]]; then
            module_name=$(uv run python "$deployment_utils_script" "$project" services-with-modules "$HOSTNAME_OVERRIDE" 2>/dev/null | grep "^${service_name}:" | cut -d: -f2 || echo "")
        else
            module_name=$(uv run python "$deployment_utils_script" "$project" services-with-modules 2>/dev/null | grep "^${service_name}:" | cut -d: -f2 || echo "")
        fi
    elif command -v python3 >/dev/null 2>&1; then
        if [[ -n "$HOSTNAME_OVERRIDE" ]]; then
            module_name=$(python3 "$deployment_utils_script" "$project" services-with-modules "$HOSTNAME_OVERRIDE" 2>/dev/null | grep "^${service_name}:" | cut -d: -f2 || echo "")
        else
            module_name=$(python3 "$deployment_utils_script" "$project" services-with-modules 2>/dev/null | grep "^${service_name}:" | cut -d: -f2 || echo "")
        fi
    fi
    
    if [[ -n "$module_name" ]]; then
        echo "$module_name"
        return 0
    else
        # Fall back to default naming
        if [[ "$service_name" == "core" || "$service_name" == "agent" ]]; then
            echo "${project}_${service_name}"
        else
            echo "experimance_${service_name}"
        fi
        return 0
    fi
}

# Default values
PROJECT="${1:-experimance}"
ACTION="${2:-install}"
MODE="${3:-}"

# Parse hostname override from remaining arguments
HOSTNAME_OVERRIDE=""
for arg in "${@:4}"; do
    case $arg in
        --hostname=*)
            HOSTNAME_OVERRIDE="${arg#*=}"
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Auto-detect mode if not specified for install action
if [[ "$ACTION" == "install" && -z "$MODE" ]]; then
    # Try deployment config first
    deployment_script="$SCRIPT_DIR/deployment_utils.py"
    if [[ -f "$deployment_script" ]]; then
        # Try to get mode from deployment config
        if command -v python3 >/dev/null 2>&1; then
            config_mode=$(python3 "$deployment_script" "$PROJECT" mode "$HOSTNAME_OVERRIDE" 2>/dev/null || echo "")
            if [[ -n "$config_mode" ]]; then
                MODE="$config_mode"
                log "Auto-detected mode from deployment config: $MODE"
            fi
        fi
    fi
    
    # Fall back to original logic if no mode from config
    if [[ -z "$MODE" ]]; then
        if [[ "${EXPERIMANCE_ENV:-}" == "development" ]]; then
            MODE="dev"
        elif id experimance &>/dev/null; then
            MODE="prod" 
            warn "Auto-detected production mode. Use 'dev' mode explicitly for development: $0 $PROJECT install dev"
        else
            MODE="dev"
            warn "No experimance user found, defaulting to development mode"
        fi
    fi
fi

# Determine user and environment based on mode
if [[ "$MODE" == "dev" ]]; then
    # Development mode: use current user, local directories
    RUNTIME_USER="$(whoami)"
    USE_SYSTEMD=false
    warn "Running in DEVELOPMENT mode with user: $RUNTIME_USER"
elif [[ "$MODE" == "prod" ]]; then
    # Production mode: try deployment config first, fall back to experimance
    DEPLOYMENT_USER=$(get_deployment_user "$PROJECT" 2>/dev/null || echo "")
    if [[ -n "$DEPLOYMENT_USER" ]]; then
        RUNTIME_USER="$DEPLOYMENT_USER"
        log "Using deployment config user: $RUNTIME_USER"
    else
        RUNTIME_USER="experimance"
        log "No deployment config user found, using default: $RUNTIME_USER"
    fi
    USE_SYSTEMD=true
    log "Running in PRODUCTION mode with user: $RUNTIME_USER"
else
    # For non-install actions, auto-detect based on environment
    if [[ "${EXPERIMANCE_ENV:-}" == "development" ]]; then
        MODE="dev"
        RUNTIME_USER="$(whoami)"
        USE_SYSTEMD=false
        warn "Running in development mode with user: $RUNTIME_USER"
    else
        # Try deployment config for production mode detection
        DEPLOYMENT_USER=$(get_deployment_user "$PROJECT" 2>/dev/null || echo "")
        if [[ -n "$DEPLOYMENT_USER" ]] && id "$DEPLOYMENT_USER" &>/dev/null; then
            MODE="prod"
            RUNTIME_USER="$DEPLOYMENT_USER"
            USE_SYSTEMD=true
            log "Auto-detected production mode with deployment config user: $RUNTIME_USER"
        elif id experimance &>/dev/null; then
            MODE="prod"
            RUNTIME_USER="experimance"
            USE_SYSTEMD=true
            log "Auto-detected production mode with experimance user"
        else
            error "Cannot determine runtime mode. For install, specify 'dev' or 'prod' mode. For other actions, ensure user exists or set EXPERIMANCE_ENV=development"
        fi
    fi
fi

# Get services dynamically for the project
get_project_services() {
    local project="$1"
    local deployment_script="$SCRIPT_DIR/get_deployment_services.py"
    local fallback_script="$SCRIPT_DIR/get_project_services.py"
    
    # Try deployment-aware script first (supports multi-machine)
    local services_script="$deployment_script"
    if [[ ! -f "$services_script" ]]; then
        log "Deployment services script not found, using fallback: $fallback_script"
        services_script="$fallback_script"
    fi
    
    if [[ ! -f "$services_script" ]]; then
        error "Service detection script not found: $services_script"
    fi
    
    # In production mode, we need to run as the experimance user to access their environment
    if [[ "$MODE" == "prod" && "$RUNTIME_USER" == "experimance" && "$EUID" -eq 0 ]]; then
        # Running as root in production mode, delegate to experimance user
        sudo -u experimance bash -c "
            # Initialize pyenv for experimance user
            if [[ -d \"/home/experimance/.pyenv\" ]]; then
                export PYENV_ROOT=\"/home/experimance/.pyenv\"
                export PATH=\"\$PYENV_ROOT/bin:\$PATH\"
                eval \"\$(pyenv init --path)\" 2>/dev/null || true
                eval \"\$(pyenv init -)\" 2>/dev/null || true
            fi
            
            # Add uv to PATH
            export PATH=\"/home/experimance/.local/bin:\$PATH\"
            
            # Try to find uv
            uv_cmd=\"\"
            if command -v uv >/dev/null 2>&1; then
                uv_cmd=\"uv\"
            elif [[ -x \"/home/experimance/.local/bin/uv\" ]]; then
                uv_cmd=\"/home/experimance/.local/bin/uv\"
            else
                echo 'ERROR: uv is not installed or not found. Install dependencies first.' >&2
                exit 1
            fi
            
            # Use uv run to execute the Python script in the proper environment
            cd '$REPO_DIR'
            if ! \"\$uv_cmd\" run python '$services_script' '$project' '$HOSTNAME_OVERRIDE'; then
                echo 'ERROR: Failed to detect services for project $project' >&2
                exit 1
            fi
        "
    else
        # Development mode or already running as the correct user
        # Initialize pyenv if available
        if command -v pyenv >/dev/null 2>&1; then
            export PYENV_ROOT="$HOME/.pyenv"
            export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init --path)" 2>/dev/null || true
            eval "$(pyenv init -)" 2>/dev/null || true
        fi
        
        # Try to find uv (check PATH first, then common locations)
        local uv_cmd=""
        if command -v uv >/dev/null 2>&1; then
            uv_cmd="uv"
        elif [[ -x "$HOME/.local/bin/uv" ]]; then
            uv_cmd="$HOME/.local/bin/uv"
        else
            error "uv is not installed or not found. Install dependencies first."
        fi
        
        # Use uv run to execute the Python script in the proper environment
        cd "$REPO_DIR"
        if ! "$uv_cmd" run python "$services_script" "$project" "$HOSTNAME_OVERRIDE"; then
            error "Failed to detect services for project '$project'. Check that the project exists and is properly configured."
        fi
    fi
}

# Portable alternative to readarray for zsh compatibility
load_services_array() {
    local project="$1"
    SERVICES=()
    while IFS= read -r line; do
        SERVICES+=("$line")
    done < <(get_project_services "$project")
}

# Services to manage (dynamically determined)
# For install action, we'll populate this after installing dependencies
if [[ "$ACTION" != "install" ]]; then
    load_services_array "$PROJECT"
else
    # For install action, we'll detect services after dependencies are installed
    SERVICES=()
fi

check_root() {
    # Only require root for Linux systemd operations
    if [[ "$USE_SYSTEMD" == true ]] && [[ "$PLATFORM" != "macos" ]] && [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for systemd operations in production mode"
    fi
    
    # On macOS, LaunchAgents should NOT be run as root
    if [[ "$PLATFORM" == "macos" ]] && [[ $EUID -eq 0 ]]; then
        error "Do not run this script as root on macOS. LaunchAgents run as the current user."
    fi
}

check_user() {
    if [[ "$RUNTIME_USER" == "experimance" ]] && ! id "$RUNTIME_USER" &>/dev/null; then
        error "User $RUNTIME_USER does not exist. Create it with: sudo useradd -m -s /bin/bash experimance"
    fi
    
    log "Using user: $RUNTIME_USER"
}

check_project() {
    if [[ ! -d "$REPO_DIR/projects/$PROJECT" ]]; then
        error "Project $PROJECT does not exist in $REPO_DIR/projects/"
    fi
    
    # Validate that we have services detected
    if [[ ${#SERVICES[@]} -eq 0 ]]; then
        error "No services detected for project $PROJECT. Service detection failed."
    fi
    
    # Show detected services for this project
    log "Detected services for project $PROJECT:"
    for service in "${SERVICES[@]}"; do
        echo "  - $service"
    done
}

# Update shell profiles to include pyenv and uv in PATH
update_shell_profile() {
    local bashrc="$HOME/.bashrc"
    local pyenv_init_added=false
    local uv_path_added=false
    
    # Check if pyenv is already configured in .bashrc
    if [[ -f "$bashrc" ]] && grep -q "PYENV_ROOT" "$bashrc"; then
        pyenv_init_added=true
    fi
    
    # Check if uv path is already configured in .bashrc
    if [[ -f "$bashrc" ]] && grep -q "\$HOME/\.local/bin" "$bashrc"; then
        uv_path_added=true
    fi
    
    # Add pyenv initialization to .bashrc if not present
    if [[ "$pyenv_init_added" == false ]]; then
        log "Adding pyenv initialization to ~/.bashrc..."
        cat >> "$bashrc" << 'EOF'

# Added by experimance deploy script - pyenv configuration
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
fi
EOF
    fi
    
    # Add uv path to .bashrc if not present
    if [[ "$uv_path_added" == false ]]; then
        log "Adding uv to PATH in ~/.bashrc..."
        cat >> "$bashrc" << 'EOF'

# Added by experimance deploy script - uv configuration
export PATH="$HOME/.local/bin:$PATH"
EOF
    fi
    
    if [[ "$pyenv_init_added" == false ]] || [[ "$uv_path_added" == false ]]; then
        log "Shell profile updated. Changes will take effect in new terminal sessions."
        log "Run 'source ~/.bashrc' to use the new settings in this terminal."
    fi
}

# Create symlink for standard installation directory
create_symlink() {
    if [[ "$USE_SYSTEMD" != true ]]; then
        log "Skipping symlink creation in development mode"
        return
    fi
    
    local install_dir="$1"
    local symlink_target="/opt/experimance"
    
    log "Creating symlink for standard directory layout..."
    
    # Create /opt directory if it doesn't exist
    if [[ ! -d "/opt" ]]; then
        mkdir -p /opt
        log "Created /opt directory"
    fi
    
    # Remove existing symlink or directory
    if [[ -L "$symlink_target" ]]; then
        rm "$symlink_target"
        log "Removed existing symlink $symlink_target"
    elif [[ -d "$symlink_target" ]]; then
        log "Warning: $symlink_target exists as directory, not creating symlink"
        return 1
    fi
    
    # Create symlink
    ln -s "$install_dir" "$symlink_target"
    chown -h experimance:experimance "$symlink_target"
    log "Created symlink: $symlink_target -> $install_dir"
}

install_reset_on_input() {
    if [[ "$USE_SYSTEMD" != true ]]; then
        log "Skipping reset on input installation in development mode"
        return
    fi
    
    # Check if reset-on-input is in the services list for this machine
    local install_reset=false
    for service in "${SERVICES[@]}"; do
        if [[ "$service" == "reset-on-input"* ]] || [[ "$service" == *"reset-on-input" ]]; then
            install_reset=true
            break
        fi
    done
    
    if [[ "$install_reset" != true ]]; then
        log "Skipping reset-on-input installation (not configured for this machine)"
        return
    fi
    
    log "Installing reset on input listener (keyboard/controller)..."
    
    # Copy the reset on input systemd service
    local reset_service_file="$SCRIPT_DIR/../systemd/reset-on-input.service"
    if [[ -f "$reset_service_file" ]]; then
        cp "$reset_service_file" "$SYSTEMD_DIR/"
        chmod 644 "$SYSTEMD_DIR/reset-on-input.service"
        log "✓ Copied reset-on-input.service"
        
        # Enable the service
        systemctl daemon-reload
        systemctl enable reset-on-input.service
        log "✓ Enabled reset-on-input.service"
        
        log "Reset on input listener installed successfully"
        log "  Start: sudo systemctl start reset-on-input.service"
        log "  Status: sudo systemctl status reset-on-input.service"
        log "  Logs: sudo journalctl -u reset-on-input.service -f"
    else
        warn "Reset on input service file not found: $reset_service_file"
        return 1
    fi
}

update_health_config() {
    log "Updating health service configuration for deployment..."
    
    local health_config_path="$REPO_DIR/projects/$PROJECT/health.toml"
    
    # Check if health.toml exists
    if [[ ! -f "$health_config_path" ]]; then
        log "No health.toml found at $health_config_path, skipping health config update"
        return
    fi
    
    # Get the service types (without @project suffix) for this machine, excluding health service
    local service_types=()
    for service in "${SERVICES[@]}"; do
        # Extract service type from "service_type@project" format
        local service_type="${service%@*}"
        # Skip the health service - it shouldn't monitor itself
        if [[ "$service_type" != "health" ]]; then
            service_types+=("$service_type")
        fi
    done
    
    # Get the module names for each service type
    local expected_services=()
    for service_type in "${service_types[@]}"; do
        local module_name
        if [[ "$MODE" == "prod" && "$RUNTIME_USER" == "experimance" && "$EUID" -eq 0 ]]; then
            # Running as root in production mode, delegate to experimance user
            module_name=$(sudo -u experimance bash -c "
                cd '$REPO_DIR'
                export PATH=\"/home/experimance/.local/bin:\$PATH\"
                uv run python infra/scripts/get_service_module.py '$PROJECT' '$service_type'
            " 2>/dev/null || echo "experimance_$service_type")
        else
            # Development mode or already running as correct user
            module_name=$(cd "$REPO_DIR" && uv run python infra/scripts/get_service_module.py "$PROJECT" "$service_type" 2>/dev/null || echo "experimance_$service_type")
        fi
        expected_services+=("$module_name")
    done
    
    # Create the expected_services array string for TOML
    local services_toml="expected_services = ["
    for i in "${!expected_services[@]}"; do
        services_toml+="\n    \"${expected_services[i]}\""
        if [[ $i -lt $((${#expected_services[@]} - 1)) ]]; then
            services_toml+=","
        fi
    done
    services_toml+="\n]"
    
    # Create a backup of the original file
    cp "$health_config_path" "$health_config_path.backup"
    
    # Update the expected_services in the health.toml file
    # Use sed to replace the expected_services array
    local temp_file="$health_config_path.tmp"
    
    # Use awk to replace the expected_services section
    awk -v new_services="$services_toml" '
    /^expected_services = \[/ {
        print new_services
        # Skip lines until we find the closing bracket
        while (getline && !/^\]/ && !/^[a-zA-Z_]/) {
            # Skip lines that are part of the array
        }
        # If we stopped on a line that is not a closing bracket, print it
        if (!/^\]/) {
            print
        }
        next
    }
    { print }
    ' "$health_config_path" > "$temp_file"
    
    # Replace the original file
    mv "$temp_file" "$health_config_path"
    
    log "Updated health.toml expected_services:"
    for service in "${expected_services[@]}"; do
        log "  - $service"
    done
    
    log "Health configuration updated successfully"
    log "Original config backed up to: $health_config_path.backup"
}

install_systemd_files() {
    case "$PLATFORM" in
        linux)
            install_systemd_files_linux
            ;;
        macos)
            install_launchd_files_macos
            ;;
        *)
            log "Skipping service installation on unsupported platform: $PLATFORM"
            ;;
    esac
}

# Linux systemd installation (renamed from original function)
install_systemd_files_linux() {
    if [[ "$USE_SYSTEMD" != true ]]; then
        log "Skipping systemd installation in development mode"
        return
    fi
    
    log "Installing systemd service files..."
    
    # Determine which service template files are needed for this machine
    declare -A needed_service_types
    declare -A service_modules
    for service in "${SERVICES[@]}"; do
        # service is in format "service_type@project", e.g., "core@experimance"
        local service_type="${service%@*}"
        needed_service_types["$service_type"]=1
        
        # Get the module name for this service using the helper script
        local module_name
        if [[ "$MODE" == "prod" && "$RUNTIME_USER" == "experimance" && "$EUID" -eq 0 ]]; then
            # Running as root in production mode, delegate to experimance user
            module_name=$(sudo -u experimance bash -c "
                cd '$REPO_DIR'
                export PATH=\"/home/experimance/.local/bin:\$PATH\"
                uv run python infra/scripts/get_service_module.py '$PROJECT' '$service_type'
            " 2>/dev/null || echo "experimance_$service_type")
        else
            # Development mode or already running as correct user
            module_name=$(cd "$REPO_DIR" && uv run python infra/scripts/get_service_module.py "$PROJECT" "$service_type" 2>/dev/null || echo "experimance_$service_type")
        fi
        service_modules["$service_type"]="$module_name"
        log "Service $service_type will use module: $module_name"
    done
    
    # Copy only the needed template service files (e.g., core@.service, display@.service)
    # These are TEMPLATES that systemd uses to create instances like core@experimance.service
    local files_copied=0
    for service_file in "$SCRIPT_DIR"/../systemd/*.service; do
        if [[ -f "$service_file" ]]; then
            local basename_file=$(basename "$service_file")
            
            # Handle template services (contains @) and standalone services
            if [[ "$basename_file" == *"@.service" ]]; then
                # Template service - extract service type
                local service_type="${basename_file%@*}"
                if [[ "${needed_service_types[$service_type]:-}" == "1" ]]; then
                    # Get the module name for this service type
                    local module_name="${service_modules[$service_type]:-experimance_$service_type}"
                    
                    # Create a customized version of the template file
                    local temp_file="/tmp/${basename_file}.$$"
                    cp "$service_file" "$temp_file"
                    
                    # Replace the ExecStart line with the correct module name
                    sed -i "s|ExecStart=/home/experimance/.local/bin/uv run -m experimance_${service_type}|ExecStart=/home/experimance/.local/bin/uv run -m ${module_name}|g" "$temp_file"
                    
                    # Copy the customized file to systemd directory with correct name
                    if cp "$temp_file" "$SYSTEMD_DIR/$basename_file"; then
                        log "✓ Copied $basename_file (module: $module_name)"
                        files_copied=$((files_copied + 1))
                    else
                        error "✗ Failed to copy $basename_file"
                    fi
                    
                    # Clean up temp file
                    rm -f "$temp_file"
                else
                    log "Skipping $basename_file (not needed for this machine)"
                fi
            else
                # Standalone service - check if it's in the services list
                local service_name="${basename_file%.service}"
                local found_service=false
                for service in "${SERVICES[@]}"; do
                    if [[ "$service" == "$service_name"* ]] || [[ "$service" == *"$service_name" ]]; then
                        found_service=true
                        break
                    fi
                done
                
                if [[ "$found_service" == true ]]; then
                    if cp "$service_file" "$SYSTEMD_DIR/"; then
                        log "✓ Copied $basename_file"
                        files_copied=$((files_copied + 1))
                    else
                        error "✗ Failed to copy $basename_file"
                    fi
                else
                    log "Skipping $basename_file (not needed for this machine)"
                fi
            fi
        fi
    done
    
    # Copy target template file (e.g., experimance@.target)
    if [[ -f "$SCRIPT_DIR/../systemd/experimance@.target" ]]; then
        if cp "$SCRIPT_DIR/../systemd/experimance@.target" "$SYSTEMD_DIR/"; then
            log "✓ Copied experimance@.target"
            files_copied=$((files_copied + 1))
        else
            error "✗ Failed to copy experimance@.target"
        fi
    else
        warn "Target template file not found: $SCRIPT_DIR/../systemd/experimance@.target"
    fi
    
    if [[ $files_copied -eq 0 ]]; then
        log "ERROR: No systemd files found to copy!"
        return 1
    fi
    
    # Set proper permissions on template files
    sudo chmod 644 "$SYSTEMD_DIR"/*.service "$SYSTEMD_DIR"/*.target 2>/dev/null || true
    
    # Reload systemd to recognize new template files
    systemctl daemon-reload
    log "Reloaded systemd configuration"
    log "Installed $files_copied systemd template files"
    
    # Enable and link all service instances to the target
    log "Enabling service instances and linking to target..."
    local services_enabled=0
    for service in "${SERVICES[@]}"; do
        # service is in format "service_type@project", e.g., "core@experimance"
        local full_service_name="$service.service"
        local service_type="${service%@*}"
        local template_file="$SYSTEMD_DIR/${service_type}@.service"
        
        if [ -f "$template_file" ]; then
            log "Enabling $full_service_name..."
            if systemctl enable "$full_service_name"; then
                log "✓ Enabled $full_service_name"
                services_enabled=$((services_enabled + 1))
            else
                warn "Failed to enable $full_service_name"
            fi
        else
            warn "Template file not found for $full_service_name: $template_file"
        fi
    done
    
    # Enable the target instance
    local target="experimance@${PROJECT}.target"
    if [ -f "$SYSTEMD_DIR/experimance@.target" ]; then
        log "Enabling target $target..."
        if systemctl enable "$target"; then
            log "✓ Enabled $target"
        else
            warn "Failed to enable $target"
        fi
    else
        warn "Target template file not found: $SYSTEMD_DIR/experimance@.target"
    fi
    
    # Verify service linking
    log "Verifying service linking to target..."
    local linked_services=0
    local deps_output
    
    # Get dependencies once to avoid multiple calls
    if deps_output=$(systemctl list-dependencies "$target" 2>/dev/null); then
        for service in "${SERVICES[@]}"; do
            local full_service_name="$service.service"
            if echo "$deps_output" | grep -q "$full_service_name"; then
                log "✓ $full_service_name is linked to $target"
                linked_services=$((linked_services + 1))
            else
                warn "✗ $full_service_name is NOT linked to $target"
            fi
        done
    else
        warn "Could not retrieve dependencies for $target"
        log "This may indicate the target is not properly configured or systemd needs to be reloaded"
    fi
    
    log "Installation summary:"
    log "  Template files copied: $files_copied"
    log "  Service instances enabled: $services_enabled/${#SERVICES[@]}"
    log "  Services linked to target: $linked_services/${#SERVICES[@]}"
    
    if [[ $services_enabled -eq ${#SERVICES[@]} ]] && [[ $linked_services -eq ${#SERVICES[@]} ]]; then
        log "✓ All services successfully installed, enabled, and linked"
    else
        warn "⚠ Some services may not be properly configured. Run 'sudo ./deploy.sh $PROJECT diagnose' for details"
    fi
    
    log "Note: Instance services are now created and linked. Use 'sudo ./deploy.sh $PROJECT start' to start them"
}

# macOS launchd installation
install_launchd_files_macos() {
    log "Installing launchd service files for macOS..."
    
    # On macOS, always use LaunchAgents (user-level), never LaunchDaemons
    local launchd_dir="$HOME/Library/LaunchAgents"
    
    mkdir -p "$launchd_dir"
    
    # Create log directory for LaunchAgent logs
    mkdir -p "$HOME/Library/Logs/experimance"
    
    # Convert systemd templates to launchd plist files
    local files_created=0
    for service in "${SERVICES[@]}"; do
        if create_launchd_plist "$service" "$launchd_dir"; then
            files_created=$((files_created + 1))
        fi
    done
    
    log "launchd service files installed: $files_created services"
    
    # Show Full Disk Access instructions for LaunchAgents
    if [[ "$MODE" != "prod" ]]; then
        show_macos_full_disk_access_instructions
    fi
    
    log "Note: Use './deploy.sh $PROJECT start' to load and start services"
}

create_launchd_plist() {
    local service="$1"
    local launchd_dir="$2"
    
    # On macOS, always use LaunchAgents, never LaunchDaemons
    create_launchd_agent "$service" "$launchd_dir"
}

# Create LaunchDaemon plist (for production mode)
create_launchd_daemon() {
    local service="$1"
    local launchd_dir="$2"
    local service_with_project="${service%@*}"  # e.g., "agent@fire" -> "agent"
    local project="${service#*@}"                # e.g., "agent@fire" -> "fire"
    local service_type="$service_with_project"  # e.g., "agent"
    local plist_file="$launchd_dir/com.experimance.$service.plist"
    
    # Get the correct module name from deployment config
    local module_name
    module_name=$(get_service_module_name "$project" "$service")
    
    # Determine the correct paths based on platform
    local uv_path="/opt/homebrew/bin/uv"
    if [[ ! -x "$uv_path" ]]; then
        # Try Intel Mac path
        uv_path="/usr/local/bin/uv"
        if [[ ! -x "$uv_path" ]]; then
            # Try user installation
            uv_path="$HOME/.local/bin/uv"
        fi
    fi
    
    cat > "$plist_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.experimance.$service</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>$uv_path</string>
        <string>run</string>
        <string>-m</string>
        <string>$module_name</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>$REPO_DIR</string>
    
    <key>KeepAlive</key>
    <true/>
    
    <key>RunAtLoad</key>
    <false/>
    
    <key>StandardErrorPath</key>
    <string>/var/log/experimance_$service.log</string>
    
    <key>StandardOutPath</key>
    <string>/var/log/experimance_$service.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>EXPERIMANCE_ENV</key>
        <string>production</string>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:\$HOME/.local/bin:/usr/bin:/bin</string>
    </dict>
    
    <key>UserName</key>
    <string>$RUNTIME_USER</string>
</dict>
</plist>
EOF

    # Set proper ownership and permissions for LaunchDaemon
    chown root:wheel "$plist_file"
    chmod 644 "$plist_file"
    
    log "✓ Created LaunchDaemon service: $plist_file"
    return 0
}

# Create LaunchAgent plist (for development mode)
create_launchd_agent() {
    local service="$1"
    local launchd_dir="$2"
    local service_with_project="${service%@*}"  # e.g., "agent@fire" -> "agent"
    local project="${service#*@}"                # e.g., "agent@fire" -> "fire"
    local service_type="$service_with_project"  # e.g., "agent"
    local plist_file="$launchd_dir/com.experimance.${project}.${service_type}.plist"
    
    # Get the correct module name from deployment config
    local module_name
    module_name=$(get_service_module_name "$project" "$service")
    
    # Determine uv path - prefer Homebrew location if available
    local uv_path
    if [[ -x "/opt/homebrew/bin/uv" ]]; then
        uv_path="/opt/homebrew/bin/uv"
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        uv_path="$HOME/.local/bin/uv"
    elif command -v uv &> /dev/null; then
        uv_path=$(command -v uv)
    else
        error "uv not found. Please install uv first."
        return 1
    fi
    
    cat > "$plist_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.experimance.${service_type}.$project</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>$uv_path</string>
        <string>run</string>
        <string>-m</string>
        <string>$module_name</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>$REPO_DIR</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PROJECT_ENV</key>
        <string>$project</string>
    </dict>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/experimance/${project}_${service_type}_launchd_error.log</string>
    
    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/experimance/${project}_${service_type}_launchd.log</string>
    
    <!-- LaunchAgent version - no UserName needed, runs as logged-in user -->
    
    <!-- Restart on failure after 10 seconds -->
    <key>ThrottleInterval</key>
    <integer>10</integer>
    
    <!-- Start after a delay to ensure system is ready -->
    <key>StartInterval</key>
    <integer>5</integer>
</dict>
</plist>
EOF

    # Set proper ownership and permissions for LaunchAgent
    chmod 644 "$plist_file"
    
    log "✓ Created LaunchAgent service: $plist_file"
    log "✓ Using uv path: $uv_path"
    return 0
}

# Show Full Disk Access instructions for macOS LaunchAgents
show_macos_full_disk_access_instructions() {
    echo ""
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║                        macOS FULL DISK ACCESS REQUIRED                        ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}LaunchAgents require Full Disk Access for uv to execute Python modules.${NC}"
    echo ""
    echo -e "${BLUE}Manual steps required:${NC}"
    echo -e "${GREEN}1.${NC} Open System Settings → Privacy & Security → Full Disk Access"
    echo -e "${GREEN}2.${NC} Click the + button to add applications"
    echo -e "${GREEN}3.${NC} Navigate to and add the uv binary:"
    
    # Determine uv path
    local uv_path
    if [[ -x "/opt/homebrew/bin/uv" ]]; then
        uv_path="/opt/homebrew/bin/uv"
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        uv_path="$HOME/.local/bin/uv"
    elif command -v uv &> /dev/null; then
        uv_path=$(command -v uv)
    else
        uv_path="<uv_path_not_found>"
    fi
    
    echo "   - $uv_path"
    echo ""
    echo -e "${GREEN}4.${NC} Toggle the switch to grant Full Disk Access to uv"
    echo -e "${GREEN}5.${NC} After granting access, run: ${BLUE}./deploy.sh $PROJECT start${NC}"
    echo ""
    echo -e "${YELLOW}Note: This is only required once for the uv binary.${NC}"
    echo -e "${YELLOW}Without Full Disk Access, LaunchAgents will fail with 'Operation not permitted'.${NC}"
    echo ""
    echo -e "${BLUE}To find uv location manually, run: ${GREEN}which uv${NC}"
    echo ""
}

# Download file from Google Drive using uvx gdown
download_google_drive_file() {
    local file_id="$1"
    local output_file="$2"
    
    log "Using uvx gdown to download from Google Drive"
    if uvx gdown "$file_id" -O "$output_file" --quiet; then
        # Verify the downloaded file is actually a zip file
        if file "$output_file" | grep -q -i zip; then
            return 0
        else
            warn "Downloaded file is not a valid zip file"
            log "File type: $(file "$output_file")"
            rm -f "$output_file"
            return 1
        fi
    else
        warn "uvx gdown failed to download the file"
        log "Manual download required: https://drive.google.com/file/d/${file_id}/view"
        return 1
    fi
}

setup_directories() {
    log "Setting up directories..."
    
    # Get correct group for this platform
    CHOWN_GROUP=$(get_chown_group)
    
    # Create cache directory
    if [[ "$MODE" == "prod" ]]; then
        # Production: use /var/cache/experimance
        mkdir -p /var/cache/experimance
        chown "$RUNTIME_USER:$CHOWN_GROUP" /var/cache/experimance
        log "Created production cache directory: /var/cache/experimance"
        
        # Production: use /var/log/experimance
        mkdir -p /var/log/experimance
        chown "$RUNTIME_USER:$CHOWN_GROUP" /var/log/experimance
        chmod 775 /var/log/experimance
        log "Created production log directory: /var/log/experimance"
    else
        # Development: use local cache directory
        mkdir -p "$REPO_DIR/cache"
        chown "$RUNTIME_USER:$CHOWN_GROUP" "$REPO_DIR/cache"
        log "Created development cache directory: $REPO_DIR/cache"
    fi
    
    # Create log directory if it doesn't exist
    mkdir -p "$REPO_DIR/logs"
    chown "$RUNTIME_USER:$CHOWN_GROUP" "$REPO_DIR/logs"
    
    # Create images directory if it doesn't exist
    mkdir -p "$REPO_DIR/media/images/generated"
    mkdir -p "$REPO_DIR/media/images/mocks"
    mkdir -p "$REPO_DIR/media/video"
    chown "$RUNTIME_USER:$CHOWN_GROUP" "$REPO_DIR/media/images"
    chown "$RUNTIME_USER:$CHOWN_GROUP" "$REPO_DIR/media/images/generated"
    chown "$RUNTIME_USER:$CHOWN_GROUP" "$REPO_DIR/media/images/mocks"
    chown "$RUNTIME_USER:$CHOWN_GROUP" "$REPO_DIR/media/video"
    
    # create transcripts directory
    mkdir -p "$REPO_DIR/transcripts"
    chown "$RUNTIME_USER:$CHOWN_GROUP" "$REPO_DIR/transcripts"

    # Download and extract media bundle from Google drive
    # https://drive.google.com/file/d/1JPf4biReYj1qwWXsb0R_ZVcVrzM94XeE/view?usp=drive_link
    # Note: This function will automatically install gdown if needed

    MEDIA_ZIP_ID="${MEDIA_ZIP_ID:-1JPf4biReYj1qwWXsb0R_ZVcVrzM94XeE}"
    if [[ -n "${MEDIA_ZIP_ID:-}" ]]; then
        ZIP_FILE="$REPO_DIR/media/experimance_installation_media_bundle.zip"
        # Ask user if they want to download media files (default Yes)
        read -p "Download and extract media files? (Y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            MEDIA_URL="https://docs.google.com/uc?export=download&id=${MEDIA_ZIP_ID}"
            if [[ ! -f "$ZIP_FILE" ]]; then
                log "Downloading media bundle to $ZIP_FILE"
                if ! download_google_drive_file "$MEDIA_ZIP_ID" "$ZIP_FILE"; then
                    warn "Failed to download media bundle automatically"
                    log "You can download manually from: https://drive.google.com/file/d/${MEDIA_ZIP_ID}/view"
                    log "Save as: $ZIP_FILE"
                    log "Continuing installation without media files..."
                    return
                fi
                chown "$RUNTIME_USER:$CHOWN_GROUP" "$ZIP_FILE"
            else
                log "Media bundle zip already present at $ZIP_FILE"
                
                # Verify existing file is actually a zip file
                if ! file "$ZIP_FILE" | grep -q -i zip; then
                    warn "Existing media bundle file is not a valid zip file, re-downloading..."
                    rm -f "$ZIP_FILE"
                    
                    log "Downloading media bundle to $ZIP_FILE"
                    if ! download_google_drive_file "$MEDIA_ZIP_ID" "$ZIP_FILE"; then
                        warn "Failed to download media bundle automatically"
                        log "You can download manually from: https://drive.google.com/file/d/${MEDIA_ZIP_ID}/view"
                        log "Save as: $ZIP_FILE"
                        log "Continuing installation without media files..."
                        return
                    fi
                    chown "$RUNTIME_USER:$CHOWN_GROUP" "$ZIP_FILE"
                fi
            fi

            log "Extracting media bundle into $REPO_DIR/media"
            if command -v unzip &>/dev/null; then
                # Update existing files only if archive files are newer, extract media/* into REPO_DIR to avoid nested media/media
                if ! unzip -qu "$ZIP_FILE" "media/*" -d "$REPO_DIR"; then
                    warn "Failed to extract media bundle automatically"
                    log "You can extract manually with: unzip '$ZIP_FILE' -d '$REPO_DIR'"
                    log "Or download the media bundle manually from: https://drive.google.com/file/d/${MEDIA_ZIP_ID}/view"
                    log "Continuing installation without media files..."
                else
                    log "Media bundle extracted successfully"
                    chown -R "$RUNTIME_USER:$CHOWN_GROUP" "$REPO_DIR/media"
                fi
            else
                warn "Cannot extract media bundle: 'unzip' not found"
                log "Please install 'unzip' or extract manually: unzip '$ZIP_FILE' -d '$REPO_DIR'"
                log "Or download the media bundle manually from: https://drive.google.com/file/d/${MEDIA_ZIP_ID}/view"
            fi
        else
            log "Skipping media bundle download and extraction."
        fi
    fi

    log "Directories created and permissions set"
}

# Define the complete list of required system packages
get_required_packages() {
    local package_type="${1:-default}"
    
    case "$PLATFORM" in
        linux)
            get_required_packages_linux "$package_type"
            ;;
        macos)
            get_required_packages_macos "$package_type"
            ;;
        *)
            error "Unsupported platform: $PLATFORM"
            ;;
    esac
}

# Linux package requirements
get_required_packages_linux() {
    # Essential build tools that should be checked by command
    local build_tools=("make" "gcc")
    
    # Complete list of packages for apt/yum/dnf
    # NOTE: We use `dpkg -l package_name` instead of `dpkg -l | grep package_name`
    # because with `set -euo pipefail`, the pipeline approach can fail unexpectedly
    # when dpkg has warnings or the grep doesn't match, even with 2>/dev/null
    local apt_packages=(
        "make"
        "build-essential"
        "libssl-dev"
        "zlib1g-dev" 
        "libbz2-dev"
        "libreadline-dev"
        "libsqlite3-dev"
        "curl"
        "git"
        "libncurses-dev"
        "xz-utils"
        "tk-dev"
        "libxml2-dev"
        "libxmlsec1-dev"
        "libffi-dev"
        "liblzma-dev"
        "libgdbm-dev"
        "libdb-dev"
        "portaudio19-dev"  # needed for pyaudio
        "libasound2-dev"   # needed for pyaudio
        "supercollider"    # needed for audio synthesis
        "v4l-utils"        # Video4Linux utilities for webcam support
        "libv4l-dev"       # Video4Linux development libraries  
        "uvcdynctrl"       # UVC (USB Video Class) control tools
        "guvcview"         # GTK+ UVC Viewer and control tool (optional)
        "ffmpeg"           # FFmpeg for video/audio processing
        "evtest"           # evtest for USB input monitoring
        "lm-sensors"       # lm-sensors for hardware monitoring
    )
    
    # Export arrays for use by calling functions
    case "${1:-apt}" in
        "build_tools")
            printf '%s\n' "${build_tools[@]}"
            ;;
        "apt")
            printf '%s\n' "${apt_packages[@]}"
            ;;
        *)
            printf '%s\n' "${apt_packages[@]}"
            ;;
    esac
}

# macOS package requirements
get_required_packages_macos() {
    # Essential build tools (checked differently on macOS)
    local build_tools=("xcode-select")
    
    # Homebrew packages (equivalent to apt_packages)
    local brew_packages=(
        # Build tools (make is included in Xcode Command Line Tools)
        # "build-essential" -> Xcode Command Line Tools
        
        # Development libraries
        "openssl"           # libssl-dev
        "zlib"              # zlib1g-dev  
        "bzip2"             # libbz2-dev
        "readline"          # libreadline-dev
        "sqlite"            # libsqlite3-dev
        "curl"              # curl (usually pre-installed)
        "git"               # git (usually pre-installed via Xcode)
        "xz"                # xz-utils
        "tcl-tk"            # tk-dev (different name on macOS)
        "libxml2"           # libxml2-dev
        "xmlsec1"           # libxmlsec1-dev
        "libffi"            # libffi-dev
        "gdbm"              # libgdbm-dev
        
        # Audio/Video
        "portaudio"         # portaudio19-dev
        "ffmpeg"            # ffmpeg
        
        # Note: v4l-utils, libv4l-dev, uvcdynctrl, guvcview -> Not needed on macOS
        # macOS uses AVFoundation framework instead
        # evtest -> Not available on macOS (use different input monitoring)
        # lm-sensors -> Not available on macOS (use built-in sensors)
        # supercollider available via brew if needed
    )
    
    case "${1:-brew}" in
        "build_tools")
            printf '%s\n' "${build_tools[@]}"
            ;;
        "brew")
            printf '%s\n' "${brew_packages[@]}"
            ;;
        *)
            printf '%s\n' "${brew_packages[@]}"
            ;;
    esac
}

# Check if system build dependencies are installed
check_build_dependencies() {
    local missing_deps=()
    
    # Check for essential build tools
    while IFS= read -r tool; do
        case "$tool" in
            "gcc")
                if ! command -v gcc >/dev/null 2>&1; then 
                    if [[ "$PLATFORM" == "linux" ]]; then
                        missing_deps+=("build-essential")
                    elif [[ "$PLATFORM" == "macos" ]]; then
                        missing_deps+=("xcode-command-line-tools")
                    fi
                fi
                ;;
            "xcode-select")
                if ! xcode-select -p &>/dev/null; then
                    missing_deps+=("xcode-command-line-tools")
                fi
                ;;
            *)
                if ! command -v "$tool" >/dev/null 2>&1; then missing_deps+=("$tool"); fi
                ;;
        esac
    done < <(get_required_packages "build_tools")
    
    # Check each package (skip build tools as they're handled above)
    if [[ "$PLATFORM" == "linux" ]]; then
        while IFS= read -r package; do
            if [[ "$package" != "make" && "$package" != "build-essential" ]]; then
                if ! dpkg -l "$package" 2>/dev/null | grep -q "^ii"; then
                    missing_deps+=("$package")
                fi
            fi
        done < <(get_required_packages "apt")
    elif [[ "$PLATFORM" == "macos" ]]; then
        while IFS= read -r package; do
            if [[ "$package" != "make" ]]; then  # make comes with Xcode tools
                if ! brew list "$package" &>/dev/null; then
                    missing_deps+=("$package")
                fi
            fi
        done < <(get_required_packages "brew")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        warn "Missing system build dependencies required for Python compilation:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        
        if [[ "$PLATFORM" == "linux" ]]; then
            echo "The following packages need to be installed:"
            echo "  sudo apt install $(get_required_packages "apt" | tr '\n' ' ')"
        elif [[ "$PLATFORM" == "macos" ]]; then
            if [[ " ${missing_deps[*]} " =~ " xcode-command-line-tools " ]]; then
                echo "Xcode Command Line Tools need to be installed:"
                echo "  xcode-select --install"
            fi
            local brew_deps=()
            for dep in "${missing_deps[@]}"; do
                if [[ "$dep" != "xcode-command-line-tools" ]]; then
                    brew_deps+=("$dep")
                fi
            done
            if [[ ${#brew_deps[@]} -gt 0 ]]; then
                echo "The following Homebrew packages need to be installed:"
                echo "  brew install ${brew_deps[*]}"
            fi
        fi
        echo ""
        
        if [[ "$MODE" == "prod" ]]; then
            # In production mode, we're already running as root, so just install
            return 1
        else
            # In development mode, ask for permission
            read -p "Install system build dependencies now? (y/N): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                return 1
            else
                error "Cannot proceed without build dependencies. Please install them manually and run again."
            fi
        fi
    fi
    
    return 0
}

install_system_dependencies() {
    log "Installing system build dependencies for Python compilation..."
    
    if command -v apt >/dev/null 2>&1; then
        # Ubuntu/Debian/Mint
        local packages_to_install
        packages_to_install=$(get_required_packages "apt" | tr '\n' ' ')
        
        if [[ "$MODE" == "prod" ]]; then
            # Already running as root in production
            apt update
            apt install -y $packages_to_install
        else
            # Use sudo in development
            sudo apt update
            sudo apt install -y $packages_to_install
        fi
        log "System build dependencies installed"
    elif command -v yum >/dev/null 2>&1; then
        # CentOS/RHEL/Amazon Linux
        local yum_packages=(
            "gcc" "make" "patch" "zlib-devel" "bzip2" "bzip2-devel"
            "readline-devel" "sqlite" "sqlite-devel" "openssl-devel"
            "tk-devel" "libffi-devel" "xz-devel"
            "portaudio-devel" "alsa-lib-devel"
            # Note: v4l-utils equivalent packages for RHEL/CentOS would be different
        )
        if [[ "$MODE" == "prod" ]]; then
            yum install -y "${yum_packages[@]}"
        else
            sudo yum install -y "${yum_packages[@]}"
        fi
        log "System build dependencies installed"
    elif command -v dnf >/dev/null 2>&1; then
        # Fedora
        local dnf_packages=(
            "make" "gcc" "patch" "zlib-devel" "bzip2" "bzip2-devel"
            "readline-devel" "sqlite" "sqlite-devel" "openssl-devel"
            "tk-devel" "libffi-devel" "xz-devel" "libuuid-devel" "gdbm-libs" "libnsl2"
            "portaudio-devel" "alsa-lib-devel"
            # Note: v4l-utils equivalent packages for Fedora would be different
        )
        if [[ "$MODE" == "prod" ]]; then
            dnf install -y "${dnf_packages[@]}"
        else
            sudo dnf install -y "${dnf_packages[@]}"
        fi
        log "System build dependencies installed"
    else
        warn "Unknown package manager. Please install build dependencies manually:"
        warn "  sudo apt install $(get_required_packages "apt" | tr '\n' ' ')"
        error "Cannot proceed without build dependencies"
    fi
}

# macOS dependency installation
install_system_dependencies_macos() {
    log "Installing system dependencies for macOS..."
    
    # Determine the actual user (not root if using sudo)
    local actual_user="$RUNTIME_USER"
    if [[ "$EUID" -eq 0 && -n "${SUDO_USER:-}" ]]; then
        actual_user="$SUDO_USER"
        log "Running as root via sudo, will install Homebrew as user: $actual_user"
    fi
    
    # Check for Xcode Command Line Tools
    if ! xcode-select -p &>/dev/null; then
        log "Installing Xcode Command Line Tools..."
        xcode-select --install
        log "Please complete the Xcode Command Line Tools installation and re-run this script"
        exit 1
    fi
    
    # Function to run commands as the actual user
    run_as_user() {
        if [[ "$EUID" -eq 0 && -n "${SUDO_USER:-}" ]]; then
            # Running as root via sudo, delegate to actual user
            sudo -u "$actual_user" -H bash -c "$1"
        else
            # Running as regular user
            bash -c "$1"
        fi
    }
    
    # Check for Homebrew
    if ! run_as_user "command -v brew &>/dev/null"; then
        log "Installing Homebrew as user $actual_user..."
        if [[ "$EUID" -eq 0 && -n "${SUDO_USER:-}" ]]; then
            # Install Homebrew as the sudo user, not root
            sudo -u "$actual_user" -H bash -c '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        else
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Add Homebrew to PATH for current session
        if [[ -x /opt/homebrew/bin/brew ]]; then
            export PATH="/opt/homebrew/bin:$PATH"
        elif [[ -x /usr/local/bin/brew ]]; then
            export PATH="/usr/local/bin:$PATH"
        fi
    fi
    
    # Install packages as the actual user (Homebrew requirement)
    local packages_to_install
    packages_to_install=$(get_required_packages_macos "brew" | tr '\n' ' ')
    
    log "Installing Homebrew packages as user $actual_user: $packages_to_install"
    
    # Create the brew command with proper PATH
    local brew_cmd='
        if [[ -x /opt/homebrew/bin/brew ]]; then
            export PATH="/opt/homebrew/bin:$PATH"
            BREW_CMD="/opt/homebrew/bin/brew"
        elif [[ -x /usr/local/bin/brew ]]; then
            export PATH="/usr/local/bin:$PATH" 
            BREW_CMD="/usr/local/bin/brew"
        else
            BREW_CMD="brew"
        fi
        $BREW_CMD install '"$packages_to_install"'
    '
    
    if ! run_as_user "$brew_cmd"; then
        error "Failed to install Homebrew packages. Some packages may have different names on macOS or may not be available."
    fi
    
    log "System dependencies installed via Homebrew"
}

# Ask about Python optimization flags
ask_python_optimization() {
    if [[ "$MODE" == "prod" ]]; then
        # In production, default to optimized builds
        log "Production mode: Building Python with optimization flags for better performance"
        return 0
    else
        # In development, ask the user
        echo ""
        echo "Python can be built with optimization flags for ~30% better performance."
        echo "This includes Profile Guided Optimization (PGO) and Link Time Optimization (LTO)."
        echo "Note: Optimized builds take significantly longer to compile (20-30 minutes vs 5 minutes)."
        echo ""
        read -p "Build Python with optimization flags? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Building Python with optimization flags..."
            return 0
        else
            log "Building Python without optimization flags (faster compilation)..."
            return 1
        fi
    fi
}

install_dependencies() {
    log "Installing Python dependencies..."
    
    # Check and install system build dependencies if needed
    if check_build_dependencies; then
        log "System build dependencies already installed"
    else
        case "$PLATFORM" in
            linux)
                install_system_dependencies
                ;;
            macos)
                install_system_dependencies_macos
                ;;
            *)
                error "Unsupported platform: $PLATFORM"
                ;;
        esac
    fi
    
    if [[ "$MODE" == "prod" ]]; then
        # Production mode: install uv and dependencies as the runtime user
        log "Installing uv and Python 3.11 for $RUNTIME_USER user..."
        sudo -u "$RUNTIME_USER" bash -c "
            # Install pyenv if not already installed
            if ! command -v pyenv >/dev/null 2>&1; then
                # Check if pyenv directory exists but not in PATH
                if [[ -d \"\$HOME/.pyenv\" ]]; then
                    echo 'pyenv directory exists, initializing...'
                    export PYENV_ROOT=\"\$HOME/.pyenv\"
                    export PATH=\"\$PYENV_ROOT/bin:\$PATH\"
                    eval \"\$(pyenv init --path)\" 2>/dev/null || true
                    eval \"\$(pyenv init -)\" 2>/dev/null || true
                else
                    echo 'Installing pyenv...'
                    curl https://pyenv.run | bash
                    
                    # Add pyenv to PATH for this session
                    export PYENV_ROOT=\"\$HOME/.pyenv\"
                    export PATH=\"\$PYENV_ROOT/bin:\$PATH\"
                    eval \"\$(pyenv init --path)\"
                    eval \"\$(pyenv init -)\"
                fi
            else
                # Initialize pyenv if already installed
                export PYENV_ROOT=\"\$HOME/.pyenv\"
                export PATH=\"\$PYENV_ROOT/bin:\$PATH\"
                eval \"\$(pyenv init --path)\"
                eval \"\$(pyenv init -)\"
            fi
            
            # Install Python 3.11 if not available
            if ! pyenv versions | grep -q '3.11'; then
                echo 'Installing latest Python 3.11...'
                LATEST_311=\$(pyenv install --list | grep -E '^  3\.11\.[0-9]+$' | tail -1 | xargs)
                echo \"Installing Python \$LATEST_311\"
                
                # Ask about optimization in production mode (default yes)
                echo 'Production mode: Building Python with optimization flags for better performance'
                PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install \$LATEST_311
            else
                LATEST_311=\$(pyenv versions | grep '3.11' | tail -1 | sed 's/[* ]//g' | sed 's/(.*$//')
            fi
            
            # Set latest Python 3.11 for this project
            cd '$REPO_DIR'
            pyenv local \$LATEST_311
            
            # Add uv to PATH first
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            
            # Install uv if not already installed
            if ! command -v uv >/dev/null 2>&1 && ! [[ -x \"\$HOME/.local/bin/uv\" ]]; then
                echo 'Installing uv...'
                curl -LsSf https://astral.sh/uv/install.sh | sh
                # Re-export PATH after installation
                export PATH=\"\$HOME/.local/bin:\$PATH\"
            else
                echo 'uv is already installed'
            fi
            
            # Verify uv is available
            if ! command -v uv >/dev/null 2>&1 && ! [[ -x \"\$HOME/.local/bin/uv\" ]]; then
                echo 'ERROR: uv installation failed or not found' >&2
                exit 1
            fi
            
            # Use uv (either from PATH or direct path)
            UV_CMD=\$(command -v uv 2>/dev/null || echo \"\$HOME/.local/bin/uv\")
            
            # Install project dependencies with Python 3.11
            cd '$REPO_DIR'
            if ! \"\$UV_CMD\" sync; then
                echo 'ERROR: Failed to install project dependencies' >&2
                exit 1
            fi
        " || error "Failed to install dependencies for production"
    else
        # Development mode: install for current user
        
        # First, try to initialize pyenv if it exists but isn't in PATH
        if [[ -d "$HOME/.pyenv" ]] && ! command -v pyenv >/dev/null 2>&1; then
            log "pyenv directory exists, initializing..."
            export PYENV_ROOT="$HOME/.pyenv"
            export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init --path)" 2>/dev/null || true
            eval "$(pyenv init -)" 2>/dev/null || true
        fi
        
        # Now check if both pyenv and uv are available
        if command -v pyenv >/dev/null 2>&1 && command -v uv >/dev/null 2>&1; then
            log "pyenv and uv already installed, checking Python version..."
            cd "$REPO_DIR"
            
            # Check if we have Python 3.11 available and set
            if ! pyenv local 2>/dev/null | grep -q "3.11"; then
                if pyenv versions | grep -q "3.11"; then
                    log "Setting latest Python 3.11 for this project..."
                    LATEST_311=$(pyenv versions | grep '3.11' | tail -1 | sed 's/[* ]//g' | sed 's/(.*$//' | xargs)
                    
                    # Verify the version is actually installed and usable
                    if pyenv local "$LATEST_311" 2>/dev/null && uv run python --version >/dev/null 2>&1; then
                        log "Successfully set Python $LATEST_311 for this project"
                    else
                        warn "Python $LATEST_311 appears to be corrupted or not properly installed"
                        log "Will install a fresh Python 3.11..."
                        # Fall through to install a new Python
                        pyenv local --unset 2>/dev/null || true
                        
                        # Ask about optimization for new Python installation
                        local use_optimization=false
                        if ask_python_optimization; then
                            use_optimization=true
                        fi
                        
                        log "Installing latest Python 3.11..."
                        LATEST_311=$(pyenv install --list | grep -E '^  3\.11\.[0-9]+$' | tail -1 | xargs)
                        log "Installing Python $LATEST_311"
                        if [[ "$use_optimization" == true ]]; then
                            log "Building with optimization flags for better performance (this will take longer)..."
                            if ! PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install $LATEST_311; then
                                error "Failed to install optimized Python $LATEST_311"
                            fi
                        else
                            if ! pyenv install $LATEST_311; then
                                error "Failed to install Python $LATEST_311"
                            fi
                        fi
                        pyenv local $LATEST_311
                    fi
                    
                    # Check if current Python was built with optimizations
                    if python -c "import sys; print('Optimized build detected' if hasattr(sys, 'flags') and any('optimiz' in str(sys.version).lower() for _ in [1]) else 'Standard build detected')"; then
                        log "Current Python build status checked"
                    fi
                else
                    # Ask about optimization for new Python installation
                    local use_optimization=false
                    if ask_python_optimization; then
                        use_optimization=true
                    fi
                    
                    log "Installing latest Python 3.11..."
                    LATEST_311=$(pyenv install --list | grep -E '^  3\.11\.[0-9]+$' | tail -1 | xargs)
                    log "Installing Python $LATEST_311"
                    if [[ "$use_optimization" == true ]]; then
                        log "Building with optimization flags for better performance (this will take longer)..."
                        if ! PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install $LATEST_311; then
                            error "Failed to install optimized Python $LATEST_311"
                        fi
                    else
                        if ! pyenv install $LATEST_311; then
                            error "Failed to install Python $LATEST_311"
                        fi
                    fi
                    pyenv local $LATEST_311
                fi
            else
                # Already have 3.11 set, check optimization status
                log "Python 3.11 already set for this project"
                python -c "
import sys
import sysconfig
config = sysconfig.get_config_vars()
opts = config.get('CONFIG_ARGS', '')
if '--enable-optimizations' in opts:
    print('✓ Current Python was built with optimization flags')
else:
    print('ℹ Current Python was built without optimization flags')
    print('  To rebuild with optimizations: pyenv uninstall $(pyenv version --bare) && ./deploy.sh experimance install dev')
"
            fi
            
            if ! uv sync; then
                error "Failed to install project dependencies"
            fi
            
            # Ensure shell profile is updated for persistent access
            update_shell_profile
            
            log "Dependencies installed for development with Python 3.11"
        else
            log "Installing pyenv and Python 3.11 for development..."
            
            # Install pyenv if not present
            if ! command -v pyenv >/dev/null 2>&1; then
                # Check if pyenv directory exists but not in PATH
                if [[ -d "$HOME/.pyenv" ]]; then
                    log "pyenv directory exists, initializing..."
                    export PYENV_ROOT="$HOME/.pyenv"
                    export PATH="$PYENV_ROOT/bin:$PATH"
                    eval "$(pyenv init --path)" 2>/dev/null || true
                    eval "$(pyenv init -)" 2>/dev/null || true
                else
                    log "Installing pyenv..."
                    if ! curl https://pyenv.run | bash; then
                        error "Failed to install pyenv"
                    fi
                    
                    # Add pyenv to PATH for this session
                    export PYENV_ROOT="$HOME/.pyenv"
                    export PATH="$PYENV_ROOT/bin:$PATH"
                    eval "$(pyenv init --path)"
                    eval "$(pyenv init -)"
                fi
                
                log "pyenv initialized. Note: You may need to restart your shell or run:"
                log "  export PATH=\"\$HOME/.pyenv/bin:\$PATH\""
                log "  eval \"\$(pyenv init --path)\""
                log "  eval \"\$(pyenv init -)\""
            else
                # Initialize pyenv if already installed
                export PYENV_ROOT="$HOME/.pyenv"
                export PATH="$PYENV_ROOT/bin:$PATH"
                eval "$(pyenv init --path)"
                eval "$(pyenv init -)"
            fi
            
            # Install Python 3.11
            if ! pyenv versions | grep -q "3.11"; then
                # Ask about optimization for new Python installation
                local use_optimization=false
                if ask_python_optimization; then
                    use_optimization=true
                fi
                
                log "Installing latest Python 3.11..."
                LATEST_311=$(pyenv install --list | grep -E '^  3\.11\.[0-9]+$' | tail -1 | xargs)
                log "Installing Python $LATEST_311"
                if [[ "$use_optimization" == true ]]; then
                    log "Building with optimization flags for better performance (this will take longer)..."
                    if ! PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install $LATEST_311; then
                        error "Failed to install optimized Python $LATEST_311"
                    fi
                else
                    if ! pyenv install $LATEST_311; then
                        error "Failed to install Python $LATEST_311"
                    fi
                fi
            else
                # Check if we have a working Python 3.11 installation
                LATEST_311=$(pyenv versions | grep '3.11' | tail -1 | sed 's/[* ]//g' | sed 's/(.*$//' | xargs)
                log "Found Python 3.11: $LATEST_311"
                
                # Verify the version is actually installed and usable
                cd "$REPO_DIR"
                if pyenv local "$LATEST_311" 2>/dev/null && uv run python --version >/dev/null 2>&1; then
                    log "Python 3.11 already available and working: $LATEST_311"
                else
                    warn "Found Python $LATEST_311 but it appears corrupted or not properly installed"
                    log "Will install a fresh Python 3.11..."
                    pyenv local --unset 2>/dev/null || true
                    
                    # Ask about optimization for new Python installation
                    local use_optimization=false
                    if ask_python_optimization; then
                        use_optimization=true
                    fi
                    
                    log "Installing latest Python 3.11..."
                    LATEST_311=$(pyenv install --list | grep -E '^  3\.11\.[0-9]+$' | tail -1 | xargs)
                    log "Installing Python $LATEST_311"
                    if [[ "$use_optimization" == true ]]; then
                        log "Building with optimization flags for better performance (this will take longer)..."
                        if ! PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install $LATEST_311; then
                            error "Failed to install optimized Python $LATEST_311"
                        fi
                    else
                        if ! pyenv install $LATEST_311; then
                            error "Failed to install Python $LATEST_311"
                        fi
                    fi
                    pyenv local $LATEST_311
                fi
            fi
            
            # Set latest Python 3.11 for this project
            cd "$REPO_DIR"
            pyenv local $LATEST_311
            
            # Install uv if not present
            if ! command -v uv >/dev/null 2>&1; then
                log "Installing uv for development..."
                if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
                    error "Failed to install uv"
                fi
                
                # Add uv to PATH for this session
                export PATH="$HOME/.local/bin:$PATH"
            fi
            
            # Verify uv is available (try both PATH and direct path)
            if command -v uv >/dev/null 2>&1; then
                log "uv successfully installed and available in PATH"
            elif [[ -x "$HOME/.local/bin/uv" ]]; then
                log "uv installed, using direct path"
                export PATH="$HOME/.local/bin:$PATH"
            else
                error "uv installation failed or not found. Expected at: $HOME/.local/bin/uv"
            fi
            
            # Update shell profile for persistent access
            update_shell_profile
            
            cd "$REPO_DIR"
            if ! uv sync; then
                error "Failed to install project dependencies"
            fi
            log "Dependencies installed for development with Python 3.11"
        fi
    fi
}

start_services() {
    if [[ "$USE_SYSTEMD" != true ]]; then
        warn "Development mode: Services not started via system service manager"
        warn "Use './scripts/dev <service>' to run individual services in development"
        return
    fi
    
    case "$PLATFORM" in
        linux)
            start_services_linux
            ;;
        macos)
            start_services_macos
            ;;
        *)
            error "Service management not supported on platform: $PLATFORM"
            ;;
    esac
}

start_services_linux() {
    log "Starting services for project $PROJECT..."
    
    # First, reload systemd configuration to pick up any changes
    log "Reloading systemd daemon..."
    systemctl daemon-reload
    
    # Start individual services first
    for service in "${SERVICES[@]}"; do
        # service is in format "service_type@project", e.g., "core@experimance"
        local full_service_name="$service.service"
        log "Processing service: $full_service_name"
        
        # Check if service template exists (e.g., core@.service)
        local service_type="${service%@*}"
        local template_file="$SYSTEMD_DIR/${service_type}@.service"
        
        if [ -f "$template_file" ]; then
            log "Service template found: $template_file"
            
            # Enable the service instance if not already enabled
            if ! systemctl is-enabled "$full_service_name" &>/dev/null; then
                log "Enabling $full_service_name..."
                systemctl enable "$full_service_name"
            else
                log "$full_service_name is already enabled"
            fi
            
            # Start the service instance
            if systemctl is-active "$full_service_name" &>/dev/null; then
                log "$full_service_name is already active"
            else
                log "Starting $full_service_name..."
                if systemctl start "$full_service_name"; then
                    log "✓ Started $full_service_name"
                    
                    # Wait a moment and check if it's actually running
                    sleep 1
                    if systemctl is-active "$full_service_name" &>/dev/null; then
                        log "✓ Confirmed $full_service_name is active"
                    else
                        warn "⚠ $full_service_name started but is not active. Checking status..."
                        systemctl status "$full_service_name" --no-pager -l || true
                    fi
                else
                    error "✗ Failed to start $full_service_name"
                    log "Service status:"
                    systemctl status "$full_service_name" --no-pager -l || true
                fi
            fi
        else
            error "Service template file not found: $template_file"
        fi
    done
    
    # Now start the target
    local target="experimance@${PROJECT}.target"
    log "Processing target: $target"
    
    # Check if the target template exists and if systemd recognizes the instance
    if [ -f "$SYSTEMD_DIR/experimance@.target" ]; then
        log "Target template found: experimance@.target"
        
        # Enable the target instance if not already enabled
        if ! systemctl is-enabled "$target" &>/dev/null; then
            log "Enabling $target..."
            systemctl enable "$target"
        else
            log "$target is already enabled"
        fi
        
        # Start the target instance
        if systemctl is-active "$target" &>/dev/null; then
            log "$target is already active"
        else
            log "Starting $target..."
            if systemctl start "$target"; then
                log "✓ Started $target"
                
                # Wait a moment and verify
                sleep 2
                if systemctl is-active "$target" &>/dev/null; then
                    log "✓ Confirmed $target is active"
                else
                    warn "⚠ $target started but is not active. Checking status..."
                    systemctl status "$target" --no-pager -l || true
                fi
            else
                error "✗ Failed to start $target"
                log "Target status:"
                systemctl status "$target" --no-pager -l || true
            fi
        fi
    else
        error "Target template file not found: $SYSTEMD_DIR/experimance@.target"
    fi
    
    log "Start operation complete"
}

start_services_macos() {
    log "Starting services for project $PROJECT using launchd..."
    
    # On macOS, always use LaunchAgents (user-level), never LaunchDaemons
    local launchd_dir="$HOME/Library/LaunchAgents"
    
    # Start individual services
    for service in "${SERVICES[@]}"; do
        local service_with_project="${service%@*}"  # e.g., "agent@fire" -> "agent"
        local project="${service#*@}"                # e.g., "agent@fire" -> "fire"
        local service_type="$service_with_project"  # e.g., "agent"
        local plist_file="$launchd_dir/com.experimance.${project}.${service_type}.plist"
        local service_label="com.experimance.${service_type}.$project"
        
        if [[ -f "$plist_file" ]]; then
            log "Processing service: $service_label"
            
            # Check if already loaded
            if launchctl list | grep -q "$service_label"; then
                log "$service_label is already loaded"
                # Try to start if not running
                if ! launchctl start "$service_label" 2>/dev/null; then
                    log "$service_label may already be running"
                fi
            else
                log "Loading and starting $service_label..."
                if launchctl bootstrap gui/$(id -u) "$plist_file"; then
                    log "✓ Loaded $service_label"
                    sleep 1
                    # Verify it's running
                    if launchctl list | grep -q "$service_label"; then
                        log "✓ Confirmed $service_label is active"
                    else
                        warn "⚠ $service_label loaded but not found in active services"
                    fi
                else
                    error "✗ Failed to load $service_label"
                fi
            fi
        else
            error "Service plist file not found: $plist_file"
        fi
    done
    
    log "Start operation complete"
}

stop_services() {
    if [[ "$USE_SYSTEMD" != true ]]; then
        warn "Development mode: No system services to stop"
        return
    fi
    
    case "$PLATFORM" in
        linux)
            stop_services_linux
            ;;
        macos)
            stop_services_macos
            ;;
        *)
            error "Service management not supported on platform: $PLATFORM"
            ;;
    esac
}

stop_services_linux() {
    log "Stopping services for project $PROJECT..."
    
    # Check if target exists and is loaded
    local target="experimance@${PROJECT}.target"
    if systemctl list-units --all "$target" | grep -q "$target"; then
        log "Target $target exists"
        if systemctl is-active "$target" &>/dev/null; then
            log "Stopping active target: $target"
            systemctl stop "$target"
            
            # Wait for it to actually stop
            local timeout=10
            local count=0
            while systemctl is-active "$target" &>/dev/null && [ $count -lt $timeout ]; do
                log "Waiting for $target to stop... ($count/$timeout)"
                sleep 1
                count=$((count + 1))
            done
            
            if systemctl is-active "$target" &>/dev/null; then
                warn "$target did not stop within $timeout seconds"
            else
                log "✓ Stopped $target"
            fi
        else
            log "Target $target is not active (state: $(systemctl is-active "$target" 2>/dev/null || echo "unknown"))"
        fi
    else
        warn "Target $target not found or not loaded"
        log "Available targets matching 'experimance':"
        systemctl list-units --all | grep experimance || log "  No experimance units found"
    fi
    
    # Stop individual services
    log "Stopping individual services..."
    for service in "${SERVICES[@]}"; do
        # service is in format "service_type@project", e.g., "core@experimance"
        local full_service_name="$service.service"
        
        # Check if service template exists
        local service_type="${service%@*}"
        local template_file="$SYSTEMD_DIR/${service_type}@.service"
        
        if [ -f "$template_file" ]; then
            if systemctl is-active "$full_service_name" &>/dev/null; then
                log "Stopping $full_service_name..."
                systemctl stop "$full_service_name"
                
                # Wait for it to stop
                local timeout=5
                local count=0
                while systemctl is-active "$full_service_name" &>/dev/null && [ $count -lt $timeout ]; do
                    sleep 1
                    count=$((count + 1))
                done
                
                if systemctl is-active "$full_service_name" &>/dev/null; then
                    warn "$full_service_name did not stop within $timeout seconds"
                else
                    log "✓ Stopped $full_service_name"
                fi
            else
                log "$full_service_name is not active (state: $(systemctl is-active "$full_service_name" 2>/dev/null || echo "unknown"))"
            fi
        else
            warn "Service template not found: $template_file"
        fi
    done
    
    log "Stop operation complete"
}

stop_services_macos() {
    log "Stopping services for project $PROJECT using launchd..."
    
    # On macOS, always use LaunchAgents (user-level), never LaunchDaemons
    local launchd_dir="$HOME/Library/LaunchAgents"
    
    # Stop individual services
    for service in "${SERVICES[@]}"; do
        local service_with_project="${service%@*}"  # e.g., "agent@fire" -> "agent"
        local project="${service#*@}"                # e.g., "agent@fire" -> "fire"
        local service_type="$service_with_project"  # e.g., "agent"
        local plist_file="$launchd_dir/com.experimance.${project}.${service_type}.plist"
        local service_label="com.experimance.${service_type}.$project"
        
        if [[ -f "$plist_file" ]]; then
            log "Processing service: $service_label"
            
            # Check if loaded
            if launchctl list | grep -q "$service_label"; then
                log "Unloading $service_label..."
                if launchctl bootout gui/$(id -u) "$plist_file"; then
                    log "✓ Unloaded $service_label"
                    
                    # Wait and verify it's gone
                    sleep 1
                    if ! launchctl list | grep -q "$service_label"; then
                        log "✓ Confirmed $service_label is stopped"
                    else
                        warn "⚠ $service_label may still be running"
                    fi
                else
                    warn "Failed to unload $service_label"
                fi
            else
                log "$service_label is not loaded"
            fi
        else
            warn "Service plist file not found: $plist_file"
        fi
    done
    
    log "Stop operation complete"
}

restart_services() {
    log "Restarting services for project $PROJECT..."
    stop_services
    sleep 2
    start_services
}

status_services() {
    log "Checking status of services for project $PROJECT..."
    
    case "$PLATFORM" in
        linux)
            status_services_linux
            ;;
        macos)
            status_services_macos
            ;;
        *)
            warn "Service status checking not supported on platform: $PLATFORM"
            ;;
    esac
}

status_services_linux() {
    echo -e "\n${BLUE}=== Systemd Template Files Status ===${NC}"
    
    # Check if template files exist (these are the actual files we install)
    local target="experimance@${PROJECT}.target"
    if [ -f "$SYSTEMD_DIR/experimance@.target" ]; then
        echo -e "${GREEN}✓${NC} Target template exists: experimance@.target"
    else
        echo -e "${RED}✗${NC} Target template missing: experimance@.target"
    fi
    
    for service in "${SERVICES[@]}"; do
        # Extract service type from service@project format (e.g., "core" from "core@experimance")
        local service_type="${service%@*}"
        local template_file="${service_type}@.service"
        if [ -f "$SYSTEMD_DIR/$template_file" ]; then
            echo -e "${GREEN}✓${NC} Service template exists: $template_file"
        else
            echo -e "${RED}✗${NC} Service template missing: $template_file"
        fi
    done
    
    echo -e "\n${BLUE}=== Service Instance Status ===${NC}"
    echo "Note: Instances are created from templates when services start"
    for service in "${SERVICES[@]}"; do
        local full_service_name="$service.service"
        local status=$(systemctl is-active "$full_service_name" 2>/dev/null || echo "not-found")
        local enabled=$(systemctl is-enabled "$full_service_name" 2>/dev/null || echo "not-found")
        
        case $status in
            active)
                echo -e "${GREEN}✓${NC} $full_service_name: $status (enabled: $enabled)"
                ;;
            inactive)
                echo -e "${YELLOW}○${NC} $full_service_name: $status (enabled: $enabled)"
                ;;
            failed)
                echo -e "${RED}✗${NC} $full_service_name: $status (enabled: $enabled)"
                ;;
            *)
                echo -e "${RED}?${NC} $full_service_name: $status (enabled: $enabled)"
                ;;
        esac
    done
    
    echo -e "\n${BLUE}=== Target Status ===${NC}"
    local status=$(systemctl is-active "$target" 2>/dev/null || echo "not-found")
    local enabled=$(systemctl is-enabled "$target" 2>/dev/null || echo "not-found")
    
    case $status in
        active)
            echo -e "${GREEN}✓${NC} $target: $status (enabled: $enabled)"
            ;;
        inactive)
            echo -e "${YELLOW}○${NC} $target: $status (enabled: $enabled)"
            ;;
        failed)
            echo -e "${RED}✗${NC} $target: $status (enabled: $enabled)"
            ;;
        *)
            echo -e "${RED}?${NC} $target: $status (enabled: $enabled)"
            ;;
    esac
    
    # Show failed services details
    echo -e "\n${BLUE}=== Failed Services Details ===${NC}"
    local any_failed=false
    for service in "${SERVICES[@]}"; do
        local full_service_name="$service.service"
        if systemctl is-failed "$full_service_name" &>/dev/null; then
            any_failed=true
            echo -e "${RED}Failed service: $full_service_name${NC}"
            systemctl status "$full_service_name" --no-pager -l | head -10
            echo ""
        fi
    done
    
    if systemctl is-failed "$target" &>/dev/null; then
        any_failed=true
        echo -e "${RED}Failed target: $target${NC}"
        systemctl status "$target" --no-pager -l | head -10
        echo ""
    fi
    
    if [ "$any_failed" = false ]; then
        echo "No failed services"
    fi
    
    echo -e "\n${BLUE}=== Recent Logs ===${NC}"
    # Look for core service logs using the correct service instance name
    local core_service="core@${PROJECT}.service"
    if systemctl list-units --all | grep -q "$core_service"; then
        echo "Logs for $core_service:"
        journalctl --no-pager -n 10 -u "$core_service" || true
    else
        log "No core service logs available for $core_service"
    fi
}

status_services_macos() {
    echo -e "\n${BLUE}=== Launchd Service Files Status ===${NC}"
    
    # On macOS, always use LaunchAgents (user-level), never LaunchDaemons
    local launchd_dir="$HOME/Library/LaunchAgents"
    
    # Check if plist files exist
    for service in "${SERVICES[@]}"; do
        local service_with_project="${service%@*}"  # e.g., "agent@fire" -> "agent"
        local project="${service#*@}"                # e.g., "agent@fire" -> "fire"
        local service_type="$service_with_project"  # e.g., "agent"
        local plist_file="$launchd_dir/com.experimance.${project}.${service_type}.plist"
        local service_label="com.experimance.${service_type}.$project"
        
        if [[ -f "$plist_file" ]]; then
            echo -e "${GREEN}✓${NC} Service plist exists: com.experimance.${project}.${service_type}.plist"
        else
            echo -e "${RED}✗${NC} Service plist missing: com.experimance.${project}.${service_type}.plist"
        fi
    done
    
    echo -e "\n${BLUE}=== Service Status ===${NC}"
    for service in "${SERVICES[@]}"; do
        local service_with_project="${service%@*}"  # e.g., "agent@fire" -> "agent"
        local project="${service#*@}"                # e.g., "agent@fire" -> "fire"
        local service_type="$service_with_project"  # e.g., "agent"
        local service_label="com.experimance.${service_type}.$project"
        
        if launchctl list | grep -q "$service_label"; then
            # Service is loaded, check its status
            local pid=$(launchctl list | grep "$service_label" | awk '{print $1}')
            local exit_code=$(launchctl list | grep "$service_label" | awk '{print $2}')
            
            if [[ "$pid" != "-" ]]; then
                echo -e "${GREEN}✓${NC} $service_label: running (PID: $pid)"
            elif [[ "$exit_code" == "0" ]]; then
                echo -e "${YELLOW}○${NC} $service_label: loaded but not running (last exit: $exit_code)"
            else
                echo -e "${RED}✗${NC} $service_label: failed (exit code: $exit_code)"
            fi
        else
            echo -e "${RED}?${NC} $service_label: not loaded"
        fi
    done
    
    echo -e "\n${BLUE}=== Recent Logs ===${NC}"
    echo "Check LaunchAgent logs:"
    echo "  tail -f ~/Library/Logs/experimance/${PROJECT}_*_launchd_error.log"
    echo "  ls -la ~/Library/Logs/experimance/*launchd*.log"
}

# Scheduling functions
setup_schedule() {
    local start_schedule="$1"
    local stop_schedule="$2"
    
    log "Setting up schedule..."
    log "  Start: $start_schedule"
    log "  Stop: $stop_schedule"
    
    # Create crontab entries
    local temp_cron=$(mktemp)
    local startup_script="$SCRIPT_DIR/reset.sh"    # Use reset.sh for startup (includes audio reset)
    local shutdown_script="$SCRIPT_DIR/shutdown.sh"
    
    # Get existing crontab (excluding our entries)
    crontab -l 2>/dev/null | grep -v "experimance" > "$temp_cron" || true
    
    # Add startup schedule
    echo "# experimance-start" >> "$temp_cron"
    echo "$start_schedule cd '$REPO_DIR' && '$startup_script' --project '$PROJECT' >/dev/null 2>&1" >> "$temp_cron"
    
    # Add shutdown schedule  
    echo "# experimance-stop" >> "$temp_cron"
    echo "$stop_schedule cd '$REPO_DIR' && '$shutdown_script' --project '$PROJECT' >/dev/null 2>&1" >> "$temp_cron"
    
    # Install new crontab
    crontab "$temp_cron"
    rm "$temp_cron"
    
    log "✓ Schedule installed successfully"
}

show_schedule() {
    log "Current schedule:"
    local schedule_found=false
    
    if crontab -l 2>/dev/null | grep -q "experimance"; then
        crontab -l | grep "experimance" -A1 -B0
        schedule_found=true
    fi
    
    if [ "$schedule_found" = false ]; then
        log "No schedule currently set"
    fi
}

remove_schedule() {
    log "Removing schedule..."
    
    # Remove our crontab entries
    local temp_cron=$(mktemp)
    crontab -l 2>/dev/null | grep -v "experimance" > "$temp_cron" || true
    crontab "$temp_cron"
    rm "$temp_cron"
    
    log "✓ Schedule removed successfully"
}

# Preset schedules for common use cases
setup_gallery_schedule() {
    # Gallery hours: Monday-Friday, 12PM-5PM
    local start_schedule="50 11 * * 1-5"  # Start at 11:50 AM, Monday-Friday
    local stop_schedule="10 17 * * 1-5"   # Stop at 5:10 PM, Monday-Friday

    log "Setting up gallery schedule (Monday-Friday, 12PM-5PM)..."
    setup_schedule "$start_schedule" "$stop_schedule"
}

main() {
    case "$ACTION" in
        install)
            if [[ "$MODE" == "dev" ]]; then
                log "=== DEVELOPMENT INSTALL ==="
                log "This will set up the project for development testing only"
                log "No system services will be installed"
                
                # Warn if running dev mode with sudo
                if [[ "$EUID" -eq 0 ]]; then
                    warn "Development mode does not require root privileges!"
                    if [[ -n "${SUDO_USER:-}" ]]; then
                        warn "Consider running without sudo: ./infra/scripts/deploy.sh $PROJECT install dev"
                        warn "Continuing with Homebrew running as user: $SUDO_USER"
                    else
                        warn "Running as root user. This may cause issues with Homebrew on macOS."
                    fi
                fi
            elif [[ "$MODE" == "prod" ]]; then
                log "=== PRODUCTION INSTALL ==="
                log "This will set up the project for production deployment"
                if [[ "$PLATFORM" == "linux" ]]; then
                    log "Systemd services will be installed and can be managed"
                    check_root
                elif [[ "$PLATFORM" == "macos" ]]; then
                    log "Launchd services will be installed and can be managed"
                    if [[ "$USE_SYSTEMD" == true ]] && [[ $EUID -ne 0 ]]; then
                        error "Production mode on macOS requires root privileges for LaunchDaemons. Run with sudo."
                    fi
                fi
            fi
            
            check_user
            install_dependencies
            # Re-populate SERVICES array after dependencies are installed
            load_services_array "$PROJECT"
            check_project
            update_health_config
            install_systemd_files
            if [[ "$PLATFORM" == "linux" ]]; then
                install_reset_on_input
            fi
            if [[ "$PLATFORM" == "linux" ]]; then
                create_symlink "$(pwd)"
            fi
            setup_directories

            # Group management for hardware access (Linux only)
            if [[ "$PLATFORM" == "linux" ]]; then
                # add user to groups as needed
                local group_added=false
                # add user to video group if not already a member
                if ! id -nG "$RUNTIME_USER" | grep -qw "video"; then
                    log "Adding $RUNTIME_USER to video group for webcam access"
                    if [[ "$MODE" == "prod" ]]; then
                        usermod -aG video "$RUNTIME_USER"
                    else
                        sudo usermod -aG video "$RUNTIME_USER"
                    fi
                    group_added=true
                else
                    log "$RUNTIME_USER is already a member of the video group"
                fi
                
                # add user to audio group if not already a member
                if ! id -nG "$RUNTIME_USER" | grep -qw "audio"; then
                    log "Adding $RUNTIME_USER to audio group for audio device access"
                    if [[ "$MODE" == "prod" ]]; then
                        usermod -aG audio "$RUNTIME_USER"
                    else
                        sudo usermod -aG audio "$RUNTIME_USER"
                    fi
                    group_added=true
                else
                    log "$RUNTIME_USER is already a member of the audio group"
                fi

                # add user to input group if not already a member
                if ! id -nG "$RUNTIME_USER" | grep -qw "input"; then
                    log "Adding $RUNTIME_USER to input group for input device access"
                    if [[ "$MODE" == "prod" ]]; then
                        usermod -aG input "$RUNTIME_USER"
                    else
                        sudo usermod -aG input "$RUNTIME_USER"
                    fi
                    group_added=true
                else
                    log "$RUNTIME_USER is already a member of the input group"
                fi
                
                # Inform user about group membership activation if groups were added
                if [[ "$group_added" == true ]] && [[ "$MODE" == "dev" ]]; then
                    echo ""
                    warn "New group membership added. To activate group access:"
                    warn "  Option 1: Log out and log back in (recommended)"
                    warn "  Option 2: Run 'newgrp video', 'newgrp audio', and/or 'newgrp input' to start a new shell with groups active"
                    warn "  Option 3: Restart your terminal session"
                    echo ""
                fi
            else
                log "Device access permissions are managed by macOS system settings"
            fi
            
            if [[ "$MODE" == "dev" ]]; then
                log "Development installation complete!"
                log ""
                log "IMPORTANT: To use Python and uv in new terminal sessions, run:"
                log "  source ~/.bashrc"
                log "Or restart your terminal."
                log ""
                log "To verify installation:"
                log "  python --version    # Should show Python 3.11.x"
                log "  uv --version        # Should show uv version"
                log ""
                log "To test services: Use './scripts/dev <service>'"
                log "To install for production: sudo ./deploy.sh $PROJECT install prod"
            else
                log "Production installation complete!"
                log "To start services: sudo ./deploy.sh $PROJECT start"
                log "You may want to run 'sudo systemctl daemon-reload' if you made changes to systemd files"
            fi
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
        reset-on-input-start)
            check_root
            log "Starting reset on input listener..."
            systemctl start reset-on-input.service
            systemctl status reset-on-input.service
            ;;
        reset-on-input-stop)
            check_root
            log "Stopping reset on input listener..."
            systemctl stop reset-on-input.service
            ;;
        reset-on-input-status)
            log "Reset on input listener status:"
            systemctl status reset-on-input.service || true
            echo ""
            log "Recent logs:"
            journalctl --no-pager -n 20 -u reset-on-input.service || true
            ;;
        reset-on-input-logs)
            log "Following reset on input logs (Ctrl+C to exit):"
            journalctl -f -u reset-on-input.service
            ;;
        reset-on-input-test)
            log "Testing reset on input device detection..."
            log "This will run the listener script manually for testing"
            log "Note: Using --test-mode and --bypass-ssh-check for safe testing"
            python3 "$REPO_DIR/infra/scripts/reset_on_input.py" --test-mode --bypass-ssh-check
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
        schedule-gallery)
            log "Setting up gallery schedule (Monday-Friday, 12PM-5PM)"
            setup_gallery_schedule
            show_schedule
            ;;
        schedule-custom)
            # Expect start and stop schedules as additional arguments
            local start_schedule="${3:-}"
            local stop_schedule="${4:-}"
            
            if [[ -z "$start_schedule" || -z "$stop_schedule" ]]; then
                error "Custom schedule requires start and stop cron expressions"
                error "Usage: $0 $PROJECT schedule-custom 'start-cron' 'stop-cron'"
                error "Example: $0 $PROJECT schedule-custom '0 8 * * 1-5' '0 20 * * 1-5'"
                exit 1
            fi
            
            log "Setting up custom schedule"
            setup_schedule "$start_schedule" "$stop_schedule"
            show_schedule
            ;;
        schedule-reset)
            # Schedule a one-time reset for a specific time (uses reset.sh which does an audio reset)
            local reset_time="${3:-}"
            if [[ -z "$reset_time" ]]; then
                error "Reset time required in format 'HH:MM' (24-hour format)"
                error "Usage: $0 $PROJECT schedule-reset 'HH:MM'"
                error "Example: $0 $PROJECT schedule-reset '12:00'"
                exit 1
            fi
            
            # Parse the time and create a cron expression for today
            local hour="${reset_time%:*}"
            local minute="${reset_time#*:}"
            local today=$(date '+%d')
            local month=$(date '+%m')
            local reset_cron="$minute $hour $today $month *"

            log "Scheduling one-time reset for today at $reset_time"
            log "Cron expression: $reset_cron"

            # Create a temporary cron entry that will be removed after execution
            local temp_cron=$(mktemp)
            crontab -l 2>/dev/null | grep -v "experimance-onetime-reset" > "$temp_cron" || true

            echo "# experimance-onetime-reset" >> "$temp_cron"
            echo "$reset_cron cd '$REPO_DIR' && '$SCRIPT_DIR/reset.sh' --project '$PROJECT' && (crontab -l 2>/dev/null | grep -v 'experimance-onetime-reset' | crontab -) >/dev/null 2>&1" >> "$temp_cron"

            crontab "$temp_cron"
            rm "$temp_cron"

            log "✓ Scheduled one-time reset for $reset_time today"
            log "The cron job will automatically remove itself after execution"
            log "Note: Using reset.sh which will reset service instances"
            show_schedule
            ;;
        schedule-shutdown)
            # Schedule a one-time shutdown for a specific time (uses shutdown.sh which destroys instances)
            local stop_time="${3:-}"
            if [[ -z "$stop_time" ]]; then
                error "Stop time required in format 'HH:MM' (24-hour format)"
                error "Usage: $0 $PROJECT schedule-shutdown 'HH:MM'"
                error "Example: $0 $PROJECT schedule-shutdown '12:00'"
                exit 1
            fi
            
            # Parse the time and create a cron expression for today
            local hour="${stop_time%:*}"
            local minute="${stop_time#*:}"
            local today=$(date '+%d')
            local month=$(date '+%m')
            local stop_cron="$minute $hour $today $month *"
            
            log "Scheduling one-time stop for today at $stop_time"
            log "Cron expression: $stop_cron"
            
            # Create a temporary cron entry that will be removed after execution
            local temp_cron=$(mktemp)
            crontab -l 2>/dev/null | grep -v "experimance-onetime-shutdown" > "$temp_cron" || true
            
            echo "# experimance-onetime-shutdown" >> "$temp_cron"
            echo "$stop_cron cd '$REPO_DIR' && '$SCRIPT_DIR/shutdown.sh' --project '$PROJECT' && (crontab -l 2>/dev/null | grep -v 'experimance-onetime-shutdown' | crontab -) >/dev/null 2>&1" >> "$temp_cron"
            
            crontab "$temp_cron"
            rm "$temp_cron"
            
            log "✓ Scheduled one-time shutdown for $stop_time today"
            log "The cron job will automatically remove itself after execution"
            log "Note: Using shutdown.sh which will destroy service instances"
            show_schedule
            ;;
        schedule-remove)
            log "Removing schedule"
            remove_schedule
            show_schedule
            ;;
        schedule-show)
            show_schedule
            ;;
        diagnose)
            log "Running systemd diagnostics for project $PROJECT..."
            
            echo -e "\n${BLUE}=== System Information ===${NC}"
            echo "User: $(whoami)"
            echo "Project: $PROJECT"
            echo "Mode: $MODE"
            echo "Use Systemd: $USE_SYSTEMD"
            echo "Runtime User: $RUNTIME_USER"
            
            echo -e "\n${BLUE}=== Systemd Files ===${NC}"
            echo "Systemd directory: $SYSTEMD_DIR"
            local file_count=$(ls -1 "$SYSTEMD_DIR" | grep experimance | wc -l)
            echo "Experimance systemd files installed: $file_count"
            
            echo -e "\n${BLUE}=== Unit Files Check ===${NC}"
            local target="experimance@${PROJECT}.target"
            
            echo "Target template file: $SYSTEMD_DIR/experimance@.target"
            if [ -f "$SYSTEMD_DIR/experimance@.target" ]; then
                echo -e "${GREEN}✓${NC} Target template file exists"
            else
                echo -e "${RED}✗${NC} Target template file missing"
            fi
            
            echo -e "\n${BLUE}=== Service Template Files ===${NC}"
            for service in "${SERVICES[@]}"; do
                # Extract service type from service@project format (e.g., "core" from "core@experimance")
                local service_type="${service%@*}"
                local template_file="$SYSTEMD_DIR/${service_type}@.service"
                echo "Service template: $template_file"
                if [ -f "$template_file" ]; then
                    echo -e "${GREEN}✓${NC} Template exists"
                    echo "  Permissions: $(ls -la "$template_file")"
                else
                    echo -e "${RED}✗${NC} Template missing"
                fi
            done
            
            echo -e "\n${BLUE}=== Service Instance Status ===${NC}"
            echo "Note: systemd creates instances from templates when you start them"
            for service in "${SERVICES[@]}"; do
                # service is in format "service_type@project", e.g., "core@experimance"
                local full_service_name="$service.service"
                echo "Instance: $full_service_name"
                
                # Check enabled status (this is the key check for instances)
                local enabled_status=$(systemctl is-enabled "$full_service_name" 2>/dev/null || echo "not-found")
                local active_status=$(systemctl is-active "$full_service_name" 2>/dev/null || echo "inactive")
                
                case $enabled_status in
                    enabled)
                        echo -e "${GREEN}✓${NC} Enabled (active: $active_status)"
                        ;;
                    disabled)
                        echo -e "${YELLOW}○${NC} Disabled (active: $active_status)"
                        ;;
                    not-found)
                        echo -e "${RED}?${NC} Not found (never created)"
                        ;;
                    *)
                        echo -e "${YELLOW}?${NC} Status: $enabled_status (active: $active_status)"
                        ;;
                esac
            done
            
            echo -e "\n${BLUE}=== Systemd Daemon Status ===${NC}"
            systemctl status --no-pager | head -5
            
            echo -e "\n${BLUE}=== All Experimance Units ===${NC}"
            echo "Loaded units:"
            systemctl list-units --all | grep experimance || echo "  No experimance units found"
            
            echo -e "\n${BLUE}=== Unit File Status ===${NC}"
            local unit_count=$(systemctl list-unit-files | grep experimance | wc -l)
            echo "Experimance unit files registered: $unit_count"
            
            if systemctl list-unit-files | grep -q "$target"; then
                echo -e "\n${BLUE}=== Target Dependencies ===${NC}"
                echo "Dependencies for $target:"
                if systemctl list-dependencies "$target"; then
                    echo ""
                    echo -e "${BLUE}=== Service Linking Analysis ===${NC}"
                    echo "Checking if services are properly linked to target..."
                    
                    # Check each service's enabled status and linking
                    for service in "${SERVICES[@]}"; do
                        local full_service_name="$service.service"
                        local enabled_status=$(systemctl is-enabled "$full_service_name" 2>/dev/null || echo "not-found")
                        echo "Service: $full_service_name"
                        echo "  Enabled status: $enabled_status"
                        
                        # Check if service appears in target dependencies
                        if systemctl list-dependencies "$target" 2>/dev/null | grep -q "$full_service_name"; then
                            echo -e "  ${GREEN}✓${NC} Linked to target"
                        else
                            echo -e "  ${RED}✗${NC} NOT linked to target"
                            echo "  To fix: sudo systemctl enable $full_service_name"
                        fi
                        
                        # Check if symlink exists in target.wants directory
                        local wants_dir="/etc/systemd/system/${target}.wants"
                        if [ -L "$wants_dir/$full_service_name" ]; then
                            echo -e "  ${GREEN}✓${NC} Symlink exists in $wants_dir"
                        else
                            echo -e "  ${YELLOW}○${NC} No symlink in $wants_dir"
                        fi
                        echo ""
                    done
                else
                    echo "Could not list dependencies for $target"
                fi
            else
                echo -e "\n${BLUE}=== Target Not Found ===${NC}"
                echo "Target $target is not recognized by systemd"
                echo "This usually means:"
                echo "  1. Target template file is missing"
                echo "  2. systemctl daemon-reload was not run"
                echo "  3. Target was never enabled"
            fi
            ;;
        *)
            error "Unknown action: $ACTION. Use: install, start, stop, restart, status, services, diagnose, schedule-gallery, schedule-demo, schedule-weekend, schedule-custom, schedule-remove, schedule-show"
            ;;
    esac
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [project_name] [action] [mode]"
    echo "Projects: $(ls "$REPO_DIR/projects" 2>/dev/null | tr '\n' ' ')"
    echo "Actions: install, start, stop, restart, status, services, diagnose"
    echo "USB Reset on Input: reset-on-input-start, reset-on-input-stop, reset-on-input-status, reset-on-input-logs, reset-on-input-test"
    echo "Schedule Actions: schedule-gallery, schedule-demo, schedule-weekend, schedule-custom, schedule-shutdown, schedule-remove, schedule-show"
    echo "Modes: dev, prod (only for install action)"
    echo ""
    echo "Install Modes:"
    echo "  dev     # Development setup - no systemd, local directories, current user"
    echo "  prod    # Production setup - systemd services, system directories, experimance user"
    echo ""
    echo "Basic Examples:"
    echo "  $0 experimance services              # Show services (no sudo needed)"
    echo "  $0 experimance install dev           # Development setup (no sudo needed)"
    echo "  sudo $0 experimance install prod     # Production setup (sudo needed)"
    echo "  sudo $0 experimance start            # Start services in production (sudo needed)"
    echo ""
    echo "Schedule Examples:"
    echo "  $0 experimance schedule-gallery      # Monday-Friday, 12PM-5PM"
    echo "  $0 experimance schedule-custom '0 8 * * 1-5' '0 20 * * 1-5'  # Custom: Weekdays 8AM-8PM"
    echo "  $0 experimance schedule-reset '11:30'    # Call shutdown.sh at 11:30 today"
    echo "  $0 experimance schedule-shutdown '12:30' # Call shutdown.sh at 12:30 today"
    echo "  $0 experimance schedule-show         # Show current schedule"
    echo "  $0 experimance schedule-remove       # Remove schedule"
    echo ""
    echo "Development workflow:"
    echo "  $0 experimance install dev           # Setup dependencies and directories"
    echo "  ./scripts/dev <service>              # Run individual services"
    echo ""
    echo "Production workflow:"
    echo "  sudo useradd -m -s /bin/bash experimance    # Create experimance user (first time)"
    echo "  sudo $0 experimance install prod            # Install for production"
    echo "  sudo $0 experimance start                   # Start all services"
    echo ""
    echo "Reset on Input (Gallery Staff):"
    echo "  sudo $0 experimance reset-on-input-start   # Start input listener"
    echo "  sudo $0 experimance reset-on-input-status  # Check listener status"
    echo "  sudo $0 experimance reset-on-input-test    # Test listener manually"
    echo "  sudo $0 experimance reset-on-input-logs    # View listener logs"
    echo ""
    echo "Reset on Input Safety Features:"
    echo "  • Escape sequences: Ctrl+Alt+E or Escape key disables listener for 5 minutes (admin access)"
    echo "  • SSH protection: Only works on local console, not remote sessions"
    echo "  • Physical device priority: Prefers external USB controllers over built-in keyboards"
    echo "  sudo $0 experimance reset-on-input-logs    # View listener logs"
    echo ""
    echo "Note: sudo is only needed for production systemd operations and system setup."
    exit 1
fi

main "$@"
