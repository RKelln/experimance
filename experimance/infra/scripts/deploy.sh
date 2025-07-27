#!/bin/bash

# Experimance Deployment Script
# Usage: ./deploy.sh [project_name] [action] [mode]
# Actions: install, start, stop, restart, status
# Modes: dev, prod (only for install action)
#
# SYSTEMD TEMPLATE SYSTEM:
# This script uses systemd template services for multi-project support:
# - Template files: core@.service, display@.service, experimance@.target (installed to /etc/systemd/system/)
# - Instance services: core@experimance.service, display@sohkepayin.service (created when started)
# - The @ symbol makes it a template, %i gets replaced with project name
# - Multiple projects can share the same templates with different instances

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

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SYSTEMD_DIR="/etc/systemd/system"

# Default values
PROJECT="${1:-experimance}"
ACTION="${2:-install}"
MODE="${3:-}"

# Auto-detect mode if not specified for install action
if [[ "$ACTION" == "install" && -z "$MODE" ]]; then
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

# Determine user and environment based on mode
if [[ "$MODE" == "dev" ]]; then
    # Development mode: use current user, local directories
    RUNTIME_USER="$(whoami)"
    USE_SYSTEMD=false
    warn "Running in DEVELOPMENT mode with user: $RUNTIME_USER"
elif [[ "$MODE" == "prod" ]]; then
    # Production mode: use experimance user, system directories, systemd
    RUNTIME_USER="experimance"
    USE_SYSTEMD=true
    log "Running in PRODUCTION mode with user: $RUNTIME_USER"
else
    # For non-install actions, auto-detect based on environment
    if [[ "${EXPERIMANCE_ENV:-}" == "development" ]]; then
        MODE="dev"
        RUNTIME_USER="$(whoami)"
        USE_SYSTEMD=false
        warn "Running in development mode with user: $RUNTIME_USER"
    elif id experimance &>/dev/null; then
        MODE="prod"
        RUNTIME_USER="experimance"
        USE_SYSTEMD=true
        log "Auto-detected production mode with experimance user"
    else
        error "Cannot determine runtime mode. For install, specify 'dev' or 'prod' mode. For other actions, ensure experimance user exists or set EXPERIMANCE_ENV=development"
    fi
fi

# Get services dynamically for the project
get_project_services() {
    local project="$1"
    local services_script="$SCRIPT_DIR/get_project_services.py"
    
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
            if ! \"\$uv_cmd\" run python '$services_script' '$project'; then
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
        if ! "$uv_cmd" run python "$services_script" "$project"; then
            error "Failed to detect services for project '$project'. Check that the project exists and is properly configured."
        fi
    fi
}

# Services to manage (dynamically determined)
# For install action, we'll populate this after installing dependencies
if [[ "$ACTION" != "install" ]]; then
    readarray -t SERVICES < <(get_project_services "$PROJECT")
else
    # For install action, we'll detect services after dependencies are installed
    SERVICES=()
fi

check_root() {
    if [[ "$USE_SYSTEMD" == true ]] && [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for systemd operations in production mode"
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

# Create symlink for standard installation directory
create_symlink() {
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

install_systemd_files() {
    if [[ "$USE_SYSTEMD" != true ]]; then
        log "Skipping systemd installation in development mode"
        return
    fi
    
    log "Installing systemd service files..."
    
    # Debug: Show what files we're looking for
    log "Looking for service files in: $SCRIPT_DIR/../systemd/"
    ls -la "$SCRIPT_DIR/../systemd/"*.service 2>&1 | while read line; do log "  $line"; done || log "  No service files found"
    
    # Copy all template service files (e.g., core@.service, display@.service)
    # These are TEMPLATES that systemd uses to create instances like core@experimance.service
    local files_copied=0
    for service_file in "$SCRIPT_DIR"/../systemd/*.service; do
        if [[ -f "$service_file" ]]; then
            local basename_file=$(basename "$service_file")
            log "Copying $basename_file..."
            if cp "$service_file" "$SYSTEMD_DIR/"; then
                log "✓ Copied $basename_file"
                files_copied=$((files_copied + 1))
            else
                error "✗ Failed to copy $basename_file"
            fi
        fi
    done
    
    # Copy target template file (e.g., experimance@.target)
    if [[ -f "$SCRIPT_DIR/../systemd/experimance@.target" ]]; then
        log "Copying experimance@.target..."
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
    
    # Create cache directory
    if [[ "$MODE" == "prod" ]]; then
        # Production: use /var/cache/experimance
        mkdir -p /var/cache/experimance
        chown "$RUNTIME_USER:$RUNTIME_USER" /var/cache/experimance
        log "Created production cache directory: /var/cache/experimance"
    else
        # Development: use local cache directory
        mkdir -p "$REPO_DIR/cache"
        chown "$RUNTIME_USER:$RUNTIME_USER" "$REPO_DIR/cache"
        log "Created development cache directory: $REPO_DIR/cache"
    fi
    
    # Create log directory if it doesn't exist
    mkdir -p "$REPO_DIR/logs"
    chown "$RUNTIME_USER:$RUNTIME_USER" "$REPO_DIR/logs"
    
    # Create images directory if it doesn't exist
    mkdir -p "$REPO_DIR/media/images/generated"
    mkdir -p "$REPO_DIR/media/images/mocks"
    mkdir -p "$REPO_DIR/media/video"
    chown "$RUNTIME_USER:$RUNTIME_USER" "$REPO_DIR/media/images"
    chown "$RUNTIME_USER:$RUNTIME_USER" "$REPO_DIR/media/images/generated"
    chown "$RUNTIME_USER:$RUNTIME_USER" "$REPO_DIR/media/images/mocks"
    chown "$RUNTIME_USER:$RUNTIME_USER" "$REPO_DIR/media/video"
    
    # create transcripts directory
    mkdir -p "$REPO_DIR/transcripts"
    chown "$RUNTIME_USER:$RUNTIME_USER" "$REPO_DIR/transcripts"

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
                chown "$RUNTIME_USER:$RUNTIME_USER" "$ZIP_FILE"
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
                    chown "$RUNTIME_USER:$RUNTIME_USER" "$ZIP_FILE"
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
                    chown -R "$RUNTIME_USER:$RUNTIME_USER" "$REPO_DIR/media"
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

# Check if system build dependencies are installed
check_build_dependencies() {
    local missing_deps=()
    
    # Check for essential build tools
    while IFS= read -r tool; do
        case "$tool" in
            "gcc")
                if ! command -v gcc >/dev/null 2>&1; then missing_deps+=("build-essential"); fi
                ;;
            *)
                if ! command -v "$tool" >/dev/null 2>&1; then missing_deps+=("$tool"); fi
                ;;
        esac
    done < <(get_required_packages "build_tools")
    
    # Check each package (skip build tools as they're handled above)
    while IFS= read -r package; do
        if [[ "$package" != "make" && "$package" != "build-essential" ]]; then
            if ! dpkg -l "$package" 2>/dev/null | grep -q "^ii"; then
                missing_deps+=("$package")
            fi
        fi
    done < <(get_required_packages "apt")
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        warn "Missing system build dependencies required for Python compilation:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "The following packages need to be installed:"
        echo "  sudo apt install $(get_required_packages "apt" | tr '\n' ' ')"
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
        install_system_dependencies
    fi
    
    if [[ "$MODE" == "prod" ]]; then
        # Production mode: install uv and dependencies as the experimance user
        log "Installing uv and Python 3.11 for experimance user..."
        sudo -u experimance bash -c "
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
        warn "Development mode: Services not started via systemd"
        warn "Use './scripts/dev <service>' to run individual services in development"
        return
    fi
    
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
            log "Available service templates matching pattern:"
            ls -la "$SYSTEMD_DIR" | grep "@\.service" || log "  No template service files found"
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
        log "Available target templates matching 'experimance':"
        ls -la "$SYSTEMD_DIR" | grep experimance | grep target || log "  No experimance target files found"
    fi
    
    log "Start operation complete"
}

stop_services() {
    if [[ "$USE_SYSTEMD" != true ]]; then
        warn "Development mode: No systemd services to stop"
        return
    fi
    
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

restart_services() {
    log "Restarting services for project $PROJECT..."
    stop_services
    sleep 2
    start_services
}

status_services() {
    log "Checking status of services for project $PROJECT..."
    
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

# Scheduling functions
setup_schedule() {
    local start_schedule="$1"
    local stop_schedule="$2"
    
    log "Setting up schedule..."
    log "  Start: $start_schedule"
    log "  Stop: $stop_schedule"
    
    # Create crontab entries
    local temp_cron=$(mktemp)
    local startup_script="$SCRIPT_DIR/startup.sh"
    local shutdown_script="$SCRIPT_DIR/shutdown.sh"
    
    # Get existing crontab (excluding our entries)
    crontab -l 2>/dev/null | grep -v "experimance-auto" > "$temp_cron" || true
    
    # Add startup schedule
    echo "# experimance-auto-start" >> "$temp_cron"
    echo "$start_schedule $startup_script --project $PROJECT >/dev/null 2>&1" >> "$temp_cron"
    
    # Add shutdown schedule  
    echo "# experimance-auto-stop" >> "$temp_cron"
    echo "$stop_schedule $shutdown_script --project $PROJECT >/dev/null 2>&1" >> "$temp_cron"
    
    # Install new crontab
    crontab "$temp_cron"
    rm "$temp_cron"
    
    success "Schedule installed successfully"
    log "Current schedule:"
    crontab -l | grep "experimance-auto" || true
}

remove_schedule() {
    log "Removing schedule..."
    
    # Remove our crontab entries
    local temp_cron=$(mktemp)
    crontab -l 2>/dev/null | grep -v "experimance-auto" > "$temp_cron" || true
    crontab "$temp_cron"
    rm "$temp_cron"
    
    success "Schedule removed successfully"
}

show_schedule() {
    log "Current schedule:"
    local schedule_found=false
    
    if crontab -l 2>/dev/null | grep -q "experimance-auto"; then
        crontab -l | grep "experimance-auto" -A1 -B0
        schedule_found=true
    fi
    
    if [ "$schedule_found" = false ]; then
        log "No schedule currently set"
    fi
}

# Preset schedules for common use cases
setup_gallery_schedule() {
    # Gallery hours: Monday-Friday, 12PM-5PM
    local start_schedule="0 12 * * 1-5"  # Start at 12:00 PM, Monday-Friday
    local stop_schedule="0 17 * * 1-5"   # Stop at 5:00 PM, Monday-Friday
    
    log "Setting up gallery schedule (Monday-Friday, 12PM-5PM)..."
    setup_schedule "$start_schedule" "$stop_schedule"
}

setup_demo_schedule() {
    # Demo schedule: Every day, 9AM-6PM
    local start_schedule="0 9 * * *"   # Start at 9:00 AM every day
    local stop_schedule="0 18 * * *"   # Stop at 6:00 PM every day
    
    log "Setting up demo schedule (Daily, 9AM-6PM)..."
    setup_schedule "$start_schedule" "$stop_schedule"
}

setup_weekend_schedule() {
    # Weekend schedule: Saturday-Sunday, 10AM-4PM
    local start_schedule="0 10 * * 6,0"  # Start at 10:00 AM, Saturday-Sunday
    local stop_schedule="0 16 * * 6,0"   # Stop at 4:00 PM, Saturday-Sunday
    
    log "Setting up weekend schedule (Saturday-Sunday, 10AM-4PM)..."
    setup_schedule "$start_schedule" "$stop_schedule"
}

main() {
    case "$ACTION" in
        install)
            if [[ "$MODE" == "dev" ]]; then
                log "=== DEVELOPMENT INSTALL ==="
                log "This will set up the project for development testing only"
                log "No systemd services will be installed"
            elif [[ "$MODE" == "prod" ]]; then
                log "=== PRODUCTION INSTALL ==="
                log "This will set up the project for production deployment"
                log "Systemd services will be installed and can be managed"
                check_root
            fi
            
            check_user
            install_dependencies
            # Re-populate SERVICES array after dependencies are installed
            readarray -t SERVICES < <(get_project_services "$PROJECT")
            check_project
            install_systemd_files
            create_symlink "$(pwd)"
            setup_directories

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
            
            # Inform user about group membership activation if groups were added
            if [[ "$group_added" == true ]] && [[ "$MODE" == "dev" ]]; then
                echo ""
                warn "New group membership added. To activate group access:"
                warn "  Option 1: Log out and log back in (recommended)"
                warn "  Option 2: Run 'newgrp video' and/or 'newgrp audio' to start a new shell with audio group active"
                warn "  Option 3: Restart your terminal session"
                echo ""
            fi
            
            if [[ "$MODE" == "dev" ]]; then
                log "Development installation complete!"
                log "To test services: Use './scripts/dev <service>'"
                log "To install for production: sudo ./deploy.sh $PROJECT install prod"
            else
                log "Production installation complete!"
                log "To start services: sudo ./deploy.sh $PROJECT start"
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
            ;;
        schedule-demo)
            log "Setting up demo schedule (Daily, 9AM-6PM)"
            setup_demo_schedule
            ;;
        schedule-weekend)
            log "Setting up weekend schedule (Saturday-Sunday, 10AM-4PM)"
            setup_weekend_schedule
            ;;
        schedule-custom)
            # Expect start and stop schedules as additional arguments
            local start_schedule="${4:-}"
            local stop_schedule="${5:-}"
            
            if [[ -z "$start_schedule" || -z "$stop_schedule" ]]; then
                error "Custom schedule requires start and stop cron expressions"
                error "Usage: $0 $PROJECT schedule-custom 'start-cron' 'stop-cron'"
                error "Example: $0 $PROJECT schedule-custom '0 8 * * 1-5' '0 20 * * 1-5'"
                exit 1
            fi
            
            log "Setting up custom schedule"
            setup_schedule "$start_schedule" "$stop_schedule"
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
            echo "$stop_cron $SCRIPT_DIR/shutdown.sh --project $PROJECT && (crontab -l 2>/dev/null | grep -v 'experimance-onetime-shutdown' | crontab -) >/dev/null 2>&1" >> "$temp_cron"
            
            crontab "$temp_cron"
            rm "$temp_cron"
            
            log "✓ Scheduled one-time shutdown for $stop_time today"
            log "The cron job will automatically remove itself after execution"
            log "Note: Using shutdown.sh which will destroy service instances"
            ;;
        schedule-remove)
            log "Removing schedule"
            remove_schedule
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
            echo "Available experimance systemd files:"
            ls -la "$SYSTEMD_DIR" | grep experimance || echo "  No experimance files found"
            
            echo -e "\n${BLUE}=== Unit Files Check ===${NC}"
            local target="experimance@${PROJECT}.target"
            
            echo "Target template file: $SYSTEMD_DIR/experimance@.target"
            if [ -f "$SYSTEMD_DIR/experimance@.target" ]; then
                echo -e "${GREEN}✓${NC} Target template file exists"
                echo "Target file permissions: $(ls -la "$SYSTEMD_DIR/experimance@.target")"
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
            echo "Unit files:"
            systemctl list-unit-files | grep experimance || echo "  No experimance unit files found"
            
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
    echo "  $0 experimance schedule-demo         # Daily, 9AM-6PM"  
    echo "  $0 experimance schedule-weekend      # Saturday-Sunday, 10AM-4PM"
    echo "  $0 experimance schedule-custom '0 8 * * 1-5' '0 20 * * 1-5'  # Custom: Weekdays 8AM-8PM"
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
    echo "Note: sudo is only needed for production systemd operations and system setup."
    exit 1
fi

main "$@"
