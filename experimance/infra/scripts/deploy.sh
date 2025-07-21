#!/bin/bash

# Experimance Deployment Script
# Usage: ./deploy.sh [project_name] [action] [mode]
# Actions: install, start, stop, restart, status
# Modes: dev, prod (only for install action)

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
        RUNTIME_USER="$(whoami)"
        USE_SYSTEMD=false
        warn "Running in development mode with user: $RUNTIME_USER"
    elif id experimance &>/dev/null; then
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

install_systemd_files() {
    if [[ "$USE_SYSTEMD" != true ]]; then
        log "Skipping systemd installation in development mode"
        return
    fi
    
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
                if command -v gdown &>/dev/null; then
                    gdown --quiet --id "$MEDIA_ZIP_ID" -O "$ZIP_FILE" || error "Failed to download media bundle via gdown"
                else
                    wget --quiet --no-check-certificate "$MEDIA_URL" -O "$ZIP_FILE" || error "Failed to download media bundle via wget"
                fi
                chown "$RUNTIME_USER:$RUNTIME_USER" "$ZIP_FILE"
            else
                log "Media bundle zip already present at $ZIP_FILE"
            fi

            log "Extracting media bundle into $REPO_DIR/media"
            if command -v unzip &>/dev/null; then
                # Update existing files only if archive files are newer, extract media/* into REPO_DIR to avoid nested media/media
                unzip -qu "$ZIP_FILE" "media/*" -d "$REPO_DIR" || error "Failed to extract media bundle"
            else
                error "Cannot extract media bundle: 'unzip' not found. Please install 'unzip'."
            fi
            chown -R "$RUNTIME_USER:$RUNTIME_USER" "$REPO_DIR/media"
        else
            log "Skipping media bundle download and extraction."
        fi
    fi

    log "Directories created and permissions set"
}

# Check if system build dependencies are installed
check_build_dependencies() {
    local missing_deps=()
    
    # Check for essential build tools
    if ! command -v make >/dev/null 2>&1; then missing_deps+=("make"); fi
    if ! command -v gcc >/dev/null 2>&1; then missing_deps+=("build-essential"); fi
    
    # Define list of required development packages
    # NOTE: We use `dpkg -l package_name` instead of `dpkg -l | grep package_name`
    # because with `set -euo pipefail`, the pipeline approach can fail unexpectedly
    # when dpkg has warnings or the grep doesn't match, even with 2>/dev/null
    local required_packages=(
        "libssl-dev"
        "zlib1g-dev" 
        "libreadline-dev"
        "libgdbm-dev"
        "libdb-dev"
        "portaudio19-dev"  # needed for pyaudio
        "libasound2-dev"   # needed for pyaudio
        "supercollider"    # needed for audio synthesis
    )
    
    # Check each package
    for package in "${required_packages[@]}"; do
        if ! dpkg -l "$package" 2>/dev/null | grep -q "^ii"; then
            missing_deps+=("$package")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        warn "Missing system build dependencies required for Python compilation:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "The following packages need to be installed:"
        echo "  sudo apt install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev libgdbm-dev libdb-dev portaudio19-dev libasound2-dev supercollider"
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
        if [[ "$MODE" == "prod" ]]; then
            # Already running as root in production
            apt update
            apt install -y make build-essential libssl-dev zlib1g-dev \
                libbz2-dev libreadline-dev libsqlite3-dev curl git \
                libncursesw5-dev xz-utils tk-dev libxml2-dev \
                libxmlsec1-dev libffi-dev liblzma-dev libgdbm-dev libdb-dev \
                portaudio19-dev libasound2-dev supercollider
        else
            # Use sudo in development
            sudo apt update
            sudo apt install -y make build-essential libssl-dev zlib1g-dev \
                libbz2-dev libreadline-dev libsqlite3-dev curl git \
                libncursesw5-dev xz-utils tk-dev libxml2-dev \
                libxmlsec1-dev libffi-dev liblzma-dev libgdbm-dev libdb-dev \
                portaudio19-dev libasound2-dev supercollider
        fi
        log "System build dependencies installed"
    elif command -v yum >/dev/null 2>&1; then
        # CentOS/RHEL/Amazon Linux
        if [[ "$MODE" == "prod" ]]; then
            yum install -y gcc make patch zlib-devel bzip2 bzip2-devel \
                readline-devel sqlite sqlite-devel openssl-devel \
                tk-devel libffi-devel xz-devel \
                portaudio-devel alsa-lib-devel
        else
            sudo yum install -y gcc make patch zlib-devel bzip2 bzip2-devel \
                readline-devel sqlite sqlite-devel openssl-devel \
                tk-devel libffi-devel xz-devel \
                portaudio-devel alsa-lib-devel
        fi
        log "System build dependencies installed"
    elif command -v dnf >/dev/null 2>&1; then
        # Fedora
        if [[ "$MODE" == "prod" ]]; then
            dnf install -y make gcc patch zlib-devel bzip2 bzip2-devel \
                readline-devel sqlite sqlite-devel openssl-devel \
                tk-devel libffi-devel xz-devel libuuid-devel gdbm-libs libnsl2 \
                portaudio-devel alsa-lib-devel
        else
            sudo dnf install -y make gcc patch zlib-devel bzip2 bzip2-devel \
                readline-devel sqlite sqlite-devel openssl-devel \
                tk-devel libffi-devel xz-devel libuuid-devel gdbm-libs libnsl2 \
                portaudio-devel alsa-lib-devel
        fi
        log "System build dependencies installed"
    else
        warn "Unknown package manager. Please install build dependencies manually:"
        warn "  sudo apt install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev libgdbm-dev libdb-dev portaudio19-dev libasound2-dev supercollider"
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
    
    # Ask about Python optimization (only for new installations)
    local use_optimization=false
    if ask_python_optimization; then
        use_optimization=true
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
                if [ '$use_optimization' = true ]; then
                    echo 'Building with optimization flags for better performance...'
                    PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install \$LATEST_311
                else
                    pyenv install \$LATEST_311
                fi
            else
                LATEST_311=\$(pyenv versions | grep '3.11' | tail -1 | sed 's/[* ]//g')
            fi
            
            # Set latest Python 3.11 for this project
            cd '$REPO_DIR'
            pyenv local \$LATEST_311
            
            # Install uv if not already installed
            if ! command -v uv >/dev/null 2>&1; then
                curl -LsSf https://astral.sh/uv/install.sh | sh
            fi
            
            # Add uv to PATH for this session
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            
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
        if command -v pyenv >/dev/null 2>&1 && command -v uv >/dev/null 2>&1; then
            log "pyenv and uv already installed, checking Python version..."
            cd "$REPO_DIR"
            
            # Check if we have Python 3.11 available and set
            if ! pyenv local 2>/dev/null | grep -q "3.11"; then
                if pyenv versions | grep -q "3.11"; then
                    log "Setting latest Python 3.11 for this project..."
                    LATEST_311=$(pyenv versions | grep '3.11' | tail -1 | sed 's/[* ]//g')
                    pyenv local $LATEST_311
                    
                    # Check if current Python was built with optimizations
                    if python -c "import sys; print('Optimized build detected' if hasattr(sys, 'flags') and any('optimiz' in str(sys.version).lower() for _ in [1]) else 'Standard build detected')"; then
                        log "Current Python build status checked"
                    fi
                else
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
                # Check if we need to rebuild for optimization
                LATEST_311=$(pyenv versions | grep '3.11' | tail -1 | sed 's/[* ]//g')
                if [[ "$use_optimization" == true ]]; then
                    # Check if current version was built with optimizations
                    log "Checking if current Python 3.11 was built with optimizations..."
                    cd "$REPO_DIR"
                    
                    # Temporarily set the version to check build flags
                    if pyenv local $LATEST_311 2>/dev/null && python -c "
import sysconfig
config = sysconfig.get_config_vars()
opts = config.get('CONFIG_ARGS', '')
exit(0 if '--enable-optimizations' in opts else 1)
" 2>/dev/null; then
                        log "Current Python $LATEST_311 already built with optimizations"
                    else
                        log "Current Python $LATEST_311 not optimized, rebuilding with optimization flags..."
                        pyenv uninstall -f $LATEST_311
                        if ! PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto' PYTHON_CFLAGS='-march=native -mtune=native' pyenv install $LATEST_311; then
                            error "Failed to install optimized Python $LATEST_311"
                        fi
                    fi
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
    if [[ "$USE_SYSTEMD" != true ]]; then
        warn "Development mode: No systemd services to stop"
        return
    fi
    
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
        echo -e "${GREEN}✓${NC} $target: $(systemctl is_active "$target")"
    else
        echo -e "${RED}✗${NC} $target: $(systemctl is_active "$target")"
    fi
    
    echo -e "\n${BLUE}=== Recent Logs ===${NC}"
    journalctl --no-pager -n 20 -u "experimance-core@${PROJECT}" || true
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
            setup_directories
            
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
        *)
            error "Unknown action: $ACTION. Use: install, start, stop, restart, status, services"
            ;;
    esac
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [project_name] [action] [mode]"
    echo "Projects: $(ls "$REPO_DIR/projects" 2>/dev/null | tr '\n' ' ')"
    echo "Actions: install, start, stop, restart, status, services"
    echo "Modes: dev, prod (only for install action)"
    echo ""
    echo "Install Modes:"
    echo "  dev     # Development setup - no systemd, local directories, current user"
    echo "  prod    # Production setup - systemd services, system directories, experimance user"
    echo ""
    echo "Examples:"
    echo "  $0 experimance services              # Show services (no sudo needed)"
    echo "  $0 experimance install dev           # Development setup (no sudo needed)"
    echo "  sudo $0 experimance install prod     # Production setup (sudo needed)"
    echo "  sudo $0 experimance start            # Start services in production (sudo needed)"
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
