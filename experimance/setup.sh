#!/bin/bash

# Modern setup script for Experimance
# This script installs all services in development mode

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo -e "Install uv with: ${YELLOW}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo -e "${BLUE}Setting up Experimance in ${SCRIPT_DIR}${NC}"

# Check for system dependencies
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]] || [[ "$ID" == "debian" ]]; then
        echo -e "${BLUE}Checking for required system packages...${NC}"
        MISSING_PKGS=""
        
        # Check for required packages
        for pkg in libssl-dev libusb-1.0-0-dev libsdl2-dev ffmpeg libasound2-dev portaudio19-dev; do
            if ! dpkg -l | grep -q "^ii  $pkg "; then
                MISSING_PKGS+=" $pkg"
            fi
        done
        
        # Install missing packages if necessary
        if [ -n "$MISSING_PKGS" ]; then
            echo -e "${YELLOW}Installing required system packages: ${MISSING_PKGS}${NC}"
            sudo apt-get update
            sudo apt-get install -y $MISSING_PKGS
            if [ $? -ne 0 ]; then
                echo -e "${RED}Failed to install required system packages${NC}"
                exit 1
            fi
        fi
    else
        echo -e "${YELLOW}Non-Debian based system detected. You may need to install these dependencies manually:${NC}"
        echo -e "${YELLOW}- SDL2 development libraries${NC}"
        echo -e "${YELLOW}- libusb-1.0 development libraries${NC}"
        echo -e "${YELLOW}- OpenSSL development libraries${NC}"
        echo -e "${YELLOW}- ffmpeg${NC}"
        echo -e "${YELLOW}- ALSA and PortAudio development libraries${NC}"
    fi
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
uv venv --python=3.11 "${SCRIPT_DIR}/.venv"
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment created${NC}"

# Activate virtual environment
source "${SCRIPT_DIR}/.venv/bin/activate"

# Install development tools
echo -e "${BLUE}Installing development tools...${NC}"
uv pip install setuptools wheel pytest mypy ruff
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install development tools${NC}"
    exit 1
fi

# Install all packages in development mode
echo -e "${BLUE}Installing all services in development mode...${NC}"

# First install common library
echo -e "${BLUE}Installing experimance-common...${NC}"
cd "${SCRIPT_DIR}/libs/common"
uv pip install -e .
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install experimance-common${NC}"
    exit 1
fi
echo -e "${GREEN}Installed experimance-common${NC}"

# Install all service packages
SERVICES=(
    "${SCRIPT_DIR}/services/transition"   # Install transition before display (dependency)
    "${SCRIPT_DIR}/services/display"      # Install before core (dependency)
    "${SCRIPT_DIR}/services/audio"
    "${SCRIPT_DIR}/services/agent"
    "${SCRIPT_DIR}/services/image_server" # Install before core (dependency)
    "${SCRIPT_DIR}/services/core"
)

for service in "${SERVICES[@]}"; do
    if [ -f "${service}/pyproject.toml" ]; then
        echo -e "${BLUE}Installing $(basename ${service})...${NC}"
        cd "${service}"
        uv pip install -e .
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Warning: Failed to install $(basename ${service})${NC}"
        else
            echo -e "${GREEN}Installed $(basename ${service})${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping $(basename ${service}) - no pyproject.toml found${NC}"
    fi
done

cd "${SCRIPT_DIR}"

# Install special dependencies that might need special handling
echo -e "${BLUE}Installing special dependencies...${NC}"

# PySDL2
echo -e "${BLUE}Installing PySDL2...${NC}"
uv pip install pysdl2-dll pysdl2
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install PySDL2${NC}"
    echo -e "${YELLOW}You may need to install SDL2 manually${NC}"
fi

# PyRealSense2
echo -e "${BLUE}Installing PyRealSense2...${NC}"
if uv pip install pyrealsense2; then
    echo -e "${GREEN}PyRealSense2 installed successfully${NC}"
else
    echo -e "${YELLOW}Attempting to install PyRealSense2 with specific version for Python 3.11...${NC}"
    if uv pip install --ignore-requires-python pyrealsense2; then
        echo -e "${GREEN}PyRealSense2 installed with version override${NC}"
    else
        echo -e "${RED}Failed to install PyRealSense2${NC}"
        echo -e "${YELLOW}You may need to install it manually from: https://github.com/IntelRealSense/librealsense${NC}"
    fi
fi

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p "${SCRIPT_DIR}/services/core/logs"
mkdir -p "${SCRIPT_DIR}/services/core/saved_data"
mkdir -p "${SCRIPT_DIR}/services/image_server/images"
mkdir -p "${SCRIPT_DIR}/infra/grafana/data"
echo -e "${GREEN}Directories created${NC}"

echo -e "${GREEN}Setup complete!${NC}"
echo -e "To activate the virtual environment, run: ${YELLOW}source ${SCRIPT_DIR}/.venv/bin/activate${NC}"

# Run a simple import test
echo -e "${BLUE}Testing imports...${NC}"
source "${SCRIPT_DIR}/.venv/bin/activate"
python -c "import experimance_common; print('Successfully imported experimance_common'); import experimance_core; print('Successfully imported experimance_core')" || echo -e "${YELLOW}Import test failed.${NC}"
