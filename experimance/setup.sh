#!/bin/bash

# Setup script for Experimance
# This script installs the project and its dependencies using uv

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

# Get script directory (software folder)
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
            if ! dpkg -s $pkg &> /dev/null; then
                MISSING_PKGS="$MISSING_PKGS $pkg"
            fi
        done
        
        # Install missing packages if necessary
        if [ -n "$MISSING_PKGS" ]; then
            echo -e "${YELLOW}Installing required system packages: $MISSING_PKGS ${NC}"
            sudo apt-get update && sudo apt-get install -y $MISSING_PKGS
            if [ $? -ne 0 ]; then
                echo -e "${RED}Failed to install system dependencies${NC}"
                echo -e "${YELLOW}You may need to install these manually: $MISSING_PKGS${NC}"
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

# Install main dependencies
echo -e "${BLUE}Installing main dependencies...${NC}"
uv pip install -r "${SCRIPT_DIR}/requirements.txt"
if [ $? -ne 0 ]; then
    echo -e "${RED}Warning: Some main dependencies may have failed to install${NC}"
    echo -e "${YELLOW}Continuing with installation...${NC}"
fi
echo -e "${GREEN}Main dependencies installed${NC}"

# Install special dependencies
echo -e "${BLUE}Installing PySDL2...${NC}"
uv pip install pysdl2-dll pysdl2
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install PySDL2${NC}"
    echo -e "${YELLOW}You may need to install SDL2 manually${NC}"
fi

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

# Install experimance-common in development mode
echo -e "${BLUE}Installing experimance-common library...${NC}"
cd "${SCRIPT_DIR}/libs/common"
uv pip install -e .
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install experimance-common library${NC}"
    exit 1
fi
cd "${SCRIPT_DIR}"
echo -e "${GREEN}experimance-common library installed${NC}"

# Clean any existing installation
if pip show experimance &> /dev/null; then
    echo -e "${YELLOW}Removing existing experimance package...${NC}"
    uv pip uninstall -y experimance
fi

# Install experimance in development mode but without extras
echo -e "${BLUE}Installing experimance package...${NC}"
cd "${SCRIPT_DIR}"

# Make sure we have setuptools
uv pip install setuptools wheel
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install setuptools and wheel${NC}"
    exit 1
fi

# Install without extras to avoid dependency issues
uv pip install -e . --no-deps
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}First installation method failed, trying alternative...${NC}"
    
    # Try with python setup.py develop as a fallback but without extras
    python setup.py develop --no-deps
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Second installation method failed, trying direct copy...${NC}"
        
        # Create an __init__.py file in the site-packages directory to make experimance importable
        SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "import sys; sys.path.append('${SCRIPT_DIR}')" > "$SITE_PACKAGES/experimance.pth"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}All installation methods failed${NC}"
            exit 1
        fi
    fi
fi
echo -e "${GREEN}experimance package installed${NC}"

# Add PYTHONPATH setup to .venv/bin/activate
# ACTIVATE_FILE="${SCRIPT_DIR}/.venv/bin/activate"
# if ! grep -q "PYTHONPATH" "$ACTIVATE_FILE"; then
#     echo -e "\n# Add experimance to PYTHONPATH" >> "$ACTIVATE_FILE"
#     echo "export PYTHONPATH=\$PYTHONPATH:${SCRIPT_DIR}" >> "$ACTIVATE_FILE"
#     echo -e "${GREEN}Added PYTHONPATH to virtual environment activation script${NC}"
# fi

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p "${SCRIPT_DIR}/services/experimance/logs"
mkdir -p "${SCRIPT_DIR}/services/experimance/saved_data"
mkdir -p "${SCRIPT_DIR}/services/image_server/images"
mkdir -p "${SCRIPT_DIR}/infra/grafana/data"
echo -e "${GREEN}Directories created${NC}"

# Fix Python imports
# echo -e "${BLUE}Fixing Python imports for experimance package...${NC}"
# python "${SCRIPT_DIR}/fix_imports.py"
# if [ $? -ne 0 ]; then
#     echo -e "${YELLOW}Warning: Could not automatically fix Python imports${NC}"
#     echo -e "${YELLOW}If you have import issues, run: python ${SCRIPT_DIR}/fix_imports.py${NC}"
# else
#     echo -e "${GREEN}Python imports fixed successfully${NC}"
# fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "To activate the virtual environment, run: ${YELLOW}source ${SCRIPT_DIR}/.venv/bin/activate${NC}"
echo -e "To start the experimance service: ${YELLOW}cd ${SCRIPT_DIR} && python -m services.experimance.experimance${NC}"
echo -e ""
echo -e "${YELLOW}Note: If you encountered any errors with special dependencies like PySDL2 or PyRealSense2,${NC}"
echo -e "${YELLOW}you may need to install them manually based on your specific system.${NC}"
echo -e ""
echo -e "${BLUE}Testing import...${NC}"
source "${SCRIPT_DIR}/.venv/bin/activate"
python -c "import experimance; print(f'Successfully imported experimance v{experimance.__version__}'); import experimance_common; print('Successfully imported experimance_common')" || echo -e "${YELLOW}Import test failed. If you have import issues, run one of the test scripts: python ${SCRIPT_DIR}/utils/tests/simple_test.py${NC}"
