#!/bin/bash
# Vast.ai Provisioning Script for Experimance Image Generation
# This script is designed to be used with the PROVISIONING_SCRIPT environment variable
# URL: https://gist.githubusercontent.com/RKelln/21ad3ecb4be1c1d0d55a8f1524ff9b14/raw/vast_experimance_provisioning.sh

set -eo pipefail

echo "=== Experimance Image Generation - Vast.ai Provisioning ==="

# Verify Python environment
echo "Python environment:"
which python
python --version

# Install dependencies for image generation (preserving existing PyTorch)
echo "Installing image generation dependencies..."
pip install tokenizers regex pillow requests numpy importlib_metadata
pip install diffusers transformers accelerate safetensors --no-deps
pip install controlnet-aux peft
pip install fastapi uvicorn pydantic python-multipart

# Install xformers for current PyTorch version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")

echo "Detected PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION}"

# Determine the correct PyTorch wheel index URL based on CUDA version
INDEX_URL=""
if [[ "$CUDA_VERSION" == *"12.1"* ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cu121"
elif [[ "$CUDA_VERSION" == *"12.4"* ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cu124"
elif [[ "$CUDA_VERSION" == *"12.6"* ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cu126"
elif [[ "$CUDA_VERSION" == *"12.8"* ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cu128"
elif [[ "$CUDA_VERSION" == 12.* ]]; then
    # Default to cu121 for any other 12.x CUDA version
    INDEX_URL="https://download.pytorch.org/whl/cu121"
    echo "⚠️  Unknown CUDA 12.x version, defaulting to cu121 index"
else
    echo "⚠️  Unsupported CUDA version: ${CUDA_VERSION}, skipping xformers"
fi

# Install xformers if we have a valid index URL
if [ -n "$INDEX_URL" ]; then
    # Use latest xformers only for PyTorch 2.7+ with CUDA 12.6/12.8
    if [[ "$PYTORCH_VERSION" == 2.7* ]] && ([[ "$CUDA_VERSION" == *"12.6"* ]] || [[ "$CUDA_VERSION" == *"12.8"* ]]); then
        echo "Installing latest xformers for PyTorch 2.7+ with CUDA 12.6/12.8..."
        pip install xformers --no-deps --index-url "$INDEX_URL" || echo "⚠️  xformers install failed"
    else
        echo "Installing xformers==0.0.28.post3 for PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION}..."
        pip install xformers==0.0.28.post3 --no-deps --index-url "$INDEX_URL" || echo "⚠️  xformers install failed"
    fi
else
    echo "⚠️  Skipping xformers installation due to unsupported CUDA version"
fi

# Clone or update the experimance repository
echo "Setting up experimance repository..."
cd /workspace

# Use the GitHub token if provided (GITHUB_TOKEN or GITHUB_ACCESS_TOKEN)
GITHUB_TOKEN=${GITHUB_TOKEN:-$GITHUB_ACCESS_TOKEN}

# Debug: Show current directory and contents
echo "Current directory: $(pwd)"
echo "Contents: $(ls -la)"

if [ ! -d "experimance" ]; then
    echo "Experimance directory not found, cloning repository..."
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "Using GitHub token for private repo access..."
        git clone https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git
    else
        echo "No GitHub token provided, trying public access..."
        git clone https://github.com/RKelln/experimance.git
    fi
elif [ ! -d "experimance/.git" ]; then
    echo "Experimance directory exists but is not a git repository, removing and re-cloning..."
    rm -rf experimance
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "Using GitHub token for private repo access..."
        git clone https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git
    else
        echo "No GitHub token provided, trying public access..."
        git clone https://github.com/RKelln/experimance.git
    fi
else
    echo "Experimance repository already exists, updating..."
    cd experimance
    
    # Reset any local changes and pull latest
    echo "Resetting local changes and cleaning untracked files..."
    git reset --hard HEAD
    git clean -fd
    
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "Using GitHub token for private repo access..."
        git remote set-url origin https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git
    fi
    
    echo "Pulling latest changes from main branch..."
    git pull origin main || {
        echo "⚠️  Git pull failed, trying to re-clone..."
        cd ..
        rm -rf experimance
        if [ -n "$GITHUB_TOKEN" ]; then
            git clone https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git
        else
            git clone https://github.com/RKelln/experimance.git
        fi
    }
    echo "Repository updated to latest version"
    cd ..
fi

# Create directories
mkdir -p /workspace/{models,logs}

# Set up paths and directories
PROJECT_ROOT="/workspace/experimance/experimance"
IMAGE_SERVER_PATH="$PROJECT_ROOT/services/image_server/src/image_server/generators/vastai"
WORKER_DIR="$IMAGE_SERVER_PATH/server"
MODELS_DIR="/workspace/models"
LOG_DIR="/var/log/portal"

# Create supervisor configuration for the image server
cat > /etc/supervisor/conf.d/experimance-image-server.conf << EOF
[program:experimance-image-server]
command=/venv/main/bin/python model_server.py --host 0.0.0.0 --port 8000
directory=$WORKER_DIR
user=root
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$LOG_DIR/experimance-image-server.log
environment=MODELS_DIR="$MODELS_DIR",LOG_LEVEL="info",PRELOAD_MODEL="lightning"
EOF

# Create supervisor wrapper script (vast.ai pattern)
cat > /opt/supervisor-scripts/experimance-image-server.sh << EOF
#!/bin/bash
# Experimance Image Server wrapper script for supervisor

# Check if application is configured in portal.yaml
if ! grep -q "Experimance Image Server" /etc/portal.yaml 2>/dev/null; then
    echo "Experimance Image Server not found in portal.yaml - not starting"
    exit 0
fi

# Set environment variables
export MODELS_DIR=\${MODELS_DIR:-$MODELS_DIR}
export LOG_LEVEL=\${LOG_LEVEL:-info}
export PRELOAD_MODEL=\${PRELOAD_MODEL:-lightning}

# Create directories
mkdir -p "\$MODELS_DIR"
mkdir -p /workspace/logs

# Start the image server
cd $WORKER_DIR
exec python model_server.py --host 0.0.0.0 --port 8000
EOF

# Make wrapper script executable
chmod +x /opt/supervisor-scripts/experimance-image-server.sh

# Reload Supervisor
supervisorctl reload