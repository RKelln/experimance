#!/bin/bash
# Vast.ai Provisioning Script for Experimance Image Generation
# This script is designed to be used with the PROVISIONING_SCRIPT environment variable
# URL: https://raw.githubusercontent.com/RKelln/experimance/refs/heads/main/experimance/services/image_server/src/image_server/generators/vastai/server/vast_provisioning.sh

cd /workspace/

# Note: Using more permissive error handling for VastAI compatibility
# VastAI expects scripts to be resilient and not fail the entire container startup
set -e  # Exit on error, but not as strict as pipefail

echo "=== Experimance Image Generation - Vast.ai Provisioning ==="

# Debug environment
echo "Environment debugging:"
echo "PATH: $PATH"
echo "SHELL: $SHELL"
echo "USER: $USER"
echo "PWD: $PWD"

# Check if we're in a virtual environment and activate if needed
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Already in virtual environment: $VIRTUAL_ENV"
elif [ -f "/venv/main/bin/activate" ]; then
    echo "Activating virtual environment..."
    # Use more permissive activation - don't fail if it doesn't work
    source /venv/main/bin/activate 2>/dev/null && echo "Virtual environment activated: $VIRTUAL_ENV" || echo "Virtual environment activation failed, continuing with system Python..."
else
    echo "No virtual environment found, trying to continue with system Python..."
fi

# Verify Python environment
echo "Python environment:"
if which python >/dev/null 2>&1; then
    echo "Python found: $(which python)"
    python --version
    PYTHON_CMD="python"
elif which python3 >/dev/null 2>&1; then
    echo "Python3 found: $(which python3)"
    python3 --version
    PYTHON_CMD="python3"
elif [ -f "/venv/main/bin/python" ]; then
    echo "Using venv Python: /venv/main/bin/python"
    /venv/main/bin/python --version
    PYTHON_CMD="/venv/main/bin/python"
else
    echo "ERROR: No Python found!"
    exit 1
fi

if which pip >/dev/null 2>&1; then
    echo "Pip found: $(which pip)"
    PIP_CMD="pip"
elif which pip3 >/dev/null 2>&1; then
    echo "Pip3 found: $(which pip3)"
    PIP_CMD="pip3"
elif [ -f "/venv/main/bin/pip" ]; then
    echo "Using venv pip: /venv/main/bin/pip"
    PIP_CMD="/venv/main/bin/pip"
else
    echo "ERROR: No pip found!"
    exit 1
fi

# Install dependencies for image generation (preserving existing PyTorch)
echo "Installing image generation dependencies..."
$PIP_CMD install --root-user-action=ignore tokenizers regex pillow requests numpy importlib_metadata || echo "⚠️ Some basic packages failed to install"
$PIP_CMD install --root-user-action=ignore diffusers transformers accelerate safetensors --no-deps || echo "⚠️ Some ML packages failed to install"
$PIP_CMD install --root-user-action=ignore controlnet-aux peft || echo "⚠️ Some auxiliary packages failed to install"
$PIP_CMD install --root-user-action=ignore fastapi uvicorn pydantic python-multipart || echo "⚠️ Some web server packages failed to install"

# Install DeepCache for acceleration (separate package)
echo "Installing DeepCache for performance acceleration..."
$PIP_CMD install --root-user-action=ignore DeepCache || echo "⚠️ DeepCache installation failed, acceleration will be disabled"

# Install xformers for current PyTorch version
PYTORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
CUDA_VERSION=$($PYTHON_CMD -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")

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

# Install xformers with compatibility testing
XFORMERS_INSTALLED=false

if [ -n "$INDEX_URL" ]; then
    # Use latest xformers only for PyTorch 2.7+ with CUDA 12.6/12.8
    if [[ "$PYTORCH_VERSION" == 2.7* ]] && ([[ "$CUDA_VERSION" == *"12.6"* ]] || [[ "$CUDA_VERSION" == *"12.8"* ]]); then
        echo "Installing latest xformers for PyTorch 2.7+ with CUDA 12.6/12.8..."
        if $PIP_CMD install --root-user-action=ignore xformers --no-deps --index-url "$INDEX_URL"; then
            XFORMERS_INSTALLED=true
        fi
    else
        echo "Installing xformers==0.0.28.post3 for PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION}..."
        if $PIP_CMD install --root-user-action=ignore xformers==0.0.28.post3 --no-deps --index-url "$INDEX_URL"; then
            XFORMERS_INSTALLED=true
        fi
    fi
else
    echo "⚠️  Skipping xformers installation due to unsupported CUDA version"
fi

# Test xformers compatibility if installed
if [ "$XFORMERS_INSTALLED" = true ]; then
    echo "Testing xformers compatibility..."
    
    # Test if xformers can be imported and used with current CUDA/PyTorch
    XFORMERS_TEST_RESULT=$($PYTHON_CMD -c "
import torch
try:
    import xformers
    import xformers.ops
    # Try to access flash attention module that's causing the issue
    from xformers import flash_attn_3
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
    exit(1)
" 2>&1)

    if echo "$XFORMERS_TEST_RESULT" | grep -q "SUCCESS"; then
        echo "✅ xformers compatibility test passed"
    else
        echo "❌ xformers compatibility test failed: $XFORMERS_TEST_RESULT"
        echo "Uninstalling incompatible xformers to prevent runtime errors..."
        
        # Uninstall xformers
        $PIP_CMD uninstall -y xformers || echo "⚠️ Failed to uninstall xformers"
        
        # Set environment variables to disable xformers in applications
        echo "export XFORMERS_DISABLED=1" >> /etc/environment
        echo "export DISABLE_XFORMERS=1" >> /etc/environment
        
        # Also set for current session
        export XFORMERS_DISABLED=1
        export DISABLE_XFORMERS=1
        
        echo "✅ xformers disabled due to compatibility issues - diffusers will use standard PyTorch attention"
    fi
else
    echo "⚠️ xformers installation failed or skipped"
    # Set environment variables to disable xformers
    echo "export XFORMERS_DISABLED=1" >> /etc/environment
    echo "export DISABLE_XFORMERS=1" >> /etc/environment
    export XFORMERS_DISABLED=1
    export DISABLE_XFORMERS=1
    echo "✅ xformers disabled - diffusers will use standard PyTorch attention"
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
        git clone https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git || {
            echo "⚠️ Failed to clone with token, trying public access..."
            git clone https://github.com/RKelln/experimance.git || echo "⚠️ Git clone failed completely, continuing without repo update..."
        }
    else
        echo "No GitHub token provided, trying public access..."
        git clone https://github.com/RKelln/experimance.git || echo "⚠️ Git clone failed, continuing without repo update..."
    fi
elif [ ! -d "experimance/.git" ]; then
    echo "Experimance directory exists but is not a git repository, removing and re-cloning..."
    rm -rf experimance || echo "⚠️ Failed to remove directory"
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "Using GitHub token for private repo access..."
        git clone https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git || {
            echo "⚠️ Failed to clone with token, trying public access..."
            git clone https://github.com/RKelln/experimance.git || echo "⚠️ Git clone failed completely, continuing without repo update..."
        }
    else
        echo "No GitHub token provided, trying public access..."
        git clone https://github.com/RKelln/experimance.git || echo "⚠️ Git clone failed, continuing without repo update..."
    fi
else
    echo "Experimance repository already exists, updating..."
    cd experimance || {
        echo "⚠️ Failed to cd into experimance directory, skipping update"
        cd /workspace
    }
    
    if [ -d ".git" ]; then
        # Reset any local changes and pull latest (but don't fail if it doesn't work)
        echo "Resetting local changes and cleaning untracked files..."
        
        # Check if git repository is in a valid state
        if git rev-parse --verify HEAD >/dev/null 2>&1; then
            echo "Git repository is valid, resetting to HEAD..."
            git reset --hard HEAD 2>/dev/null || echo "⚠️ Git reset failed, but continuing..."
            git clean -fd 2>/dev/null || echo "⚠️ Git clean failed, but continuing..."
        else
            echo "Git repository appears corrupted or shallow, attempting to fix..."
            # Try to fetch and set HEAD properly
            if [ -n "$GITHUB_TOKEN" ]; then
                git remote set-url origin https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git 2>/dev/null || echo "⚠️ Failed to set remote URL"
            fi
            git fetch origin main --depth=1 2>/dev/null || {
                echo "⚠️ Git fetch failed, repository may be corrupted. Trying re-clone..."
                cd ..
                rm -rf experimance 2>/dev/null || echo "⚠️ Failed to remove directory"
                if [ -n "$GITHUB_TOKEN" ]; then
                    git clone https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git 2>/dev/null || git clone https://github.com/RKelln/experimance.git 2>/dev/null || echo "⚠️ Re-clone failed"
                else
                    git clone https://github.com/RKelln/experimance.git 2>/dev/null || echo "⚠️ Re-clone failed"
                fi
                [ -d "experimance" ] && cd experimance
            }
            # Try reset again after fetch
            git reset --hard origin/main 2>/dev/null || git reset --hard FETCH_HEAD 2>/dev/null || echo "⚠️ Could not reset, but continuing..."
            git clean -fd 2>/dev/null || echo "⚠️ Could not clean, but continuing..."
        fi
        
        if [ -n "$GITHUB_TOKEN" ]; then
            echo "Using GitHub token for private repo access..."
            git remote set-url origin https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git 2>/dev/null || echo "⚠️ Failed to set remote URL"
        fi
        
        echo "Pulling latest changes from main branch..."
        git pull origin main 2>/dev/null || {
            echo "⚠️ Git pull failed, but continuing with existing version..."
        }
        echo "Repository update completed (or skipped if failed)"
    else
        echo "⚠️ Not a git repository, skipping update"
    fi
    cd /workspace
fi

# Create directories
mkdir -p /workspace/{models,logs} || echo "⚠️ Failed to create some directories"

# Set up paths and directories
PROJECT_ROOT="/workspace/experimance/experimance"
IMAGE_SERVER_PATH="$PROJECT_ROOT/services/image_server/src/image_server/generators/vastai"
WORKER_DIR="$IMAGE_SERVER_PATH/server"
MODELS_DIR="/workspace/models"
LOG_DIR="/var/log/portal"

# Only proceed with supervisor configuration if the experimance directory exists
if [ -d "$PROJECT_ROOT" ]; then
    echo "Configuring supervisor for experimance image server..."
    
    # Create supervisor configuration for the image server
    cat > /etc/supervisor/conf.d/experimance-image-server.conf << EOF
[program:experimance-image-server]
command=/opt/supervisor-scripts/experimance-image-server.sh
directory=$WORKER_DIR
user=root
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$LOG_DIR/experimance-image-server.log
environment=MODELS_DIR="$MODELS_DIR",LOG_LEVEL="info",PRELOAD_MODEL="lightning"
EOF
    if [ $? -ne 0 ]; then
        echo "⚠️ Failed to create supervisor config"
    fi

    # Create supervisor wrapper script (vast.ai pattern)
    cat > /opt/supervisor-scripts/experimance-image-server.sh << EOF
#!/bin/bash
# Experimance Image Server wrapper script for supervisor

# Check if application is configured in portal.yaml
if ! grep -q "WebServer" /etc/portal.yaml 2>/dev/null; then
    echo "WebServer not found in portal.yaml - not starting"
    exit 0
fi

# Activate virtual environment if it exists
if [ -f "/venv/main/bin/activate" ]; then
    echo "Activating virtual environment..."
    source /venv/main/bin/activate
    export PATH="/venv/main/bin:\$PATH"
fi

# Determine Python command
if [ -f "/venv/main/bin/python" ]; then
    PYTHON_CMD="/venv/main/bin/python"
elif which python >/dev/null 2>&1; then
    PYTHON_CMD="python"
elif which python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "ERROR: No Python found!"
    exit 1
fi

echo "Using Python: \$PYTHON_CMD"
\$PYTHON_CMD --version

# Set environment variables
export MODELS_DIR=\${MODELS_DIR:-$MODELS_DIR}
export LOG_LEVEL=\${LOG_LEVEL:-info}
export PRELOAD_MODEL=\${PRELOAD_MODEL:-lightning}

# Import xformers compatibility settings from system environment
if grep -q "XFORMERS_DISABLED=1" /etc/environment 2>/dev/null; then
    export XFORMERS_DISABLED=1
    export DISABLE_XFORMERS=1
    echo "xformers disabled due to compatibility issues"
fi

# Create directories
mkdir -p "\$MODELS_DIR"
mkdir -p /workspace/logs

# Start the image server
cd $WORKER_DIR
echo "Starting Experimance Image Server from \$(pwd)"
exec \$PYTHON_CMD model_server.py --host 0.0.0.0 --port 8000
EOF
    if [ $? -ne 0 ]; then
        echo "⚠️ Failed to create supervisor wrapper script"
    fi

    # Make wrapper script executable
    chmod +x /opt/supervisor-scripts/experimance-image-server.sh || echo "⚠️ Failed to make wrapper script executable"

    # Reload Supervisor
    echo "Reloading supervisor configuration..."
    supervisorctl reload || echo "⚠️ Supervisor reload failed, but continuing..."
else
    echo "⚠️ Experimance repository not found, skipping supervisor configuration"
fi

echo ""
echo "=== Provisioning script completed successfully ==="

# Wait a moment for supervisor to start the service
sleep 5

# Check if the service is running (only if supervisor config was created)
if [ -f "/etc/supervisor/conf.d/experimance-image-server.conf" ]; then
    echo "Checking if Experimance Image Server is running..."
    if supervisorctl status experimance-image-server 2>/dev/null | grep -q "RUNNING"; then
        echo "✅ Experimance Image Server is running successfully!"
        exit 0
    else
        echo "⚠️ Service may not be running yet, but configuration is complete"
        # Still exit successfully since configuration completed
        exit 0
    fi
else
    echo "⚠️ Supervisor configuration was not created, but basic setup completed"
    exit 0
fi