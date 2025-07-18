#!/bin/bash
# Vast.ai Provisioning Script for Experimance Image Generation
# This script is designed to be used with the PROVISIONING_SCRIPT environment variable
# URL: https://raw.githubusercontent.com/RKelln/experimance/main/services/image_server/src/image_server/generators/vastai/vast_provisioning.sh

#$ vastai create instance <OFFER_ID> --image vastai/pytorch:@vastai-automatic-tag --env '-p 1111:1111 -p 8000:8000 -p 72299:72299 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8000:8000:/:WebServer"' --onstart-cmd 'entrypoint.sh' --disk 50 --ssh --direct


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

# Clone the experimance repository
echo "Cloning experimance repository..."
cd /workspace
if [ ! -d "experimance" ]; then
    # Use the GitHub token if provided (GITHUB_TOKEN or GITHUB_ACCESS_TOKEN)
    GITHUB_TOKEN=${GITHUB_TOKEN:-$GITHUB_ACCESS_TOKEN}
    
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "Using GitHub token for private repo access..."
        git clone https://${GITHUB_TOKEN}@github.com/RKelln/experimance.git
    else
        echo "No GitHub token provided, trying public access..."
        git clone https://github.com/RKelln/experimance.git
    fi
fi

# Create directories
mkdir -p /workspace/{models,logs}

# Set up paths and directories
PROJECT_ROOT="/workspace/experimance/experimance"
IMAGE_SERVER_PATH="$PROJECT_ROOT/services/image_server/src/image_server/generators/vastai"
WORKER_DIR="$IMAGE_SERVER_PATH/workers/experimance_controlnet"
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

# Test installation
echo "Testing installation..."
python -c "
try:
    import torch
    import diffusers
    import transformers
    import fastapi
    import controlnet_aux
    print('✅ All packages imported successfully')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Reload supervisor configuration and start service
#echo "Reloading supervisor configuration..."
#supervisorctl reread
#supervisorctl update
#supervisorctl start experimance-image-server || echo "Service may already be running"

echo ""
echo "=== Experimance Image Generation Setup Complete ==="
echo ""
echo "🚀 Your image generation server is ready!"
echo ""
echo "📋 What was installed:"
echo "   • Diffusers, Transformers, ControlNet support"
echo "   • FastAPI image generation server"
echo "   • Supervisor service configuration"
echo "   • Portal.yaml integration"
echo ""
echo "🔧 To manage the service:"
echo "   • Start: supervisorctl start experimance-image-server"
echo "   • Stop: supervisorctl stop experimance-image-server"
echo "   • Status: supervisorctl status experimance-image-server"
echo "   • Logs: supervisorctl tail experimance-image-server"
echo ""
echo "🌐 Access your server:"
echo "   • Via Instance Portal: Click 'Open' → 'Experimance Image Server'"
echo "   • Direct: http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo ""
echo "🧪 Test generation:"
echo "   curl -X POST http://localhost:8000/generate \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"prompt\": \"a serene lake at sunset\", \"mock_depth\": true}'"
echo ""
echo "Happy generating! 🎨"
