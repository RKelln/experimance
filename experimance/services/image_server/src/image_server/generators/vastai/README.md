# VastAI Image Generation

This directory contains both the VastAI generator for the image server and the deployment scripts for running the experimance ControlNet model server on VastAI instances.

## 🚀 **NEW: Automatic Deployment (Recommended)**

The easiest way to deploy is using vast.ai's provisioning system:

1. **Create instance** with PyTorch template
2. **Set environment variable**: `PROVISIONING_SCRIPT=https://raw.githubusercontent.com/RKelln/experimance/main/services/image_server/src/image_server/generators/vastai/vast_provisioning.sh`
3. **Start instance** - Everything installs automatically!
4. **Access**: Click "Open" → "Experimance Image Server" in vast.ai dashboard

## VastAI Generator Integration

The VastAI generator allows the image server to automatically use VastAI instances for remote image generation. This provides cost-effective, scalable image generation without requiring local GPU resources.

### Generator Features

- **Remote Generation**: Generate images using VastAI cloud instances
- **Single Instance Management**: Manages one instance at a time for cost efficiency  
- **Automatic Instance Lifecycle**: Finds existing instances or creates new ones as needed
- **Health Monitoring**: Monitors instance health and switches to new instances if needed
- **ControlNet Support**: Full support for depth-conditioned image generation
- **Era-specific LoRAs**: Support for experimance and drone era models
- **Multiple Models**: Support for lightning, hyper, and base SDXL models

### Quick Start

```toml
[image_server]
strategy = "vastai"

[image_server.config]
model_name = "hyper"
era = "experimance" 
steps = 6
cfg = 2.0
```

See `config_example.toml` for full configuration options.

### Testing the Integration

```bash
cd services/image_server/src/image_server/generators/vastai
python test_integration.py
```

See [VAST_PROVISIONING.md](VAST_PROVISIONING.md) for full details.

---

# Set up vastai tool

```bash
uv tool install vastai
uv tool run vastai set api-key <VASTAI_API_KEY>
```

---

## Manual Setup (For Development/Testing)

If you want to manually install or test individual scripts:

## 1. SSH in (the console shows the exact port)
```bash
ssh -p 42602 root@166.113.52.39 -L 8080:localhost:8080
export VAST_PORT=42602
export VAST_ADDRESS=166.113.52.39
export VAST_SSH=root@$VAST_ADDRESS
ssh -p $VAST_PORT $VAST_SSH -L 8080:localhost:8080
```

## 2. Environment Setup 

The vast.ai PyTorch template provides a pre-configured environment at `/venv/main/`. **Always activate this first:**

```bash
# Activate the vast.ai PyTorch environment
source /venv/main/bin/activate

# Check what PyTorch version is installed
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# In tmux, enable mouse support:
# Press Ctrl+b then :
set -g mouse on
```

## 3. Install Dependencies (Carefully!)

**Important:** Install packages without breaking the pre-installed PyTorch:

```bash
# Install basic dependencies
pip install tokenizers regex pillow requests numpy importlib_metadata

# Install diffusers and transformers WITHOUT dependencies to avoid PyTorch conflicts
pip install diffusers transformers accelerate safetensors --no-deps

# Install ControlNet and LoRA support
pip install controlnet-aux peft

# Install xformers based on your PyTorch version
# Check your versions first:
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# For PyTorch 2.5.1 with CUDA 12.1:
pip install xformers==0.0.28.post3 --no-deps --index-url https://download.pytorch.org/whl/cu121

# For PyTorch 2.5.1 with CUDA 12.4:
pip install xformers==0.0.28.post3 --no-deps --index-url https://download.pytorch.org/whl/cu124

# For PyTorch 2.7.1 with CUDA 12.6:
pip install xformers --no-deps --index-url https://download.pytorch.org/whl/cu126

# If you break PyTorch, restore it:
# pip install torch==2.7.1+cu126 torchaudio==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

## 4. Copy Scripts

### Copy code from local to instance:
```bash
scp -P $VAST_PORT -r /home/ryankelln/Documents/Projects/Art/experimance/installation/software/experimance/services/image_server/src/image_server/generators/vastai/ $VAST_SSH:/workspace/experimance/experimance/services/image_server/src/image_server/generators/
```

## 5. Run Scripts

### Basic SDXL test:
```bash
python vast_simple_test.py --prompt "Lightning striking a gothic tower" --steps 25
```

### ControlNet test (original workflow):
```bash
# ControlNet with mock depth map
python vast_controlnet_test.py --prompt "Lightning striking a gothic tower" --mock_depth --steps 4 --cfg 1.5

# ControlNet with era-specific LoRA
python vast_controlnet_test.py --prompt "ancient temple ruins" --mock_depth --era wilderness --lora_strength 0.8
```

### **NEW: FAL Workflow Test (Exact fal_comfy_generator.py workflow):**
```bash
# Test with mock depth map (fast setup)
python fal_workflow_test.py --prompt "Lightning striking a gothic tower" --mock_depth --steps 6 --cfg 1.5

# Test with LoRA
python fal_workflow_test.py --prompt "ancient temple ruins" --mock_depth --steps 6 --cfg 1.5 --lora_strength 0.8

# Test with base64 depth map (like your real workflow)
echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==" | python fal_workflow_test.py --prompt "modern cityscape" --depth_b64 /dev/stdin --steps 4 --cfg 2.0

# Fast generation (4 steps, minimal CFG)
python fal_workflow_test.py --prompt "futuristic landscape" --mock_depth --steps 4 --cfg 1.0 --width 512 --height 512
```

### **Comprehensive Test Suite:**
```bash
# Run all tests to verify everything works
./test_fal_workflow.sh

# This tests:
# - Lightning fast generation (4 steps, mock depth)
# - Higher quality with LoRA (6 steps)
# - Full resolution (1024x1024)
# - Base64 depth input example
```

### **Models Used in FAL Workflow Script:**
- **ControlNet**: Kohya-ss ControlNet-LLLite depth (ultra-fast)
- **Base Model**: Lightning SDXL (4-6 steps optimal)
- **Scheduler**: DPM++ SDE (as specified)
- **LoRA**: Optional era-specific LoRA support

## 6. Download Generated Images

```bash
# From your local machine
scp -P 51157 root@70.27.143.168:/workspace/output/*.png media/images/generated/
scp -P 51157 root@70.27.143.168:/workspace/output/*.b64 media/images/generated/
```

## Notes

**Vast.ai Base Image Pre-installed:**
- **Caddy** web server (already running via Supervisor)
- **Supervisor** for process management
- **Python 3.10+** virtual environment at `/venv/main/`
- **Instance Portal** for easy web access
- **TLS certificates** and authentication

**Your Setup:**
- **Models auto-download** on first run to local `models/` directory
- **Subsequent runs** use cached models (much faster)
- **Output** saved in `output/` directory with both PNG and base64 formats
- **FAL workflow script** uses exact same models/settings as your fal_comfy_generator.py
- **Memory optimization** with CPU offload and xformers acceleration






OK now I'd like to create a vast ai test script that uses the same models and workflow as #file:fal_comfy_generator.py :
- takes a depth map in b64 and uses controlnet depth
  - I was testing with various controlnets, this one seemed fast: https://huggingface.co/kohya-ss/controlnet-lllite/resolve/main/controllllite_v01032064e_sdxl_depth_500-1000.safetensors?download=true
- https://civitai.com/api/download/models/920957?type=Model&format=SafeTensor&size=full&fp=fp16
  - this is a lightning model using:
    - Sampler: DPM++ SDE
    - Steps: 4-6
    - CFG: 1-2
- lora: "https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor"
