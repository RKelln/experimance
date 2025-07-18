# Vast.ai Automatic Deployment Guide

This is the simplest way to deploy the Experimance image generation server on vast.ai using their built-in provisioning system.

## Quick Start

1. **Create vast.ai instance** with PyTorch template
2. **Set environment variables**:
   ```
   PROVISIONING_SCRIPT=https://gist.githubusercontent.com/RKelln/21ad3ecb4be1c1d0d55a8f1524ff9b14/raw/vast_experimance_provisioning.sh
   GITHUB_ACCESS_TOKEN=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
   ```
3. **Start instance** - the script will automatically install everything
4. **Access via Instance Portal** - Click "Open" → "Experimance Image Server"

## Environment Variables

- **`GITHUB_ACCESS_TOKEN`** - Your GitHub token for private repo access (required)
- `PRELOAD_MODEL` - Model to preload on startup (default: "lightning")
- `MODELS_DIR` - Where to store models (default: "/workspace/models")

### Gist-based Deployment

The provisioning script is hosted as a public gist for easy access:
- **Gist URL**: https://gist.github.com/RKelln/21ad3ecb4be1c1d0d55a8f1524ff9b14
- **Raw Script URL**: https://gist.githubusercontent.com/RKelln/21ad3ecb4be1c1d0d55a8f1524ff9b14/raw/vast_experimance_provisioning.sh

This allows vast.ai to download and execute the script without authentication while keeping your main repository private.

## What Gets Installed

✅ **Image Generation Stack**
- Diffusers, Transformers, ControlNet-aux
- FastAPI server with automatic API docs
- CUDA-optimized xformers

✅ **Vast.ai Integration**
- Supervisor service management
- Instance Portal integration
- Automatic startup/restart
- Proper logging to `/var/log/portal/`

✅ **Ready-to-Use Models**
- Lightning SDXL (4-6 steps)
- Hyper SDXL (6-8 steps) 
- Base SDXL (20+ steps)
- ControlNet depth conditioning
- LoRA support

## Usage

### Via Instance Portal
1. Click "Open" button in vast.ai dashboard
2. Select "Experimance Image Server"
3. Use the automatic API documentation

### Direct API Access
```bash
# Health check
curl http://localhost:8000/healthcheck

# Generate image
curl -X POST http://localhost:8000/generate \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "a serene lake at sunset", "mock_depth": true}'
```

### Supervisor Management
```bash
# Check status
supervisorctl status experimance-image-server

# View logs
supervisorctl tail -f experimance-image-server

# Restart if needed
supervisorctl restart experimance-image-server
```

## Features

- **Zero configuration** - Works out of the box
- **Auto-scaling models** - Downloads on first use
- **Memory optimized** - CPU offload + xformers
- **Production ready** - Health monitoring and auto-restart
- **Portal integrated** - Easy web access with HTTPS/auth

## Troubleshooting

**Service not starting?**
```bash
supervisorctl status experimance-image-server
supervisorctl tail experimance-image-server
```

**Portal not showing server?**
Check `/etc/portal.yaml` contains "Experimance Image Server" entry

**Models not downloading?**
Check disk space: `df -h /workspace/models`

