# Vast.ai Automatic Deployment Guide

This is the simplest way to deploy the Experimance image generation server on vast.ai using their built-in provisioning system.

## Quick Start

1. **Create vast.ai instance** with PyTorch template
2. **Set environment variables**:
   ```
   PROVISIONING_SCRIPT=https://raw.githubusercontent.com/RKelln/experimance/refs/heads/main/experimance/services/image_server/src/image_server/generators/vastai/server/vast_provisioning.sh
   ```
3. **Start instance** - the script will automatically install everything
4. **Access via Instance Portal** - Click "Open" → "Experimance Image Server"

## Environment Variables

- `PRELOAD_MODEL` - Model to preload on startup (default: "lightning")
- `MODELS_DIR` - Where to store models (default: "/workspace/models")

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

**Vast.ai GUI rejecting PROVISIONING_SCRIPT URL?**
Try these alternatives:
1. Use the shorter URL: `https://raw.githubusercontent.com/RKelln/experimance/refs/heads/main/experimance/services/image_server/src/image_server/generators/vastai/server/vast_provisioning.sh`
2. Use Vast.ai CLI instead: `vastai create instance OFFER_ID --env PROVISIONING_SCRIPT=https://raw.githubusercontent.com/RKelln/experimance/refs/heads/main/experimance/services/image_server/src/image_server/generators/vastai/server/vast_provisioning.sh`
3. Copy the script content directly into the "On-Start Script" field instead of using PROVISIONING_SCRIPT

**PROVISIONING_SCRIPT environment variable not working?**
The VastAIManager now includes an automatic SCP fallback that will:
1. Wait for the instance to be running and accessible via SSH
2. SCP the local `vast_provisioning.sh` script to the instance
3. Execute it remotely with the required environment variables

This happens automatically when using `VastAIManager.find_or_create_instance()` for new instances. For existing instances, you can force provisioning with:
```python
manager = VastAIManager()
endpoint = manager.find_or_create_instance(provision_existing=True)
```

Or manually provision a specific instance:
```python
manager = VastAIManager()
success = manager.provision_existing_instance(instance_id)
```

You can check the provisioning script log on the instance:
```bash
cat /var/log/portal/provisioning.log
```


**Git "ambiguous argument 'HEAD'" error?**
This can happen when the git repository is in a shallow or corrupted state. The provisioning script now:
- Checks if HEAD is valid before attempting reset
- Falls back to fetching and resetting to origin/main
- Re-clones the repository if git operations fail
- Continues execution even if some git operations fail

**Pip warnings about running as root?**
The provisioning script now uses `--root-user-action=ignore` to suppress pip warnings about running as root user, which is normal in container environments.

**Exit code 128 but service works?**
Exit code 128 typically indicates a git error. The provisioning script now handles this more gracefully:
- Attempts to fix git issues automatically with better error handling
- Falls back to re-cloning if necessary  
- Continues with service setup even if some git operations fail
- Explicitly exits with 0 when service setup is successful
- Vast.ai manager now checks service health even if script reports failure
- If the service is running and healthy, provisioning is considered successful regardless of intermediate failures

This ensures that temporary git issues don't prevent successful deployment of a working service.

### Dedicated Python CLI for Vast.ai

Use the `vastai_cli.py` script located in the `scripts` directory to manage Vast.ai instances with a unified interface.
Once you create an instance using `provision` it will default to that instance id for the rest of the commands 
and `provision` with no running instance and no id will search offers, select the best and provision automatically.

FIXME: Until vastai fixes their provisioning, you wil need to run `fix` after `provision`.

If you make changes to the `model_server.py` or other server instance files then use `update` to up those.

```bash
python scripts/vastai_cli.py <command> [options]
```

Available commands:
- list           List all Vast.ai instances
- search         Search for suitable GPU offers
- provision      Find or create an Experimance instance
- fix            Fix an existing instance via SCP provisioning
- update         Update server code and restart service
- ssh            Show SSH command for an instance
- endpoint       Show model server endpoint for an instance
- start          Start an instance
- stop           Stop an instance
- restart        Restart an instance
- destroy        Destroy an instance
- health         Check the health of the model server

Examples:
```bash
python scripts/vastai_cli.py list
python scripts/vastai_cli.py update 12345 --debug --verbose
```

