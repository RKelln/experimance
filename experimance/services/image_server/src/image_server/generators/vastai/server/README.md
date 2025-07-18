# ControlNet PyWorker for Vast.ai

A FastAPI implementation for ControlNet-based image generation on vast.ai infrastructure. This worker provides serverless SDXL image generation with depth conditioning, multiple model support, and era-specific LoRA loading.

### Direct GPU Rental 
- **For**: Development, testing, custom applications
- **Benefits**: Simple setup, direct control, easy debugging
- **Setup**: Just rent GPU and run scripts directly
- **Best for**: Learning, prototyping, one-off generation tasks
- **Guide**: See [DIRECT_DEPLOYMENT.md](../DIRECT_DEPLOYMENT.md) for step-by-step instructions

## Features

- **ControlNet Depth Conditioning**: Generate images guided by depth maps
- **Multiple Base Models**: Support for Lightning (6-step), Hyper (8-step), and standard SDXL models
- **Era-Specific LoRAs**: Historical themes (wilderness, pre-industrial, future) for artistic consistency
- **Automatic Scaling**: Integrates with vast.ai autoscaler for demand-based scaling
- **Workload Estimation**: Calculates resource requirements based on generation parameters
- **Model Caching**: Efficient model loading and memory management

## Architecture

The worker consists of two main components:

1. **Model Server** (`model_server.py`): FastAPI server that handles the actual image generation

```
Client Request → Model Server → Image Generation → Response
```

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- PyTorch with CUDA support
- vast.ai PyWorker framework

### Dependencies

The dependencies are installed automatically using the `vast_provisioning.sh` script.

### Model Downloads

Models are automatically downloaded from Google Cloud Storage on first use:

- **Base Models**: Juggernaut XL Lightning, Hyper-SDXL
- **ControlNet**: depth-sdxl-1.0-small
- **LoRAs**: experimance_wilderness, experimance_historical, experimance_future

## Usage


### Development Mode

```bash
# Start model server
python model_server.py

```

### API Endpoints

#### Generate Image
```http
POST /generate
Content-Type: application/json

{
  "prompt": "A majestic mountain landscape",
  "negative_prompt": "blurry, low quality",
  "mock_depth": true,
  "model": "lightning",
  "era": "wilderness",
  "steps": 6,
  "cfg": 2.0,
  "width": 1024,
  "height": 1024
}
```

#### Health Check
```http
GET /ping
```

#### List Models
```http
GET /models
```

### Parameters

| Parameter             | Type    | Default     | Description                                    |
| --------------------- | ------- | ----------- | ---------------------------------------------- |
| `prompt`              | string  | required    | Text description of desired image              |
| `negative_prompt`     | string  | optional    | What to avoid in the image                     |
| `depth_map_b64`       | string  | optional    | Base64-encoded depth map                       |
| `mock_depth`          | boolean | false       | Generate mock depth map for testing            |
| `model`               | string  | "lightning" | Model type: lightning, hyper, base             |
| `era`                 | string  | optional    | LoRA theme: wilderness, pre_industrial, future |
| `steps`               | integer | auto        | Inference steps (model-dependent)              |
| `cfg`                 | float   | auto        | Guidance scale (model-dependent)               |
| `seed`                | integer | random      | Random seed for reproducibility                |
| `lora_strength`       | float   | 1.0         | LoRA application strength                      |
| `controlnet_strength` | float   | 0.8         | ControlNet conditioning strength               |
| `scheduler`           | string  | "auto"      | Scheduler: auto, euler, dpm_sde                |
| `width`               | integer | 1024        | Output width (multiple of 64)                  |
| `height`              | integer | 1024        | Output height (multiple of 64)                 |

### Model Types

#### Lightning (6-step)
- **Best for**: Fast generation, real-time applications
- **Steps**: 6 (recommended)
- **CFG**: 2.0 (recommended)
- **Speed**: ~3-5 seconds

#### Hyper (8-step)
- **Best for**: Balanced quality and speed
- **Steps**: 8 (recommended)
- **CFG**: 6.0 (recommended)
- **Speed**: ~5-8 seconds

#### Base SDXL
- **Best for**: Highest quality output
- **Steps**: 20+ (recommended)
- **CFG**: 7.5 (recommended)
- **Speed**: ~15-30 seconds

### Era LoRAs

#### Drone
- Drone photography (aerial shots), recommended LoRA wight 0.8
- https://civitai.com/models/159324/drone-photography-for-xl

#### Experimance
- Custom Experimance LoRA, recommended weight 0.8-1.2

## Testing

### Using the VastAI Manager (locally)

```bash
uv run python services/image_server/src/image_server/generators/vastai/test_vastai_manager.py
```


## Configuration

### Environment Variables

| Variable            | Default                            | Description                 |
| ------------------- | ---------------------------------- | --------------------------- |
| `MODEL_SERVER_URL`  | `http://0.0.0.0:5001`              | Model server endpoint       |
| `MODEL_SERVER_HOST` | `0.0.0.0`                          | Model server bind address   |
| `MODEL_SERVER_PORT` | `5001`                             | Model server port           |
| `MODELS_DIR`        | `/workspace/models`                | Model storage directory     |
| `MODEL_LOG`         | `/workspace/logs/model_server.log` | Log file path               |
| `PRELOAD_MODEL`     | `lightning`                        | Model to preload on startup |
| `LOG_LEVEL`         | `info`                             | Logging level               |

### Performance Tuning

#### GPU Memory Optimization
- Models use `float16` precision
- CPU offloading for unused components
- xformers memory efficient attention
- Automatic garbage collection

#### Model Caching
- Models persist in memory after first load
- Automatic model selection based on request
- LoRA weights cached and reused

#### Workload Calculation
The worker calculates workload units based on:
- Image resolution (pixel count)
- Inference steps
- Model complexity
- LoRA usage
- ControlNet processing

## Deployment on Vast.ai

### Template Requirements

1. PyTorch container with CUDA support
2. PyWorker framework installed
3. Sufficient GPU memory (12GB+ recommended)
4. Fast storage for model caching


## Troubleshooting

### Common Issues

#### Model Download Failures
- Check internet connectivity
- Verify Google Cloud Storage access
- Ensure sufficient disk space

#### Out of Memory Errors
- Enable CPU offloading

#### Slow Generation
- Verify GPU utilization with `nvidia-smi`
- Check model caching is working
- Use Lightning model for speed
- Enable xformers optimization

#### Connection Errors
- Verify model server is running on port 8000

### Monitoring

#### Health Checks
The `/checkhealth` endpoint provides:
- Model server connectivity
- Loaded models list
- Memory usage statistics

#### Logging
- Model server logs: `/workspace/logs/model_server.log`
- Error logs: separate stderr files

#### Performance Metrics
- Generation time per request
- Memory usage (RAM and GPU)
- Model loading times
- Workload calculations

#### Monitor Performance
```bash
# GPU usage
nvidia-smi

# Memory usage
htop

# Disk space
df -h

# Network connections
ss -tuln | grep 8000
```

## Contributing

### Code Structure

```
experimance_controlnet/
├── data_types.py            # Request/response data structures
├── model_server.py          # FastAPI model server
└── README.md                # This file
```

### Development Guidelines

1. Use type hints throughout
2. Implement proper error handling
3. Add comprehensive logging
4. Test with different model combinations
5. Document API changes



