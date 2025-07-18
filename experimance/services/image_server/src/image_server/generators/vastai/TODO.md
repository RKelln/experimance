# Vast.ai PyWorker Setup for ControlNet Image Generation

## Project Overview
Create a serverless ControlNet image generation worker using vast.ai's PyWorker framework that can:
- Generate SDXL images with ControlNet depth conditioning
- Support multiple base models (Lightning, Hyper, standard SDXL)
- Handle era-specific LoRA loading (wilderness, pre-industrial, future)
- Automatically download and cache models from Google Storage
- Scale automatically based on demand

## Architecture Plan

### 1. Worker Structure
```
workers/experimance_controlnet/
├── data_types.py          # Request/response data structures
├── server.py              # Main PyWorker server with endpoints
├── model_server.py        # Standalone diffusers model server
├── client.py              # Test client for development
├── start_model_server.sh  # Script to start the model server
├── requirements.txt       # Python dependencies
└── README.md              # Setup and usage instructions
```

### 2. Data Flow
```
Client Request → PyWorker → Model Server → PyWorker → Client Response
     ↓              ↓            ↓             ↓           ↓
  JSON Payload  → Validate → Generate Image → Process → Return Image
```

## TODO List

### Phase 1: Core Infrastructure Setup ✅ COMPLETE
- [x] **1.1** Create worker directory structure under `workers/experimance_controlnet/`
- [x] **1.2** Set up `data_types.py` with ControlNet-specific payload classes
- [x] **1.3** Create `model_server.py` based on `vast_controlnet_test.py`
- [x] **1.4** Implement `server.py` with PyWorker handlers
- [x] **1.5** Create `start_model_server.sh` startup script
- [x] **1.6** Define `requirements.txt` with all dependencies
- [x] **1.7** Create `test_load.py` for load testing
- [x] **1.8** Create comprehensive `README.md` documentation
- [x] **1.9** Create `client.py` for testing and examples

### Phase 2: Model Server Implementation ✅ COMPLETE
- [x] **2.1** Convert `vast_controlnet_test.py` into a FastAPI server
- [x] **2.2** Implement `/generate` endpoint for image generation
- [x] **2.3** Add `/healthcheck` endpoint for monitoring
- [x] **2.4** Add model loading and caching logic
- [x] **2.5** Implement proper error handling and logging
- [x] **2.6** Add startup/shutdown lifecycle management

### Phase 3: PyWorker Integration
cancelled

### Phase 4: Vast.ai Template Integration
- [x] **4.1** Create Docker image based on PyTorch template
- [x] **4.2** Set up environment variables and configuration
- [x] **4.3** Configure supervisor for process management
- [x] **4.4** Set up model pre-loading on instance startup
- [x] **4.5** Configure Instance Portal integration
- [x] **4.6** Set up logging and monitoring

### Phase 5: Testing and Optimization
- [ ] **5.1** Create comprehensive test client
- [ ] **5.2** Test with different model combinations
- [ ] **5.3** Benchmark performance and memory usage
- [ ] **5.4** Optimize model loading times
- [ ] **5.6** Load testing with multiple concurrent requests

### Phase 6: Documentation and Deployment
- [ ] **6.1** Write comprehensive README with setup instructions
- [ ] **6.2** Create deployment guide for vast.ai
- [ ] **6.3** Document API endpoints and parameters
- [ ] **6.4** Create troubleshooting guide
- [ ] **6.5** Set up monitoring and alerting
- [ ] **6.6** Deploy to production vast.ai template

## Technical Specifications

### API Payload Structure
```python
@dataclass
class ControlNetGenerateData(ApiPayload):
    prompt: str
    negative_prompt: Optional[str] = None
    depth_map_b64: Optional[str] = None  # base64 encoded depth map
    mock_depth: bool = False
    model: str = "lightning"  # lightning, hyper, base
    era: Optional[str] = None  # wilderness, pre_industrial, future
    steps: int = 6
    cfg: float = 2.0
    seed: Optional[int] = None
    lora_strength: float = 1.0
    controlnet_strength: float = 0.8
    scheduler: str = "auto"  # euler, dpm_sde, auto
    width: int = 1024
    height: int = 1024
```

### Model Server Endpoints
- `POST /generate` - Generate image with ControlNet
- `GET /healthcheck` - Health status
- `GET /models` - List available models
- `POST /preload` - Preload specific models

### Environment Variables
```bash
MODEL_SERVER_URL=http://0.0.0.0:5001
MODEL_LOG=/workspace/model_server.log
WORKSPACE=/workspace
MODELS_DIR=/workspace/models
GOOGLE_STORAGE_BASE=https://storage.googleapis.com/experimance_models/
```

### Dependencies
- torch (PyTorch with CUDA)
- diffusers
- transformers
- accelerate
- safetensors
- peft (for LoRA support)
- controlnet-aux
- xformers (memory optimization)
- fastapi
- uvicorn
- aiohttp
- Pillow
- numpy
- requests

