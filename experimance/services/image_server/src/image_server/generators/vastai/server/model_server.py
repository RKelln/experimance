"""
FastAPI model server for ControlNet image generation.

This server provides REST endpoints for image generation using SDXL with ControlNet
depth conditioning. It supports multiple base models, LoRA loading,
and various schedulers optimized for different use cases.
"""

import argparse
import base64
import gc
import io
import logging
import os
import psutil
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import requests
import torch
import uvicorn # type: ignore (loaded on server not locally)
from fastapi import FastAPI, HTTPException # type: ignore (loaded on server not locally)
from fastapi.responses import JSONResponse # type: ignore (loaded on server not locally)
from PIL import Image
from pydantic import BaseModel

from diffusers import ( # type: ignore (loaded on server not locally)
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DDIMScheduler,
    LCMScheduler
)

from data_types import ( # type: ignore (loaded on server not locally)
    ControlNetGenerateData,
    ControlNetGenerateResponse,
    HealthCheckResponse,
    ModelListResponse
)


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model management
loaded_models: Dict[str, Any] = {}
loaded_controlnets: Dict[str, ControlNetModel] = {}
startup_time = None

# Model configuration with optimized scheduler settings
MODEL_CONFIG = {
    "lightning": {
        "repo_id": "https://storage.googleapis.com/experimance_models/juggernautXL_juggXILightningByRD.safetensors",
        "filename": "juggernaut-xl-lightning.safetensors",
        "scheduler": "EulerDiscreteScheduler",  # Try Euler with proper config
        # From: https://huggingface.co/RunDiffusion/Juggernaut-XI-Lightning/blob/main/scheduler/scheduler_config.json
        "scheduler_config": {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "interpolation_type": "linear",
            "prediction_type": "epsilon",
            "rescale_betas_zero_snr": False,
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "timestep_type": "discrete",
            "use_karras_sigmas": False
        },
        "steps": 6,  # Creator recommends 6 steps for Lightning models
        "cfg": 1.0  # Lightning models often work better with lower CFG
    },
    "hyper": {
        "repo_id": "https://storage.googleapis.com/experimance_models/Juggernaut_X_RunDiffusion_Hyper.safetensors",
        "filename": "juggernaut-x-hyper.safetensors",
        # Official scheduler config from https://huggingface.co/RunDiffusion/Juggernaut-X-Hyper/blob/fd7ff232cdea4fda67ad3a8b90bd87fa8c85f51a/scheduler/scheduler_config.json
        "scheduler": "EulerDiscreteScheduler",
        "scheduler_config": {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "clip_sample": False,
            "interpolation_type": "linear",
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
            "use_karras_sigmas": False
        },
        "steps": 6,  # Creator recommends 6 steps for Hyper models
        "cfg": 2.0
    },
    "base": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "scheduler": "DPMSolverMultistepScheduler",
        "scheduler_config": {
            "use_karras_sigmas": True,
            "algorithm_type": "sde-dpmsolver++",
            "solver_type": "midpoint",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear"
        },
        "steps": 20,
        "cfg": 7.5
    }
}

LORA_CONFIG = {
    "drone": "https://storage.googleapis.com/experimance_models/drone_photo_v1.0_XL.safetensors",
    "experimance": "https://storage.googleapis.com/experimance_models/civitai_experimance_sdxl_lora_step_1000_1024x1024.safetensors"
}

CONTROLNET_CONFIG = {
    "sdxl_small": {
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0-small",
        "filename": "controlnet-depth-sdxl-1.0-small.safetensors"
    },
    "llite": {
        "repo_id": "https://storage.googleapis.com/experimance_models/controllllite_v01032064e_sdxl_depth_500-1000.safetensors",
        "filename": "controllllite_v01032064e_sdxl_depth_500-1000.safetensors"
    }
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events using modern lifespan context manager."""
    global startup_time
    
    # Startup
    logger.info("Starting ControlNet Model Server...")
    startup_time = time.time()
    
    # Preload ControlNet and depth estimator
    load_controlnet()
    
    # Optionally preload default model
    default_model = os.getenv("PRELOAD_MODEL", "lightning")
    default_controlnet = os.getenv("PRELOAD_CONTROLNET", "sdxl_small")
    if default_model in MODEL_CONFIG:
        try:
            load_model(default_model, default_controlnet)
            logger.info(f"Preloaded {default_model} model with {default_controlnet} ControlNet")
        except Exception as e:
            logger.error(f"Failed to preload {default_model}: {e}")
    
    logger.info("Model server startup complete")
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down model server...")
    
    # Clear GPU memory
    for model_name in list(loaded_models.keys()):
        del loaded_models[model_name]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()
    logger.info("Model server shutdown complete")


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Experimance Model Server", 
    version="1.0.0",
    lifespan=lifespan
)


class GenerateRequest(BaseModel):
    """Request model for the generate endpoint."""
    prompt: str
    negative_prompt: Optional[str] = None
    depth_map_b64: Optional[str] = None
    mock_depth: bool = False
    model: str = "lightning"
    controlnet: str = "sdxl_small"  # Which ControlNet to use
    era: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    seed: Optional[int] = None
    lora_strength: float = 1.0
    controlnet_strength: float = 0.8
    scheduler: str = "auto"  # auto, euler, euler_a, dpm_multi, dpm_single, ddim
    use_karras_sigmas: Optional[bool] = None  # Override Karras setting
    width: int = 1024
    height: int = 1024


def download_model(url: str, filename: str, models_dir: Path) -> Path:
    """Download a model file from URL if it doesn't exist locally."""
    file_path = models_dir / filename
    
    if file_path.exists():
        logger.info(f"Model {filename} already exists at {file_path}")
        return file_path
    
    logger.info(f"Downloading {filename} from {url}")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded {filename}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        raise


def load_controlnet(controlnet_id: str = "sdxl_small") -> ControlNetModel:
    """Load ControlNet model for depth conditioning."""
    # Check if this ControlNet is already loaded
    if controlnet_id in loaded_controlnets:
        logger.info(f"Using cached ControlNet: {controlnet_id}")
        return loaded_controlnets[controlnet_id]
    
    logger.info(f"Loading ControlNet depth model: {controlnet_id}")
    
    if controlnet_id not in CONTROLNET_CONFIG:
        logger.warning(f"Unknown ControlNet ID: {controlnet_id}, falling back to sdxl_small")
        controlnet_id = "sdxl_small"
    
    config = CONTROLNET_CONFIG[controlnet_id]
    
    # Check if this is a Hugging Face model or a custom URL
    if config["repo_id"].startswith("http"):
        # Custom model - download it first
        models_dir = Path(os.getenv("MODELS_DIR", "/workspace/models"))
        controlnet_path = download_model(config["repo_id"], config["filename"], models_dir)
        
        # Load from local file
        controlnet = ControlNetModel.from_single_file(
            str(controlnet_path),
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    else:
        # Hugging Face model
        controlnet = ControlNetModel.from_pretrained(
            config["repo_id"],
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    
    # Cache the loaded ControlNet
    loaded_controlnets[controlnet_id] = controlnet
    logger.info(f"ControlNet {controlnet_id} loaded successfully")
    return controlnet


def load_model(model_name: str, controlnet_id: str = "sdxl_small") -> StableDiffusionXLControlNetPipeline:
    """Load and cache a specific model with specified ControlNet."""
    # Create a cache key that includes both model and controlnet
    cache_key = f"{model_name}_{controlnet_id}"
    
    if cache_key in loaded_models:
        logger.info(f"Using cached model: {cache_key}")
        return loaded_models[cache_key]
    
    logger.info(f"Loading model: {model_name} with ControlNet: {controlnet_id}")
    models_dir = Path(os.getenv("MODELS_DIR", "/workspace/models"))
    
    config = MODEL_CONFIG[model_name]
    controlnet = load_controlnet(controlnet_id)
    
    if model_name == "base":
        # Load base SDXL model
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            config["repo_id"],
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    else:
        # Load Lightning or Hyper model as full checkpoint
        # Download the specific model first
        model_path = download_model(config["repo_id"], config["filename"], models_dir)
        
        # Load from single file checkpoint
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            str(model_path),
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    
    # Set up scheduler with optimized settings
    scheduler_name = config["scheduler"]
    scheduler_config = config.get("scheduler_config", {})
    
    logger.info(f"Setting up scheduler: {scheduler_name} with config: {scheduler_config}")

    if scheduler_name == "EulerDiscreteScheduler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
        logger.info(f"Configured EulerDiscreteScheduler with Karras sigmas: {scheduler_config.get('use_karras_sigmas', False)}")
    elif scheduler_name == "LCMScheduler":
        pipe.scheduler = LCMScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
        logger.info(f"Configured LCMScheduler for Lightning model")
    elif scheduler_name == "DPMSolverMultistepScheduler":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
        logger.info(f"Configured DPMSolverMultistepScheduler with Karras sigmas: {scheduler_config.get('use_karras_sigmas', False)}")
    elif scheduler_name == "EulerAncestralDiscreteScheduler":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
    elif scheduler_name == "DPMSolverSinglestepScheduler":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
    
    # Enable optimizations
    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory optimization enabled")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    # Move to GPU for maximum performance (no CPU offloading for production speed)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        logger.info("Pipeline moved to GPU for maximum performance")
    
    loaded_models[cache_key] = pipe
    logger.info(f"Model {cache_key} loaded successfully")
    return pipe


def generate_mock_depth(width: int, height: int) -> Image.Image:
    """Generate a mock depth map for testing."""
    # Create a simple radial gradient depth map
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Normalize to 0-255 range and invert (closer = brighter)
    depth_array = (255 * (1 - distance / max_distance)).astype(np.uint8)
    
    return Image.fromarray(depth_array, mode='L')


def setup_scheduler(pipe: StableDiffusionXLControlNetPipeline, 
                   scheduler_name: str, 
                   use_karras_sigmas: Optional[bool] = None,
                   model_name: str = "lightning") -> None:
    """
    Set up scheduler for the pipeline with optimized settings.
    
    Args:
        pipe: The pipeline to configure
        scheduler_name: Name of scheduler (auto, euler, euler_a, dpm_multi, dpm_single, ddim)
        use_karras_sigmas: Override Karras sigma setting
        model_name: Model name for default scheduler selection
    """
    # Use model's default scheduler if "auto"
    if scheduler_name == "auto":
        config = MODEL_CONFIG.get(model_name, {})
        scheduler_name = config.get("scheduler", "EulerDiscreteScheduler")
        # Convert class name to short name
        scheduler_map = {
            "EulerDiscreteScheduler": "euler",
            "DPMSolverMultistepScheduler": "dpm_multi",
            "EulerAncestralDiscreteScheduler": "euler_a",
            "LCMScheduler": "lcm"
        }
        scheduler_name = scheduler_map.get(scheduler_name, "euler")
    
    # Get model-specific scheduler config
    config = MODEL_CONFIG.get(model_name, {})
    scheduler_config = config.get("scheduler_config", {}).copy()
    
    # Override Karras setting if specified
    if use_karras_sigmas is not None:
        scheduler_config["use_karras_sigmas"] = use_karras_sigmas
    
    # Set up the scheduler
    if scheduler_name == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
        logger.info(f"Using Euler scheduler with Karras: {scheduler_config.get('use_karras_sigmas', False)}")
        
    elif scheduler_name == "lcm":
        pipe.scheduler = LCMScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
        logger.info("Using LCM scheduler for Lightning model")
        
    elif scheduler_name == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
        logger.info(f"Using Euler Ancestral scheduler with Karras: {scheduler_config.get('use_karras_sigmas', False)}")
        
    elif scheduler_name == "dpm_multi":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
        logger.info(f"Using DPM++ Multistep scheduler with Karras: {scheduler_config.get('use_karras_sigmas', False)}")
        
    elif scheduler_name == "dpm_single":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
        logger.info(f"Using DPM++ Singlestep scheduler with Karras: {scheduler_config.get('use_karras_sigmas', False)}")
        
    elif scheduler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_config
        )
        logger.info("Using DDIM scheduler")
        
    else:
        logger.warning(f"Unknown scheduler: {scheduler_name}, keeping default")


def load_loras(pipe: StableDiffusionXLControlNetPipeline, lora_id: str, strength: float):
    """Load era-specific LoRA weights."""
    if lora_id not in LORA_CONFIG:
        logger.warning(f"Unknown era: {lora_id}")
        return
    
    models_dir = Path(os.getenv("MODELS_DIR", "/workspace/models"))
    lora_url = LORA_CONFIG[lora_id]
    lora_filename = f"experimance_{lora_id}_sdxl.safetensors"
    
    try:
        lora_path = download_model(lora_url, lora_filename, models_dir)
        
        # Unload existing LoRAs first
        if hasattr(pipe, 'unload_lora_weights'):
            pipe.unload_lora_weights()
        
        pipe.load_lora_weights(str(lora_path), adapter_name=lora_id)
        pipe.set_adapters([lora_id], adapter_weights=[strength])
        logger.info(f"Loaded {lora_id} LoRA with strength {strength}")
        
    except Exception as e:
        logger.error(f"Failed to load {lora_id} LoRA: {e}")
        raise


@app.get("/healthcheck")
async def healthcheck() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Check GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
                "total": torch.cuda.get_device_properties(0).total_memory
            }
        
        response = HealthCheckResponse(
            status="healthy",
            model_server_healthy=True,
            models_loaded=list(loaded_models.keys()),
            memory_usage={
                "ram_mb": memory_info.rss / 1024 / 1024,
                "gpu_memory": gpu_memory
            },
            uptime=time.time() - startup_time if startup_time else None
        )
        
        return response.to_dict()
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_server_healthy": False,
            "error": str(e)
        }


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available models and configurations."""
    response = ModelListResponse(
        available_models=list(MODEL_CONFIG.keys()),
        available_controlnets=list(CONTROLNET_CONFIG.keys()),
        available_eras=list(LORA_CONFIG.keys()),
        available_schedulers=["auto", "euler", "euler_a", "dpm_multi", "dpm_single", "ddim", "lcm"]
    )
    return response.to_dict()


@app.post("/generate")
async def generate_image(request: GenerateRequest) -> Dict[str, Any]:
    """Generate an image using ControlNet."""
    start_time = time.time()
    
    logger.info(f"Received generation request: {request.model} model, prompt: {request.prompt[:50]}...")
    
    try:
        # Convert request to our data type for validation
        data = ControlNetGenerateData(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            depth_map_b64=request.depth_map_b64,
            mock_depth=request.mock_depth,
            model=request.model,
            controlnet=request.controlnet,
            era=request.era,
            steps=request.steps or MODEL_CONFIG[request.model]["steps"],
            cfg=request.cfg or MODEL_CONFIG[request.model]["cfg"],
            seed=request.seed,
            lora_strength=request.lora_strength,
            controlnet_strength=request.controlnet_strength,
            scheduler=request.scheduler,
            use_karras_sigmas=request.use_karras_sigmas,
            width=request.width,
            height=request.height
        )
        
        # Validate the request
        validation_errors = data.validate()
        if validation_errors:
            raise HTTPException(status_code=400, detail=f"Validation errors: {validation_errors}")
        
        # Load the model (this applies the MODEL_CONFIG scheduler when scheduler="auto")
        pipe = load_model(data.model, data.controlnet)
        
        # Only override scheduler if user explicitly requests a different one
        if data.scheduler != "auto":
            setup_scheduler(pipe, data.scheduler, data.use_karras_sigmas, data.model)
        elif data.use_karras_sigmas is not None:
            # If user wants to override just the Karras setting, do that
            setup_scheduler(pipe, "auto", data.use_karras_sigmas, data.model)
        
        # Load era-specific LoRA if specified
        if data.era:
            load_loras(pipe, data.era, data.lora_strength)
        
        # Get or generate depth map
        logger.info(f"Debug - depth_map_b64: {'Present' if data.depth_map_b64 else 'None'}, mock_depth: {data.mock_depth}")
        if data.depth_map_b64:
            depth_image = data.get_depth_image()
            logger.info("Got depth map")
        elif data.mock_depth:
            logger.info("Generating mock depth map")
            depth_image = generate_mock_depth(data.width, data.height)
        else:
            raise HTTPException(status_code=400, detail="Either depth_map_b64 or mock_depth=true must be provided")
        
        # Resize depth map to match output dimensions
        depth_image = depth_image.resize((data.width, data.height))
        
        # Set seed if provided
        if data.seed is not None:
            generator = torch.Generator().manual_seed(data.seed)
            seed_used = data.seed
        else:
            random_seed = int(torch.randint(0, 2**32-1, (1,)).item())
            generator = torch.Generator().manual_seed(random_seed)
            seed_used = random_seed
        
        # Generate the image
        logger.info(f"Generating image with prompt: {data.prompt[:50]}...")
        
        result = pipe(
            prompt=data.prompt,
            negative_prompt=data.negative_prompt,
            image=depth_image,
            num_inference_steps=data.steps,
            guidance_scale=data.cfg,
            controlnet_conditioning_scale=data.controlnet_strength,
            width=data.width,
            height=data.height,
            generator=generator
        )
        
        generation_time = time.time() - start_time
        
        # Create response
        response = ControlNetGenerateResponse.success_response(
            image=result.images[0],
            generation_time=generation_time,
            seed_used=seed_used,
            model_used=data.model,
            era_used=data.era,
            metadata={
                "steps": data.steps,
                "cfg": data.cfg,
                "scheduler": pipe.scheduler.__class__.__name__,
                "controlnet": data.controlnet,
                "controlnet_strength": data.controlnet_strength,
                "lora_strength": data.lora_strength if data.era else None
            }
        )
        
        logger.info(f"Image generated successfully in {generation_time:.2f}s")
        return response.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.error(traceback.format_exc())
        
        response = ControlNetGenerateResponse.error_response(
            error_message=f"Generation failed: {str(e)}"
        )
        return response.to_dict()


@app.post("/preload")
async def preload_model(model_name: str, controlnet_id: str = "sdxl_small") -> Dict[str, Any]:
    """Preload a specific model with specified ControlNet."""
    try:
        if model_name not in MODEL_CONFIG:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        
        if controlnet_id not in CONTROLNET_CONFIG:
            raise HTTPException(status_code=400, detail=f"Unknown ControlNet: {controlnet_id}")
        
        load_model(model_name, controlnet_id)
        return {"status": "success", "message": f"Model {model_name} with ControlNet {controlnet_id} preloaded"}
        
    except Exception as e:
        logger.error(f"Failed to preload model {model_name} with ControlNet {controlnet_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preload model: {str(e)}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ControlNet Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()
    
    # Override with environment variables if set
    host = os.getenv("MODEL_SERVER_HOST", args.host)
    port = int(os.getenv("MODEL_SERVER_PORT", str(args.port)))
    log_level = os.getenv("LOG_LEVEL", args.log_level).lower()
    
    logger.info(f"Starting model server on {host}:{port}")
    logger.info(f"Command line args: host={args.host}, port={args.port}")
    logger.info(f"Final values: host={host}, port={port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )
