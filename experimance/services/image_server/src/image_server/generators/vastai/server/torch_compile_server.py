#!/usr/bin/env python3
"""
Torch.compile optimized Lightning server with clean GPU memory.
Uses PyTorch VAE (known working) with torch.compile optimization.
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
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    EulerDiscreteScheduler
)

from data_types import (
    ControlNetGenerateData,
    ControlNetGenerateResponse,
    HealthCheckResponse,
    ModelListResponse,
    LoraData
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
loaded_models: Dict[str, Any] = {}
loaded_controlnets: Dict[str, ControlNetModel] = {}
startup_time = None

# MODEL CONFIGURATION
MODEL_CONFIG = {
    "lightning": {
        "repo_id": "https://storage.googleapis.com/experimance_models/juggernautXL_juggXILightningByRD.safetensors",
        "filename": "juggernaut-xl-lightning.safetensors",
        "scheduler": "EulerDiscreteScheduler",
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
        "steps": 4,
        "cfg": 1.0
    }
}

CONTROLNET_CONFIG = {
    "sdxl_small": {
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0-small",
        "filename": "controlnet-depth-sdxl-1.0-small.safetensors"
    }
}

def setup_torch_compile_optimizations():
    """Setup torch.compile optimizations as recommended by HuggingFace."""
    print("🔧 Setting up torch.compile optimizations...")
    
    # HuggingFace recommended compiler flags
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True
    
    # Additional optimizations
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True
    
    print("✅ Torch compile flags configured")

def enable_torch_compile_optimizations(pipe: StableDiffusionXLControlNetPipeline) -> StableDiffusionXLControlNetPipeline:
    """Enable torch.compile optimizations with clean GPU memory."""
    
    logger.info("🚀 Starting torch.compile optimization (this may take 5-10 minutes)...")
    compile_start = time.time()
    
    setup_torch_compile_optimizations()
    
    # 1. Standard optimizations first
    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("✅ xformers memory optimization enabled")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    # 2. VAE optimizations 
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
        logger.info("✅ VAE slicing enabled")
    
    if hasattr(pipe, 'enable_vae_tiling'):
        pipe.enable_vae_tiling()
        logger.info("✅ VAE tiling enabled")
    
    # 3. Attention slicing
    if hasattr(pipe, 'enable_attention_slicing'):
        try:
            pipe.enable_attention_slicing("auto")
            logger.info("✅ Attention slicing enabled")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
    
    # 4. Move to GPU FIRST
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        logger.info("✅ Pipeline moved to GPU")
    
    # 5. CRITICAL: Set channels_last memory format BEFORE compilation (HuggingFace recommendation)
    logger.info("🧠 Converting to channels_last memory format...")
    if torch.cuda.is_available():
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
            logger.info("✅ UNet and VAE converted to channels_last")
        except Exception as e:
            logger.warning(f"Could not optimize memory format: {e}")
    
    # 6. torch.compile optimization (the main event!) - HuggingFace recommendations
    logger.info("🔥 Compiling UNet with torch.compile...")
    try:
        # Compile UNet for maximum speed (HuggingFace: most compute-intensive)
        pipe.unet = torch.compile(
            pipe.unet,
            mode="max-autotune",  # HuggingFace recommendation for max speed
            fullgraph=True        # HuggingFace recommendation
        )
        logger.info("✅ UNet compiled successfully")
        
        # Compile VAE decoder for speed (HuggingFace: second most compute-intensive)
        logger.info("🔥 Compiling VAE decoder...")
        pipe.vae.decode = torch.compile(
            pipe.vae.decode,
            mode="max-autotune",  # HuggingFace recommendation
            fullgraph=True        # HuggingFace recommendation
        )
        logger.info("✅ VAE decode method compiled successfully")
        
    except Exception as e:
        logger.error(f"❌ torch.compile failed: {e}")
        logger.info("Continuing without torch.compile optimization")
    
    compile_time = time.time() - compile_start
    logger.info(f"🎯 torch.compile optimization completed in {compile_time:.1f}s")
    
    return pipe

def download_model(url: str, filename: str, models_dir: Path) -> Path:
    """Download a model file from URL if it doesn't exist locally."""
    import requests
    
    file_path = models_dir / filename
    
    if file_path.exists():
        logger.info(f"Model {filename} already exists")
        return file_path
    
    logger.info(f"Downloading {filename}...")
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
    """Load ControlNet model."""
    if controlnet_id in loaded_controlnets:
        logger.info(f"Using cached ControlNet: {controlnet_id}")
        return loaded_controlnets[controlnet_id]
    
    logger.info(f"Loading ControlNet: {controlnet_id}")
    
    config = CONTROLNET_CONFIG[controlnet_id]
    controlnet = ControlNetModel.from_pretrained(
        config["repo_id"],
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    loaded_controlnets[controlnet_id] = controlnet
    logger.info(f"ControlNet {controlnet_id} loaded successfully")
    return controlnet

def unload_model(cache_key: Optional[str] = None):
    """Unload models from VRAM."""
    
    if cache_key and cache_key in loaded_models:
        logger.info(f"Unloading model: {cache_key}")
        del loaded_models[cache_key]
    elif cache_key is None:
        for model_key in list(loaded_models.keys()):
            logger.info(f"Unloading model: {model_key}")
            del loaded_models[model_key]
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")

def load_model(model_name: str, controlnet_id: str = "sdxl_small") -> StableDiffusionXLControlNetPipeline:
    """Load and cache a model with torch.compile optimizations."""
    cache_key = f"{model_name}_{controlnet_id}_compiled"
    
    if cache_key in loaded_models:
        logger.info(f"Using cached compiled model: {cache_key}")
        return loaded_models[cache_key]
    
    # Unload existing models to ensure clean memory
    if loaded_models:
        logger.info("Unloading existing models for clean torch.compile")
        unload_model()
    
    logger.info(f"Loading torch.compile optimized model: {model_name}")
    models_dir = Path(os.getenv("MODELS_DIR", "~/projects/experimance/models")).expanduser()
    
    config = MODEL_CONFIG[model_name]
    controlnet = load_controlnet(controlnet_id)
    
    # Download Lightning model
    model_path = download_model(config["repo_id"], config["filename"], models_dir)
    
    # Load pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        str(model_path),
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    # Set up scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        **config["scheduler_config"]
    )
    
    # Apply torch.compile optimizations
    pipe = enable_torch_compile_optimizations(pipe)
    
    loaded_models[cache_key] = pipe
    logger.info(f"torch.compile optimized model {cache_key} loaded successfully")
    return pipe

def generate_mock_depth(width: int, height: int) -> Image.Image:
    """Generate a mock depth map."""
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    depth_array = (255 * (1 - distance / max_distance)).astype(np.uint8)
    return Image.fromarray(depth_array, mode='L')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global startup_time
    
    # Startup
    logger.info("Starting torch.compile Optimized Lightning Model Server...")
    startup_time = time.time()
    
    # Preload ControlNet and Lightning model
    load_controlnet()
    
    default_model = os.getenv("PRELOAD_MODEL", "lightning")
    if default_model in MODEL_CONFIG:
        try:
            load_model(default_model, "sdxl_small")
            logger.info(f"Preloaded compiled {default_model} model")
        except Exception as e:
            logger.error(f"Failed to preload {default_model}: {e}")
    
    startup_duration = time.time() - startup_time
    logger.info(f"torch.compile server startup complete in {startup_duration:.1f}s")
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down torch.compile server...")
    unload_model()
    logger.info("torch.compile server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="torch.compile Optimized Lightning Server", 
    version="5.0.0",
    lifespan=lifespan
)

@app.get("/healthcheck")
async def healthcheck() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        gpu_memory = {}
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_reserved()
            
            gpu_memory = {
                "allocated_gb": allocated_memory / (1024**3),
                "cached_gb": cached_memory / (1024**3),
                "total_gb": total_memory / (1024**3),
                "free_gb": (total_memory - cached_memory) / (1024**3),
                "usage_percent": (allocated_memory / total_memory) * 100
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
    """List available models."""
    response = ModelListResponse(
        available_models=["lightning"],
        available_controlnets=["sdxl_small"],
        available_schedulers=["euler"]
    )
    return response.to_dict()

@app.post("/generate")
async def generate_image_compiled(request: ControlNetGenerateData) -> Dict[str, Any]:
    """Generate an image using torch.compile optimized pipeline."""
    start_time = time.time()
    
    logger.info(f"torch.compile generation request: {request.model}, prompt: {request.prompt[:50]}...")
    
    try:
        # Use Lightning model with optimal defaults
        model_name = request.model if request.model in MODEL_CONFIG else "lightning"
        
        config = MODEL_CONFIG[model_name]
        steps = request.steps if request.steps is not None else config["steps"]
        cfg = request.cfg if request.cfg is not None else config["cfg"]
        
        data = request.model_copy(update={"steps": steps, "cfg": cfg})
        
        # Load the compiled model
        pipe = load_model(model_name, data.controlnet)
        
        # Get depth map
        if data.depth_map_b64:
            depth_image = data.get_depth_image()
            if depth_image is None:
                logger.warning("Depth map decode failed, using mock depth")
                depth_image = generate_mock_depth(data.width, data.height)
        elif data.mock_depth:
            depth_image = generate_mock_depth(data.width, data.height)
        else:
            raise HTTPException(status_code=400, detail="Either depth_map_b64 or mock_depth=true must be provided")
        
        depth_image = depth_image.resize((data.width, data.height))
        
        # Set seed
        if data.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(data.seed)
            seed_used = data.seed
        else:
            random_seed = int(torch.randint(0, 2**32-1, (1,)).item())
            generator = torch.Generator(device="cuda").manual_seed(random_seed)
            seed_used = random_seed
        
        # torch.compile GENERATION
        logger.info(f"torch.compile generating: {data.width}x{data.height}, {data.steps} steps")
        
        generation_start = time.time()
        
        with torch.inference_mode():
            result = pipe(
                prompt=data.prompt,
                negative_prompt=data.negative_prompt,
                image=depth_image,
                num_inference_steps=data.steps,
                guidance_scale=data.cfg,
                controlnet_conditioning_scale=data.controlnet_strength,
                control_guidance_start=data.control_guidance_start,
                control_guidance_end=data.control_guidance_end,
                width=data.width,
                height=data.height,
                generator=generator,
                output_type="pil",
                return_dict=True
            )
            final_image = result.images[0]
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Create response
        response = ControlNetGenerateResponse.success_response(
            image=final_image,
            generation_time=total_time,
            seed_used=seed_used,
            model_used=f"{model_name}_torch_compile",
            metadata={
                "steps": data.steps,
                "cfg": data.cfg,
                "scheduler": pipe.scheduler.__class__.__name__,
                "controlnet": data.controlnet,
                "controlnet_strength": data.controlnet_strength,
                "generation_time_only": generation_time,
                "torch_compile_optimized": True,
                "optimization_type": "max-autotune"
            }
        )
        
        logger.info(f"🚀 torch.compile generation completed in {total_time:.2f}s (pure: {generation_time:.2f}s)")
        return response.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"torch.compile generation failed: {e}")
        logger.error(traceback.format_exc())
        
        response = ControlNetGenerateResponse.error_response(
            error_message=f"torch.compile generation failed: {str(e)}"
        )
        return response.to_dict()

@app.post("/unload")
async def unload_models(model_key: Optional[str] = None) -> Dict[str, Any]:
    """Unload models from VRAM."""
    try:
        if model_key:
            unload_model(model_key)
            message = f"Unloaded model: {model_key}"
        else:
            unload_model()
            message = "Unloaded all models"
        
        return {
            "status": "success",
            "message": message,
            "loaded_models": list(loaded_models.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to unload models: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/preload")
async def preload_model(model_name: str = "lightning", controlnet_id: str = "sdxl_small") -> Dict[str, Any]:
    """Preload a model and ControlNet into VRAM."""
    try:
        logger.info(f"Preloading compiled model: {model_name} with ControlNet: {controlnet_id}")
        start_time = time.time()
        
        pipe = load_model(model_name, controlnet_id)
        
        load_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": f"Preloaded compiled model {model_name} with ControlNet {controlnet_id}",
            "model_name": model_name,
            "controlnet_id": controlnet_id,
            "load_time": load_time,
            "loaded_models": list(loaded_models.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to preload model {model_name}: {e}")
        return {
            "status": "error",
            "message": str(e),
            "model_name": model_name,
            "controlnet_id": controlnet_id
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch.compile Optimized Lightning Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8005, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()
    
    host = os.getenv("MODEL_SERVER_HOST", args.host)
    port = int(os.getenv("MODEL_SERVER_PORT", str(args.port)))
    log_level = os.getenv("LOG_LEVEL", args.log_level).lower()
    
    logger.info(f"Starting torch.compile Lightning model server on {host}:{port}")
    logger.info("🔥 This server uses torch.compile for maximum performance")
    logger.info("⚠️  First generation will be slow due to compilation")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )