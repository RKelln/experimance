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
from pydantic import BaseModel, Field

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
    ModelListResponse,
    LoraData
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
loaded_loras: Dict[str, List[LoraData]] = {}  # Track loaded LoRAs per model
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

KNOWN_LORAS = {
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
    
    # Clear GPU memory using our unload function
    unload_model()  # Unload all models
    
    logger.info("Model server shutdown complete")


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Experimance Model Server", 
    version="1.0.0",
    lifespan=lifespan
)


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


def unload_model(cache_key: Optional[str] = None):
    """
    Unload a specific model or all models from VRAM.
    
    Args:
        cache_key: Specific model to unload. If None, unloads all models.
    """
    if cache_key and cache_key in loaded_models:
        logger.info(f"Unloading model: {cache_key}")
        del loaded_models[cache_key]
        # Also clear LoRA cache for this model
        if cache_key in loaded_loras:
            del loaded_loras[cache_key]
    elif cache_key is None:
        # Unload all models
        for model_key in list(loaded_models.keys()):
            logger.info(f"Unloading model: {model_key}")
            del loaded_models[model_key]
        # Clear all LoRA caches
        loaded_loras.clear()
    
    # Force garbage collection and clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")


def load_model(model_name: str, controlnet_id: str = "sdxl_small") -> StableDiffusionXLControlNetPipeline:
    """Load and cache a specific model with specified ControlNet."""
    # Create a cache key that includes both model and controlnet
    cache_key = f"{model_name}_{controlnet_id}"
    
    if cache_key in loaded_models:
        logger.info(f"Using cached model: {cache_key}")
        return loaded_models[cache_key]
    
    # Unload all existing models to free VRAM (single-model cache)
    if loaded_models:
        logger.info("Unloading existing models to free VRAM")
        unload_model()  # Unload all models
    
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


def load_loras(pipe: StableDiffusionXLControlNetPipeline, loras: List[LoraData]):
    """Load multiple LoRA weights, optimized for known LoRAs that stay loaded."""
    
    # Create a cache key for the current model
    model_cache_key = None
    for key, model in loaded_models.items():
        if model is pipe:
            model_cache_key = key
            break
    
    if model_cache_key is None:
        logger.warning("Could not find model cache key for LoRA optimization")
        model_cache_key = "unknown"
    
    # Get currently loaded LoRAs
    current_loras = loaded_loras.get(model_cache_key, [])
    
    # Check if the LoRA set is exactly the same as last time (including strengths)
    if current_loras == loras:
        logger.info(f"LoRA set unchanged, skipping reload: {[f'{lora.name}({lora.strength})' for lora in loras]}")
        return
    
    # Check if we have any custom LoRAs (not in KNOWN_LORAS)
    custom_loras = [lora for lora in loras if lora.name not in KNOWN_LORAS]
    current_custom_loras = [lora for lora in current_loras if lora.name not in KNOWN_LORAS]
    
    # If custom LoRAs changed, we need to unload and reload everything
    if custom_loras != current_custom_loras:
        logger.info(f"Custom LoRAs changed, full reload required")
        if hasattr(pipe, 'unload_lora_weights') and current_loras:
            pipe.unload_lora_weights()
            logger.info("Unloaded all LoRAs for custom LoRA change")
        need_full_reload = True
    else:
        # Only known LoRAs, check if we can just update strengths
        need_full_reload = False
        
        # If no LoRAs are currently loaded, we need to load them
        if not current_loras:
            need_full_reload = True
            logger.info("No LoRAs currently loaded, loading known LoRAs")
    
    models_dir = Path(os.getenv("MODELS_DIR", "/workspace/models"))
    
    try:
        if need_full_reload:
            # Load all LoRAs (known + custom)
            adapter_names = []
            adapter_weights = []
            
            for lora in loras:
                # Check if this is a known LoRA or a custom one
                if lora.name in KNOWN_LORAS:
                    # Known LoRA
                    lora_url = KNOWN_LORAS[lora.name]
                    lora_filename = f"experimance_{lora.name}_sdxl.safetensors"
                    lora_path = download_model(lora_url, lora_filename, models_dir)
                else:
                    # Custom LoRA
                    if lora.name.startswith("http"):
                        # Download from URL
                        lora_filename = f"custom_{lora.name.split('/')[-1]}"
                        lora_path = download_model(lora.name, lora_filename, models_dir)
                    else:
                        # Assume it's a local file path
                        lora_path = Path(lora.name)
                        if not lora_path.exists():
                            logger.error(f"LoRA file not found: {lora_path}")
                            continue
                
                # Load the LoRA
                adapter_name = f"lora_{len(adapter_names)}"
                pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                adapter_names.append(adapter_name)
                adapter_weights.append(lora.strength)
                
                logger.info(f"Loaded LoRA {lora.name} with strength {lora.strength} as adapter {adapter_name}")
            
            # Set all adapters at once if any were loaded
            if adapter_names:
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                logger.info(f"Activated {len(adapter_names)} LoRAs: {[f'{lora.name}({lora.strength})' for lora in loras]}")
                
                # Cache the loaded LoRAs for this model
                loaded_loras[model_cache_key] = loras.copy()
        else:
            # Only update strengths for known LoRAs (no custom LoRAs changed)
            logger.info("Only known LoRA strengths changed, updating weights without reloading files")
            
            # We can only do strength-only updates if we have the same number of LoRAs
            # and they're in the same order (since adapter names are sequential)
            if len(loras) != len(current_loras):
                logger.info(f"LoRA count changed ({len(current_loras)} -> {len(loras)}), need full reload")
                need_full_reload = True
            else:
                # Build adapter names and weights for current LoRAs
                adapter_names = [f"lora_{i}" for i in range(len(loras))]
                adapter_weights = [lora.strength for lora in loras]
                
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                logger.info(f"Updated LoRA strengths: {[f'{lora.name}({lora.strength})' for lora in loras]}")
                
                # Cache the updated LoRAs
                loaded_loras[model_cache_key] = loras.copy()
            
            # If we discovered we need a full reload, do it now
            if need_full_reload:
                logger.info("Falling back to full reload due to LoRA count/order change")
                if hasattr(pipe, 'unload_lora_weights') and current_loras:
                    pipe.unload_lora_weights()
                    logger.info("Unloaded all LoRAs for full reload")
                
                # Load all LoRAs (known + custom)
                adapter_names = []
                adapter_weights = []
                
                for lora in loras:
                    # Check if this is a known LoRA or a custom one
                    if lora.name in KNOWN_LORAS:
                        # Known LoRA
                        lora_url = KNOWN_LORAS[lora.name]
                        lora_filename = f"experimance_{lora.name}_sdxl.safetensors"
                        lora_path = download_model(lora_url, lora_filename, models_dir)
                    else:
                        # Custom LoRA
                        if lora.name.startswith("http"):
                            # Download from URL
                            lora_filename = f"custom_{lora.name.split('/')[-1]}"
                            lora_path = download_model(lora.name, lora_filename, models_dir)
                        else:
                            # Assume it's a local file path
                            lora_path = Path(lora.name)
                            if not lora_path.exists():
                                logger.error(f"LoRA file not found: {lora_path}")
                                continue
                    
                    # Load the LoRA
                    adapter_name = f"lora_{len(adapter_names)}"
                    pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                    adapter_names.append(adapter_name)
                    adapter_weights.append(lora.strength)
                    
                    logger.info(f"Loaded LoRA {lora.name} with strength {lora.strength} as adapter {adapter_name}")
                
                # Set all adapters at once if any were loaded
                if adapter_names:
                    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    logger.info(f"Activated {len(adapter_names)} LoRAs: {[f'{lora.name}({lora.strength})' for lora in loras]}")
                    
                    # Cache the loaded LoRAs for this model
                    loaded_loras[model_cache_key] = loras.copy()
        
        # Handle the case where no LoRAs are requested
        if not loras and current_loras:
            logger.info("No LoRAs requested, unloading existing LoRAs")
            if hasattr(pipe, 'unload_lora_weights'):
                pipe.unload_lora_weights()
            loaded_loras[model_cache_key] = []
        elif not loras:
            logger.info("No LoRAs to load")
        
    except Exception as e:
        logger.error(f"Failed to load LoRAs: {e}")
        raise


def load_loras_legacy(pipe: StableDiffusionXLControlNetPipeline, lora_id: str, strength: float):
    """Load era-specific LoRA weights (legacy function for backward compatibility)."""
    if lora_id not in KNOWN_LORAS:
        logger.warning(f"Unknown LoRA: {lora_id}")
        return
    
    # Convert to new format and use new function
    loras = [LoraData(name=lora_id, strength=strength)]
    load_loras(pipe, loras)


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
    """List available models and configurations."""
    response = ModelListResponse(
        available_models=list(MODEL_CONFIG.keys()),
        available_controlnets=list(CONTROLNET_CONFIG.keys()),
        available_schedulers=["auto", "euler", "euler_a", "dpm_multi", "dpm_single", "ddim", "lcm"]
    )
    return response.to_dict()


@app.post("/unload")
async def unload_models(model_key: Optional[str] = None) -> Dict[str, Any]:
    """Unload models from VRAM."""
    try:
        # Get memory usage before unloading
        gpu_memory_before = {}
        if torch.cuda.is_available():
            gpu_memory_before = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "cached_gb": torch.cuda.memory_reserved() / (1024**3),
                "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        
        # Unload models
        if model_key:
            unload_model(model_key)
            message = f"Unloaded model: {model_key}"
        else:
            unload_model()
            message = "Unloaded all models"
        
        # Get memory usage after unloading
        gpu_memory_after = {}
        if torch.cuda.is_available():
            gpu_memory_after = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "cached_gb": torch.cuda.memory_reserved() / (1024**3),
                "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        
        return {
            "status": "success",
            "message": message,
            "loaded_models": list(loaded_models.keys()),
            "gpu_memory_before": gpu_memory_before,
            "gpu_memory_after": gpu_memory_after
        }
        
    except Exception as e:
        logger.error(f"Failed to unload models: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/generate")
async def generate_image(request: ControlNetGenerateData) -> Dict[str, Any]:
    """Generate an image using ControlNet."""
    start_time = time.time()
    
    logger.info(f"Received generation request: {request.model} model, prompt: {request.prompt[:50]}...")
    
    try:
        # Set default values based on model if not provided
        steps = request.steps if request.steps is not None else MODEL_CONFIG[request.model]["steps"]
        cfg = request.cfg if request.cfg is not None else MODEL_CONFIG[request.model]["cfg"]
        
        # Create a new instance with resolved defaults for internal use
        data = request.model_copy(update={"steps": steps, "cfg": cfg})
        
        # Pydantic validation is automatic, no need for manual validation
        
        # Load the model (this applies the MODEL_CONFIG scheduler when scheduler="auto")
        pipe = load_model(data.model, data.controlnet)
        
        # Only override scheduler if user explicitly requests a different one
        if data.scheduler != "auto":
            setup_scheduler(pipe, data.scheduler, data.use_karras_sigmas, data.model)
        elif data.use_karras_sigmas is not None:
            # If user wants to override just the Karras setting, do that
            setup_scheduler(pipe, "auto", data.use_karras_sigmas, data.model)
        
        # Load LoRAs if specified
        if data.loras:
            load_loras(pipe, data.loras)
        
        # Get or generate depth map
        logger.info(f"Debug - depth_map_b64: {'Present' if data.depth_map_b64 else 'None'}, mock_depth: {data.mock_depth}")
        if data.depth_map_b64:
            depth_image = data.get_depth_image()
            if depth_image is None:
                logger.error("🚨 DEPTH MAP DECODE FAILED! Provided depth_map_b64 could not be decoded")
                logger.error(f"Depth map prefix: {data.depth_map_b64[:50]}..." if len(data.depth_map_b64) > 50 else f"Full depth map: {data.depth_map_b64}")
                logger.warning("Falling back to mock depth map generation")
                depth_image = generate_mock_depth(data.width, data.height)
            else:
                logger.info(f"✅ Successfully decoded depth map: {depth_image.size}")
        elif data.mock_depth:
            logger.info("Generating mock depth map as requested")
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
            control_guidance_start=data.control_guidance_start,
            control_guidance_end=data.control_guidance_end,
            width=data.width,
            height=data.height,
            generator=generator
        )
        
        generation_time = time.time() - start_time
        
        # Create response with LoRA metadata
        lora_metadata = []
        if data.loras:
            lora_metadata = [{"name": lora.name, "strength": lora.strength} for lora in data.loras]
        
        response = ControlNetGenerateResponse.success_response(
            image=result.images[0],
            generation_time=generation_time,
            seed_used=seed_used,
            model_used=data.model,
            metadata={
                "steps": data.steps,
                "cfg": data.cfg,
                "scheduler": pipe.scheduler.__class__.__name__,
                "controlnet": data.controlnet,
                "controlnet_strength": data.controlnet_strength,
                "control_guidance_start": data.control_guidance_start,
                "control_guidance_end": data.control_guidance_end,
                "loras": lora_metadata
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
