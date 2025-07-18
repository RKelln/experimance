#!/usr/bin/env python3
"""
SDXL ControlNet depth inference script for vast.ai
Uses the same models and workflow as fal_comfy_generator.py

Models are automatically downloaded from Google Storage to local paths on first run.
Subsequent runs use the cached local files for faster startup.

Usage:
  python vast_controlnet_test.py --prompt "Lightning striking a gothic tower" --depth_map depth.png
  python vast_controlnet_test.py --prompt "test" --mock_depth --lora_strength 0.8 --era wilderness
"""
import argparse
import torch
import uuid
import base64
import io
import os
import requests
from PIL import Image
import numpy as np
from diffusers import (
    StableDiffusionXLControlNetPipeline, 
    ControlNetModel,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)

# Model configuration - URLs and local paths
MODEL_CONFIG = {
    "controlnet": {
        "url": "https://storage.googleapis.com/experimance_models/controllllite_v01032064e_sdxl_blur-500-1000.safetensors",
        "local_path": "models/depth.safetensors"
    },
    "juggernaut_xl_lightning": {
        "url": "https://storage.googleapis.com/experimance_models/juggernautXL_juggXILightningByRD.safetensors",
        "local_path": "models/lightning.safetensors"
    },
    "hyper_model": {
        "url": "https://storage.googleapis.com/experimance_models/Juggernaut_X_RunDiffusion_Hyper.safetensors",
        "local_path": "models/hyper.safetensors"
    },
    "historical_lora": {
        "url": "https://storage.googleapis.com/experimance_models/drone_photo_v1.0_XL.safetensors",
        "local_path": "models/historical.safetensors"
    },
    "experimance_lora": {
        "url": "https://storage.googleapis.com/experimance_models/civitai_experimance_sdxl_lora_step_1000_1024x1024.safetensors",
        "local_path": "models/experimance.safetensors"
    }
}

def download_model(url, local_path):
    """Download model from URL to local path if not already exists."""
    if os.path.exists(local_path):
        print(f"Model already exists at {local_path}")
        return local_path
    
    print(f"Downloading {url} to {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded {local_path}")
        return local_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def create_mock_depth_map(width=1024, height=1024):
    """Create a simple mock depth map for testing."""
    # Create a simple gradient depth map
    depth = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Create a radial gradient from center
            center_x, center_y = width // 2, height // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            depth[y, x] = int(255 * (distance / max_distance))
    
    return Image.fromarray(depth, mode='L')

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    if base64_string.startswith('data:image'):
        # Remove data URL prefix
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--depth_map", type=str, help="Path to depth map image file")
    parser.add_argument("--depth_b64", type=str, help="Base64 encoded depth map")
    parser.add_argument("--mock_depth", action="store_true", help="Use mock depth map for testing")
    parser.add_argument("--steps", type=int, default=6, help="Number of inference steps (4-6 for lightning)")
    parser.add_argument("--cfg", type=float, default=2, help="CFG scale (1-2 for lightning)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, choices=['lightning', 'hyper', 'base'], default='lightning',
                       help="Base model to use: lightning (Juggernaut XI Lightning), hyper (Hyper model), or base (SDXL base)")
    parser.add_argument("--era", type=str, choices=['wilderness', 'pre_industrial', 'future'], 
                       help="Era for LoRA selection")
    parser.add_argument("--lora_strength", type=float, default=1.0, help="LoRA strength multiplier")
    parser.add_argument("--controlnet_strength", type=float, default=0.8, 
                       help="ControlNet conditioning scale (default 0.8)")
    parser.add_argument("--scheduler", type=str, choices=['euler', 'dpm_sde', 'auto'], default='auto',
                       help="Scheduler to use: euler (recommended for Lightning), dpm_sde, or auto")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated images")
    parser.add_argument("--negative_prompt", type=str, 
                       default="distorted, warped, blurry, text, cartoon, illustration, low quality, lowres")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    
    # Load ControlNet from Google Storage
    controlnet_config = MODEL_CONFIG["controlnet"]
    
    # Try to download the custom ControlNet (for future use)
    local_controlnet_path = download_model(controlnet_config["url"], controlnet_config["local_path"])
    
    # For now, use the standard depth ControlNet since the custom one has loading issues
    # TODO: Figure out how to properly load the ControlNet-Lite model
    try:
        print("Loading standard SDXL depth ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        print("Successfully loaded ControlNet")
        if local_controlnet_path:
            print(f"Note: Custom ControlNet downloaded to {local_controlnet_path} but not used yet")
    except Exception as e:
        print(f"Failed to load ControlNet: {e}")
        raise RuntimeError("Could not load ControlNet model")

    # Load the selected base model from Google Storage
    model_loaded = False
    local_model_path = None
    
    if args.model == 'lightning':
        # Load Juggernaut XI Lightning
        juggernaut_config = MODEL_CONFIG["juggernaut_xl_lightning"]
        local_model_path = download_model(juggernaut_config["url"], juggernaut_config["local_path"])
        if local_model_path:
            try:
                print(f"Loading Juggernaut XI Lightning from local file: {local_model_path}")
                pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                    local_model_path,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                ).to("cuda")
                print("Successfully loaded Juggernaut XI Lightning from local file")
                model_loaded = True
            except Exception as e:
                print(f"Failed to load Juggernaut from local file: {e}")
    
    elif args.model == 'hyper':
        # Load Hyper model
        hyper_config = MODEL_CONFIG["hyper_model"]
        local_model_path = download_model(hyper_config["url"], hyper_config["local_path"])
        if local_model_path:
            try:
                print(f"Loading Hyper model from local file: {local_model_path}")
                pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                    local_model_path,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                ).to("cuda")
                print("Successfully loaded Hyper model from local file")
                model_loaded = True
            except Exception as e:
                print(f"Failed to load Hyper model from local file: {e}")
    
    # Fallback to standard SDXL base model if nothing else worked or if base was explicitly requested
    if not model_loaded or args.model == 'base':
        try:
            print("Loading standard SDXL base model...")
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to("cuda")
            print("Loaded SDXL base model")
            local_model_path = None  # Mark as not using custom model
        except Exception as e:
            print(f"Fallback model also failed: {e}")
            raise RuntimeError("Could not load any base model")

    # Set the scheduler based on user choice or auto-detection
    if args.scheduler == 'euler' or (args.scheduler == 'auto' and local_model_path and args.model in ['lightning', 'hyper']):
        # Use Euler for Lightning and Hyper models
        try:
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True
            )
            print(f"Using Euler scheduler (recommended for {args.model} models)")
        except:
            print("Euler scheduler failed, falling back to DPM++ SDE")
            args.scheduler = 'dpm_sde'
    
    if args.scheduler == 'dpm_sde' or args.scheduler != 'euler':
        # Use DPM++ SDE as fallback or when explicitly requested
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            final_sigmas_type="zero"  # Important for Lightning/Hyper models
        )
        print("Using DPM++ SDE scheduler")

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    # Load LoRA based on era
    lora_strength = args.lora_strength
    if args.era:
        if args.era in ['wilderness', 'pre_industrial']:
            # Use the historical LoRA
            lora_config = MODEL_CONFIG["historical_lora"]
            lora_strength *= 0.8  # Reduce strength for historical eras
            print(f"Loading historical LoRA for {args.era} era (strength: {lora_strength})")
        else:
            # Use the experimance LoRA for future era
            lora_config = MODEL_CONFIG["experimance_lora"]
            lora_strength *= 1.2  # Increase strength for future era
            print(f"Loading experimance LoRA for future era (strength: {lora_strength})")
        
        # Load the LoRA into the pipeline
        try:
            # Download LoRA model
            local_lora_path = download_model(lora_config["url"], lora_config["local_path"])
            if local_lora_path:
                print(f"Loading LoRA from local file: {local_lora_path}")
                
                # Check if peft is available
                try:
                    import peft
                    pipe.load_lora_weights(local_lora_path)
                    print(f"LoRA loaded with strength: {lora_strength}")
                    
                    # Set the LoRA strength - use the actual adapter name that was created
                    # Try to get the adapter names from the UNet component specifically
                    try:
                        # Check if the UNet has adapters
                        if hasattr(pipe.unet, 'peft_config') and pipe.unet.peft_config:
                            adapter_names = list(pipe.unet.peft_config.keys())
                            if adapter_names:
                                adapter_name = adapter_names[0]
                                print(f"Setting LoRA strength for adapter: {adapter_name}")
                                pipe.set_adapters([adapter_name], adapter_weights=[lora_strength])
                            else:
                                print("Warning: No adapters found in UNet peft_config")
                        else:
                            # Fallback: just use 'default_0' which is the common name
                            print("Using default adapter name: default_0")
                            pipe.set_adapters(["default_0"], adapter_weights=[lora_strength])
                    except Exception as adapter_error:
                        print(f"Error setting adapter: {adapter_error}")
                        print("LoRA loaded but strength not set - will use default strength")
                except ImportError:
                    print("Error: peft package is required for LoRA loading. Install with: pip install peft")
                    print("Continuing without LoRA...")
            else:
                print("Failed to download LoRA, continuing without LoRA...")
        except Exception as e:
            print(f"Failed to load LoRA: {e}")
            if "PEFT backend is required" in str(e):
                print("Install peft with: pip install peft")
            print("Continuing without LoRA...")

    # Prepare depth map
    depth_image = None
    if args.mock_depth:
        depth_image = create_mock_depth_map()
        print("Using mock depth map")
    elif args.depth_b64:
        depth_image = base64_to_image(args.depth_b64)
        print("Using base64 depth map")
    elif args.depth_map:
        depth_image = Image.open(args.depth_map).convert('L')
        print(f"Loaded depth map from {args.depth_map}")
    else:
        print("No depth map provided, using mock depth map")
        depth_image = create_mock_depth_map()

    # Ensure depth map is the right size
    depth_image = depth_image.resize((1024, 1024))

    # Set up generator
    seed = None
    if args.seed is not None:
        seed = args.seed
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        generator = None

    print(f"Generating image...")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}, CFG: {args.cfg}")

    # Generate image
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=depth_image,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
        controlnet_conditioning_scale=args.controlnet_strength,
        height=1024,
        width=1024
    )

    image = result.images[0]

    fname = f"out_controlnet_{args.model}_seed_{seed}_{args.scheduler}_era_{args.era}_{uuid.uuid4()}.png"
    output_path = os.path.join(args.output_dir, fname)
    image.save(output_path)
    print(f"Saved {output_path}")

    # Also save the depth map for reference
    depth_fname = f"depth_{uuid.uuid4()}.png"
    depth_output_path = os.path.join(args.output_dir, depth_fname)
    depth_image.save(depth_output_path)
    print(f"Saved depth map: {depth_output_path}")

if __name__ == "__main__":
    main()
