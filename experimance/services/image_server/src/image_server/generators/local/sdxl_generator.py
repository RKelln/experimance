"""Local SDXL generator implementations.

This module adds a production-quality SDXL Lightning generator integrated into the
generic ImageGenerator interface. It supersedes the ad-hoc juggernaut_lightning module.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Any, Callable
from pathlib import Path

from PIL import Image
from pydantic import Field

from experimance_common.constants import MODELS_DIR
from experimance_common.image_utils import base64url_to_image
from image_server.generators.generator import ImageGenerator, GeneratorCapabilities
from image_server.generators.config import BaseGeneratorConfig
from pydantic import Field

logger = logging.getLogger(__name__)

try:  # Optional heavy imports (loaded lazily where possible)
    import torch
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLControlNetPipeline,
        ControlNetModel,
        EulerDiscreteScheduler,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0
    _DIFFUSERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _DIFFUSERS_AVAILABLE = False
    torch = None  # type: ignore
    logger.warning("diffusers/torch not installed, local SDXL generation will be unavailable. Install extras: `uv sync --extra local_gen`")


class LocalSDXLConfig(BaseGeneratorConfig):
    """Configuration for LocalSDXLGenerator supporting various SDXL model sources.

    model can be:
    - URL (http/https): Downloads single-file checkpoint and uses from_single_file
      Example: "https://storage.googleapis.com/experimance_models/juggernautXL_juggXILightningByRD.safetensors"
    - HuggingFace model ID (org/name): Uses from_pretrained 
      Example: "stabilityai/stable-diffusion-xl-base-1.0"
    - Local filename: Uses from_single_file with existing file in models_dir
      Example: "my_model.safetensors"
    """
    
    # Strategy identification (required for factory)
    strategy: str = "local_sdxl"
    
    # Single model attribute - auto-detects URL, model ID, or filename
    model: str = "https://storage.googleapis.com/experimance_models/juggernautXL_juggXILightningByRD.safetensors"
    controlnet_id: Optional[str] = None  # Disabled by default - enable for depth control
    steps: int = 6
    guidance_scale: float = 1.5
    strength: float = 0.4 # image-to-image strength, if image is passed into generator, 0-1, where 0 keeps the original image and 1 is and completely new image
    width: int = 1024
    height: int = 1024
    compile_unet: bool = False
    enable_xformers: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = False
    enable_cpu_offload: bool = False  # CPU offloading - not needed on GPU 0
    device: str = "cuda"  # Use generic cuda - explicit GPU IDs (cuda:0, cuda:1) cause issues
    models_dir: Path = MODELS_DIR
    warmup_on_start: bool = True

    # Dependency injection hooks (used for lightweight/unit testing)
    # Using Any type and exclude from serialization since these are callables
    pipeline_factory: Optional[Callable[..., Any]] = Field(default=None, exclude=True)
    controlnet_factory: Optional[Callable[..., Any]] = Field(default=None, exclude=True)
    scheduler_config: Optional[dict[str, Any]] = Field(default=None)


class LocalSDXLGenerator(ImageGenerator):
    """Local SDXL Lightning image generator.

    Optimized for interactive generation with lightning-fast sampling.
    
    Backward-compatible with previous SDXLLightningGenerator naming.
    """
    
    # Declare generator capabilities
    supported_capabilities = {
        GeneratorCapabilities.IMAGE_TO_IMAGE,
        GeneratorCapabilities.CONTROLNET,
        GeneratorCapabilities.NEGATIVE_PROMPTS,
        GeneratorCapabilities.SEEDS,
        GeneratorCapabilities.CUSTOM_SCHEDULERS
    }

    def __init__(self, config: BaseGeneratorConfig, output_dir: str = "/tmp", **kwargs):
        """Initialize LocalSDXL generator with sequential processing for thread safety."""
        # Force max_concurrent=1 for SDXL to prevent tensor dimension conflicts
        super().__init__(config, output_dir, max_concurrent=1, **kwargs)

    def _configure(self, config: Any, **kwargs):  # type: ignore[override]
        # Use provided config if it's a LocalSDXLConfig, otherwise fall back to defaults
        if isinstance(config, LocalSDXLConfig):
            self.cfg: LocalSDXLConfig = config
        else:
            raise ValueError("Invalid config type, expected LocalSDXLConfig")

        # Dynamic pipeline management - initialize as None, create on demand
        self._txt2img_pipeline: Optional[StableDiffusionXLPipeline] = None
        self._img2img_pipeline: Optional[StableDiffusionXLImg2ImgPipeline] = None  # type: ignore
        self._controlnet_pipeline: Optional[StableDiffusionXLControlNetPipeline] = None  # type: ignore
        self._current_mode = None
        self._base_components = None  # Shared components between pipelines
        
        # For compatibility with existing code that expects _pipeline
        self._pipeline = None

        # Allow overrides
        for field in ("model", "steps", "guidance_scale", "width", "height", "controlnet_id", "strength"):
            if field in kwargs:
                setattr(self.cfg, field, kwargs[field])

        # Resolve device using environment-based approach instead of explicit GPU IDs
        if not self._cuda_available():  # Fall back to cpu if cuda missing
            self.cfg.device = "cpu"
        else:    
            # Use environment-based GPU selection instead of explicit device IDs
            # This avoids the memory/performance issues with cuda:0, cuda:1 etc.
            self.cfg.device = self._get_optimal_device()
                
            # Clear GPU memory cache
            if torch is not None:
                torch.cuda.empty_cache()
                
            if _DIFFUSERS_AVAILABLE:
                self._setup_pytorch_optimizations()

    def _get_optimal_device(self) -> str:
        """Select the optimal CUDA device using environment variables and PyTorch APIs.
        
        This avoids explicit device IDs (cuda:0, cuda:1) which cause memory/performance issues.
        Instead uses PyTorch's native device selection and CUDA_VISIBLE_DEVICES.
        
        Returns:
            str: Device string, either "cuda" or "cpu"
        """
        if not torch or not torch.cuda.is_available():
            return "cpu"
            
        # Check if CUDA_VISIBLE_DEVICES is set to restrict devices
        import os
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_devices:
            logger.info(f"CUDA_VISIBLE_DEVICES={visible_devices} - using environment device selection")
        
        # Use PyTorch's current device selection - much more reliable than explicit IDs
        current_device = torch.cuda.current_device()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(current_device) if device_count > 0 else "Unknown"
        
        if visible_devices:
            # When CUDA_VISIBLE_DEVICES is set, show both logical and physical GPU info
            physical_gpu = visible_devices.split(',')[current_device] if ',' in visible_devices else visible_devices
            logger.info(f"Using physical GPU {physical_gpu} ({device_name}) - appears as logical GPU {current_device}/{device_count-1} to PyTorch")
        else:
            logger.info(f"Using GPU {current_device}/{device_count-1}: {device_name}")
        
        # Always use generic "cuda" - PyTorch handles device selection internally
        return "cuda"


    async def start(self):  # noqa: D401
        """Initialize the queue system, then warm load the model pipeline if not already loaded."""
        # Start the base class queue system first
        await super().start()
        
        if self._pipeline is None:
            await self._init_pipeline()
            
        # Warm up the pipeline with a dummy generation to populate caches
        if self.cfg.warmup_on_start and not self.cfg.pipeline_factory:
            await self._warmup_pipeline()

    async def _warmup_pipeline(self):
        """Generate a dummy image to warm up caches and trigger compilation for maximum performance."""
        try:
            import time
            warmup_start = time.time()
            logger.info("LocalSDXL: warming up pipeline (this may take several minutes with compilation enabled)...")
            # Use very fast settings for warmup
            old_output_dir = self.output_dir
            self.output_dir = Path("/tmp")  # Don't save warmup image permanently
            
            # CRITICAL: Use the same dimensions and settings as production to trigger compilation
            # for the actual computation graph that will be used in production
            warmup_steps = self.cfg.steps if self.cfg.compile_unet else 1
            
            # Call the implementation directly, bypassing the queue system during warmup
            await self._generate_image_impl(
                "warmup compilation test", 
                width=self.cfg.width,   # Use production width
                height=self.cfg.height, # Use production height  
                steps=warmup_steps,     # Use production steps
                guidance_scale=self.cfg.guidance_scale  # Use production guidance_scale
            )
            warmup_duration = time.time() - warmup_start
            logger.info("LocalSDXL: pipeline warmup complete in %.1f seconds - subsequent renders will be fast", warmup_duration)
            
            # Restore original output directory
            self.output_dir = old_output_dir
            
        except Exception as e:
            logger.warning("LocalSDXL: warmup failed, continuing anyway: %s", e)

    async def stop(self):  # noqa: D401
        """Release pipeline resources and stop queue processing."""
        # Call parent stop first to handle queue cleanup
        await super().stop()
        
        # Then handle SDXL-specific cleanup
        if self._pipeline is not None and torch is not None:  # type: ignore[attr-defined]
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.cuda.empty_cache()  # type: ignore[attr-defined]

    def _cuda_available(self) -> bool:
        return bool(torch and torch.cuda.is_available())  # type: ignore[attr-defined]

    def _setup_pytorch_optimizations(self) -> None:
        """Configure PyTorch optimizations for better performance and memory usage."""
        if torch is None or not torch.cuda.is_available():
            return

        # 1) Enable TF32 for Ampere+ GPUs and CuDNN autotuner
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        
        # 2) Enable scaled dot product attention optimizations
        torch.backends.cuda.enable_math_sdp(True)  # type: ignore[attr-defined]
        torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined] 
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # type: ignore[attr-defined]
        
        # 3) Reduce memory fragmentation
        if hasattr(torch.cuda, 'memory_pool_empty_cache'):
            torch.cuda.memory_pool_empty_cache()  # type: ignore[attr-defined]

    def _setup_environment_variables(self) -> None:
        """Configure environment variables for optimal performance."""
        import os
        
        # Reduce inductor warnings for smaller GPUs
        os.environ.setdefault("TORCH_LOGS", "-inductor")
        # Enable expandable segments to avoid memory fragmentation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # XFormers environment variables for GPU compatibility
        os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
        
        # Disable max autotune for smaller GPUs to avoid warnings
        if torch is not None:
            torch._inductor.config.max_autotune = False  # type: ignore[attr-defined]

    def _optimize_pipeline(self, pipe: Any) -> Any:  # type: ignore[type-arg]
        """Apply optimizations to the pipeline for better performance and memory usage.
        
        Args:
            pipe: The pipeline to optimize
            
        Returns:
            The optimized pipeline
        """
        # XFormers optimization - do this first to know if we should skip attention slicing
        xformers_enabled = False
        if self.cfg.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            xformers_enabled = self._enable_xformers(pipe)
            
        # Memory optimizations - but avoid conflicts with XFormers/SDPA
        if self.cfg.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            if xformers_enabled:
                logger.info("Skipping attention slicing - XFormers is enabled (would cause serious slowdowns)")
            elif torch is not None and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.info("Skipping attention slicing - PyTorch SDPA is available (would cause serious slowdowns)")
            else:
                # Use standard attention slicing with moderate slice size
                pipe.enable_attention_slicing(1)
                logger.debug("Enabled attention slicing with slice_size=1")
            
        if self.cfg.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
            logger.debug("Enabled VAE slicing for memory optimization")
            
        # Set attention processor - but only if XFormers wasn't enabled
        if not xformers_enabled and hasattr(pipe, "unet") and hasattr(pipe.unet, "set_attn_processor"):
            pipe.unet.set_attn_processor(AttnProcessor2_0())
            logger.debug("Set AttnProcessor2_0 for optimized attention")

        # CPU offloading if needed
        self._setup_cpu_offloading(pipe)
        
        # Move to device and optimize memory format
        pipe = pipe.to(self.cfg.device)
        if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
            self._setup_memory_format(pipe)
            # Set TF32 for this pipeline specifically
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]

        # Compile UNet if requested
        if self.cfg.compile_unet and hasattr(torch, "compile"):
            self._compile_unet(pipe)
            
        return pipe

    def _enable_xformers(self, pipe: Any) -> bool:  # type: ignore[type-arg]
        """Enable XFormers memory efficient attention with proper error handling.
        
        Returns:
            bool: True if XFormers was successfully enabled, False otherwise
        """
        try:
            if torch is not None and torch.cuda.is_available() and "cuda:" in self.cfg.device:  # type: ignore[attr-defined]
                gpu_id = int(self.cfg.device.split(":")[1]) if ":" in self.cfg.device else 0
                with torch.cuda.device(gpu_id):  # type: ignore[attr-defined]
                    import xformers
                    logger.info(f"XFormers version: {xformers.__version__} on {self.cfg.device}")
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info(f"XFormers memory efficient attention enabled on {self.cfg.device}")
                    return True
            else:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info(f"XFormers memory efficient attention enabled on {self.cfg.device}")
                return True
        except ImportError:
            logger.warning("XFormers not installed - falling back to standard attention (may use more VRAM)")
            return False
        except Exception as e:
            logger.warning(f"XFormers failed to enable on {self.cfg.device}: {e}")
            logger.info("Falling back to standard attention (may use more VRAM)")
            try:
                import xformers
                logger.info(f"XFormers is installed (version {xformers.__version__}) but failed to activate")
            except ImportError:
                logger.info("XFormers is not installed")
            return False

    def _setup_cpu_offloading(self, pipe: Any) -> None:  # type: ignore[type-arg]
        """Setup CPU offloading for memory-constrained situations."""
        # Only enable if explicitly requested via config
        if not self.cfg.enable_cpu_offload:
            return
            
        if hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload()
                logger.info(f"Enabled CPU offloading for {self.cfg.device} to save VRAM")
            except Exception as e:
                logger.warning(f"CPU offloading failed: {e}")
        elif hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                pipe.enable_sequential_cpu_offload()
                logger.info(f"Enabled sequential CPU offloading for {self.cfg.device} to save VRAM")
            except Exception as e:
                logger.warning(f"Sequential CPU offloading failed: {e}")

    def _setup_memory_format(self, pipe: Any) -> None:  # type: ignore[type-arg]
        """Setup optimal memory format for pipeline components."""
        if torch is None:
            return
            
        if hasattr(pipe, "unet") and pipe.unet is not None:
            pipe.unet.to(memory_format=torch.channels_last)  # type: ignore[attr-defined]
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to(memory_format=torch.channels_last)  # type: ignore[attr-defined]
        if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
            pipe.controlnet.to(memory_format=torch.channels_last)  # type: ignore[attr-defined]

    def _compile_unet(self, pipe: Any) -> None:  # type: ignore[type-arg]
        """Compile UNet for better performance."""
        if not hasattr(pipe, "unet") or pipe.unet is None:
            return
            
        try:  # pragma: no cover - compile path environment dependent
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug("torch.compile failed: %s", e)

    def _detect_model_type(self, model: str) -> tuple[str, str, Optional[str]]:
        """Detect model type and return (type, resolved_path_or_id, filename_if_applicable).
        
        Returns:
            tuple: (type, path_or_id, filename) where type is 'url', 'hf_id', or 'file'
        """
        if model.startswith(('http://', 'https://')):
            filename = Path(model).name
            return ('url', model, filename)
        elif '/' in model and not Path(model).exists():
            # Looks like HF model ID (org/name format)
            return ('hf_id', model, None)
        else:
            # Treat as local filename
            return ('file', model, model)

    async def _generate_image_impl(self, prompt: str, *, 
        image: Optional[Image.Image] = None, 
        image_b64: Optional[str] = None,
        depth_map: Optional[Image.Image] = None, 
        depth_map_b64: Optional[str] = None,
        **kwargs) -> str:  # type: ignore[override]
        """Internal implementation of image generation.

        Args:
            prompt: Text prompt
            image: Optional PIL Image for image-to-image generation
            image_b64: Optional base64-encoded image for image-to-image generation
            depth_map: Optional PIL Image depth map (applied if ControlNet active)
            depth_map_b64: Optional base64-encoded depth map (applied if ControlNet active)

        Returns:
            Path to saved output image.
        """
        self._validate_prompt(prompt)
        if self._pipeline is None:
            await self.start()
        assert self._pipeline is not None  # for mypy

        gen_steps = kwargs.get("steps", self.cfg.steps)
        guidance = kwargs.get("guidance_scale", self.cfg.guidance_scale)
        width = kwargs.get("width", self.cfg.width)
        height = kwargs.get("height", self.cfg.height)
        seed = kwargs.get("seed")
        strength = kwargs.get("strength", self.cfg.strength)

        logger.debug(f"Generating image with prompt: {prompt}, strength: {strength}")

        # Check if we'll be doing img2img generation
        will_do_img2img = image is not None or image_b64 is not None
        
        # Apply step compensation for img2img to maintain quality
        if will_do_img2img and strength > 0:
            # Calculate effective steps: total_steps * strength
            effective_steps = gen_steps * strength
            # We want exactly 4 effective steps for good quality
            target_effective_steps = 4
            if effective_steps < target_effective_steps:
                # Calculate the exact steps needed to get target_effective_steps
                compensated_steps = int(target_effective_steps / strength)
                if compensated_steps > gen_steps:
                    logger.info(f"LocalSDXL: Step compensation - increasing from {gen_steps} to {compensated_steps} steps "
                              f"(strength={strength:.2f}, effective={effective_steps:.1f}→{compensated_steps * strength:.1f})")
                    gen_steps = compensated_steps

        generator = None
        if seed is not None and torch is not None:  # type: ignore[attr-defined]
            generator = torch.Generator(device=self.cfg.device).manual_seed(int(seed))  # type: ignore[attr-defined]

        pipe_inputs: dict[str, Any] = dict(
            prompt=prompt,
            num_inference_steps=gen_steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
        )

        # Handle different generation modes
        if image_b64 is not None and image is None:
            image = base64url_to_image(image_b64)

        if depth_map_b64 is not None and depth_map is None:
            depth_map = base64url_to_image(depth_map_b64)

        # Determine pipeline mode and get appropriate pipeline
        if depth_map and self.cfg.controlnet_id:
            generation_mode = "controlnet"
            # Use ControlNet pipeline if available, otherwise txt2img
            if hasattr(self._pipeline, '__class__') and 'ControlNet' in self._pipeline.__class__.__name__:
                current_pipeline = self._pipeline
            else:
                # For now, fall back to txt2img for ControlNet
                current_pipeline = self._pipeline
                logger.warning("ControlNet requested but no ControlNet pipeline available, using txt2img")
            pipe_inputs["image"] = depth_map
            pipe_inputs["width"] = width
            pipe_inputs["height"] = height
        elif image is not None:
            generation_mode = "img2img"
            # We need to create or use an img2img pipeline
            if hasattr(self, '_img2img_pipeline') and self._img2img_pipeline is not None:
                current_pipeline = self._img2img_pipeline
            elif hasattr(self._pipeline, '__class__') and 'Img2Img' in self._pipeline.__class__.__name__:
                current_pipeline = self._pipeline
            else:
                logger.info("Creating img2img pipeline for image-to-image generation")
                # Create img2img pipeline from the existing txt2img pipeline components
                txt2img_pipeline = self._pipeline
                if _DIFFUSERS_AVAILABLE:
                    current_pipeline = StableDiffusionXLImg2ImgPipeline(
                        vae=txt2img_pipeline.vae,
                        text_encoder=txt2img_pipeline.text_encoder,
                        text_encoder_2=getattr(txt2img_pipeline, 'text_encoder_2', None),
                        unet=txt2img_pipeline.unet,
                        scheduler=txt2img_pipeline.scheduler,
                        tokenizer=txt2img_pipeline.tokenizer,
                        tokenizer_2=getattr(txt2img_pipeline, 'tokenizer_2', None),
                    )
                    # Keep the same optimizations
                    current_pipeline = current_pipeline.to(self.cfg.device)
                    # Cache it for future use
                    self._img2img_pipeline = current_pipeline
                else:
                    current_pipeline = self._pipeline
            pipe_inputs["image"] = image
            pipe_inputs["strength"] = strength
            # img2img doesn't need width/height as it uses the image dimensions
        else:
            generation_mode = "txt2img"
            # Always use the original txt2img pipeline for txt2img
            current_pipeline = self._pipeline
            pipe_inputs["width"] = width
            pipe_inputs["height"] = height

        model_type, _, _ = self._detect_model_type(self.cfg.model)
        logger.info(f"LocalSDXL(): generating {model_type} {generation_mode}({width}x{height}, {gen_steps} steps, cfg={guidance:.2f}, strength={strength if 'strength' in pipe_inputs else 'N/A'})")

        result = await asyncio.to_thread(current_pipeline, **pipe_inputs)
        if not hasattr(result, "images") or not result.images:
            raise RuntimeError("Pipeline returned no images")
        output_path = self._get_output_path(self.config.image_file_type)
        result.images[0].save(output_path)
        logger.info("LocalSDXL: saved %s", output_path)
        return output_path

    async def _init_pipeline(self) -> None:
        # Testing path: use injected pipeline factory if provided
        if self.cfg.pipeline_factory:
            logger.warning("using injected pipeline factory (test mode)")
            self._pipeline = self.cfg.pipeline_factory()
            return
            
        # Lightweight testing path: allow injected pipeline factory even when diffusers absent
        if not _DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers/torch not installed. Install extras: local_gen")

        # Configure PyTorch for better performance and fewer warnings
        if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
            self._setup_environment_variables()

        self.cfg.models_dir.mkdir(parents=True, exist_ok=True)
        
        model_type, model_path_or_id, filename = self._detect_model_type(self.cfg.model)
        
        if model_type == "url":
            # Download from URL if not already cached
            model_path = self.cfg.models_dir / filename
            if not model_path.exists():
                await self._download_model(model_path_or_id, model_path)
            logger.info("LocalSDXL(url): loading downloaded model %s", model_path)
            factory = self.cfg.pipeline_factory or StableDiffusionXLPipeline.from_single_file
            base_pipe = await asyncio.to_thread(
                factory,
                str(model_path),
                torch_dtype=torch.float16 if self.cfg.device == "cuda" else torch.float32,  # type: ignore[attr-defined]
                use_safetensors=True,
                variant="fp16" if self.cfg.device == "cuda" else None,
            )
        elif model_type == "file":
            # Local file path
            model_path = self.cfg.models_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(f"Local model file not found: {model_path}")
            logger.info("LocalSDXL(file): loading local model %s", model_path)
            factory = self.cfg.pipeline_factory or StableDiffusionXLPipeline.from_single_file
            base_pipe = await asyncio.to_thread(
                factory,
                str(model_path),
                torch_dtype=torch.float16 if self.cfg.device == "cuda" else torch.float32,  # type: ignore[attr-defined]
                use_safetensors=True,
                variant="fp16" if self.cfg.device == "cuda" else None,
            )
        else:  # hf_id
            logger.info("LocalSDXL(hf_id): loading HuggingFace model %s", model_path_or_id)
            factory = self.cfg.pipeline_factory or StableDiffusionXLPipeline.from_pretrained
            base_pipe = await asyncio.to_thread(
                factory,
                model_path_or_id,
                torch_dtype=torch.float16 if self.cfg.device == "cuda" else torch.float32,  # type: ignore[attr-defined]
                use_safetensors=True,
                variant="fp16" if self.cfg.device == "cuda" else None,
            )

        controlnet = None
        if self.cfg.controlnet_id:
            logger.info("LocalSDXL: loading ControlNet %s", self.cfg.controlnet_id)
            cn_factory = self.cfg.controlnet_factory or ControlNetModel.from_pretrained
            controlnet = await asyncio.to_thread(
                cn_factory,
                self.cfg.controlnet_id,
                torch_dtype=torch.float16 if self.cfg.device == "cuda" else torch.float32,  # type: ignore[attr-defined]
                use_safetensors=True,
            )

        if controlnet is not None:
            pipe = StableDiffusionXLControlNetPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                text_encoder_2=base_pipe.text_encoder_2,
                tokenizer=base_pipe.tokenizer,
                tokenizer_2=base_pipe.tokenizer_2,
                unet=base_pipe.unet,
                image_encoder=getattr(base_pipe, "image_encoder", None),
                scheduler=base_pipe.scheduler,
                feature_extractor=getattr(base_pipe, "feature_extractor", None),
                controlnet=controlnet,
            )
        else:
            pipe = base_pipe

        scheduler_cfg = self.cfg.scheduler_config or {
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
            "use_karras_sigmas": False,
        }
        pipe.scheduler = EulerDiscreteScheduler.from_config(scheduler_cfg)

        # Apply all optimizations to the pipeline
        pipe = self._optimize_pipeline(pipe)

        self._pipeline = pipe
        model_type, _, _ = self._detect_model_type(self.cfg.model)
        logger.info("LocalSDXL: pipeline ready (%s)", model_type)

    async def _download_model(self, url: str, target: Path) -> None:
        """Download model with progress bar and resume capability."""
        try:
            from tqdm.asyncio import tqdm
            import aiohttp
        except ImportError:
            logger.warning("tqdm not available, downloading without progress bar")
            return await self._download_model_simple(url, target)
        
        # Check if already fully downloaded
        if target.exists():
            logger.info("LocalSDXL: model already cached at %s", target)
            return
            
        logger.info("LocalSDXL: downloading model from %s", url)
        tmp = target.with_suffix(target.suffix + ".tmp")
        
        # Check for partial download
        resume_pos = 0
        if tmp.exists():
            resume_pos = tmp.stat().st_size
            logger.info("LocalSDXL: resuming download from byte %d", resume_pos)
        
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                
                # Get total size for progress bar
                content_length = resp.headers.get('content-length')
                if content_length:
                    total_size = int(content_length) + resume_pos
                else:
                    total_size = None
                
                # Open file in append mode if resuming
                mode = "ab" if resume_pos > 0 else "wb"
                with open(tmp, mode) as f:
                    with tqdm(
                        desc=f"Downloading {target.name}",
                        total=total_size,
                        initial=resume_pos,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        async for chunk in resp.content.iter_chunked(1 << 20):  # 1MB chunks
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        # Move completed download to final location
        tmp.rename(target)
        logger.info("LocalSDXL: model downloaded to %s", target)

    async def _download_model_simple(self, url: str, target: Path) -> None:
        """Simple download without progress bar (fallback).""" 
        import aiohttp
        
        if target.exists():
            logger.info("LocalSDXL: model already cached at %s", target)
            return
            
        logger.info("LocalSDXL: downloading model from %s (no progress display)", url)
        tmp = target.with_suffix(target.suffix + ".tmp")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1 << 20):
                        f.write(chunk)
        tmp.rename(target)


"""Backward compatibility aliases (can be deprecated later)."""
SDXLLightningConfig = LocalSDXLConfig  # type: ignore
SDXLLightningGenerator = LocalSDXLGenerator  # type: ignore

__all__ = [
    "LocalSDXLGenerator",
    "LocalSDXLConfig",
    "SDXLLightningGenerator",
    "SDXLLightningConfig",
]