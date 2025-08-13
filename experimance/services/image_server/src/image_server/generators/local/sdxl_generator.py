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
from image_server.generators.generator import ImageGenerator
from image_server.generators.config import BaseGeneratorConfig
from pydantic import Field

logger = logging.getLogger(__name__)

try:  # Optional heavy imports (loaded lazily where possible)
    import torch
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLControlNetPipeline,
        ControlNetModel,
        EulerDiscreteScheduler,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0
    _DIFFUSERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _DIFFUSERS_AVAILABLE = False
    torch = None  # type: ignore


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
    controlnet_id: Optional[str] = "diffusers/controlnet-depth-sdxl-1.0-small"
    steps: int = 6
    guidance_scale: float = 1.5
    strength: float = 0.4 # image-to-image strength, if image is passed into generator
    width: int = 1024
    height: int = 1024
    compile_unet: bool = True
    enable_xformers: bool = True
    enable_attention_slicing: bool = True
    device: str = "cuda"
    models_dir: Path = MODELS_DIR

    # Dependency injection hooks (used for lightweight/unit testing)
    # Using Any type and exclude from serialization since these are callables
    pipeline_factory: Optional[Callable[..., Any]] = Field(default=None, exclude=True)
    controlnet_factory: Optional[Callable[..., Any]] = Field(default=None, exclude=True)
    scheduler_config: Optional[dict[str, Any]] = Field(default=None)


class LocalSDXLGenerator(ImageGenerator):
    """Local SDXL generator supporting lightning (single-file) and base diffusers models.

    Backward-compatible with previous SDXLLightningGenerator naming.
    """

    def _configure(self, config: Any, **kwargs):  # type: ignore[override]
        # Use provided config if it's a LocalSDXLConfig, otherwise fall back to defaults
        if isinstance(config, LocalSDXLConfig):
            self.cfg: LocalSDXLConfig = config
        else:
            # config may be a BaseGeneratorConfig coming from factory; we ignore most fields for now
            cfg = kwargs.get("lightning_config") or kwargs.get("local_config") or LocalSDXLConfig()
            self.cfg: LocalSDXLConfig = cfg
        
        self._pipeline: Optional[StableDiffusionXLControlNetPipeline] = None  # type: ignore

        # Allow overrides
        for field in ("model", "steps", "guidance_scale", "width", "height", "controlnet_id"):
            if field in kwargs:
                setattr(self.cfg, field, kwargs[field])

        # Resolve device
        if not self._cuda_available():  # Fall back to cpu if cuda missing
            self.cfg.device = "cpu"

    async def start(self):  # noqa: D401
        """Warm load the model pipeline if not already loaded."""
        if self._pipeline is None:
            await self._init_pipeline()

    async def stop(self):  # noqa: D401
        """Release pipeline resources"""
        if self._pipeline is not None and torch is not None:  # type: ignore[attr-defined]
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.cuda.empty_cache()  # type: ignore[attr-defined]

    def _cuda_available(self) -> bool:
        return bool(torch and torch.cuda.is_available())  # type: ignore[attr-defined]

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

    async def generate_image(self, prompt: str, *, image: Optional[Image.Image] = None, depth_map: Optional[Image.Image] = None, **kwargs) -> str:  # type: ignore[override]
        """Generate an image for the given prompt.

        Args:
            prompt: Text prompt
            image: Optional PIL Image for image-to-image generation
            depth_map: Optional PIL Image depth map (applied if ControlNet active)

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
        if image is not None:
            # Image-to-image generation
            pipe_inputs["image"] = image
            pipe_inputs["strength"] = kwargs.get("strength", 0.8)  # Default strength for i2i
            
            # For image-to-image, we typically don't specify exact dimensions
            # The pipeline will use the input image dimensions
            if "width" not in kwargs and "height" not in kwargs:
                # Don't set width/height, let pipeline use input image dimensions
                pass
            else:
                pipe_inputs["width"] = width
                pipe_inputs["height"] = height
        else:
            # Text-to-image generation
            pipe_inputs["width"] = width
            pipe_inputs["height"] = height

        # Add ControlNet depth map if provided (takes precedence over img2img image)
        if depth_map and isinstance(self._pipeline, StableDiffusionXLControlNetPipeline):
            pipe_inputs["image"] = depth_map
            # For ControlNet, we always need to specify dimensions
            pipe_inputs["width"] = width
            pipe_inputs["height"] = height
            # Remove img2img strength if using ControlNet
            pipe_inputs.pop("strength", None)

        model_type, _, _ = self._detect_model_type(self.cfg.model)
        if depth_map and isinstance(self._pipeline, StableDiffusionXLControlNetPipeline):
            generation_mode = "controlnet"
        elif image is not None:
            generation_mode = "img2img"
        else:
            generation_mode = "txt2img"
        logger.info("LocalSDXL(%s): generating %s (%s steps, cfg=%.2f)", model_type, generation_mode, gen_steps, guidance)

        result = await asyncio.to_thread(self._pipeline, **pipe_inputs)
        if not hasattr(result, "images") or not result.images:
            raise RuntimeError("Pipeline returned no images")
        output_path = self._get_output_path("png")
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

        # Optimisations
        if self.cfg.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing(1)
        if self.cfg.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.debug("xFormers not available")
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "unet") and hasattr(pipe.unet, "set_attn_processor"):
            pipe.unet.set_attn_processor(AttnProcessor2_0())

        pipe = pipe.to(self.cfg.device)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
            if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
                pipe.controlnet.to(memory_format=torch.channels_last)
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]

        if self.cfg.compile_unet and hasattr(torch, "compile"):
            try:  # pragma: no cover - compile path environment dependent
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)  # type: ignore[attr-defined]
            except Exception as e:
                logger.debug("torch.compile failed: %s", e)

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