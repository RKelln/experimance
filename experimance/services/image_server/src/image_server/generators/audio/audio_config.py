"""
Audio generation configuration classes for the image server.
"""

import os
from typing import Literal, Optional
from pathlib import Path
from pydantic import Field

from image_server.generators.config import BaseGeneratorConfig
from experimance_common.constants import MODELS_DIR


class BaseAudioGeneratorConfig(BaseGeneratorConfig):
    """Base configuration for audio generators."""
    
    # Model storage location
    models_dir: Path = Field(default=MODELS_DIR, description="Directory for storing audio generation models")
    
    # GPU configuration
    audio_gpu_id: Optional[int] = Field(default=0, description="GPU ID for audio generation (None for CPU)")
    cuda_visible_devices: Optional[str] = Field(default=None, description="CUDA_VISIBLE_DEVICES for subprocess (e.g., '0' or '1,2')")
    use_subprocess: bool = Field(default=False, description="Run audio generation in separate subprocess with isolated GPU")
    subprocess_timeout_seconds: int = Field(default=300, description="Timeout for subprocess operations")
    subprocess_max_retries: int = Field(default=3, description="Maximum retry attempts for subprocess operations")
    
    # Common audio generation parameters
    duration_s: int = Field(default=24, description="Duration of generated audio in seconds")
    sample_rate: int = Field(default=44100, description="Sample rate for generated audio")
    
    # Quality and processing options
    normalize_loudness: bool = Field(default=True, description="Apply EBU R128 loudness normalization")
    target_lufs: float = Field(default=-23.0, description="Target loudness in LUFS (EBU R128 standard)")
    true_peak_dbfs: float = Field(default=-2.0, description="True peak limit in dBFS")
    
    # Seamless loop configuration
    enable_seamless_loop: bool = Field(default=True, description="Make audio seamlessly loopable")
    tail_duration_s: float = Field(default=1.5, description="Tail duration for crossfade in seconds")


class Prompt2AudioConfig(BaseAudioGeneratorConfig):
    """Configuration for prompt-to-audio generation using TangoFlux."""
    
    strategy: Literal["prompt2audio"] = "prompt2audio"
    
    # TangoFlux generation parameters
    steps: int = Field(default=30, description="Number of diffusion steps")
    guidance_scale: float = Field(default=4.5, description="Guidance scale for generation")
    candidates: int = Field(default=2, description="Number of candidates to generate per request")
    
    # Cache and selection parameters
    tau_use: float = Field(default=0.35, description="Minimum CLAP similarity threshold for using cached audio")
    tau_accept_new: float = Field(default=0.40, description="Minimum CLAP similarity threshold for caching new audio")
    tau_prompt_sem: float = Field(default=0.70, description="Semantic similarity threshold for prompt matching")
    temperature: float = Field(default=0.25, description="Temperature for weighted random selection")
    reuse_k: int = Field(default=8, description="Number of semantic matches to consider")
    
    # Prefetch and caching configuration
    prefetch_new: bool = Field(default=False, description="Prefetch new variants synchronously after serving")
    prefetch_in_background: bool = Field(default=True, description="Prefetch new variants in background thread")
    target_per_prompt: int = Field(default=3, description="Target number of variants to maintain per prompt")
    max_new_when_prefetch: int = Field(default=2, description="Maximum new variants to generate during prefetch")
    max_per_prompt: int = Field(default=5, description="Maximum variants to store per prompt")
    cap_strategy: Literal["quality", "diversity"] = Field(default="quality", description="Strategy for pruning variants")
    
    # Model configuration
    model_name: str = Field(default="declare-lab/TangoFlux", description="TangoFlux model name")
    clap_model: str = Field(default="laion/clap-htsat-unfused", description="CLAP model for similarity scoring")
    use_bge: bool = Field(default=True, description="Use BGE embeddings for semantic text matching")
    bge_model: str = Field(default="BAAI/bge-small-en-v1.5", description="BGE model for text embeddings")
    
    # File management
    cache_dir: str = Field(default="audio_cache", description="Directory for audio cache")
    render_dir: str = Field(default="renders/audio", description="Directory for rendered audio files")
