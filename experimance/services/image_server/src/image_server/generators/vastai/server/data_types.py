"""
Data types for the ControlNet image generation server.

This module defines the request and response data structures for ControlNet-based image generation.
"""

import base64
import dataclasses
import random
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from io import BytesIO
from PIL import Image


# Sample prompts for testing
SAMPLE_PROMPTS = [
    "A majestic mountain landscape with snow-capped peaks and pristine lakes",
    "An ancient forest with towering trees and mystical atmosphere",
    "A futuristic cityscape with flying vehicles and neon lights",
    "A serene beach at sunset with golden waves",
    "A medieval castle on a hilltop surrounded by rolling hills",
    "An alien planet with purple skies and floating islands",
    "A cozy cabin in the woods during winter snowfall",
    "A bustling marketplace in an ancient civilization"
]


@dataclass
class ControlNetGenerateData:
    """
    Request payload for ControlNet image generation.
    
    This class defines all the parameters needed for generating images using ControlNet models.
    """
    prompt: str
    negative_prompt: Optional[str] = None
    depth_map_b64: Optional[str] = None  # base64 encoded depth map
    mock_depth: bool = False
    model: str = "lightning"  # lightning, hyper, base
    controlnet: str = "sdxl_small"  # sdxl_small, llite
    era: Optional[str] = None  # drone, experimance
    steps: int = 6
    cfg: float = 2.0
    seed: Optional[int] = None
    lora_strength: float = 1.0
    controlnet_strength: float = 0.8
    scheduler: str = "auto"  # auto, euler, euler_a, dpm_multi, dpm_single, ddim, lcm
    use_karras_sigmas: Optional[bool] = None
    width: int = 1024
    height: int = 1024
    
    @classmethod
    def for_test(cls) -> "ControlNetGenerateData":
        """Create a test payload for development and testing."""
        prompt = random.choice(SAMPLE_PROMPTS)
        era = random.choice([None, "drone", "experimance"])
        model = random.choice(["lightning", "hyper", "base"])
        controlnet = random.choice(["sdxl_small", "llite"])
        
        return cls(
            prompt=prompt,
            negative_prompt="blurry, low quality, artifacts, distorted",
            mock_depth=True,
            model=model,
            controlnet=controlnet,
            era=era,
            steps=6 if model == "lightning" else 6 if model == "hyper" else 20,
            cfg=1.0 if model == "lightning" else 2.0 if model == "hyper" else 7.5,
            width=1024,
            height=1024
        )
    
    def generate_payload_json(self) -> Dict[str, Any]:
        """Generate JSON representation of the payload for sending to model API."""
        return asdict(self)
    
    def get_depth_image(self) -> Optional[Image.Image]:
        """Decode base64 depth map to PIL Image if available."""
        if not self.depth_map_b64:
            return None
        
        try:
            image_data = base64.b64decode(self.depth_map_b64)
            return Image.open(BytesIO(image_data)).convert('RGB')
        except Exception:
            return None
    
    def validate(self) -> List[str]:
        """
        Validate the payload and return list of validation errors.
        """
        errors = []
        
        if not self.prompt or len(self.prompt.strip()) == 0:
            errors.append("Prompt cannot be empty")
        
        if len(self.prompt) > 1000:
            errors.append("Prompt must be less than 1000 characters")
        
        if self.model not in ["lightning", "hyper", "base"]:
            errors.append("Model must be one of: lightning, hyper, base")
        
        if self.controlnet not in ["sdxl_small", "llite"]:
            errors.append("ControlNet must be one of: sdxl_small, llite")
        
        if self.era and self.era not in ["drone", "experimance"]:
            errors.append("Era must be one of: drone, experimance")
        
        if not (1 <= self.steps <= 50):
            errors.append("Steps must be between 1 and 50")
        
        if not (0.1 <= self.cfg <= 20.0):
            errors.append("CFG must be between 0.1 and 20.0")
        
        if not (0.0 <= self.lora_strength <= 2.0):
            errors.append("LoRA strength must be between 0.0 and 2.0")
        
        if not (0.0 <= self.controlnet_strength <= 2.0):
            errors.append("ControlNet strength must be between 0.0 and 2.0")
        
        if self.width % 64 != 0 or self.height % 64 != 0:
            errors.append("Width and height must be multiples of 64")
        
        if not (256 <= self.width <= 2048) or not (256 <= self.height <= 2048):
            errors.append("Width and height must be between 256 and 2048")
        
        if self.scheduler not in ["auto", "euler", "euler_a", "dpm_multi", "dpm_single", "ddim", "lcm"]:
            errors.append("Scheduler must be one of: auto, euler, euler_a, dpm_multi, dpm_single, ddim, lcm")
        
        return errors


@dataclass
class ControlNetGenerateResponse:
    """
    Response payload for ControlNet image generation.
    """
    success: bool
    image_b64: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    seed_used: Optional[int] = None
    model_used: Optional[str] = None
    era_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def success_response(
        cls,
        image: Image.Image,
        generation_time: float,
        seed_used: int,
        model_used: str,
        era_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ControlNetGenerateResponse":
        """Create a successful response with the generated image."""
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return cls(
            success=True,
            image_b64=image_b64,
            generation_time=generation_time,
            seed_used=seed_used,
            model_used=model_used,
            era_used=era_used,
            metadata=metadata or {}
        )
    
    @classmethod
    def error_response(cls, error_message: str) -> "ControlNetGenerateResponse":
        """Create an error response."""
        return cls(
            success=False,
            error_message=error_message
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class HealthCheckResponse:
    """Response for health check endpoint."""
    status: str
    model_server_healthy: bool
    models_loaded: List[str]
    memory_usage: Optional[Dict[str, Any]] = None
    uptime: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ModelListResponse:
    """Response for model listing endpoint."""
    available_models: List[str]
    available_controlnets: List[str]
    available_eras: List[str]
    available_schedulers: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return asdict(self)
