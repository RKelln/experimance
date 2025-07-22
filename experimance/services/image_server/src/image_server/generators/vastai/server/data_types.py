"""
Data types for the ControlNet image generation server.

This module defines the request and response data structures for ControlNet-based image generation.
"""

import base64
import logging
import random
from typing import Optional, Dict, Any, List
from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator


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

class LoraData(BaseModel):
    """
    Data structure for LoRA (Low-Rank Adaptation) parameters.
    
    This class holds the LoRA model name and its strength.
    """
    name: str
    strength: float = Field(default=1.0, ge=0.0, le=2.0, description="LoRA strength between 0.0 and 2.0")


class ControlNetGenerateData(BaseModel):
    """
    Request payload for ControlNet image generation.
    
    This class defines all the parameters needed for generating images using ControlNet models.
    Uses Pydantic for validation and works directly with FastAPI.
    """
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, max_length=1000, description="Negative prompt to avoid certain elements")
    depth_map_b64: Optional[str] = Field(None, description="Base64 encoded depth map")
    mock_depth: bool = Field(False, description="Generate a mock depth map for testing")
    model: str = Field("lightning", pattern="^(lightning|hyper|base)$", description="Model to use for generation")
    controlnet: str = Field("sdxl_small", pattern="^(sdxl_small|llite)$", description="ControlNet model to use")
    steps: Optional[int] = Field(None, ge=1, le=50, description="Number of inference steps")
    cfg: Optional[float] = Field(None, ge=0.1, le=20.0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(None, ge=0, description="Random seed for reproducible generation")
    loras: List[LoraData] = Field(default_factory=list, description="List of LoRA models to apply")
    controlnet_strength: float = Field(0.8, ge=0.0, le=2.0, description="ControlNet conditioning strength")
    control_guidance_start: float = Field(0.0, ge=0.0, le=1.0, description="Percentage of total steps at which ControlNet starts applying")
    control_guidance_end: float = Field(1.0, ge=0.0, le=1.0, description="Percentage of total steps at which ControlNet stops applying")
    scheduler: str = Field("auto", pattern="^(auto|euler|euler_a|dpm_multi|dpm_single|ddim|lcm)$", description="Sampling scheduler")
    use_karras_sigmas: Optional[bool] = Field(None, description="Override Karras sigma setting for scheduler")
    width: int = Field(1024, ge=256, le=2048, description="Output image width")
    height: int = Field(1024, ge=256, le=2048, description="Output image height")
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Ensure dimensions are multiples of 64."""
        if v % 64 != 0:
            raise ValueError(f"Dimension must be a multiple of 64, got {v}")
        return v
    
    @model_validator(mode='after')
    def validate_depth_input(self) -> 'ControlNetGenerateData':
        """Ensure either depth_map_b64 or mock_depth is provided."""
        if not self.depth_map_b64 and not self.mock_depth:
            raise ValueError("Either depth_map_b64 or mock_depth=true must be provided")
        return self
    
    @model_validator(mode='after')
    def validate_control_guidance_timing(self) -> 'ControlNetGenerateData':
        """Ensure control_guidance_start <= control_guidance_end."""
        if self.control_guidance_start > self.control_guidance_end:
            raise ValueError(f"control_guidance_start ({self.control_guidance_start}) must be <= control_guidance_end ({self.control_guidance_end})")
        return self
    
    @classmethod
    def for_test(cls) -> "ControlNetGenerateData":
        """Create a test payload for development and testing."""
        prompt = random.choice(SAMPLE_PROMPTS)
        model = random.choice(["lightning", "hyper", "base"])
        controlnet = random.choice(["sdxl_small", "llite"])
        
        # Create with explicit values for all parameters
        return cls(
            prompt=prompt,
            negative_prompt="blurry, low quality, artifacts, distorted",
            depth_map_b64=None,  # Will use mock_depth instead
            mock_depth=True,
            model=model,
            controlnet=controlnet,
            steps=6 if model == "lightning" else 6 if model == "hyper" else 20,
            cfg=1.0 if model == "lightning" else 2.0 if model == "hyper" else 7.5,
            seed=None,  # Will be generated randomly
            loras=[],  # No LoRAs for test
            controlnet_strength=0.8,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            scheduler="auto",
            use_karras_sigmas=None,
            width=1024,
            height=1024
        )
    
    def generate_payload_json(self) -> Dict[str, Any]:
        """Generate JSON representation of the payload for sending to model API."""
        return self.model_dump()
    
    def get_depth_image(self) -> Optional[Image.Image]:
        """Decode base64 depth map to PIL Image if available."""
        if not self.depth_map_b64:
            return None

        # Remove any data URL prefix if present (handles any image format)
        # Examples: "data:image/png;base64,", "data:image/jpeg;base64,", etc.
        base64_data = self.depth_map_b64
        if base64_data.startswith("data:image/"):
            # Find the comma that separates the prefix from the actual base64 data
            comma_index = base64_data.find(",")
            if comma_index != -1:
                base64_data = base64_data[comma_index + 1:]
        
        try:
            image_data = base64.b64decode(base64_data)
            depth_image = Image.open(BytesIO(image_data)).convert('RGB')
            # Log successful decode with image info
            logging.getLogger(__name__).debug(f"✅ Depth map decoded successfully: {depth_image.size}, mode: {depth_image.mode}")
            return depth_image
        except Exception as e:
            # Log detailed error information
            logger = logging.getLogger(__name__)
            logger.error(f"❌ Failed to decode depth map base64 data: {e}")
            logger.error(f"Data length: {len(self.depth_map_b64)}")
            logger.error(f"Data prefix: {self.depth_map_b64[:100]}..." if len(self.depth_map_b64) > 100 else f"Full data: {self.depth_map_b64}")
            return None


class ControlNetGenerateResponse(BaseModel):
    """
    Response payload for ControlNet image generation.
    """
    success: bool
    image_b64: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    seed_used: Optional[int] = None
    model_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def success_response(
        cls,
        image: Image.Image,
        generation_time: float,
        seed_used: int,
        model_used: str,
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
        return self.model_dump()


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    model_server_healthy: bool
    models_loaded: List[str]
    memory_usage: Optional[Dict[str, Any]] = None
    uptime: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return self.model_dump()


class ModelListResponse(BaseModel):
    """Response for model listing endpoint."""
    available_models: List[str]
    available_controlnets: List[str]
    available_schedulers: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return self.model_dump()
