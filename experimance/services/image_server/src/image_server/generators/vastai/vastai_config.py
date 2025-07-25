"""
Configuration for VastAI image generator.
"""

from typing import Literal, Optional
from typing_extensions import Annotated
from pydantic import Field, StringConstraints

from image_server.generators.config import BaseGeneratorConfig


class VastAIGeneratorConfig(BaseGeneratorConfig):
    """Configuration for VastAI image generator."""
    
    strategy: Literal["vastai"] = "vastai"
    
    pre_warm: bool = True

    # VastAI specific configuration
    vastai_api_key: Optional[str] = Field(
        default=None, 
        description="VastAI API key. If not provided, will use environment variable or vastai CLI auth"
    )
    
    instance_timeout: int = Field(
        default=600, 
        description="Timeout for VastAI instance operations in seconds",
        ge=30
    )
    
    pre_warm_timeout: int = Field(
        default=30,
        description="Timeout for pre-warming requests in seconds", 
        ge=10
    )
    
    create_if_none: bool = Field(
        default=True, 
        description="Whether to create new instances if none are available"
    )
    
    wait_for_ready: bool = Field(
        default=True, 
        description="Whether to wait for instances to be ready before using them"
    )
    
    # Model server configuration
    model_name: Annotated[Literal["hyper", "lightning", "base"], 
                          StringConstraints(to_lower=True)] = Field(
        default="hyper", 
        description="Model to use for generation (lightning, hyper, base)"
    )
    
    steps: Optional[int] = Field(
        default=None, 
        description="Number of inference steps (overrides model default)"
    )
    
    cfg: Optional[float] = Field(
        default=None, 
        description="CFG scale (overrides model default)"
    )
    
    scheduler: Annotated[Literal["auto", "euler", "euler_a", "dpm_multi", "lcm"], 
                          StringConstraints(to_lower=True)] = Field(
        default="auto", 
        description="Scheduler to use (auto, euler, euler_a, dpm_multi, lcm)"
    )
    
    use_karras_sigmas: Optional[bool] = Field(
        default=None, 
        description="Override Karras sigma setting"
    )
    
    lora_strength: float = Field(
        default=1.0, 
        description="LoRA strength for era-specific models",
        ge=0.0,
        le=2.0
    )
    
    controlnet_strength: float = Field(
        default=0.8, 
        description="ControlNet conditioning strength",
        ge=0.0,
        le=2.0
    )
    
    control_guidance_start: float = Field(
        default=0.0,
        description="Percentage of total steps at which ControlNet starts applying",
        ge=0.0,
        le=1.0
    )
    
    control_guidance_end: float = Field(
        default=1.0,
        description="Percentage of total steps at which ControlNet stops applying", 
        ge=0.0,
        le=1.0
    )
    
    width: int = Field(
        default=1024, 
        description="Output image width",
        ge=512,
        le=2048
    )
    
    height: int = Field(
        default=1024, 
        description="Output image height",
        ge=512,
        le=2048
    )
    
    use_jpeg: bool = Field(
        default=True,
        description="Use JPEG encoding for faster response (74% smaller file, 84% faster encoding)"
    )
    
    def model_post_init(self, __context):
        """Validate configuration after initialization."""
        if self.control_guidance_start > self.control_guidance_end:
            raise ValueError(f"control_guidance_start ({self.control_guidance_start}) must be <= control_guidance_end ({self.control_guidance_end})")
        pass
