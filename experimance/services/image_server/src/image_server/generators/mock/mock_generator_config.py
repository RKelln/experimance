from typing import Optional, Literal
from pathlib import Path

from pydantic import field_validator

from image_server.generators.config import BaseGeneratorConfig
from experimance_common.constants import PROJECT_ROOT

class MockGeneratorConfig(BaseGeneratorConfig):
    strategy: Literal["mock"] = "mock"
    image_size: tuple = (1024, 1024)
    background_color: tuple = (100, 150, 200)
    text_color: tuple = (255, 255, 255)
    use_existing_images: bool = False
    existing_images_dir: Optional[Path] = None
    
    @field_validator('existing_images_dir')
    @classmethod
    def resolve_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Resolve existing_images_dir path relative to project root if needed."""
        if v is None:
            return v
        
        # Convert string to Path if needed
        if isinstance(v, str):
            v = Path(v)
            
        # If path is relative, resolve it against project root
        if not v.is_absolute():
            v = PROJECT_ROOT / v
            
        return v