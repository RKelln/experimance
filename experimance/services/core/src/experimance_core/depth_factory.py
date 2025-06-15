"""
Factory function to create depth processors.
"""

from typing import Optional
from experimance_core.robust_camera import CameraConfig, DepthProcessor
from experimance_core.mock_depth_processor import MockDepthProcessor

def create_depth_processor(config: CameraConfig, mock_path: Optional[str] = None) -> DepthProcessor:
    """
    Factory function to create a depth processor.
    
    Args:
        config: Camera configuration
        mock_path: Path to mock images (if None, uses real camera)
        
    Returns:
        DepthProcessor instance (real or mock)
    """
    if mock_path:
        return MockDepthProcessor(config, mock_path)
    else:
        return DepthProcessor(config)
