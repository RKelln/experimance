"""
Factory function to create depth processors.
"""

from typing import Optional
from experimance_core.config import CoreServiceConfig, DEFAULT_CONFIG_PATH
from experimance_core.robust_camera import DepthProcessor
from experimance_core.mock_depth_processor import MockDepthProcessor

def create_depth_processor_from_config(config_path: str = DEFAULT_CONFIG_PATH, mock_path: Optional[str] = None) -> DepthProcessor:
    """
    Factory function to create a depth processor from config file.
    
    Args:
        config_path: Path to configuration file
        mock_path: Path to mock images (if None, uses real camera)
        
    Returns:
        DepthProcessor instance (real or mock)
    """
    # Load configuration
    config = CoreServiceConfig.from_overrides(config_file=config_path)
    camera_config = config.camera
    
    if mock_path:
        return MockDepthProcessor(camera_config, mock_path)
    else:
        return DepthProcessor(camera_config)

def create_depth_processor(camera_config, mock_path: Optional[str] = None) -> DepthProcessor:
    """
    Factory function to create a depth processor from camera config object.
    
    Args:
        camera_config: Camera configuration object
        mock_path: Path to mock images (if None, uses real camera)
        
    Returns:
        DepthProcessor instance (real or mock)
    """
    if mock_path:
        return MockDepthProcessor(camera_config, mock_path)
    else:
        return DepthProcessor(camera_config)
