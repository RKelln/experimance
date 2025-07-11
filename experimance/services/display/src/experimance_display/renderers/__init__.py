"""
Renderers package for the Display Service.

This package contains specialized renderers for different types of visual content:
- ImageRenderer: Standard image display with crossfade transitions
- VideoOverlayRenderer: Masked video overlays
- TextOverlayManager: Text display with positioning
- PanoramaRenderer: Panoramic display with base image + tiles
- DebugOverlayRenderer: Debug information overlay
"""

from .layer_manager import LayerManager, LayerRenderer
from .image_renderer import ImageRenderer
from .video_overlay_renderer import VideoOverlayRenderer
from .text_overlay_manager import TextOverlayManager
from .panorama_renderer import PanoramaRenderer
from .debug_overlay_renderer import DebugOverlayRenderer
from .mask_renderer import MaskRenderer

__all__ = [
    'LayerManager',
    'LayerRenderer', 
    'ImageRenderer',
    'VideoOverlayRenderer',
    'TextOverlayManager',
    'PanoramaRenderer',
    'DebugOverlayRenderer',
    'MaskRenderer'
]