"""
Experimance Display Service: Visual rendering component for the sand-table installation.

This module handles the display, transitions, and visual rendering of images
projected onto the sand surface.
"""

__version__ = '0.1.0'

from .display_service import DisplayService
from .config import DisplayServiceConfig

__all__ = ['DisplayService', 'DisplayServiceConfig']

# __all__ = ['run_display_service']
