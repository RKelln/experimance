"""
Experimance Audio Service: Sound component for the sand-table installation.

This module handles the audio playback, crossfades, and environmental sounds
accompanying the visual experience.
"""

__version__ = '0.1.0'

from .audio import run_audio_service

__all__ = ['run_audio_service']
