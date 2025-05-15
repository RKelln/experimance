"""
Experimance Agent Service: AI interaction component for the sand-table installation.

This module handles audience detection, conversation, and AI-driven interactions
with the installation visitors.
"""

__version__ = '0.1.0'

from .agent import entrypoint, prewarm

__all__ = ['entrypoint', 'prewarm']
