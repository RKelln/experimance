"""
Experimance Core Service: Main coordinator for the interactive sand-table installation.

This module handles the state machine, depth processing, and event coordination
for the Experimance art installation.
"""

__version__ = '0.1.0'

from .experimance import run_experimance_service

__all__ = ['run_experimance_service']