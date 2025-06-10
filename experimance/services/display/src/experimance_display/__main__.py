#!/usr/bin/env python3
"""
Main entry point for the Experimance Display Service.

This allows the display service to be run as a module:
    python -m experimance_display
"""

from .display_service import main_sync

if __name__ == "__main__":
    main_sync()
