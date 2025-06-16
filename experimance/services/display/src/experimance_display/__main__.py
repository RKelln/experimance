#!/usr/bin/env python3
"""
Main entry point for the Experimance Display Service.

This allows the display service to be run as a module:
    uv run -m experimance_display [--log-level DEBUG] [--config path/to/config] [--windowed] [--debug]
    uv run -m experimance_display cli      # Run the CLI tool (legacy)
"""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "cli":
    # Remove 'cli' from args and run legacy CLI
    sys.argv.pop(1)
    from .cli import main
    main()
else:
    # For now, just use the existing main_sync function
    # TODO: Integrate with common CLI system while preserving display-specific args
    from .display_service import main_sync
    main_sync()
