#!/usr/bin/env python3
"""
Main entry point for the Experimance Display Service.

This allows the display service to be run as a module:
    python -m experimance_display          # Run the display service
    python -m experimance_display.cli      # Run the CLI tool
"""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "cli":
    # Remove 'cli' from args and run CLI
    sys.argv.pop(1)
    from .cli import main
    main()
else:
    # Run the display service
    from .display_service import main_sync
    main_sync()
