#!/usr/bin/env python3
"""
Main entry point for the Experimance Display Service.

This allows the display service to be run as a module:
    uv run -m experimance_display
"""
import sys

from experimance_common.cli import create_simple_main
from experimance_display.display_service import run_display_service
from experimance_display.config import DisplayServiceConfig, DEFAULT_CONFIG_PATH

# Create the main function using the enhanced CLI utility with auto-generated args
main = create_simple_main(
    service_name="Display",
    description="Visual display and rendering service for the sand table",
    service_runner=run_display_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=DisplayServiceConfig
)

if __name__ == "__main__":
    main()
