#!/usr/bin/env python3
"""
Main entry point for the Fires Core Service.

This allows the service to be run with:
    PROJECT_ENV=fire uv run -m fire_core [--log-level DEBUG] [--config path/to/config]
"""

from experimance_common.cli import create_simple_main
from fire_core.fire_core import run_fire_core_service, SERVICE_TYPE
from fire_core.config import FireCoreConfig, DEFAULT_CONFIG_PATH


# Create the main function using the enhanced CLI utility with auto-generated args
main = create_simple_main(
    service_name="Fire Core",
    service_type=SERVICE_TYPE,
    service_runner=run_fire_core_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=FireCoreConfig
)


if __name__ == "__main__":
    main()
