#!/usr/bin/env python3
"""
Main entry point for the Sohkepayin Core Service.

This allows the service to be run with:
    PROJECT_ENV=sohkepayin uv run -m sohkepayin_core [--log-level DEBUG] [--config path/to/config]
"""

from experimance_common.cli import create_simple_main
from sohkepayin_core.sohkepayin_core import run_sohkepayin_core_service
from sohkepayin_core.config import SohkepayinCoreConfig, DEFAULT_CONFIG_PATH


# Create the main function using the enhanced CLI utility with auto-generated args
main = create_simple_main(
    service_name="Sohkepayin Core",
    description="Core orchestration service for the Sohkepayin interactive art installation",
    service_runner=run_sohkepayin_core_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=SohkepayinCoreConfig
)


if __name__ == "__main__":
    main()
