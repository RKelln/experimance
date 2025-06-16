"""
Main entry point for the Experimance Core Service.

This allows the service to be run with:
    uv run -m experimance_core [--log-level DEBUG] [--config path/to/config]
"""
from experimance_common.cli import create_simple_main
from experimance_core.experimance_core import run_experimance_core_service
from experimance_core.config import DEFAULT_CONFIG_PATH


# Create the main function using the common CLI utility
main = create_simple_main(
    service_name="Core",
    description="Central coordinator for the interactive art installation",
    service_runner=run_experimance_core_service,
    default_config_path=DEFAULT_CONFIG_PATH
)


if __name__ == "__main__":
    main()
