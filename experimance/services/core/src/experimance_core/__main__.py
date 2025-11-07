"""
Main entry point for the Experimance Core Service.

This allows the service to be run with:
    uv run -m experimance_core [--log-level DEBUG] [--config path/to/config] [--visualize]
"""
from experimance_common.cli import create_simple_main
from experimance_core.experimance_core import run_experimance_core_service, SERVICE_TYPE
from experimance_core.config import DEFAULT_CONFIG_PATH, CoreServiceConfig


# Create the main function using the enhanced CLI utility with auto-generated args
main = create_simple_main(
    service_name="Core",
    service_type=SERVICE_TYPE,
    service_runner=run_experimance_core_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=CoreServiceConfig
)


if __name__ == "__main__":
    main()
