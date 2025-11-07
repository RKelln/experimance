"""
Main entry point for the Experimance Health Service.

This allows the service to be run with:
    uv run -m experimance_health [--log-level DEBUG] [--config path/to/config]
"""
from experimance_common.cli import create_simple_main
from experimance_health.health_service import run_health_service, SERVICE_TYPE
from experimance_health.config import DEFAULT_CONFIG_PATH, HealthServiceConfig


# Create the main function using the enhanced CLI utility with auto-generated args
main = create_simple_main(
    service_name="Health",
    service_type=SERVICE_TYPE,
    service_runner=run_health_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=HealthServiceConfig
)


if __name__ == "__main__":
    main()
