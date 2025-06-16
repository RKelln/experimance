"""
Module entry point for the Experimance Image Server Service.

Allows running the service with:
    uv run -m image_server [--log-level DEBUG] [--config path/to/config] [--generator.default-strategy fal]
"""

from experimance_common.cli import create_simple_main
from image_server.image_service import run_image_server_service  
from image_server.config import ImageServerConfig, DEFAULT_CONFIG_PATH


# Create the main function using the enhanced CLI utility with auto-generated args
main = create_simple_main(
    service_name="Image Server",
    description="Image generation and publishing service",
    service_runner=run_image_server_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=ImageServerConfig
)


if __name__ == "__main__":
    main()