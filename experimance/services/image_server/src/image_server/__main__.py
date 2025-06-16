"""
Module entry point for the Experimance Image Server Service.

Allows running the service with:
    uv run -m image_server [--log-level DEBUG] [--config path/to/config] [--generator fal]
"""

from experimance_common.cli import create_simple_main
from .main import main as image_server_main

# Extra arguments specific to the image server service
extra_args = {
    '--generator': {
        'choices': ['mock', 'fal', 'openai', 'local'],
        'default': 'mock',
        'help': 'Image generation strategy (default: mock)'
    }
}

# Create a wrapper function that calls the existing main()
async def run_image_server_service(config_path=None, **kwargs):
    # The existing main() function handles its own argument parsing,
    # so we'll just call it directly
    await image_server_main()

# Create the main function using the common CLI utility
main = create_simple_main(
    service_name="Image Server",
    description="Image generation and publishing service",
    service_runner=run_image_server_service,
    default_config_path="config.toml",
    extra_args=extra_args
)

if __name__ == "__main__":
    main()