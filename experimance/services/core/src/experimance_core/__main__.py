"""
Main entry point for the Experimance Core Service.

This allows the service to be run with:
    uv run -m experimance_core [config_path]
"""
import sys
import asyncio

from experimance_core.experimance_core import run_experimance_core_service


def main():
    """Main entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    asyncio.run(run_experimance_core_service(config_path))


if __name__ == "__main__":
    main()
