"""
Main entry point for the Experimance Audio Service.

This allows the service to be run with:
    uv run -m experimance_audio [--log-level DEBUG] [--config path/to/config]
"""
from experimance_common.cli import create_simple_main
from experimance_audio.audio_service import run_audio_service
from experimance_audio.config import DEFAULT_CONFIG_PATH, AudioServiceConfig


# Create the main function using the enhanced CLI utility with auto-generated args
main = create_simple_main(
    service_name="Audio",
    description="Audio processing and playback service for the interactive art installation",
    service_runner=run_audio_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=AudioServiceConfig
)


if __name__ == "__main__":
    main()
