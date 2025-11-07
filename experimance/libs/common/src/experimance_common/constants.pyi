"""
Static-analysis stub for experimance_common.constants.

This stub provides type information for the dynamically loaded constants.
At runtime, the actual module loads project-specific extensions based on
the PROJECT_ENV environment variable and makes them available in this namespace.

For static type checking, this file conditionally imports the appropriate
project-specific constants based on the PROJECT_ENV environment variable.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

# Re-export base constants that are NOT extended by projects
from experimance_common.constants_base import (
    # Project structure constants
    PROJECT_ROOT,
    PROJECT_SPECIFIC_DIR,

    # Config helpers
    get_project_config_path,

    # Port configurations
    DEFAULT_PORTS,

    # Timeout settings
    DEFAULT_TIMEOUT,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RECV_TIMEOUT,
    TICK,

    # Service types
    SERVICE_TYPES,

    # Image transport configuration
    IMAGE_TRANSPORT_MODES,
    DEFAULT_IMAGE_TRANSPORT_MODE,
    IMAGE_TRANSPORT_SIZE_THRESHOLD,

    # Temporary file settings
    TEMP_FILE_PREFIX,
    TEMP_FILE_SUFFIX,
    TEMP_FILE_CLEANUP_AGE,
    TEMP_FILE_CLEANUP_INTERVAL,
    DEFAULT_TEMP_DIR,

    # URI and URL constants
    FILE_URI_PREFIX,
    DATA_URL_PREFIX,
    BASE64_PNG_PREFIX,

    # ZMQ address patterns
    ZMQ_TCP_BIND_PREFIX,
    ZMQ_TCP_CONNECT_PREFIX,

    # Data directory
    DATA_DIR,

    # Media directories (relative paths)
    MEDIA_DIR,
    IMAGES_DIR,
    GENERATED_IMAGES_DIR,
    MOCK_IMAGES_DIR,
    AUDIO_DIR,
    VIDEOS_DIR,
    GENERATED_AUDIO_DIR,

    # Media directories (absolute paths)
    MEDIA_DIR_ABS,
    IMAGES_DIR_ABS,
    GENERATED_IMAGES_DIR_ABS,
    MOCK_IMAGES_DIR_ABS,
    AUDIO_DIR_ABS,
    VIDEOS_DIR_ABS,
    GENERATED_AUDIO_DIR_ABS,

    # Services directories
    SERVICES_DIR,
    CORE_SERVICE_DIR,
    AUDIO_SERVICE_DIR,
    IMAGE_SERVER_SERVICE_DIR,
    AGENT_SERVICE_DIR,
    DISPLAY_SERVICE_DIR,

    # Other directories
    LOGS_DIR,
    DATA_DIR,
    MODELS_DIR,
)

# PROJECT constant that's set dynamically
PROJECT: str

# Conditionally import project-specific constants based on PROJECT_ENV
if TYPE_CHECKING:
    _PROJECT_ENV = os.getenv("PROJECT_ENV", "experimance")
    
    if _PROJECT_ENV == "experimance":
        # Import experimance-specific constants if they exist
        # For now, the experimance constants.py file is mostly empty,
        # but this allows for future project-specific constants
        pass
    elif _PROJECT_ENV == "fire":
        # Import Feed the Fires-specific constants if they exist
        # For now, the fire constants.py file is mostly empty,
        # but this allows for future project-specific constants
        pass
    else:
        # Fallback for unknown projects
        pass

__all__: list[str] = [
    # Dynamic project constant
    "PROJECT",

    "SERVICE_TYPES",
    
    # Project structure constants
    "PROJECT_ROOT",
    "PROJECT_SPECIFIC_DIR",

    # Config helpers
    "get_project_config_path",

    # Port configurations
    "DEFAULT_PORTS",

    # Timeout settings
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRY_ATTEMPTS",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_RECV_TIMEOUT",
    "TICK",

    # Image transport configuration
    "IMAGE_TRANSPORT_MODES",
    "DEFAULT_IMAGE_TRANSPORT_MODE",
    "IMAGE_TRANSPORT_SIZE_THRESHOLD",

    # Temporary file settings
    "TEMP_FILE_PREFIX",
    "TEMP_FILE_SUFFIX",
    "TEMP_FILE_CLEANUP_AGE",
    "TEMP_FILE_CLEANUP_INTERVAL",
    "DEFAULT_TEMP_DIR",

    # URI and URL constants
    "FILE_URI_PREFIX",
    "DATA_URL_PREFIX",
    "BASE64_PNG_PREFIX",

    # ZMQ address patterns
    "ZMQ_TCP_BIND_PREFIX",
    "ZMQ_TCP_CONNECT_PREFIX",

    # Data directory
    "DATA_DIR",

    # Media directories (relative paths)
    "MEDIA_DIR",
    "IMAGES_DIR",
    "GENERATED_IMAGES_DIR",
    "MOCK_IMAGES_DIR",
    "AUDIO_DIR",
    "VIDEOS_DIR",
    "GENERATED_AUDIO_DIR",

    # Media directories (absolute paths)
    "MEDIA_DIR_ABS",
    "IMAGES_DIR_ABS",
    "GENERATED_IMAGES_DIR_ABS",
    "MOCK_IMAGES_DIR_ABS",
    "AUDIO_DIR_ABS",
    "VIDEOS_DIR_ABS",
    "GENERATED_AUDIO_DIR_ABS",

    # Services directories
    "SERVICES_DIR",
    "CORE_SERVICE_DIR",
    "AUDIO_SERVICE_DIR",
    "IMAGE_SERVER_SERVICE_DIR",
    "AGENT_SERVICE_DIR",
    "DISPLAY_SERVICE_DIR",

    # Other directories
    "LOGS_DIR",
    "DATA_DIR",
    "MODELS_DIR",
]