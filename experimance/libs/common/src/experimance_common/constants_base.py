"""
Default configuration values for Experimance services.
"""
from pathlib import Path

# Project structure constants
# Find the project root by going up from libs/common/src/experimance_common
PROJECT_ROOT = (Path(__file__).parent.parent.parent.parent.parent).absolute()

# Directory for project-specific configurations
PROJECT_SPECIFIC_DIR = PROJECT_ROOT / "projects"

DEFAULT_PORTS = {
    # Unified events channel - all services publish and subscribe here
    "events": 5555,               # Pubsub from core to services, includes display media 
    # Updates from services (e.g. agent state, image generation status)
    "updates": 5556,              # Currently not functional
    "agent": 5557,                # Agent service updates (agent binding publisher, others subscribe)
    
    # Specialized high-bandwidth channels
    "depth": 5566,                # Depth camera data (high frequency) [currenly unused]
    "transition_requests": 5560,  # Work distribution for transition rendering
    "transition_results": 5561,   # Results from transition rendering
    "video_requests": 5562,       # Work distribution for video generation
    "video_results": 5563,        # Resulting generated videos
    "image_requests": 5564,       # Work distribution for image generation 
    "image_results": 5565,        # Results from image generation
    
    # Audio OSC bridge ports
    "audio_osc_send_port": 5570,  # Audio service → SuperCollider
    "audio_osc_recv_port": 5571,  # SuperCollider → Audio service
}

# Timeout settings
DEFAULT_TIMEOUT = 1000  # ms
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 0.5  # seconds
DEFAULT_RECV_TIMEOUT = 1.0  # seconds

TICK = 0.001 # seconds, used sleeping in the main loop

# Image transport configuration
IMAGE_TRANSPORT_MODES = {
    "FILE_URI": "file_uri",      # Send file path/URI (same machine)
    "BASE64": "base64",          # Send base64 encoded image (remote machines)
    "AUTO": "auto",              # Auto-detect based on target
    "HYBRID": "hybrid"           # Send both URI and base64 (receiver chooses)
}

# Default image transport mode
DEFAULT_IMAGE_TRANSPORT_MODE = IMAGE_TRANSPORT_MODES["AUTO"]

# File size threshold for auto mode (bytes)
# Images larger than this will prefer URI over base64 to reduce network load
IMAGE_TRANSPORT_SIZE_THRESHOLD = 1024 * 1024  # 1MB

# Temporary file settings
TEMP_FILE_PREFIX = "experimance_img_"
TEMP_FILE_SUFFIX = ".png"
TEMP_FILE_CLEANUP_AGE = 300  # seconds (5 minutes)
TEMP_FILE_CLEANUP_INTERVAL = 60  # seconds (1 minute)

# Default directory for temporary files - use cache dir in production
import os
if os.path.exists("/var/cache/experimance") and os.access("/var/cache/experimance", os.W_OK):
    DEFAULT_TEMP_DIR = "/var/cache/experimance"
else:
    DEFAULT_TEMP_DIR = "/tmp"  # Fallback to /tmp

# URI and URL constants
FILE_URI_PREFIX = "file://"
DATA_URL_PREFIX = "data:image/"
BASE64_PNG_PREFIX = "data:image/png;base64,"

# ZMQ address patterns
ZMQ_TCP_BIND_PREFIX = "tcp://*"
ZMQ_TCP_CONNECT_PREFIX = "tcp://localhost"

# data dir
DATA_DIR = PROJECT_ROOT / "data"

SERVICE_TYPES = [
    "core",
    "audio",
    "image_server",
    "agent",
    "display",
    "health"
]

# services directories
SERVICES_DIR = PROJECT_ROOT / "services"
CORE_SERVICE_DIR = SERVICES_DIR / "core"
AUDIO_SERVICE_DIR = SERVICES_DIR / "audio"
IMAGE_SERVER_SERVICE_DIR = SERVICES_DIR / "image_server"
AGENT_SERVICE_DIR = SERVICES_DIR / "agent"
DISPLAY_SERVICE_DIR = SERVICES_DIR / "display"
HEALTH_SERVICE_DIR = SERVICES_DIR / "health"

# media directories
MEDIA_DIR = PROJECT_ROOT / "media"
IMAGES_DIR = MEDIA_DIR / "images"
GENERATED_IMAGES_DIR = IMAGES_DIR / "generated"
MOCK_IMAGES_DIR = IMAGES_DIR / "mocks"
AUDIO_DIR = MEDIA_DIR / "audio"
VIDEOS_DIR = MEDIA_DIR / "video"

# media directories (absolute paths)
MEDIA_DIR_ABS = MEDIA_DIR.absolute()
IMAGES_DIR_ABS = IMAGES_DIR.absolute()
GENERATED_IMAGES_DIR_ABS = GENERATED_IMAGES_DIR.absolute()
MOCK_IMAGES_DIR_ABS = MOCK_IMAGES_DIR.absolute()
AUDIO_DIR_ABS = AUDIO_DIR.absolute()
VIDEOS_DIR_ABS = VIDEOS_DIR.absolute()

# other directories
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

def get_project_config_path(service_name: str, fallback_dir: Path | None = None) -> Path:
    """
    Get the configuration path for a service, with project-aware defaults.
    
    This function implements the following priority:
    1. If PROJECT_ENV is set and projects/{PROJECT_ENV}/{service_name}.toml exists, use that
    2. If fallback_dir is provided and fallback_dir/config.toml exists, use that
    3. Default to projects/{PROJECT_ENV}/{service_name}.toml (may not exist)
    4. If PROJECT_ENV is not set, default to fallback_dir/config.toml
    
    Args:
        service_name: Name of the service (e.g., "core", "display", "audio")
        fallback_dir: Directory containing the legacy config.toml (e.g., CORE_SERVICE_DIR)
        
    Returns:
        Path to the configuration file
        
    Examples:
        get_project_config_path("core", CORE_SERVICE_DIR)
        get_project_config_path("display", DISPLAY_SERVICE_DIR)
    """
    import os
    
    project_env = os.getenv("PROJECT_ENV")
    
    if project_env and project_env != "":
        # Try project-specific config first
        project_config = PROJECT_SPECIFIC_DIR / project_env / f"{service_name}.toml"
        if project_config.exists():
            return project_config
        
        # If fallback_dir is provided and its config exists, use it
        if fallback_dir and (fallback_dir / "config.toml").exists():
            return fallback_dir / "config.toml"
            
        # Default to project-specific path (even if it doesn't exist yet)
        return project_config
    else:
        # No PROJECT_ENV set, use fallback or raise error
        if fallback_dir:
            return fallback_dir / "config.toml"
        else:
            raise ValueError("PROJECT_ENV not set and no fallback_dir provided")

__all__ = [
    "PROJECT_ROOT", 
    "PROJECT_SPECIFIC_DIR",
    "DEFAULT_PORTS", 
    "DEFAULT_TIMEOUT", 
    "DEFAULT_RETRY_ATTEMPTS", 
    "DEFAULT_RETRY_DELAY", 
    "DEFAULT_RECV_TIMEOUT", 
    "TICK",
    # Image transport settings
    "IMAGE_TRANSPORT_MODES",
    "DEFAULT_IMAGE_TRANSPORT_MODE", 
    "IMAGE_TRANSPORT_SIZE_THRESHOLD",
    # Temp file settings
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
    # Media directories (relative)
    "MEDIA_DIR",
    "IMAGES_DIR", 
    "GENERATED_IMAGES_DIR",
    "MOCK_IMAGES_DIR",
    "AUDIO_DIR",
    "VIDEOS_DIR",
    # Media directories (absolute)
    "MEDIA_DIR_ABS",
    "IMAGES_DIR_ABS",
    "GENERATED_IMAGES_DIR_ABS", 
    "MOCK_IMAGES_DIR_ABS",
    "AUDIO_DIR_ABS",
    "VIDEOS_DIR_ABS",
    # Service types
    "SERVICE_TYPES",
    # Services directories
    "SERVICES_DIR",
    "CORE_SERVICE_DIR",
    "AUDIO_SERVICE_DIR",
    "IMAGE_SERVER_SERVICE_DIR",
    "AGENT_SERVICE_DIR",
    "DISPLAY_SERVICE_DIR",
    "HEALTH_SERVICE_DIR",
    # Other directories
    "LOGS_DIR",
    "DATA_DIR",
    # Config helpers
    "get_project_config_path"
]
