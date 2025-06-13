"""
Default configuration values for Experimance services.
"""
from pathlib import Path

# Project structure constants
# Find the project root by going up from libs/common/src/experimance_common
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

DEFAULT_PORTS = {
    # Unified events channel - all services publish and subscribe here
    "events": 5555,
    "events_pub": 5555,  # For backward compatibility
    "events_sub": 5555,  # For backward compatibility
    
    # Specialized high-bandwidth channels
    "depth_pub": 5556,  # Depth camera data (high frequency)
    "transitions_pull": 5561,  # Work distribution for transition rendering
    "loops_pull": 5562,  # Work distribution for animation loops
    
    # Audio OSC bridge ports
    "audio_osc_send_port": 5567,  # Audio service → SuperCollider
    "audio_osc_recv_port": 5568,  # SuperCollider → Audio service
    
    # Legacy port aliases (deprecated - use 'events' instead)
    "core": 5555,
    "image_server_pub": 5555,
    "agent_pub": 5555,
    "audio_pub": 5555,
    "image_request_pub": 5555,
    "coordinator_pub": 5555,
    "display_pull": 5555,
    "example_pub": 5555,
}

# Timeout settings
DEFAULT_TIMEOUT = 1000  # ms
HEARTBEAT_INTERVAL = 5.0  # seconds
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 0.5  # seconds
DEFAULT_RECV_TIMEOUT = 1.0  # seconds

HEARTBEAT_TOPIC = "heartbeat"

TICK = 0.01 # seconds, used sleeping in the main loop

# media directories (relative paths)
MEDIA_DIR = "media"
IMAGES_DIR = f"{MEDIA_DIR}/images"
GENERATED_IMAGES_DIR = f"{IMAGES_DIR}/generated"
MOCK_IMAGES_DIR = f"{IMAGES_DIR}/mocks"
AUDIO_DIR = f"services/audio/audio"
VIDEOS_DIR = f"{MEDIA_DIR}/video"

# media directories (absolute paths)
MEDIA_DIR_ABS = PROJECT_ROOT / MEDIA_DIR
IMAGES_DIR_ABS = PROJECT_ROOT / IMAGES_DIR
GENERATED_IMAGES_DIR_ABS = PROJECT_ROOT / GENERATED_IMAGES_DIR
MOCK_IMAGES_DIR_ABS = PROJECT_ROOT / MOCK_IMAGES_DIR
AUDIO_DIR_ABS = PROJECT_ROOT / AUDIO_DIR
VIDEOS_DIR_ABS = PROJECT_ROOT / VIDEOS_DIR

__all__ = [
    "PROJECT_ROOT", 
    "DEFAULT_PORTS", 
    "DEFAULT_TIMEOUT", 
    "HEARTBEAT_INTERVAL", 
    "DEFAULT_RETRY_ATTEMPTS", 
    "DEFAULT_RETRY_DELAY", 
    "DEFAULT_RECV_TIMEOUT", 
    "HEARTBEAT_TOPIC", 
    "TICK",
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
    "VIDEOS_DIR_ABS"
]
