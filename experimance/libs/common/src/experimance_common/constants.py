"""
Default configuration values for Experimance services.
"""

DEFAULT_PORTS = {
    "events_pub": 5555,
    "depth_pub": 5556,
    "transition_pull": 5557,
    "images_pub": 5558,
    "agent_pub": 5559,
    "display_pub": 5560,
    "transitions_pull": 5561,
    "loops_pull": 5562,
    "image_server_pub": 5558,  # Same as images_pub
    "core_pub": 5555,  # Same as events_pub
    "audio_pub": 5565,
    "audio_pull": 5566,
    "audio_osc_send_port": 5567,
    "audio_osc_recv_port": 5568,
}

# Timeout settings
DEFAULT_TIMEOUT = 1000  # ms
HEARTBEAT_INTERVAL = 5.0  # seconds
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 0.5  # seconds
DEFAULT_RECV_TIMEOUT = 1.0  # seconds

HEARTBEAT_TOPIC = "heartbeat"

TICK = 0.01 # seconds, used sleeping in the main loop

__all__ = ["DEFAULT_PORTS", "DEFAULT_TIMEOUT", "HEARTBEAT_INTERVAL", "DEFAULT_RETRY_ATTEMPTS", "DEFAULT_RETRY_DELAY", "DEFAULT_RECV_TIMEOUT", "HEARTBEAT_TOPIC", "TICK"]
