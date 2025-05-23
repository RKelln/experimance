"""
Default configuration values for Experimance services.
"""

DEFAULT_PORTS = {
    "coordinator_pub": 5555,
    "coordinator_pull": 5556,
    "transition_pub": 5557,
    "transition_pull": 5558,
    "display_pub": 5559,
    "display_pull": 5560,
    "agent_pub": 5561,
    "agent_pull": 5562,
    "image_server_pub": 5563,
    "image_server_pull": 5564,
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

__all__ = ["DEFAULT_PORTS", "DEFAULT_TIMEOUT", "HEARTBEAT_INTERVAL"]
