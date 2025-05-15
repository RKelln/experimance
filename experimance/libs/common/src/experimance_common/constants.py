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
    "example_pub": 5567,
    "example_pull": 5568,
}

# Timeout settings
DEFAULT_TIMEOUT = 5000  # milliseconds
HEARTBEAT_INTERVAL = 2.0  # seconds

__all__ = ["DEFAULT_PORTS", "DEFAULT_TIMEOUT", "HEARTBEAT_INTERVAL"]
