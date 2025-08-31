
from experimance_common.constants_base import AGENT_SERVICE_DIR, get_project_config_path
from pydantic import BaseModel, Field
from typing import Optional

from agent.config import AgentServiceConfig
from agent.vision.yolo_person_detector import YOLO11DetectionConfig

DEFAULT_CONFIG_PATH = get_project_config_path("agent", AGENT_SERVICE_DIR)

class OSCConfig(BaseModel):
    """Configuration for the Open Sound Control (OSC) interface."""
    
    enabled: bool = Field(
        default=True,
        description="Enable OSC interface"
    )
    host: str = Field(
        default="localhost",
        description="Host for OSC interface"
    )
    port: int = Field(
        default=8000,
        description="Port for OSC interface"
    )
    address_prefix: str = Field(
        default="/",
        description="Address prefix for OSC messages"
    )

class ReolinkConfig(BaseModel):
    """Base configuration for Reolink camera detection."""
    
    enabled: bool = Field(
        default=False,
        description="Enable Reolink camera detection"
    )
    host: Optional[str] = Field(
        default=None,
        description="Reolink camera IP address or hostname (e.g. '192.168.1.100')"
    )
    user: str = Field(
        default="admin",
        description="Reolink camera username"
    )
    https: bool = Field(
        default=True,
        description="Use HTTPS for Reolink camera (recommended)"
    )
    channel: int = Field(
        default=0,
        description="Reolink camera channel (0 for single-channel cameras)"
    )
    timeout: int = Field(
        default=10,
        description="Reolink camera request timeout (seconds)"
    )

class FireReolinkConfig(ReolinkConfig):
    """Fire project specific Reolink configuration with asymmetric hysteresis."""
    
    # Fire project defaults
    enabled: bool = True
    host: Optional[str] = None  # Auto-discover by default, can specify IP to hint discovery
    user: str = "admin"
    https: bool = True
    channel: int = 0
    timeout: int = 2
    
    # Asymmetric hysteresis - different thresholds for detecting vs losing audience
    hysteresis_present: int = Field(
        default=2,
        description="Number of consecutive 'present' readings needed to confirm audience detected (lower = more responsive to arrivals)"
    )
    hysteresis_absent: int = Field(
        default=5,
        description="Number of consecutive 'absent' readings needed to confirm audience left (higher = less likely to lose audience briefly)"
    )
    
    # Person detection configuration
    detection_method: str = Field(
        default="yolo11",
        description="Person detection method: 'hog', 'yolo11', or 'hybrid'"
    )

    yolo: YOLO11DetectionConfig = Field(
        default_factory=YOLO11DetectionConfig,
        description="Configuration for the YOLO11 person detection"
    )

# Project-specific configurations can inherit from these base classes
class FireOSCConfig(OSCConfig):
    """Configuration for the Fire project OSC interface."""
    
    # Fire project defaults
    enabled: bool = True
    host: str = "localhost"
    port: int = 5580

    presence_address: str = Field(
        default="/presence",
        description="Address for OSC presence messages"
    )

    person_speak_address: str = Field(
        default="/speaking",
        description="Address for OSC speaking messages"
    )

    bot_speak_address: str = Field(
        default="/bot/speaking",
        description="Address for OSC bot speaking messages"
    )


class FireAgentServiceConfig(AgentServiceConfig):
    """Main configuration for the Fire Agent Service."""

    # OSC config
    osc: FireOSCConfig = Field(
        default_factory=FireOSCConfig,
        description="Configuration for the OSC interface"
    )
    
    # Reolink camera config (Fire project specific)
    reolink: FireReolinkConfig = Field(
        default_factory=FireReolinkConfig,
        description="Configuration for the Reolink camera detection"
    )
    
    # Proactive greeting configuration
    proactive_greeting_enabled: bool = Field(
        default=True,
        description="Enable proactive greeting when visitors are detected"
    )
    greeting_delay: float = Field(
        default=2.0,
        description="Delay in seconds before greeting visitors after detection"
    )
    greeting_prompt: str = Field(
        default="Please greet me warmly as the Fire Spirit you are. I've just arrived at your fire circle.",
        description="User message to trigger proactive greeting"
    )
