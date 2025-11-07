
from experimance_common.constants_base import AGENT_SERVICE_DIR, get_project_config_path
from pydantic import BaseModel, Field

from agent.config import AgentServiceConfig


DEFAULT_CONFIG_PATH = get_project_config_path("agent", AGENT_SERVICE_DIR)
    
class ExperimanceAgentServiceConfig(AgentServiceConfig):
    """Main configuration for the Experimance Agent Service."""

    # Deep thoughts settings
    deep_thoughts_min_delay: float = Field(
        default=60.0,
        description="Minimum delay in seconds before first deep thought can be shared in a conversation"
    )

    deep_thoughts_chance: float = Field(
        default=0.5,
        description="Chance to share a deep thought (0.0 to 1.0)"
    )

    deep_thoughts_quiet_delay: float = Field(
        default=30.0,
        description="Seconds of quiet (no speaking) required before sharing a deep thought"
    )