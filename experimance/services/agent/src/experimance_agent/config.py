"""
Configuration for the Experimance Agent Service.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from experimance_common.config import BaseServiceConfig
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.schemas import MessageType
from experimance_common.constants import DEFAULT_PORTS, ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX, AGENT_SERVICE_DIR


class VisionConfig(BaseModel):
    """Configuration for vision processing components."""
    
    # Webcam configuration
    webcam_enabled: bool = Field(default=True, description="Enable webcam capture and processing")
    webcam_device_id: int = Field(default=0, description="Webcam device ID (usually 0 for default camera)")
    webcam_width: int = Field(default=640, description="Webcam capture width")
    webcam_height: int = Field(default=480, description="Webcam capture height")
    webcam_fps: int = Field(default=30, description="Webcam capture framerate")
    
    # Audience detection configuration
    audience_detection_enabled: bool = Field(default=True, description="Enable audience presence detection")
    audience_detection_interval: float = Field(default=2.0, description="Interval between audience detection checks (seconds)")
    audience_detection_threshold: float = Field(default=0.5, description="Confidence threshold for audience detection")
    
    # Vision Language Model configuration
    vlm_enabled: bool = Field(default=True, description="Enable Vision Language Model for scene understanding")
    vlm_model: str = Field(default="moondream", description="VLM model to use (moondream, llama-vision, etc.)")
    vlm_analysis_interval: float = Field(default=10.0, description="Interval between VLM scene analysis (seconds)")
    vlm_max_image_size: int = Field(default=512, description="Maximum image size for VLM processing")
    vlm_device: str = Field(default="cuda", description="Device for VLM processing (cpu, cuda, etc.)")


class TranscriptConfig(BaseModel):
    """Configuration for transcript management and display."""
    
    # Transcript display settings
    display_transcripts: bool = Field(default=True, description="Enable real-time transcript display")
    transcript_max_lines: int = Field(default=3, description="Maximum number of transcript lines to display")
    transcript_line_duration: float = Field(default=10.0, description="Duration to display each transcript line (seconds)")
    transcript_fade_duration: float = Field(default=1.0, description="Fade in/out duration for transcript text")
    
    # Speaker styling
    agent_speaker_name: str = Field(default="Experimance", description="Display name for the agent")
    human_speaker_name: str = Field(default="Visitor", description="Display name for human speakers")
    
    # Transcript archival
    save_transcripts: bool = Field(default=True, description="Save conversation transcripts to files")
    transcript_directory: str = Field(default_factory=lambda: str(AGENT_SERVICE_DIR / "transcripts"), description="Directory to save transcript files")


class AgentServiceConfig(BaseServiceConfig):
    """Main configuration for the Agent Service."""
    
    # Override service name with default
    service_name: str = Field(default="agent", description="Name of this agent service instance")
    
    # Agent backend selection
    agent_backend: str = Field(default="livekit", description="Agent backend to use (livekit, hume, ultravox)")
    
    # Backend-specific configuration
    backend_config: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Backend-specific configuration parameters"
    )
    
    # Vision processing configuration
    vision: VisionConfig = Field(
        default_factory=VisionConfig,
        description="Vision processing configuration"
    )
    
    # Transcript management configuration
    transcript: TranscriptConfig = Field(
        default_factory=TranscriptConfig,
        description="Transcript management configuration"
    )
    
    # ZMQ configuration
    zmq: PubSubServiceConfig = Field(
        default_factory=lambda: PubSubServiceConfig(
            publisher=PublisherConfig(
                address=ZMQ_TCP_BIND_PREFIX,
                port=DEFAULT_PORTS["agent"],
                default_topic=MessageType.AGENT_CONTROL_EVENT.value
            ),
            subscriber=SubscriberConfig(
                address=ZMQ_TCP_CONNECT_PREFIX,
                port=DEFAULT_PORTS["events"],
                topics=[
                    MessageType.SPACE_TIME_UPDATE.value,
                    MessageType.HEARTBEAT.value
                ]
            )
        ),
        description="ZMQ pub/sub configuration"
    )
    
    # Additional service settings
    tool_calling_enabled: bool = Field(default=True, description="Enable tool calling capabilities")
    biome_suggestions_enabled: bool = Field(default=True, description="Enable biome suggestion tool")
    speech_detection_enabled: bool = Field(default=True, description="Enable speech detection events")
