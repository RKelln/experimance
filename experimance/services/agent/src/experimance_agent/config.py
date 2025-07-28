"""
Configuration for the Experimance Agent Service.
"""

from typing import Optional, Dict, Any, Literal, Annotated
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic import field_validator
from pydantic.types import StringConstraints

from experimance_common.config import BaseServiceConfig
from experimance_common.logger import get_log_directory
from experimance_common.zmq.config import PubSubServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.schemas import MessageType
from experimance_common.constants import (
    DEFAULT_PORTS, ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX, AGENT_SERVICE_DIR, LOGS_DIR,
    get_project_config_path
)

DEFAULT_CONFIG_PATH = get_project_config_path("agent", AGENT_SERVICE_DIR)


class EnsembleSettings(BaseModel):
    """Configuration for ensemble mode settings."""
    stt: Annotated[Literal["assemblyai"], # TODO: deepgram, whsiper
                   StringConstraints(to_lower=True)] = Field(
        default="assemblyai",
        description="Speech-to-Text provider to use in ensemble mode (whisper or assemblyai)"
    )

    llm: Annotated[Literal["openai"], 
                   StringConstraints(to_lower=True)] = Field(
        default="openai",
        description="Language Model provider to use in ensemble mode (openai or anthropic)"
    )

    tts: Annotated[Literal["elevenlabs", "cartesia"], 
                   StringConstraints(to_lower=True)] = Field(
        default="elevenlabs",
        description="Text-to-Speech provider to use in ensemble mode (elevenlabs or cartesia)"
    )


class PipecatBackendConfig(BaseModel):
    """Configuration for the Pipecat backend."""
    
    # Pipeline mode selection
    mode: Annotated[Literal["ensemble", "realtime"], 
                   StringConstraints(to_lower=True)] = Field(
        default="realtime",
        description="Pipeline mode: 'realtime' for OpenAI Realtime Beta or 'ensemble' for separate STT/LLM/TTS"
    )

    # Flow configuration
    flow_file: Optional[Path] = Field(
        default=None,
        description="Path to flow configuration file. If not specified, will look for flows/{flow_name}.py"
    )
    
    # Audio settings
    audio_in_enabled: bool = Field(default=True, description="Enable audio input")
    audio_out_enabled: bool = Field(default=True, description="Enable audio output")
    audio_in_sample_rate: int = Field(default=16000, description="Audio input sample rate")
    audio_out_sample_rate: int = Field(default=16000, description="Audio output sample rate")
    
    # Audio device selection (None means use default device)
    audio_input_device_index: Optional[int] = Field(default=None, description="PyAudio device index for input (microphone). Use None for default device.")
    audio_output_device_index: Optional[int] = Field(default=None, description="PyAudio device index for output (speaker). Use None for default device.")
    
    # Audio device selection by name (alternative to index, takes precedence if both are set)
    audio_input_device_name: Optional[str] = Field(default=None, description="Partial name match for input device (e.g., 'Yealink', 'USB'). Takes precedence over index.")
    audio_output_device_name: Optional[str] = Field(default=None, description="Partial name match for output device (e.g., 'Yealink', 'USB'). Takes precedence over index.")
    
    # Audio error suppression and device handling
    suppress_audio_errors: bool = Field(default=True, description="Suppress ALSA/JACK error messages during audio initialization")
    audio_device_retry_attempts: int = Field(default=3, description="Number of times to retry audio device initialization on failure")
    audio_device_retry_delay: float = Field(default=1.0, description="Delay between audio device retry attempts (seconds)")
    
    # Voice Activity Detection settings
    vad_enabled: bool = Field(default=True, description="Enable voice activity detection using Silero VAD")
    
    # STT settings (for ensemble mode)
    whisper_model: str = Field(default="tiny", description="Whisper model size (tiny, base, small, medium, large)")
    
    # LLM settings
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use (ensemble mode) or realtime model")
    
    # OpenAI Realtime settings (for realtime mode)
    openai_realtime_model: str = Field(default="gpt-4o-realtime-preview-2025-06-03", description="OpenAI Realtime model")
    openai_voice: str = Field(default="alloy", description="OpenAI Realtime voice (alloy, echo, fable, onyx, nova, shimmer)")
    turn_detection_threshold: float = Field(default=0.5, description="Voice activity detection threshold for realtime mode")
    turn_detection_silence_ms: int = Field(default=800, description="Silence duration in ms before turn ends")
    
    # Ensemble mode settings
    ensemble: EnsembleSettings = Field(
        default_factory=lambda: EnsembleSettings(stt="assemblyai", llm="openai", tts="cartesia"),
        description="Settings for ensemble mode, including TTS and STT providers"
    )


class BackendConfig(BaseModel):
    """Container for all backend configurations."""
    
    prompt_path: Optional[Path] = Field(
        default=None,
        description="Path to custom prompt file for the agent (if any)"
    )

    pipecat: PipecatBackendConfig = Field(
        default_factory=PipecatBackendConfig,
        description="Pipecat backend configuration"
    )


class VisionConfig(BaseModel):
    """Configuration for vision processing components."""
    
    # Webcam configuration
    webcam_enabled: bool = Field(default=True, description="Enable webcam capture and processing")
    webcam_device_id: int = Field(default=0, description="Webcam device ID (use scripts/list_webcams.py to find available cameras)")
    webcam_device_name: Optional[str] = Field(default=None, description="Alternative: specify webcam by partial name match (overrides device_id if set)")
    webcam_width: int = Field(default=640, description="Webcam capture width (common: 640, 1280, 1920)")
    webcam_height: int = Field(default=480, description="Webcam capture height (common: 480, 720, 1080)")
    webcam_fps: int = Field(default=30, description="Webcam capture framerate")
    webcam_auto_detect: bool = Field(default=True, description="Auto-detect and use first available webcam if specified device fails")
    
    # Audience detection configuration
    audience_detection_enabled: bool = Field(default=True, description="Enable audience presence detection")
    audience_detection_interval: float = Field(default=2.0, description="Interval between audience detection checks (seconds)")
    stable_readings_required: int = Field(
        default=3,
        description="Number of stable readings required before confirming audience presence"
    )

    # Detection method selection
    detection_method: Annotated[Literal["cpu", "vlm", "hybrid"], 
                               StringConstraints(to_lower=True)] = Field(
        default="cpu",
        description="Detection method: 'cpu' for OpenCV-only, 'vlm' for Vision Language Model, 'hybrid' for both"
    )
    
    # CPU detection performance mode
    cpu_performance_mode: Annotated[Literal["fast", "balanced", "accurate"], 
                                   StringConstraints(to_lower=True)] = Field(
        default="balanced",
        description="CPU detection performance mode: 'fast', 'balanced', or 'accurate'"
    )
    
    # Detector profile configuration
    detector_profile: str = Field(
        default="face_detection",
        description="Detector profile name (indoor_office, gallery_dim, outdoor_bright, workshop_cluttered, face_detection, or custom)"
    )
    detector_profile_dir: Optional[Path] = Field(
        default=None,
        description="Custom directory for detector profiles (defaults to agent/profiles/)"
    )
    
    # Vision Language Model configuration (only used if detection_method includes VLM)
    vlm_enabled: bool = Field(default=False, description="Enable Vision Language Model for scene understanding (slow on CPU)")
    vlm_model: str = Field(default="moondream", description="VLM model to use (moondream, llama-vision, etc.)")
    vlm_analysis_interval: float = Field(default=30.0, description="Interval between VLM scene analysis (seconds) - increased for CPU")
    vlm_max_image_size: int = Field(default=256, description="Maximum image size for VLM processing - reduced for CPU performance")
    vlm_device: str = Field(default="cpu", description="Device for VLM processing (cpu, cuda, etc.)")
    
    @field_validator('detection_method')
    @classmethod
    def validate_detection_method(cls, v: str) -> str:
        """Validate detection method and provide warnings for performance."""
        if v == "vlm":
            import logging
            logging.getLogger(__name__).warning(
                "VLM-only detection method selected. This may be slow on CPU. Consider 'cpu' or 'hybrid' mode."
            )
        return v
    
    @field_validator('webcam_width', 'webcam_height')
    @classmethod
    def validate_resolution(cls, v):
        """Validate that resolution values are reasonable."""
        if v < 160 or v > 4096:
            raise ValueError(f"Resolution dimension must be between 160 and 4096, got {v}")
        return v
    
    @field_validator('webcam_fps')
    @classmethod
    def validate_fps(cls, v):
        """Validate that FPS value is reasonable."""
        if v < 1 or v > 120:
            raise ValueError(f"FPS must be between 1 and 120, got {v}")
        return v
    
    @field_validator('vlm_device')
    @classmethod
    def validate_vlm_device(cls, v):
        """Validate VLM device string."""
        valid_devices = ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'mps']
        if v not in valid_devices:
            # Allow cuda:N pattern
            if not (v.startswith('cuda:') and v[5:].isdigit()):
                raise ValueError(f"VLM device must be one of {valid_devices} or 'cuda:N', got '{v}'")
        return v


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
    transcript_directory: str = Field(default_factory=lambda: str(get_log_directory() / "transcripts"), description="Directory to save transcript files")


class AgentServiceConfig(BaseServiceConfig):
    """Main configuration for the Agent Service."""
    
    # Override service name with default
    service_name: str = Field(default="agent", description="Name of this agent service instance")
    
    # Agent backend selection
    agent_backend: str = Field(default="pipecat", description="Agent backend to use (pipecat only currently supported)")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    assemblyai_api_key: Optional[str] = Field(default=None, description="AssemblyAI API key")
    cartesia_api_key: Optional[str] = Field(default=None, description="Cartesia API key")
    elevenlabs_api_key: Optional[str] = Field(default=None, description="ElevenLabs API key")
    
    # TTS settings (for ensemble mode)
    elevenlabs_voice_id: str = Field(default="EXAVITQu4vr4xnSDxMaL", description="ElevenLabs voice ID")
    cartesia_voice_id: str = Field(default="bf0a246a-8642-498a-9950-80c35e9276b5", description="Cartesia voice ID")

    # Backend-specific configuration
    backend_config: BackendConfig = Field(
        default_factory=BackendConfig, 
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
                default_topic="agent"
            ),
            subscriber=SubscriberConfig(
                address=ZMQ_TCP_CONNECT_PREFIX,
                port=DEFAULT_PORTS["events"],
                topics=[
                    MessageType.SPACE_TIME_UPDATE, MessageType.PRESENCE_STATUS
                ]
            )
        ),
        description="ZMQ pub/sub configuration"
    )
    
    # Additional service settings
    tool_calling_enabled: bool = Field(default=True, description="Enable tool calling capabilities")
    biome_suggestions_enabled: bool = Field(default=True, description="Enable biome suggestion tool")
    speech_detection_enabled: bool = Field(default=True, description="Enable speech detection events")
    
    # Conversation cooldown settings
    conversation_cooldown_duration: float = Field(
        default=30.0, 
        description="Cooldown period after conversation ends before new one can start (seconds)"
    )
    cancel_cooldown_on_absence: bool = Field(
        default=True,
        description="End cooldown early if audience leaves and returns (allows immediate restart on audience change)"
    )

    # audio issues
    audio_health_monitoring: bool = Field(
        default=False,
        description="Enable audio health monitoring and recovery mechanisms"
    )