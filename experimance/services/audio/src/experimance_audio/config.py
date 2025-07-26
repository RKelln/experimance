#!/usr/bin/env python3
"""
Configuration schema for the Experimance Audio Service.

This module defines Pydantic models for validating and accessing
audio service configuration in a type-safe way.
"""

from typing import List, Optional, Literal
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from experimance_common.config import BaseConfig, BaseServiceConfig
from experimance_common.schemas import MessageType
from experimance_common.zmq.config import SubscriberConfig, PubSubServiceConfig
from experimance_common.constants import (
    DEFAULT_PORTS, AUDIO_SERVICE_DIR, AUDIO_DIR, ZMQ_TCP_CONNECT_PREFIX,
    get_project_config_path
)

DEFAULT_CONFIG_PATH = get_project_config_path("audio", AUDIO_SERVICE_DIR)

class OscConfig(BaseModel):
    """OSC configuration for SuperCollider communication."""
    
    host: str = Field(
        default="localhost",
        description="SuperCollider host address"
    )
    
    send_port: int = Field(
        default=DEFAULT_PORTS["audio_osc_send_port"],
        description="Port for sending OSC messages to SuperCollider"
    )
    
    recv_port: int = Field(
        default=DEFAULT_PORTS["audio_osc_recv_port"],
        description="Port for receiving OSC messages from SuperCollider"
    )

class SuperColliderConfig(BaseModel):
    """SuperCollider configuration."""
    
    auto_start: bool = Field(
        default=True,
        description="Whether to automatically start SuperCollider"
    )
    
    sclang_path: str = Field(
        default="sclang",
        description="Path to SuperCollider language interpreter executable"
    )
    
    script_path: Optional[str] = Field(
        default=None,
        description="Path to SuperCollider script to execute on startup"
    )
    
    startup_timeout: float = Field(
        default=10.0,
        description="Timeout in seconds for SuperCollider startup"
    )

    log_dir: Optional[str] = Field(
        default=str(Path(AUDIO_SERVICE_DIR) / "logs"),
        description="Directory to write SuperCollider logs"
    )

    output_channels: Optional[int] = Field(
        default=2,
        ge=1,
        le=64,
        description="Number of output channels for SuperCollider"
    )

    input_channels: Optional[int] = Field(
        default=2,
        ge=1,
        le=64,
        description="Number of input channels for SuperCollider"
    )

    device: Optional[str] = Field(
        default=None,
        description="Audio device to use for SuperCollider. If None, uses default device."
    )

    # JACK configuration
    auto_start_jack: bool = Field(
        default=True,
        description="Automatically start JACK if not running"
    )
    
    jack_sample_rate: int = Field(
        default=48000,
        description="JACK sample rate"
    )
    
    jack_buffer_size: int = Field(
        default=1024,
        description="JACK buffer size (frames per period)"
    )
    
    jack_periods: int = Field(
        default=2,
        description="Number of periods in JACK buffer"
    )
    
    jack_output_channels: Optional[int] = Field(
        default=None,
        description="Number of output channels for JACK (if None, uses output_channels)"
    )

    # Surround sound configuration
    enable_surround: bool = Field(
        default=False,
        description="Enable surround sound routing. Automatically enabled if output_channels > 2"
    )
    
    surround_mode: Literal["5.1", "quad"] = Field(
        default="5.1",
        description="Surround sound mode: '5.1' or 'quad'"
    )
    
    environment_channels: List[int] = Field(
        default_factory=lambda: [0, 1],  # Front L/R
        description="Output channels for environmental sounds (0-indexed)"
    )
    
    music_channels: List[int] = Field(
        default_factory=lambda: [4, 5],  # Rear L/R in 5.1
        description="Output channels for music (0-indexed)"
    )
    
    sfx_channels: List[int] = Field(
        default_factory=lambda: [0, 1],  # Front L/R
        description="Output channels for sound effects (0-indexed)"
    )
    
    @model_validator(mode='after')
    def configure_default_channels(self):
        """Configure channel defaults based on surround mode automatically."""
        # Determine if surround is enabled
        enable_surround = self.enable_surround or (self.output_channels and self.output_channels > 2)
        
        # Auto-configure channels for quad mode if using 5.1 defaults
        if enable_surround and self.surround_mode == "quad":
            # Check if music_channels are still at the 5.1 default and update for quad
            if self.music_channels == [4, 5]:  # Default 5.1 rear channels
                self.music_channels = [2, 3]  # Quad rear channels
        
        return self
    
    def configure_surround_channels(self) -> None:
        """Configure channel routing based on surround mode and settings - for manual calls."""
        enable_surround = self.enable_surround or (self.output_channels and self.output_channels > 2)
        
        if enable_surround and self.surround_mode == "quad":
            # Quad setup: 0=FL, 1=FR, 2=RL, 3=RR
            # Update music channels for quad if they're still at 5.1 defaults
            if self.music_channels == [4, 5]:  # Default 5.1 rear channels
                object.__setattr__(self, 'music_channels', [2, 3])  # Quad rear channels

class AudioConfig(BaseModel):
    """Audio playback configuration."""
    
    master_volume: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Master volume level (0.0 to 1.0)"
    )
    
    environment_volume: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Environment sounds volume level (0.0 to 1.0)"
    )
    
    music_volume: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Music volume level (0.0 to 1.0)"
    )
    
    sfx_volume: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sound effects volume level (0.0 to 1.0)"
    )
    
    config_dir: Optional[str|Path] = Field(
        default=None,
        description="Directory containing audio configuration JSON files"
    )

    audio_dir: Optional[str|Path] = Field(
        default=AUDIO_DIR,
        description="Directory containing audio files"
    )

    music_fade_time: float = Field(
        default=10.0,
        ge=0.0,
        description="Default fade time for music transitions in seconds"
    )

class AudioServiceConfig(BaseServiceConfig):
    """Configuration for the Audio Service."""
    
    # Override the default service name (BaseServiceConfig defines the field but has no default)
    service_name: str = "audio-service"
    
    zmq: PubSubServiceConfig = Field(
        default_factory=lambda: PubSubServiceConfig(
            name="audio-service-pubsub",
            publisher=None,  # Audio service doesn't publish
            subscriber=SubscriberConfig(
                address=ZMQ_TCP_CONNECT_PREFIX,
                port=DEFAULT_PORTS["events"],
                topics=[MessageType.SPACE_TIME_UPDATE, MessageType.PRESENCE_STATUS, MessageType.SPEECH_DETECTED, MessageType.CHANGE_MAP]
            )
        ),
        description="ZeroMQ pub/sub configuration"
    )
    
    osc: OscConfig = Field(
        default_factory=OscConfig,
        description="OSC configuration"
    )
    
    supercollider: SuperColliderConfig = Field(
        default_factory=SuperColliderConfig,
        description="SuperCollider configuration"
    )
    
    audio: AudioConfig = Field(
        default_factory=AudioConfig,
        description="Audio playback configuration"
    )
