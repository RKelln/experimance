#!/usr/bin/env python3
"""
Configuration schema for the Experimance Audio Service.

This module defines Pydantic models for validating and accessing
audio service configuration in a type-safe way.
"""

from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel, Field

from experimance_common.config import BaseConfig, BaseServiceConfig
from experimance_common.schemas import MessageType
from experimance_common.zmq.config import SubscriberConfig, PubSubServiceConfig
from experimance_common.constants import DEFAULT_PORTS, AUDIO_SERVICE_DIR, ZMQ_TCP_CONNECT_PREFIX

DEFAULT_CONFIG_PATH = f"{AUDIO_SERVICE_DIR}/config.toml"

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
    
    config_dir: Optional[str] = Field(
        default=None,
        description="Directory containing audio configuration JSON files"
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
                topics=[MessageType.SPACE_TIME_UPDATE, MessageType.IDLE_STATUS, MessageType.AGENT_CONTROL_EVENT]
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
