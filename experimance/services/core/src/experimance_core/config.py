#!/usr/bin/env python3
"""
Configuration schema for the Experimance Core Service.

This module defines Pydantic models for validating and accessing
core service configuration in a type-safe way.
"""

from typing import List, Tuple
from pathlib import Path

from pydantic import BaseModel, Field

from experimance_common.config import Config
from experimance_common.constants import DEFAULT_PORTS


class ExperimanceCoreConfig(BaseModel):
    """Core service configuration."""
    
    name: str = Field(
        default="experimance_core",
        description="Service instance name"
    )
    
    heartbeat_interval: float = Field(
        default=3.0,
        description="Heartbeat interval in seconds"
    )


class ZmqConfig(BaseModel):
    """ZeroMQ configuration for the Core Service."""
    
    events_sub_address: str = Field(
        default=f"tcp://localhost:{DEFAULT_PORTS['events']}",
        description="Address for subscribing to unified events channel"
    )
    
    events_pub_address: str = Field(
        default=f"tcp://*:{DEFAULT_PORTS['events']}",
        description="Address for publishing to unified events channel"
    )


class StateMachineConfig(BaseModel):
    """State machine configuration."""
    
    idle_timeout: float = Field(
        default=45.0,
        description="Idle timeout in seconds before state changes"
    )
    
    wilderness_reset: float = Field(
        default=300.0,
        description="Time in seconds for full reset to wilderness"
    )
    
    interaction_threshold: float = Field(
        default=0.3,
        description="Threshold for user interaction detection"
    )
    
    era_min_duration: float = Field(
        default=10.0,
        description="Minimum time in seconds before era can change"
    )


class DepthProcessingConfig(BaseModel):
    """Depth processing configuration."""
    
    camera_config_path: str = Field(
        default="depth_camera_config.json",
        description="Path to depth camera configuration file"
    )
    
    fps: int = Field(
        default=6,
        description="Camera frames per second (limited by resolution)"
    )

    change_threshold: int = Field(
        default=50,
        description="Threshold for depth change detection"
    )
    
    min_depth: float = Field(
        default=0.49,
        description="Minimum depth value"
    )
    
    max_depth: float = Field(
        default=0.56,
        description="Maximum depth value"
    )
    
    resolution: Tuple[int, int] = Field(
        default=(1280, 720),
        description="Depth camera resolution"
    )
    
    output_size: Tuple[int, int] = Field(
        default=(1024, 1024),
        description="Processed output size"
    )


class AudioConfig(BaseModel):
    """Audio configuration."""
    
    tag_config_path: str = Field(
        default="config/audio_tags.json",
        description="Path to audio tag configuration file"
    )
    
    interaction_sound_duration: float = Field(
        default=2.0,
        description="Duration for interaction sounds in seconds"
    )


class PromptingConfig(BaseModel):
    """Prompting configuration."""
    
    data_path: str = Field(
        default="data/",
        description="Path to prompt data directory"
    )
    
    locations_file: str = Field(
        default="locations.json",
        description="Locations data file name"
    )
    
    developments_file: str = Field(
        default="anthropocene.json",
        description="Developments data file name"
    )


class CoreServiceConfig(Config):
    """Complete configuration schema for the Core Service."""
    
    service_name: str = "experimance-core"
    
    # Main configuration sections
    experimance_core: ExperimanceCoreConfig = Field(default_factory=ExperimanceCoreConfig)
    zmq: ZmqConfig = Field(default_factory=ZmqConfig)
    state_machine: StateMachineConfig = Field(default_factory=StateMachineConfig)
    depth_processing: DepthProcessingConfig = Field(default_factory=DepthProcessingConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    prompting: PromptingConfig = Field(default_factory=PromptingConfig)
