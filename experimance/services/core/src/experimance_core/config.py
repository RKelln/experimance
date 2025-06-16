#!/usr/bin/env python3
"""
Configuration schema for the Experimance Core Service.

This module defines Pydantic models for validating and accessing
core service configuration in a type-safe way.
"""

from typing import List, Tuple, Optional
from enum import Enum

from dataclasses import dataclass
from pydantic import BaseModel, Field

import numpy as np

from experimance_common.config import Config
from experimance_common.constants import DEFAULT_PORTS, CORE_SERVICE_DIR

# Define the default configuration path relative to the project root
DEFAULT_CONFIG_PATH = f"{CORE_SERVICE_DIR}/config.toml"
DEFAULT_CAMERA_CONFIG_DIR = CORE_SERVICE_DIR
DEFAULT_CAMERA_CONFIG_PATH = f"{DEFAULT_CAMERA_CONFIG_DIR}/depth_camera_config.json"

CAMERA_RESET_TIMEOUT = 45.0 # seconds

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


class ColorizerScheme(Enum):
    """Colorizer schemes for depth visualization."""
    # from: https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.colorizer.html
    JET = 0
    CLASSIC = 1
    WHITE_TO_BLACK = 2
    BLACK_TO_WHITE = 3
    BIO = 4
    COLD = 5
    WARM = 6
    QUANTIZED = 7
    PATTERN = 8
    
class CameraConfig(BaseModel):
    """Camera configuration for the robust camera system."""
    
    resolution: Tuple[int, int] = Field(
        default=(640, 480),
        description="Camera resolution (width, height)"
    )
    
    fps: int = Field(
        default=30,
        description="Camera frames per second"
    )
    
    align_frames: bool = Field(
        default=True,
        description="Enable frame alignment"
    )
    
    min_depth: float = Field(
        default=0.0,
        description="Minimum depth value in meters"
    )
    
    max_depth: float = Field(
        default=10.0,
        description="Maximum depth value in meters"
    )
    
    json_config_path: Optional[str] = Field(
        default=None,
        description="Path to RealSense JSON config file"
    )
    
    colorizer_scheme: ColorizerScheme = Field(
        default=ColorizerScheme.CLASSIC,
        description="Colorizer scheme for depth visualization"
    )

    # Processing parameters
    output_resolution: Tuple[int, int] = Field(
        default=(1024, 1024),
        description="Output image resolution (width, height)"
    )
    
    change_threshold: int = Field(
        default=60,
        description="Threshold for change detection"
    )
    
    detect_hands: bool = Field(
        default=True,
        description="Enable hand detection"
    )
    
    crop_to_content: bool = Field(
        default=True,
        description="Crop output to content area"
    )
    
    warm_up_frames: int = Field(
        default=10,
        description="Number of frames to skip during warmup"
    )
    
    lightweight_mode: bool = Field(
        default=False,
        description="Skip some processing for higher FPS"
    )
    
    verbose_performance: bool = Field(
        default=False,
        description="Show detailed performance timing"
    )
    
    debug_mode: bool = Field(
        default=False,
        description="Include intermediate images for visualization"
    )
    
    # Mask stability parameters
    mask_stability_frames: int = Field(
        default=20,
        description="Frames to analyze for mask stability"
    )
    
    mask_stability_threshold: float = Field(
        default=0.95,
        description="Similarity threshold for mask stability"
    )
    
    mask_lock_after_stable: bool = Field(
        default=True,
        description="Lock mask once stable"
    )
    
    mask_allow_updates: bool = Field(
        default=True,
        description="Allow mask updates when bowl moves significantly"
    )
    
    mask_update_threshold: float = Field(
        default=0.7,
        description="Threshold for detecting bowl movement"
    )
    
    # Change detection parameters for core service filtering
    significant_change_threshold: float = Field(
        default=0.02,
        description="Minimum change score to process a frame (0.0-1.0)"
    )
    
    edge_erosion_pixels: int = Field(
        default=10,
        description="Pixels to erode from edges to reduce noise in change detection"
    )
    
    # Retry parameters
    max_retries: int = Field(
        default=3,
        description="Maximum camera initialization retries"
    )
    
    retry_delay: float = Field(
        default=2.0,
        description="Initial retry delay in seconds"
    )
    
    max_retry_delay: float = Field(
        default=30.0,
        description="Maximum retry delay in seconds"
    )
    
    aggressive_reset: bool = Field(
        default=False,
        description="Use more aggressive reset strategies"
    )
    
    skip_advanced_config: bool = Field(
        default=False,
        description="Skip advanced JSON config loading"
    )
    
    # RealSense filters
    enable_filters: bool = Field(
        default=True,
        description="Enable RealSense post-processing filters"
    )
    
    spatial_filter: bool = Field(
        default=True,
        description="Enable spatial filter"
    )
    
    temporal_filter: bool = Field(
        default=True,
        description="Enable temporal filter"
    )
    
    decimation_filter: bool = Field(
        default=False,
        description="Enable decimation filter"
    )
    
    hole_filling_filter: bool = Field(
        default=True,
        description="Enable hole filling filter"
    )
    
    threshold_filter: bool = Field(
        default=False,
        description="Enable threshold filter"
    )
    
    # Spatial filter settings
    spatial_filter_magnitude: float = Field(
        default=2.0,
        description="Spatial filter magnitude"
    )
    
    spatial_filter_alpha: float = Field(
        default=0.5,
        description="Spatial filter alpha"
    )
    
    spatial_filter_delta: float = Field(
        default=20.0,
        description="Spatial filter delta"
    )
    
    spatial_filter_hole_fill: int = Field(
        default=1,
        description="Spatial filter hole fill mode"
    )
    
    # Temporal filter settings
    temporal_filter_alpha: float = Field(
        default=0.4,
        description="Temporal filter alpha"
    )
    
    temporal_filter_delta: float = Field(
        default=20.0,
        description="Temporal filter delta"
    )
    
    temporal_filter_persistence: int = Field(
        default=3,
        description="Temporal filter persistence"
    )
    
    # Decimation filter settings
    decimation_filter_magnitude: int = Field(
        default=2,
        description="Decimation filter magnitude"
    )
    
    # Hole filling filter settings
    hole_filling_mode: int = Field(
        default=1,
        description="Hole filling mode (0=disabled, 1=fill_from_left, 2=farest_from_around)"
    )
    
    # Threshold filter settings
    threshold_filter_min: float = Field(
        default=0.15,
        description="Threshold filter minimum value"
    )
    
    threshold_filter_max: float = Field(
        default=4.0,
        description="Threshold filter maximum value"
    )


class DepthProcessingConfig(BaseModel):
    """Depth processing configuration (legacy compatibility)."""
    
    # Keep these for backward compatibility with existing configs
    camera_config_path: str = Field(
        default="depth_camera_config.json",
        description="Path to depth camera configuration file (legacy)"
    )
    
    fps: int = Field(
        default=6,
        description="Camera frames per second (legacy - use camera.fps instead)"
    )

    change_threshold: int = Field(
        default=50,
        description="Threshold for depth change detection (legacy - use camera.change_threshold instead)"
    )
    
    min_depth: float = Field(
        default=0.49,
        description="Minimum depth value (legacy - use camera.min_depth instead)"
    )
    
    max_depth: float = Field(
        default=0.56,
        description="Maximum depth value (legacy - use camera.max_depth instead)"
    )
    
    resolution: Tuple[int, int] = Field(
        default=(1280, 720),
        description="Depth camera resolution (legacy - use camera.resolution instead)"
    )
    
    output_size: Tuple[int, int] = Field(
        default=(1024, 1024),
        description="Processed output size (legacy - use camera.output_resolution instead)"
    )

class CameraState(Enum):
    """Camera operational states."""
    DISCONNECTED = "disconnected"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    RESETTING = "resetting"


@dataclass
class DepthFrame:
    """Depth frame data with metadata."""
    depth_image: np.ndarray
    color_image: Optional[np.ndarray] = None
    hand_detected: Optional[bool] = None
    change_score: float = 0.0
    frame_number: int = 0
    timestamp: float = 0.0
    
    # Debug/visualization intermediate images (only populated when debug_mode=True)
    raw_depth_image: Optional[np.ndarray] = None
    masked_image: Optional[np.ndarray] = None
    importance_mask: Optional[np.ndarray] = None
    cropped_before_resize: Optional[np.ndarray] = None
    change_diff_image: Optional[np.ndarray] = None
    hand_detection_image: Optional[np.ndarray] = None
    
    @property
    def has_interaction(self) -> bool:
        """Check if frame shows user interaction."""
        return self.hand_detected or self.change_score > 0.1
    
    @property
    def has_debug_images(self) -> bool:
        """Check if frame contains debug/intermediate images."""
        return self.raw_depth_image is not None

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
    camera: CameraConfig = Field(default_factory=CameraConfig)
    depth_processing: DepthProcessingConfig = Field(default_factory=DepthProcessingConfig)  # Legacy compatibility
    audio: AudioConfig = Field(default_factory=AudioConfig)
    prompting: PromptingConfig = Field(default_factory=PromptingConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
