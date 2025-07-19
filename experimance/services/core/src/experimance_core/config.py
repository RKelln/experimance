#!/usr/bin/env python3
"""
Configuration schema for the Experimance Core Service.

This module defines Pydantic models for validating and accessing
core service configuration in a type-safe way.
"""

from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

from dataclasses import dataclass
from pydantic import BaseModel, Field

import numpy as np

from experimance_common.config import BaseConfig
from experimance_common.constants import (
    DEFAULT_PORTS, CORE_SERVICE_DIR, ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX,
    get_project_config_path
)
from experimance_common.zmq.config import (
    ControllerServiceConfig, WorkerConfig, PublisherConfig, 
    SubscriberConfig, ControllerPushConfig, ControllerPullConfig, MessageType
)

# Define the default configuration path with project-aware fallback
DEFAULT_CONFIG_PATH = get_project_config_path("core", CORE_SERVICE_DIR)
DEFAULT_CAMERA_CONFIG_DIR = CORE_SERVICE_DIR
DEFAULT_CAMERA_CONFIG_PATH = DEFAULT_CAMERA_CONFIG_DIR / "depth_camera_config.json"

CAMERA_RESET_TIMEOUT = 45.0 # seconds

class ExperimanceCoreConfig(BaseModel):
    """Core service configuration."""
    
    name: str = Field(
        default="experimance_core",
        description="Service instance name"
    )
    
    transition_timeout: float = Field(
        default=120.0,
        description="Maximum time allowed for transitions in seconds"
    )
    
    change_smoothing_queue_size: int = Field(
        default=3,
        description="Size of the change score queue for smoothing (minimum values reduce hand entry/exit artifacts)"
    )
    
    render_request_cooldown: float = Field(
        default=2.0,
        description="Minimum interval between render requests in seconds (throttling)"
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
        default=60.0,
        description="Minimum time in seconds before era can change: 0 = disable"
    )

    era_max_duration: float = Field(
        default=120.0,
        description="Maximum time in seconds before era can change: 0 = disable"
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
        default=(1280, 720),
        description="Camera resolution (width, height)"
    )
    
    fps: int = Field(
        default=30,
        description="Camera frames per second"
    )
    
    align_frames: bool = Field(
        default=False,
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
        description="Threshold for depth change detection, per pixel (0-255)"
    )
    
    detect_hands: bool = Field(
        default=True,
        description="Enable hand detection"
    )
    
    crop_to_content: bool = Field(
        default=True,
        description="Crop output to content area"
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
    
    # Debug depth visualization for alignment
    debug_depth: bool = Field(
        default=False,
        description="Send depth map each frame to display service for alignment debugging"
    )
    
    flip_horizontal: bool = Field(
        default=False,
        description="Flip depth map horizontally for camera/projector alignment"
    )
    
    flip_vertical: bool = Field(
        default=False,
        description="Flip depth map vertically for camera/projector alignment"
    )

    circular_crop: bool = Field(
        default=False,
        description="Apply circular crop to depth map"
    )

    blur_depth: bool = Field(
        default=False,
        description="Apply Gaussian blur to depth map to reduce noise"
    )
    
    # Mask stability parameters
    mask_stability_frames: int = Field(
        default=20,
        description="Frames to analyze for mask stability"
    )
    
    mask_stability_threshold: float = Field(
        default=0.95,
        description="Similarity threshold for mask stability, per image (0.0-1.0)"
    )
    
    mask_lock_after_stable: bool = Field(
        default=True,
        description="Lock mask once stable"
    )
    
    mask_allow_updates: bool = Field(
        default=False,
        description="Allow mask updates when bowl moves significantly"
    )
    
    mask_update_threshold: float = Field(
        default=0.7,
        description="Threshold for detecting bowl movement"
    )
    
    # Change detection parameters for core service filtering
    significant_change_threshold: float = Field(
        default=0.01,
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
    
    def __str__(self):
        return (f"DepthFrame(frame_number={self.frame_number}, "
                f"timestamp={self.timestamp}, "
                f"hand_detected={self.hand_detected}, "
                f"change_score={self.change_score})")

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


class CoreServiceConfig(BaseConfig):
    """Configuration for Experimance core service."""
    
    service_name: str = "experimance_core"
    
    # Debugging and visualization options
    visualize: bool = Field(
        default=False,
        description="Enable visual debugging - display depth image and processing flags in real-time"
    )
    
    # Main configuration sections
    experimance_core: ExperimanceCoreConfig = Field(default_factory=ExperimanceCoreConfig)
    zmq: ControllerServiceConfig = Field(
        default_factory=lambda: ControllerServiceConfig(
            name="experimance_core",  # Changed from "experimance-core" to match service_name
            publisher=PublisherConfig(
                address=ZMQ_TCP_BIND_PREFIX,
                port=DEFAULT_PORTS["events"],
                default_topic="core.events"
            ),
            subscriber=SubscriberConfig(
                address=ZMQ_TCP_CONNECT_PREFIX,
                port=DEFAULT_PORTS["agent"],
                topics=[MessageType.AGENT_CONTROL_EVENT]
            ),
            workers={
                "image_server": WorkerConfig(
                    name="image_server",
                    push_config=ControllerPushConfig(
                        port=DEFAULT_PORTS["image_requests"]
                    ),
                    pull_config=ControllerPullConfig(
                        port=DEFAULT_PORTS["image_results"]
                    )
                ),
                # "audio": WorkerConfig(
                #     name="audio",
                #     push_config=ControllerPushConfig(
                #         port=DEFAULT_PORTS["audio_requests"]
                #     ),
                #     pull_config=ControllerPullConfig(
                #         port=DEFAULT_PORTS["audio_results"]
                #     )
                #                ),
                # "display": WorkerConfig(
                #     name="display",
                #     push_config=ControllerPushConfig(
                #         port=DEFAULT_PORTS["display_requests"]
                #     ),
                #     pull_config=ControllerPullConfig(
                #         port=DEFAULT_PORTS["display_results"]
                #     )
                # )
            }
        )
    )
    state_machine: StateMachineConfig = Field(default_factory=StateMachineConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    depth_processing: DepthProcessingConfig = Field(default_factory=DepthProcessingConfig)  # Legacy compatibility
    audio: AudioConfig = Field(default_factory=AudioConfig)
    prompting: PromptingConfig = Field(default_factory=PromptingConfig)
