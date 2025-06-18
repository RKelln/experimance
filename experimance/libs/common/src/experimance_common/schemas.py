"""
Schema definitions for Experimance inter-service messages.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class Era(str, Enum):
    """Era definitions for the Experimance installation.
    
    These values must match the 'eras' array in data/experimance_config.json
    """
    WILDERNESS = "wilderness"
    PRE_INDUSTRIAL = "pre_industrial"
    EARLY_INDUSTRIAL = "early_industrial"
    LATE_INDUSTRIAL = "late_industrial"
    MODERN = "modern"
    CURRENT = "current"
    FUTURE = "future"
    DYSTOPIA = "dystopia"
    RUINS = "ruins"


class Biome(str, Enum):
    """Biome definitions for the Experimance installation.
    
    These values must match the 'biomes' array in data/experimance_config.json
    """
    RAINFOREST = "rainforest"
    TEMPERATE_FOREST = "temperate_forest"
    BOREAL_FOREST = "boreal_forest"
    DECIDUOUS_FOREST = "deciduous_forest"
    DESERT = "desert"
    MOUNTAIN = "mountain"
    TROPICAL_ISLAND = "tropical_island"
    RIVER = "river"
    TUNDRA = "tundra"
    STEPPE = "steppe"
    COASTAL = "coastal"
    SWAMP = "swamp"
    PLAINS = "plains"
    ARCTIC = "arctic"


class TransitionStyle(str, Enum):
    """Transition style definitions."""
    DISSOLVE = "dissolve"
    MORPH = "morph"
    WIPE = "wipe"
    SIMPLE = "simple"


class DisplayContentType(str, Enum):
    """Content types for display media."""
    IMAGE = "image"
    IMAGE_SEQUENCE = "image_sequence"
    VIDEO = "video"


class DisplayTransitionType(str, Enum):
    """Transition types for display media."""
    NONE = "none"                    # No transition, direct display
    FADE = "fade"                    # Simple fade transition
    DISSOLVE = "dissolve"            # Cross-dissolve
    SLIDE = "slide"                  # Slide transition
    MORPH = "morph"                  # Morphing transition
    IMAGE_SEQUENCE = "image_sequence"  # Play image sequence
    VIDEO = "video"                  # Play video transition


class MessageBase(BaseModel):
    """Base class for all message types."""
    type: str


class EraChanged(MessageBase):
    """Event published when the era changes."""
    type: str = "EraChanged"
    era: Era
    biome: Biome


class RenderRequest(MessageBase):
    """Request to generate a new image."""
    type: str = "RenderRequest"
    request_id: str
    era: Era
    biome: Biome
    prompt: str
    depth_map_png: Optional[str] = None  # Base64 encoded PNG


class IdleStatus(MessageBase):
    """Event published when the idle status changes."""
    type: str = "Idle"
    status: bool  # True = now idle, False = exiting idle


class ImageReady(MessageBase):
    """Event published when a new image is ready."""
    type: str = "ImageReady"
    request_id: Optional[str] = None
    image_id: str
    uri: str  # URI to the image (file://, http://, etc.)


class TransitionReady(MessageBase):
    """Event published when a transition video/sequence is ready."""
    type: str = "TransitionReady"
    transition_id: str
    uri: str  # URI to video or image sequence manifest
    is_video: bool  # True if URI points to video file, False if image sequence
    loop: bool = False  # Always False for transitions
    final_frame_uri: str  # URI to the final frame to display after transition


class LoopReady(MessageBase):
    """Event published when a loop animation is ready."""
    type: str = "LoopReady"
    loop_id: str
    uri: str  # URI to video or image sequence manifest
    is_video: bool  # True if URI points to video file, False if image sequence
    duration_s: Optional[float] = None  # Duration in seconds if known


class AgentControlEventPayload(BaseModel):
    """Base class for agent control event payloads."""
    pass


class SuggestBiomePayload(AgentControlEventPayload):
    """Payload for suggesting a biome."""
    biome_suggestion: Biome


class AudiencePresentPayload(AgentControlEventPayload):
    """Payload for audience presence detection."""
    status: bool


class SpeechDetectedPayload(AgentControlEventPayload):
    """Payload for speech detection."""
    is_speaking: bool


class AgentControlEvent(MessageBase):
    """Event published by the agent to control other services."""
    type: str = "AgentControlEvent"
    sub_type: str  # "SuggestBiome", "AudiencePresent", "SpeechDetected"
    payload: Dict  # Structure varies based on sub_type


class TransitionRequest(MessageBase):
    """Request to generate a transition between two images."""
    type: str = "TransitionRequest"
    request_id: str
    from_image_uri: str  # URI to the source image
    to_image_uri: str  # URI to the destination image
    style: TransitionStyle = TransitionStyle.DISSOLVE
    duration_frames: int  # Number of frames for the transition


class LoopRequest(MessageBase):
    """Request to generate a loop animation from a still image."""
    type: str = "LoopRequest"
    request_id: str
    still_image_uri: str  # URI to the still image to animate
    style: str  # Hint for the animation style


class ContentType(str, Enum):
    """Types of content that can be displayed."""
    IMAGE = "image"                    # Single static image
    IMAGE_SEQUENCE = "image_sequence"  # Sequence of images (for transitions)
    VIDEO = "video"                    # Video file


class DisplayMedia(MessageBase):
    """Message for sending media content to display service."""
    type: str = "DisplayMedia"
    
    # Content specification
    content_type: ContentType
    
    # For IMAGE content_type
    image_data: Optional[Any] = None      # Image data (numpy array, PIL, etc.)
    uri: Optional[str] = None             # File URI for image
    
    # For IMAGE_SEQUENCE content_type  
    sequence_path: Optional[str] = None   # Path to directory with numbered images
    
    # For VIDEO content_type
    video_path: Optional[str] = None      # Path to video file
    
    # Display properties
    duration: Optional[float] = None      # Duration in seconds (for sequences/videos)
    loop: bool = False                    # Whether to loop the content
    fade_in: Optional[float] = None       # Fade in duration in seconds
    fade_out: Optional[float] = None      # Fade out duration in seconds
    
    # Context information
    era: Optional[Era] = None
    biome: Optional[Biome] = None
    source_request_id: Optional[str] = None  # Links back to original render request
