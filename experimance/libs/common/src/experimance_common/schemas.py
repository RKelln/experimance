"""
Schema definitions for Experimance inter-service messages.
"""

from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class Era(str, Enum):
    """Era definitions for the Experimance installation."""
    WILDERNESS = "wilderness"
    PRE_INDUSTRIAL = "pre_industrial"
    EARLY_INDUSTRIAL = "early_industrial"
    LATE_INDUSTRIAL = "late_industrial"
    EARLY_MODERN = "early_modern"
    MODERN = "modern"
    AI = "ai_future"
    POST_APOCALYPTIC = "post_apocalyptic"
    RUINS = "ruins"


class Biome(str, Enum):
    """Biome definitions for the Experimance installation."""
    DESERT = "desert"
    FOREST = "forest"
    TEMPERATE_FOREST = "temperate_forest"
    TROPICAL_FOREST = "tropical_forest"
    MOUNTAINS = "mountains" 
    HILLS = "hills"
    TUNDRA = "tundra"
    COASTAL = "coastal"
    PLAINS = "plains"
    SAVANNA = "savanna"


class TransitionStyle(str, Enum):
    """Transition style definitions."""
    DISSOLVE = "dissolve"
    MORPH = "morph"
    WIPE = "wipe"
    SIMPLE = "simple"


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
