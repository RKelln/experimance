"""
Type stubs for Fire project schemas.

This file provides static type information for the Fire project's schema definitions.
These schemas extend the base experimance schemas with Fire-specific types.
"""

from typing import Optional, List
from experimance_common.schemas_base import (
    StringComparableEnum,
    SpaceTimeUpdate as _BaseSpaceTimeUpdate, 
    RenderRequest as _BaseRenderRequest,
    ImageReady as _BaseImageReady,
    DisplayMedia as _BaseDisplayMedia,
    MessageBase,
    MessageType as _BaseMessageType,
    ContentType
)

# Fire-specific enums
class Emotion(StringComparableEnum):
    """Emotion types specific to Fire."""
    JOY: str
    SORROW: str
    ANGER: str
    PEACE: str
    LONGING: str
    HOPE: str

# Extended schemas
class RenderRequest(_BaseRenderRequest):
    """Extended RenderRequest with Fire-specific fields."""
    strength: Optional[float]

# Extended MessageType with Fire-specific message types
class MessageType(StringComparableEnum):
    """Message types used in the Fire system (extends base MessageType)."""
    # Base Experimance message types
    SPACE_TIME_UPDATE: str
    RENDER_REQUEST: str
    PRESENCE_STATUS: str
    IMAGE_READY: str
    TRANSITION_READY: str
    LOOP_READY: str
    AUDIENCE_PRESENT: str
    SPEECH_DETECTED: str
    TRANSITION_REQUEST: str
    LOOP_REQUEST: str
    ALERT: str
    # Display service message types
    DISPLAY_MEDIA: str
    DISPLAY_TEXT: str
    REMOVE_TEXT: str
    CHANGE_MAP: str
    
    # Fire-specific message types
    STORY_HEARD: str
    UPDATE_LOCATION: str
    TRANSCRIPT_UPDATE: str

class SuggestTimePeriodPayload(MessageBase):
    """Fire-specific SuggestTimePeriodPayload with time_period field."""
    time_period: str

# Fire-specific message types for story handling
class StoryHeard(MessageBase):
    """Message sent when a complete story is heard from the audience."""
    type: MessageType
    content: str
    speaker_id: Optional[str]
    confidence: Optional[float]
    timestamp: Optional[str]

class UpdateLocation(MessageBase):
    """Message to update the current environmental setting."""
    type: MessageType
    content: str
    update_type: Optional[str]
    timestamp: Optional[str]

class TranscriptUpdate(MessageBase):
    """Message to stream transcript utterances to fire_core for processing."""
    type: MessageType
    content: str
    speaker_id: str  # Required speaker identification
    speaker_display_name: Optional[str]
    session_id: Optional[str]
    turn_id: Optional[str]
    confidence: Optional[float]
    timestamp: Optional[str]
    is_partial: bool
    duration: Optional[float]

__all__ = [
    "Emotion",
    "RenderRequest", 
    "MessageType",
    "SuggestTimePeriodPayload",
    "StoryHeard",
    "UpdateLocation",
    "TranscriptUpdate",
]
