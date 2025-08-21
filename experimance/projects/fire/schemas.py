"""
Fires-specific schema extensions and overrides.

This file extends the base schemas with Fires project-specific
definitions. These schemas are automatically merged with the base schemas
when PROJECT_ENV=fire.
"""

from enum import Enum
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

# Fires-specific enums (different from Experimance)
class Emotion(StringComparableEnum):
    """Emotion types specific to Fires."""
    JOY = "joy"
    SORROW = "sorrow"
    ANGER = "anger"
    PEACE = "peace"
    LONGING = "longing"
    HOPE = "hope"


# Fires-specific extensions of base schemas
class RenderRequest(_BaseRenderRequest):
    """Extended RenderRequest with Fires-specific fields."""
    # Fires-specific fields can be added here
    strength: Optional[float] = None  # img2img strength (0.0-1.0, where 1.0 completely ignores reference image)


# Extended MessageType with Fires-specific message types
class MessageType(StringComparableEnum):
    """Message types used in the Fires system (extends base MessageType)."""
    # Base Experimance message types
    SPACE_TIME_UPDATE = "SpaceTimeUpdate"
    RENDER_REQUEST = "RenderRequest"
    PRESENCE_STATUS = "PresenceStatus"
    IMAGE_READY = "ImageReady"
    TRANSITION_READY = "TransitionReady"
    LOOP_READY = "LoopReady"
    AUDIENCE_PRESENT = "AudiencePresent"
    SPEECH_DETECTED = "SpeechDetected"
    TRANSITION_REQUEST = "TransitionRequest"
    LOOP_REQUEST = "LoopRequest"
    ALERT = "Alert"
    # Display service message types
    DISPLAY_MEDIA = "DisplayMedia"
    DISPLAY_TEXT = "DisplayText"
    REMOVE_TEXT = "RemoveText"
    CHANGE_MAP = "ChangeMap"
    
    # Fires-specific message types
    STORY_HEARD = "StoryHeard"
    UPDATE_LOCATION = "UpdateLocation"
    TRANSCRIPT_UPDATE = "TranscriptUpdate"


class SuggestTimePeriodPayload(MessageBase):
    """Fires-specific SuggestTimePeriodPayload with time_period field."""
    time_period: str


# Fires-specific message types for story handling

class StoryHeard(MessageBase):
    """Message sent when a complete story is heard from the audience."""
    type: MessageType = MessageType.STORY_HEARD
    content: str  # The story content/transcript
    speaker_id: Optional[str] = None  # Optional speaker identification
    confidence: Optional[float] = None  # Optional confidence score
    timestamp: Optional[str] = None  # When the story was heard


class UpdateLocation(MessageBase):
    """Message to update the current environmental setting."""
    type: MessageType = MessageType.UPDATE_LOCATION
    content: str  # Update content/instructions
    update_type: Optional[str] = None  # Type of update (e.g., "clarification", "addition")
    timestamp: Optional[str] = None


class TranscriptUpdate(MessageBase):
    """Message to stream transcript utterances to fire_core for processing."""
    type: MessageType = MessageType.TRANSCRIPT_UPDATE
    content: str  # The transcript content/utterance
    speaker_id: str  # Speaker identification (required)
    speaker_display_name: Optional[str] = None  # Human-readable speaker name
    session_id: Optional[str] = None  # Session identifier
    turn_id: Optional[str] = None  # Turn identifier within session
    confidence: Optional[float] = None  # Confidence score
    timestamp: Optional[str] = None  # When the utterance was captured
    is_partial: bool = False  # Whether this is a partial/interim result
    duration: Optional[float] = None  # Duration of the utterance
