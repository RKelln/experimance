"""
Sohkepayin-specific schema extensions and overrides.

This file extends the base schemas with Sohkepayin project-specific
definitions. These schemas are automatically merged with the base schemas
when PROJECT_ENV=sohkepayin.
"""

from enum import Enum
from typing import Optional, List
from experimance_common.schemas_base import (
    StringComparableEnum, 
    SpaceTimeUpdate as _BaseSpaceTimeUpdate,
    RenderRequest as _BaseRenderRequest,
    ImageReady as _BaseImageReady,
    AgentControlEventPayload,
    DisplayMedia as _BaseDisplayMedia,
    MessageBase,
    MessageType as _BaseMessageType,
    ContentType
)

# Sohkepayin-specific enums (different from Experimance)
class Biome(StringComparableEnum):
    """Biome definitions for the Sohkepayin installation."""
    FOREST = "forest"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    MOUNTAIN = "mountain"
    RIVER = "river"
    LAKE = "lake"
    URBAN = "urban"
    INDOORS = "indoors"
    DESERT = "desert"
    COAST = "coast"


class Emotion(StringComparableEnum):
    """Emotion types specific to Sohkepayin."""
    JOY = "joy"
    SORROW = "sorrow"
    ANGER = "anger"
    PEACE = "peace"
    LONGING = "longing"
    HOPE = "hope"


# Extended MessageType with Sohkepayin-specific message types
class MessageType(StringComparableEnum):
    """Message types used in the Sohkepayin system (extends base MessageType)."""
    # Base Experimance message types
    SPACE_TIME_UPDATE = "SpaceTimeUpdate"
    RENDER_REQUEST = "RenderRequest"
    IDLE_STATUS = "IdleStatus"
    IMAGE_READY = "ImageReady"
    TRANSITION_READY = "TransitionReady"
    LOOP_READY = "LoopReady"
    AGENT_CONTROL_EVENT = "AgentControlEvent"
    TRANSITION_REQUEST = "TransitionRequest"
    LOOP_REQUEST = "LoopRequest"
    HEARTBEAT = "Heartbeat"
    ALERT = "Alert"
    # Display service message types
    DISPLAY_MEDIA = "DisplayMedia"
    DISPLAY_TEXT = "DisplayText"
    REMOVE_TEXT = "RemoveText"
    CHANGE_MAP = "ChangeMap"
    
    # Sohkepayin-specific message types
    STORY_HEARD = "StoryHeard"
    UPDATE_LOCATION = "UpdateLocation"


class SuggestTimePeriodPayload(AgentControlEventPayload):
    """Sohkepayin-specific SuggestTimePeriodPayload with time_period field."""
    time_period: str


# Sohkepayin-specific message types for story handling

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
