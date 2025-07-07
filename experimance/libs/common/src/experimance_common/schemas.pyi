"""
Static-analysis stub for experimance_common.schemas.

It re-exports every Pydantic model that is guaranteed to exist in the
*base* definitions so that type checkers have something concrete to import.
At runtime the real module is assembled dynamically and may *extend* or
*replace* these classes with project-specific versions.
"""

# Re-export all base schemas that are available in all projects
from experimance_common.schemas_base import (
    # Base classes
    StringComparableEnum,
    MessageSchema,
    MessageBase,
    
    # Enums that are project-independent
    TransitionStyle,
    DisplayContentType,
    DisplayTransitionType,
    MessageType,
    ContentType,
    
    # Base message types (may be extended by projects)
    SpaceTimeUpdate,
    ImageSource,
    RenderRequest,
    IdleStatus,
    ImageReady,
    TransitionReady,
    LoopReady,
    AgentControlEventPayload,
    AudiencePresentPayload,
    SpeechDetectedPayload,
    AgentControlEvent,
    DisplayText,
    RemoveText,
    TransitionRequest,
    LoopRequest,
    DisplayMedia,
)

# Note: Era and Biome are defined in project-specific schema files
# The actual types available depend on the PROJECT_ENV setting:
# - When PROJECT_ENV=experimance: Era and Biome have Experimance-specific values
# - When PROJECT_ENV=sohkepayin: Era and Biome have Sohkepayin-specific values
# - Other projects can define their own Era and Biome enums

__all__: list[str] = [
    # Base classes
    "StringComparableEnum",
    "MessageSchema", 
    "MessageBase",
    
    # Enums
    "TransitionStyle",
    "DisplayContentType",
    "DisplayTransitionType", 
    "MessageType",
    "ContentType",
    
    # Message types
    "SpaceTimeUpdate",
    "ImageSource",
    "RenderRequest",
    "IdleStatus",
    "ImageReady",
    "TransitionReady",
    "LoopReady",
    "AgentControlEventPayload",
    "AudiencePresentPayload", 
    "SpeechDetectedPayload",
    "AgentControlEvent",
    "DisplayText",
    "RemoveText",
    "TransitionRequest",
    "LoopRequest",
    "DisplayMedia",
]