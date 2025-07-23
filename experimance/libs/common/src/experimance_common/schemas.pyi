"""
Static-analysis stub for experimance_common.schemas.

This stub provides type information for the dynamically loaded schemas.
At runtime, the actual module loads project-specific extensions based on
the PROJECT_ENV environment variable and makes them available in this namespace.

For static type checking, this file conditionally imports the appropriate
project-specific types based on the PROJECT_ENV environment variable.
"""

import os
from typing import TYPE_CHECKING

# Re-export base schemas that are NOT extended by projects
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
    
    # Message types that are NOT extended by projects
    PresenceStatus,
    AudiencePresent,
    SpeechDetected,
    DisplayMedia,
    DisplayText,
    ImageSource,
    LoopReady,
    LoopRequest,
    RemoveText,
    TransitionReady,
    TransitionRequest,
)

# Conditionally import project-specific types based on PROJECT_ENV
if TYPE_CHECKING:
    _PROJECT_ENV = os.getenv("PROJECT_ENV", "experimance")
    
    if _PROJECT_ENV == "experimance":
        from projects.experimance.schemas import (
            Biome,
            Era,
            ImageReady,
            RenderRequest,
            SpaceTimeUpdate,
            RequestBiome,
        )
    elif _PROJECT_ENV == "sohkepayin":
        from projects.sohkepayin.schemas import (
            Emotion,
            MessageType,
            StoryHeard,
            SuggestTimePeriodPayload,
            UpdateLocation,
        )
    else:
        # Fallback for unknown projects - use base types and create minimal stubs
        from experimance_common.schemas_base import (
            SpaceTimeUpdate,
            RenderRequest,
            ImageReady,
            DisplayMedia,
        )
        
        # Create minimal project-specific enums for unknown projects
        class Era(StringComparableEnum):
            """Fallback Era enum for unknown projects."""
            ...
        
        class Biome(StringComparableEnum):
            """Fallback Biome enum for unknown projects."""
            ...

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
    
    # Message types that are NOT extended by projects
    "PresenceStatus",
    "AudiencePresent",
    "SpeechDetected",
    "DisplayText",
    "ImageSource",
    "LoopReady",
    "LoopRequest",
    "RemoveText",
    "TransitionReady",
    "TransitionRequest",
    
    # Common project-specific types (available in all projects)
    "Biome",  # Extended by all projects
    "DisplayMedia",  # Extended by all projects
    "ImageReady",  # Extended by all projects
    "RenderRequest",  # Extended by all projects
    "SpaceTimeUpdate",  # Extended by all projects
    
    # Note: Project-specific types like Era, Emotion, RequestBiome, etc.
    # are not included here since they're not universal across all projects.
    # They are still available for import when the appropriate PROJECT_ENV is set.
]