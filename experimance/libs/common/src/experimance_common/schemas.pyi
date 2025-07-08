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
    ImageSource,
    IdleStatus,
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
)

# Conditionally import project-specific types based on PROJECT_ENV
if TYPE_CHECKING:
    _PROJECT_ENV = os.getenv("PROJECT_ENV", "experimance")
    
    if _PROJECT_ENV == "experimance":
        # Import all experimance-specific types
        #from projects.experimance.schemas import *  # type: ignore[misc]
        from projects.experimance.schemas import (
            Era,
            Biome,
            SpaceTimeUpdate,
            RenderRequest,
            ImageReady,
            # Add any other experimance-specific types here
        )
    elif _PROJECT_ENV == "sohkepayin":
        # Import all sohkepayin-specific types
        from projects.sohkepayin.schemas import *  # type: ignore[misc]
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
    "ImageSource",
    "IdleStatus",
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
    
    # Project-specific types (conditionally imported above with import *)
    # The actual symbols depend on PROJECT_ENV and what's defined in each project
]