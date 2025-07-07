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
    MessageType
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


# Sohkepayin-specific extensions of base message types

class SpaceTimeUpdate(_BaseSpaceTimeUpdate):
    """Sohkepayin-specific SpaceTimeUpdate with era, biome, and emotion."""
    time_period: Optional[str] = None  # Sohkepayin uses time_period instead of era
    biome: Biome
    emotion: Optional[Emotion] = None  # Sohkepayin adds emotional context


class RenderRequest(_BaseRenderRequest):
    """Sohkepayin-specific RenderRequest with era, biome, and emotion."""
    time_period: Optional[str] = None  # Sohkepayin uses time_period instead of era
    biome: Biome
    emotion: Optional[Emotion] = None  # Sohkepayin adds emotional context
    setting_tags: Optional[List[str]] = None  # Additional setting descriptors


class ImageReady(_BaseImageReady):
    """Sohkepayin-specific ImageReady with era, biome, and emotion."""
    time_period: Optional[str] = None  # Sohkepayin uses time_period instead of era
    biome: Biome
    emotion: Optional[Emotion] = None


class SuggestTimePeriodPayload(AgentControlEventPayload):
    """Sohkepayin-specific SuggestBiomePayload with biome_suggestion field."""
    time_period: str


class DisplayMedia(_BaseDisplayMedia):
    """Sohkepayin-specific DisplayMedia with time_period, biome, and emotion context."""
    time_period: Optional[str] = None
    biome: Optional[Biome] = None
    emotion: Optional[Emotion] = None


