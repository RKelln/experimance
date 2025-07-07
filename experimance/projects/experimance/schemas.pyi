"""
Type stubs for Experimance-specific schemas.

This file provides type information for the dynamically loaded
Experimance schemas for static type checkers like mypy.
"""

from typing import Optional
from experimance_common.schemas_base import (
    StringComparableEnum,
    MessageBase,
    SpaceTimeUpdate as _BaseSpaceTimeUpdate,
    RenderRequest as _BaseRenderRequest,
    ImageReady as _BaseImageReady,
    DisplayMedia as _BaseDisplayMedia,
    AgentControlEventPayload,
    ImageSource
)

class Era(StringComparableEnum):
    """Era definitions for the Experimance installation."""
    WILDERNESS: str
    PRE_INDUSTRIAL: str
    EARLY_INDUSTRIAL: str
    LATE_INDUSTRIAL: str
    MODERN: str
    CURRENT: str
    FUTURE: str
    DYSTOPIA: str
    RUINS: str

class Biome(StringComparableEnum):
    """Biome definitions for the Experimance installation."""
    RAINFOREST: str
    TEMPERATE_FOREST: str
    BOREAL_FOREST: str
    DECIDUOUS_FOREST: str
    DESERT: str
    MOUNTAIN: str
    TROPICAL_ISLAND: str
    RIVER: str
    TUNDRA: str
    STEPPE: str
    COASTAL: str
    SWAMP: str
    PLAINS: str
    ARCTIC: str
    JUNGLE: str

# Extended message types with Experimance-specific fields

class SpaceTimeUpdate(_BaseSpaceTimeUpdate):
    """Experimance-specific SpaceTimeUpdate with era and biome fields."""
    era: Era
    biome: Biome

class RenderRequest(_BaseRenderRequest):
    """Experimance-specific RenderRequest with era and biome fields."""
    era: Era
    biome: Biome

class ImageReady(_BaseImageReady):
    """Experimance-specific ImageReady with era and biome fields."""
    era: Era
    biome: Biome

class SuggestBiomePayload(AgentControlEventPayload):
    """Experimance-specific SuggestBiomePayload with biome_suggestion field."""
    biome_suggestion: Biome

class DisplayMedia(_BaseDisplayMedia):
    """Experimance-specific DisplayMedia with era and biome context."""
    era: Optional[Era]
    biome: Optional[Biome]
