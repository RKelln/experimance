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
    WILDERNESS = "wilderness"
    PRE_INDUSTRIAL = "pre_industrial"
    EARLY_INDUSTRIAL = "early_industrial"
    LATE_INDUSTRIAL = "late_industrial"
    MODERN = "modern"
    CURRENT = "current"
    FUTURE = "future"
    DYSTOPIA = "dystopia"
    RUINS = "ruins"

class Biome(StringComparableEnum):
    """Biome definitions for the Experimance installation."""
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
    JUNGLE = "jungle"

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
