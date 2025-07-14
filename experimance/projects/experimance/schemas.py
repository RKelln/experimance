"""
Experimance-specific schema extensions and overrides.

This file extends the base schemas with Experimance project-specific
definitions. These schemas are automatically merged with the base schemas
when PROJECT_ENV=experimance.
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

class Era(StringComparableEnum):
    """Era definitions for the Experimance installation.
    
    These values must match the 'eras' array in data/experimance_config.json
    """
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
    """Biome definitions for the Experimance installation.
    
    These values must match the 'biomes' array in data/experimance_config.json
    """
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


# Experimance-specific extensions of base message types

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


# Add any other Experimance-specific schema extensions here
# For example, completely new message types specific to Experimance:

# class ExperimanceSpecificMessage(MessageBase):
#     """A message type only used in Experimance."""
#     type: str = "ExperimanceSpecific"
#     experimance_field: str
#     # ... other fields
