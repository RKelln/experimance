#!/usr/bin/env python3
"""
Prompt builder for Sohkepayin image generation.

This module creates image generation prompts based on story analysis
and location inference, tailored for the Sohkepayin aesthetic.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from experimance_common.schemas import Biome, Emotion
from .llm import LocationInference

logger = logging.getLogger(__name__)


@dataclass
class ImagePrompt:
    """Image generation prompt with metadata."""
    
    prompt: str
    negative_prompt: str
    style_keywords: List[str]
    biome: Biome
    emotion: Emotion
    aspect_ratio: str = "16:9"  # Default panoramic aspect ratio
    seed: Optional[int] = None


class SohkepayinPromptBuilder:
    """
    Creates image generation prompts for the Sohkepayin project.
    
    Generates prompts that create dream-like, immersive panoramic scenes
    based on audience stories and environmental analysis.
    """
    
    def __init__(self):
        """Initialize prompt builder with Sohkepayin-specific templates."""
        self.base_style = [
            "panoramic view",
            "wide angle",
            "immersive landscape", 
            "dreamlike atmosphere",
            "cinematic lighting",
            "high detail",
            "photorealistic",
            "8k resolution"
        ]
        
        self.biome_templates = {
            Biome.FOREST: {
                "base": "ancient forest with towering trees",
                "elements": ["moss-covered rocks", "dappled sunlight", "forest floor", "canopy"],
                "atmosphere": "mystical and serene"
            },
            Biome.GRASSLAND: {
                "base": "vast grassland with rolling hills",
                "elements": ["wildflowers", "tall grass", "distant horizon", "scattered trees"],
                "atmosphere": "open and free"
            },
            Biome.WETLAND: {
                "base": "tranquil wetland with still waters",
                "elements": ["reeds", "lily pads", "mist", "reflections"],
                "atmosphere": "peaceful and mysterious"
            },
            Biome.MOUNTAIN: {
                "base": "majestic mountain landscape",
                "elements": ["snow-capped peaks", "rocky outcrops", "alpine meadows", "valleys"],
                "atmosphere": "grand and inspiring"
            },
            Biome.RIVER: {
                "base": "flowing river through landscape",
                "elements": ["riverbank", "flowing water", "stones", "vegetation"],
                "atmosphere": "dynamic and life-giving"
            },
            Biome.LAKE: {
                "base": "serene lake surrounded by nature",
                "elements": ["calm water", "shoreline", "reflections", "distant shores"],
                "atmosphere": "tranquil and reflective"
            },
            Biome.URBAN: {
                "base": "urban environment with architectural elements",
                "elements": ["buildings", "streets", "lights", "geometric forms"],
                "atmosphere": "dynamic and human-made"
            },
            Biome.INDOORS: {
                "base": "interior space with architectural character",
                "elements": ["walls", "windows", "light and shadow", "textures"],
                "atmosphere": "intimate and enclosed"
            },
            Biome.DESERT: {
                "base": "expansive desert landscape",
                "elements": ["sand dunes", "rock formations", "sparse vegetation", "endless sky"],
                "atmosphere": "vast and contemplative"
            },
            Biome.COAST: {
                "base": "coastal landscape where land meets sea",
                "elements": ["waves", "shoreline", "cliffs or beach", "sea spray"],
                "atmosphere": "powerful and eternal"
            }
        }
        
        self.emotion_modifiers = {
            Emotion.JOY: {
                "lighting": "warm golden hour light, bright and uplifting",
                "colors": "vibrant colors, warm tones, golden highlights",
                "atmosphere": "celebration of life, energy, movement",
                "weather": "clear skies, gentle breeze"
            },
            Emotion.SORROW: {
                "lighting": "soft overcast light, muted and gentle",
                "colors": "subdued colors, cool tones, gentle blues and grays",
                "atmosphere": "contemplative, quiet, respectful silence",
                "weather": "gentle rain, mist, or overcast skies"
            },
            Emotion.ANGER: {
                "lighting": "dramatic contrast, strong shadows",
                "colors": "intense colors, deep reds and oranges, stark contrasts",
                "atmosphere": "turbulent, powerful, intense energy",
                "weather": "storm clouds, strong winds, dramatic weather"
            },
            Emotion.PEACE: {
                "lighting": "soft even light, gentle illumination",
                "colors": "harmonious colors, natural tones, gentle pastels",
                "atmosphere": "calm, balanced, harmonious, restful",
                "weather": "still air, gentle conditions"
            },
            Emotion.LONGING: {
                "lighting": "distant light, soft focus, ethereal",
                "colors": "nostalgic tones, soft purples and blues, faded warmth",
                "atmosphere": "distant beauty, yearning, memory-like quality",
                "weather": "hazy, misty, dreamlike conditions"
            },
            Emotion.HOPE: {
                "lighting": "emerging light, dawn-like, growing brightness",
                "colors": "fresh colors, emerging greens, soft yellows and pinks",
                "atmosphere": "renewal, growth, gentle optimism",
                "weather": "clearing skies, gentle sunshine breaking through"
            }
        }
        
        logger.info("Sohkepayin prompt builder initialized")
    
    def build_prompt(
        self,
        location: LocationInference,
        width: int,
        height: int,
        is_base_image: bool = True,
        reference_description: Optional[str] = None
    ) -> ImagePrompt:
        """
        Build an image generation prompt from location inference.
        
        Args:
            location: LocationInference from story analysis
            width: Target image width
            height: Target image height  
            is_base_image: Whether this is the base panorama or a detail tile
            reference_description: Optional reference for tile consistency
            
        Returns:
            ImagePrompt with full prompt and metadata
        """
        logger.info(
            f"Building prompt for {location.biome}/{location.emotion} "
            f"({width}x{height}, base={is_base_image})"
        )
        
        # Get biome and emotion templates
        biome_info = self.biome_templates.get(location.biome, self.biome_templates[Biome.FOREST])
        emotion_info = self.emotion_modifiers.get(location.emotion, self.emotion_modifiers[Emotion.PEACE])
        
        # Build main prompt components
        components = []
        
        # Base scene description
        if location.description:
            components.append(location.description)
        else:
            components.append(f"{biome_info['base']} with {biome_info['atmosphere']} mood")
        
        # Environmental elements
        components.append(f"featuring {', '.join(biome_info['elements'][:3])}")
        
        # Emotional atmosphere
        components.append(f"with {emotion_info['atmosphere']}")
        
        # Lighting and weather
        if location.lighting:
            components.append(f"illuminated by {location.lighting}")
        else:
            components.append(f"with {emotion_info['lighting']}")
        
        if location.weather:
            components.append(f"under {location.weather} conditions")
        else:
            components.append(f"in {emotion_info['weather']}")
        
        # Time period if specified
        if location.time_period:
            components.append(f"in {location.time_period} setting")
        
        # Color guidance
        components.append(f"rendered in {emotion_info['colors']}")
        
        # Reference consistency for tiles
        if reference_description and not is_base_image:
            components.append(f"consistent with: {reference_description}")
        
        # Join components into main prompt
        main_prompt = ", ".join(components)
        
        # Add style keywords
        style_keywords = self.base_style.copy()
        
        if is_base_image:
            style_keywords.extend([
                "wide panoramic composition",
                "seamless horizontal flow",
                "detailed landscape"
            ])
        else:
            style_keywords.extend([
                "detailed section view", 
                "consistent with panorama",
                "high resolution detail"
            ])
        
        # Build final prompt
        full_prompt = f"{main_prompt}, {', '.join(style_keywords)}"
        
        # Build negative prompt
        negative_elements = [
            "people", "humans", "faces", "portraits",
            "text", "watermarks", "signatures",
            "low quality", "blurry", "distorted",
            "split screen", "borders", "frames",
            "unrealistic", "cartoonish", "anime"
        ]
        
        # Add emotion-specific negative prompts
        if location.emotion == Emotion.PEACE:
            negative_elements.extend(["violence", "chaos", "destruction"])
        elif location.emotion == Emotion.JOY:
            negative_elements.extend(["darkness", "gloom", "sadness"])
        elif location.emotion == Emotion.SORROW:
            negative_elements.extend(["bright colors", "celebration", "party"])
        
        negative_prompt = ", ".join(negative_elements)
        
        # Calculate aspect ratio
        aspect_ratio = f"{width}:{height}"
        if abs(width / height - 16/9) < 0.1:
            aspect_ratio = "16:9"
        elif abs(width / height - 21/9) < 0.1:
            aspect_ratio = "21:9"
        
        result = ImagePrompt(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            style_keywords=style_keywords,
            biome=location.biome,
            emotion=location.emotion,
            aspect_ratio=aspect_ratio
        )
        
        logger.debug(f"Generated prompt: {full_prompt[:100]}...")
        return result
    
    def build_tile_prompt(
        self,
        base_prompt: ImagePrompt,
        tile_index: int,
        total_tiles: int,
        tile_width: int,
        tile_height: int
    ) -> ImagePrompt:
        """
        Build a prompt for a specific tile based on the base panorama.
        
        Args:
            base_prompt: Original panorama prompt
            tile_index: Index of this tile (0-based)
            total_tiles: Total number of tiles
            tile_width: Tile width in pixels
            tile_height: Tile height in pixels
            
        Returns:
            ImagePrompt optimized for tile generation
        """
        logger.info(f"Building tile prompt {tile_index+1}/{total_tiles}")
        
        # Modify prompt for tile-specific generation
        tile_specific = []
        
        if tile_index == 0:
            tile_specific.append("left section of panoramic view")
        elif tile_index == total_tiles - 1:
            tile_specific.append("right section of panoramic view")
        else:
            tile_specific.append("central section of panoramic view")
        
        # Add detail focus
        tile_specific.append("highly detailed")
        tile_specific.append("seamless edges for tiling")
        
        # Combine with base prompt but emphasize detail over wide composition
        modified_prompt = base_prompt.prompt.replace(
            "wide panoramic composition", 
            "detailed panoramic section"
        ).replace(
            "seamless horizontal flow",
            "seamless tiling edges"
        )
        
        # Add tile-specific elements
        tile_prompt = f"{modified_prompt}, {', '.join(tile_specific)}"
        
        return ImagePrompt(
            prompt=tile_prompt,
            negative_prompt=base_prompt.negative_prompt,
            style_keywords=base_prompt.style_keywords + ["detailed section", "tileable"],
            biome=base_prompt.biome,
            emotion=base_prompt.emotion,
            aspect_ratio=f"{tile_width}:{tile_height}"
        )
    
    def get_style_presets(self) -> Dict[str, List[str]]:
        """Get available style presets for different moods."""
        return {
            "dreamy": [
                "soft focus", "ethereal", "misty", "atmospheric perspective",
                "gentle light rays", "subtle color gradients"
            ],
            "cinematic": [
                "dramatic lighting", "depth of field", "film grain",
                "color grading", "professional photography"
            ],
            "painterly": [
                "brush strokes", "artistic interpretation", "impressionistic",
                "canvas texture", "painted quality"
            ],
            "hyperreal": [
                "ultra detailed", "sharp focus", "perfect lighting",
                "crystal clear", "maximum detail"
            ]
        }
    
    def validate_prompt(self, prompt: ImagePrompt) -> List[str]:
        """
        Validate a prompt for potential issues.
        
        Args:
            prompt: ImagePrompt to validate
            
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        if len(prompt.prompt) < 50:
            issues.append("Prompt may be too short for good results")
        
        if len(prompt.prompt) > 1000:
            issues.append("Prompt may be too long, consider shortening")
        
        # Check for conflicting terms
        prompt_lower = prompt.prompt.lower()
        if "bright" in prompt_lower and "dark" in prompt_lower:
            issues.append("Conflicting lighting terms detected")
        
        if "joy" in prompt_lower and "sorrow" in prompt_lower:
            issues.append("Conflicting emotional terms detected")
        
        return issues
