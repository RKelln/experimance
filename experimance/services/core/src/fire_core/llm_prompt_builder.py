#!/usr/bin/env python3
"""
LLM-based prompt builder for Fire image generation.

This module uses an LLM to create contextual, nuanced image generation prompts
based purely on story analysis, relying on the LLM's understanding rather than
predefined biome/emotion classifications.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import ImagePrompt
from .llm import LLMProvider

logger = logging.getLogger(__name__)


class InsufficientContentException(Exception):
    """Raised when there's insufficient content to generate an image prompt."""
    pass


class LLMPromptBuilder:
    """
    Creates image generation prompts using LLM analysis of stories.
    
    Uses an LLM with a specialized system prompt to generate contextual,
    nuanced image generation prompts that capture the environmental and
    emotional elements of audience stories through natural language understanding.
    """
    
    def __init__(self, llm: LLMProvider, system_prompt_or_file: Optional[str|Path] = None):
        """Initialize LLM prompt builder."""
        self.llm = llm
        
        # System prompt can be loaded from file or set directly
        if system_prompt_or_file:
            if isinstance(system_prompt_or_file, Path):
                self.llm.set_system_prompt_file(system_prompt_or_file)
            else: 
                # check if string is a file
                if Path(system_prompt_or_file).exists():
                    self.llm.set_system_prompt_file(system_prompt_or_file)
                else:
                    self.llm.set_system_prompt(system_prompt_or_file)

        self.panorama_style = [
            "360 degree equirectangular view", 
        ]

        self.tile_style = [
            "ultra detailed",
            "sharp focus"
        ]

        self.quality_style = [
            "dreamlike",
            "cinematic lighting",
            "high detail",
            "8k resolution"
        ]

        self.negative_style = [
            "people", "humans", "faces", "low quality", "blurry", "watermark", "text",
            "deformed", "disfigured", "poor quality", "lowres", "bad anatomy",
            "worst quality", "jpeg artifacts", "signature", "malformed", "clone", "duplicate"
        ]
        
        logger.info("LLM prompt builder initialized")
    
    
    async def build_prompt(
        self,
        story_content: str,
        prefix: Optional[List[str]] = None,
        suffix: Optional[List[str]] = None,
        negative: Optional[List[str]] = None,
    ) -> ImagePrompt:
        """
        Build an image generation prompt using LLM analysis.
        
        Args:
            story_content: Original story text from audience
            prefix: Optional list of keywords to prepend to the prompt
            suffix: Optional list of keywords to append to the prompt
            negative: Optional list of keywords to include in negative prompt
            
        Returns:
            ImagePrompt with LLM-generated prompt and metadata
        """
        # fallback prompt
        data = {
            "prompt": "A cinematic landscape scene",
        }
        try:
            # Query the LLM
            logger.info(f"Querying LLM for prompt generation: {story_content[:100]}...")
            result = await self.llm.query(
                content=story_content,
            )
            if result is None:
                raise ValueError("LLM returned no result")
            logger.debug(f"LLM query result: {result}")
            
            if result == "<invalid>" or "invalid" in result.lower():
                logger.warning(f"LLM returned invalid response ({result}), using fallback prompt")
            else: # try to parse json
                # Attempt to parse the JSON response
                try:
                    data = json.loads(result)
                except json.JSONDecodeError:
                    # Debug: Log the raw response
                    logger.debug(f"Raw LLM response: {repr(result)}")
                    
                    # Clean up the response - remove markdown code blocks if present
                    json_content = result.strip()
                    if json_content.startswith("```json"):
                        # Remove opening ```json
                        json_content = json_content[7:]
                    elif json_content.startswith("```"):
                        # Remove opening ``` (fallback)
                        json_content = json_content[3:]
                    if json_content.endswith("```"):
                        # Remove closing ```
                        json_content = json_content[:-3]
                    
                    # Also handle single backticks
                    if json_content.startswith("`") and json_content.endswith("`"):
                        json_content = json_content[1:-1]
                    
                    json_content = json_content.strip()

                    data = json.loads(json_content)
                
                # Check the status from the new system prompt format
                status = data.get("status", "ready")  # Default to ready for backward compatibility
                logger.debug(f"LLM response status: {status}, data: {data}")
                
                if status == "insufficient":
                    reason = data.get("reason", "LLM determined insufficient content")
                    logger.info(f"LLM determined insufficient content: {reason}")
                    raise InsufficientContentException(reason)
                    
                elif status == "invalid":
                    reason = data.get("reason", "LLM determined invalid content")
                    logger.warning(f"LLM determined invalid content: {reason}")
                    # Use fallback for invalid content
                    data = {"prompt": "A cinematic landscape scene"}
                    
                elif status == "ready":
                    # Content is ready for generation
                    logger.info("LLM determined content is ready for image generation")

        except InsufficientContentException:
            # Re-raise this exception to be caught by the caller
            raise
        except Exception as e:
            logger.error(f"LLM prompt generation failed: {e}")
            data = {
                "prompt": "A cinematic landscape scene",
            }
        
        # Only return if we didn't raise an exception
        return self._elements_to_prompt(
            prompt_elements=data.get("prompt", "").strip().split(","),
            negative_elements=data.get("negative_prompt", "").strip().split(","),
            prefix=prefix,
            suffix=suffix,
            negative=negative
        )

    async def build_panorama_prompt(
        self,
        story_content: str,
    ) -> ImagePrompt:
        """
        Build a prompt for a panorama using LLM.
        
        Args:
            story_content: Original story text

        Returns:
            ImagePrompt optimized for panorama generation
        """

        return await self.build_prompt(
            story_content=story_content,
            prefix=self.panorama_style,
            suffix=self.quality_style
        )
    
    def _elements_to_prompt(
        self,
        prompt_elements: List[str],
        negative_elements: List[str],
        prefix: Optional[List[str]] = None,
        suffix: Optional[List[str]] = None,
        negative: Optional[List[str]] = None,
    ) -> ImagePrompt:
        """
        Modify prompt elements by adding prefix, suffix, and negative keywords.
        
        Args:
            prompt_elements: List of prompt elements to modify
            prefix: Optional list of keywords to prepend
            suffix: Optional list of keywords to append
            negative: Optional list of keywords for negative prompt
            
        Returns:
            Modified ImagePrompt with updated prompt and negative prompt
        """
        prompt_elements = [elem.strip() for elem in prompt_elements if elem.strip()]
        if prefix:
            prompt_elements = [elem for elem in prompt_elements if elem not in prefix]
            prompt_elements = prefix + prompt_elements
        
        if suffix:
            prompt_elements = [elem for elem in prompt_elements if elem not in suffix]
            prompt_elements += suffix
        
        negative_elements = [elem.strip() for elem in negative_elements if elem.strip()]
        if negative:
            negative_elements = [elem for elem in negative_elements if elem not in negative]
            negative_elements += negative
        else:
            negative_elements += self.negative_style
        
        return ImagePrompt(
            prompt=", ".join(prompt_elements),
            negative_prompt=", ".join(negative_elements)
        )

    def base_prompt_to_panorama_prompt(
        self,
        base_prompt: ImagePrompt,
    ) -> ImagePrompt:
        """
        Convert a base prompt to a panorama prompt.
        
        Args:
            base_prompt: Base ImagePrompt object
            
        Returns:
            ImagePrompt optimized for panorama generation
        """
        base_elements = base_prompt.prompt.split(",")
        negative_elements = base_prompt.negative_prompt.split(",") if base_prompt.negative_prompt else []
        return self._elements_to_prompt(
            prompt_elements=base_elements,
            negative_elements=negative_elements,
            prefix=self.panorama_style,
            suffix=self.quality_style
        )
    
    def base_prompt_to_tile_prompt(
        self,
        base_prompt: ImagePrompt,
    ) -> ImagePrompt:   
        """
        Convert a base prompt to a tile prompt.
        
        Args:
            base_prompt: Base ImagePrompt object
            
        Returns:
            ImagePrompt optimized for tile generation
        """
        base_elements = base_prompt.prompt.split(",")
        negative_elements = base_prompt.negative_prompt.split(",") if base_prompt.negative_prompt else []
        return self._elements_to_prompt(
            prompt_elements=base_elements,
            negative_elements=negative_elements,
            prefix=self.tile_style,
            suffix=self.quality_style
        )

    async def build_tile_prompt(
        self,
        story_content: str,
    ) -> ImagePrompt:
        """
        Build a prompt for a tile using LLM.
        
        Args:
            story_content: Original story text

        Returns:
            ImagePrompt optimized for tile generation
        """

        return await self.build_prompt(
            story_content=story_content,
            prefix=self.tile_style,
            suffix=self.quality_style
        )

    
    async def test_system_prompt(self, test_story: str) -> Dict[str, str]:
        """
        Test the current system prompt with a sample story.
        
        Args:
            test_story: Sample story to test with
            
        Returns:
            Dict with test results including generated prompt and metadata
        """
        
        try:
            result = await self.build_prompt(
                story_content=test_story,
            )
            
            return {
                "status": "success",
                "prompt": result.prompt,
                "negative_prompt": result.negative_prompt if result.negative_prompt else "",
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "fallback_used": "true"
            }


