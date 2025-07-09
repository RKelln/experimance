#!/usr/bin/env python3
"""
Experimance Test Flow Configuration

This module contains the flow configuration and function handlers for testing
Pipecat flows with the Experimance art installation conversation system.
"""

import logging
from typing import Any, Dict, Optional

from pipecat_flows import FlowConfig, FlowManager, FlowArgs

logger = logging.getLogger(__name__)


# Function handlers for the flows
async def move_to_explorer(args: FlowArgs, flow_manager: FlowManager) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Move from welcome mode to explorer mode after collecting name and location."""
    logger.info(f"[FUNCTION CALL] move_to_explorer with args: {args}")
    
    name = args.get("name", "visitor")
    location = args.get("location", "unknown")
    
    # Store data in flow state
    flow_manager.state["name"] = name
    flow_manager.state["location"] = location
    flow_manager.state["current_mode"] = "explorer"
    
    result = {
        "status": "success",
        "message": f"Welcome {name} from {location}! Switching to explorer mode."
    }
    
    return result, "explorer"


async def get_theme_info(args: FlowArgs, flow_manager: FlowManager) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Get thematic information about the art installation."""
    logger.info(f"[FUNCTION CALL] get_theme_info with args: {args}")
    
    topic = args.get("topic", "general")
    
    themes = {
        "general": (
            "Experimance explores the delicate balance between human presence and environmental change. "
            "The installation invites you to consider how our experiences shape and are shaped by "
            "the natural world around us."
        ),
        "environment": (
            "The environmental theme focuses on different biomes and how they respond to human interaction. "
            "Each biome represents different aspects of our planet's ecosystems and their fragility."
        ),
        "interaction": (
            "The interactive elements demonstrate how human presence can create ripple effects "
            "through complex systems, much like our impact on the environment."
        )
    }
    
    theme_info = themes.get(topic, themes["general"])
    
    result = {
        "status": "success",
        "topic": topic,
        "information": theme_info
    }
    
    return result, None


async def get_technical_info(args: FlowArgs, flow_manager: FlowManager) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Get technical information about how the installation works."""
    logger.info(f"[FUNCTION CALL] get_technical_info with args: {args}")
    
    aspect = args.get("aspect", "general")
    
    technical = {
        "general": (
            "Experimance uses real-time sensor data, computer vision, and AI to create responsive "
            "visual and audio experiences. The system processes audience presence and movement "
            "to drive the interactive elements."
        ),
        "sensors": (
            "The installation uses depth cameras and motion sensors to detect audience presence "
            "and movement. This data drives the real-time generation of visuals and soundscapes."
        ),
        "ai": (
            "AI systems generate unique visual content and adapt the conversation based on "
            "audience interaction. The voice agent uses advanced language models for natural conversation."
        )
    }
    
    tech_info = technical.get(aspect, technical["general"])
    
    result = {
        "status": "success",
        "aspect": aspect,
        "information": tech_info
    }
    
    return result, None


async def suggest_biome(args: FlowArgs, flow_manager: FlowManager) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Suggest a biome for the visitor to explore."""
    logger.info(f"[FUNCTION CALL] suggest_biome with args: {args}")
    
    preference = args.get("preference", "any")
    
    biomes = {
        "forest": "A lush forest biome with deep greens and the sounds of rustling leaves.",
        "ocean": "An oceanic biome with flowing blues and the rhythm of waves.",
        "desert": "A desert biome with warm earth tones and the whisper of wind through sand.",
        "arctic": "An arctic biome with cool blues and whites and the crystalline sound of ice.",
        "any": "How about exploring a forest biome? It's peaceful with deep greens and natural sounds."
    }
    
    biome_suggestion = biomes.get(preference, biomes["any"])
    
    result = {
        "status": "success",
        "preference": preference,
        "suggestion": biome_suggestion
    }
    
    return result, None


# Flow configuration for the Experimance test
flow_config: FlowConfig = {
    "initial_node": "welcome",
    "nodes": {
        "welcome": {
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a voice assistant for the Experimance art installation. "
                        "You must ALWAYS use the available functions to progress the conversation. "
                        "This is a voice conversation and your responses will be converted to audio. "
                        "Keep responses conversational, brief, and engaging. "
                        "Avoid outputting special characters and emojis."
                    )
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are now in WELCOME mode. Always announce 'I am now in WELCOME mode' "
                        "at the start of your first response. Greet visitors warmly and ask for their name. "
                        "Once you have their name, ask where they are visiting from (city/country). "
                        "After getting both name and location, say you'll switch to explorer mode "
                        "and call the move_to_explorer function."
                    )
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "move_to_explorer",
                        "handler": move_to_explorer,
                        "description": "Move from welcome mode to explorer mode after collecting name and location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "The visitor's name"},
                                "location": {"type": "string", "description": "Where the visitor is from"}
                            },
                            "required": ["name", "location"]
                        }
                    }
                }
            ]
        },
        "explorer": {
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a voice assistant for the Experimance art installation. "
                        "This is a voice conversation and your responses will be converted to audio. "
                        "Keep responses conversational, brief, and engaging."
                    )
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are now in EXPLORER mode. Always announce 'I am now in EXPLORER mode' "
                        "at the start of your first response. You help visitors explore and understand "
                        "the art installation which represents the intersection of human experience "
                        "and environmental change. When visitors ask about the installation, "
                        "you can provide information about the theme OR the technical aspects. "
                        "Use the appropriate functions to provide detailed responses."
                    )
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_theme_info",
                        "handler": get_theme_info,
                        "description": "Get thematic information about the art installation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topic": {
                                    "type": "string", 
                                    "enum": ["general", "environment", "interaction"], 
                                    "description": "The theme topic to explore"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_technical_info",
                        "handler": get_technical_info,
                        "description": "Get technical information about how the installation works",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "aspect": {
                                    "type": "string", 
                                    "enum": ["general", "sensors", "ai"], 
                                    "description": "The technical aspect to explain"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "suggest_biome",
                        "handler": suggest_biome,
                        "description": "Suggest a biome for the visitor to explore",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "preference": {
                                    "type": "string", 
                                    "enum": ["forest", "ocean", "desert", "arctic", "any"], 
                                    "description": "The type of biome preference"
                                }
                            }
                        }
                    }
                }
            ]
        }
    }
}
