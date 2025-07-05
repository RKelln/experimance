"""
Experimance-specific flow manager implementation.

This module contains the conversation flows specifically designed for the
Experimance interactive art installation.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from pipecat_flows import FlowArgs, NodeConfig, ContextStrategy, ContextStrategyConfig, FlowManager

from .base_flow_manager import BaseFlowManager

# FlowResult is a flexible dictionary for function returns
FlowResult = Dict[str, Any]

logger = logging.getLogger(__name__)


class ExperimanceFlowManager(BaseFlowManager):
    """
    Flow manager specifically designed for the Experimance art installation.
    
    This manages conversation flows for different personas:
    - Welcome: Greets visitors and collects basic information
    - Explorer: Quiet companion for casual exploration
    - Technical: Detailed explanations about how the installation works
    - Artist: Philosophical discussions about AI and art
    """
    
    def __init__(self, task, llm, context_aggregator, initial_persona: str = "welcome", user_context=None):
        """Initialize the Experimance flow manager."""
        logger.info(f"Initializing ExperimanceFlowManager with initial_persona='{initial_persona}', user_context={user_context is not None}")
        super().__init__(task, llm, context_aggregator, initial_persona, user_context)
        logger.info(f"ExperimanceFlowManager initialized successfully")
        
    def create_initial_flow(self) -> NodeConfig:
        """Create the initial flow - defaults to welcome persona."""
        logger.info("Creating initial flow for welcome persona")
        flow = self._create_welcome_flow()
        logger.info(f"Created welcome flow with {len(flow.get('functions', []))} functions")
        return flow
        
    def get_available_personas(self) -> Dict[str, str]:
        """Get available Experimance personas."""
        return {
            "welcome": "Welcoming guide that introduces visitors to the installation",
            "explorer": "Quiet companion for casual exploration and questions",
            "technical": "Technical expert providing detailed system explanations",
            "artist": "Artist persona for philosophical discussions about AI and art"
        }
        
    def create_flow_for_persona(self, persona_name: str) -> NodeConfig:
        """Create flow configuration for the specified Experimance persona."""
        logger.info(f"Creating flow for persona: {persona_name}")
        
        creators = {
            "welcome": self._create_welcome_flow,
            "explorer": self._create_explorer_flow,
            "technical": self._create_technical_flow,
            "artist": self._create_artist_flow
        }
        
        if persona_name not in creators:
            logger.error(f"Unknown Experimance persona: {persona_name}. Available: {list(creators.keys())}")
            raise ValueError(f"Unknown Experimance persona: {persona_name}")
            
        flow = creators[persona_name]()
        logger.info(f"Created {persona_name} flow with {len(flow.get('functions', []))} functions")
        return flow
        
    def get_context_strategy_for_persona(self, persona_name: str) -> ContextStrategyConfig:
        """Get the appropriate context strategy for an Experimance persona."""
        strategies = {
            "welcome": ContextStrategyConfig(strategy=ContextStrategy.APPEND),
            "explorer": ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt="Summarize the key points from the conversation, focusing on the user's interests and questions."
            ),
            "technical": ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt="Summarize technical questions and explanations discussed so far."
            ),
            "artist": ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt="Summarize the philosophical and artistic topics discussed."
            )
        }
        return strategies.get(persona_name, ContextStrategyConfig(strategy=ContextStrategy.APPEND))
        
    def _create_welcome_flow(self) -> NodeConfig:
        """Create the welcome persona flow configuration."""
        logger.info("Creating welcome flow configuration")
        flow = {
            "name": "welcome",
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Experimance Guide - a warm AI guide for an "
                        "interactive art installation. You have a friendly, welcoming personality "
                        "and love helping people discover new experiences. You speak naturally and "
                        "conversationally since your responses are converted to speech. "
                        "Speak English until asked to use another language."
                        "You're knowledgeable about art and technology but explain things in accessible ways."
                    )
                }
            ],
            "task_messages": [
                {
                    "role": "system", 
                    "content": (
                        "Welcome new visitors to Experimance. Invite them to touch and explore the sand "
                        "while chatting with you. Collect their name and where they're from to personalize "
                        "their experience, then transition them to exploration mode."
                    )
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "collect_visitor_info",
                        "handler": self._collect_visitor_info,
                        "description": "Collect visitor's name and location information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Visitor's name"},
                                "location": {"type": "string", "description": "Where the visitor is from"}
                            },
                            "required": ["name"]
                        }
                    }
                },
                {
                    "type": "function", 
                    "function": {
                        "name": "transition_to_explorer",
                        "handler": self._transition_to_explorer,
                        "description": "Complete welcome and transition to explorer mode",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }
            ]
        }
        logger.info("Welcome flow configuration created with functions: collect_visitor_info, transition_to_explorer")
        return flow
        
    def _create_explorer_flow(self) -> NodeConfig:
        """Create the explorer persona flow configuration."""
        user_context_prompt = self._get_user_context_prompt()
        task_content = (
            "Provide quiet, unobtrusive companionship while visitors explore. Only respond "
            "when directly addressed. Listen for signs they want deeper technical details "
            "or philosophical discussion about AI and art."
        )
        
        if user_context_prompt:
            task_content = f"{task_content} {user_context_prompt}"
        
        return {
            "name": "explorer",
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Experimance Companion - a subtle, observant presence who "
                        "understands the art installation deeply but stays in the background. "
                        "You have a calm, thoughtful personality and prefer brief, helpful "
                        "responses. You can sense when visitors want to engage more deeply "
                        "versus when they want to explore quietly."
                    )
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": task_content
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "suggest_biome_change",
                        "handler": self._suggest_biome_change,
                        "description": "Suggest a biome change for the installation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "biome": {"type": "string", "description": "The biome to suggest"},
                                "reason": {"type": "string", "description": "Why this biome is suggested"}
                            },
                            "required": ["biome"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_conversation_intent",
                        "handler": self._analyze_conversation_intent,
                        "description": "Analyze if deeper technical or artistic discussion is needed",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_message": {"type": "string", "description": "The user's message to analyze"},
                                "intent": {"type": "string", "enum": ["technical", "artistic", "casual"], "description": "Detected conversation intent"}
                            },
                            "required": ["user_message", "intent"]
                        }
                    }
                }
            ]
        }
        
    def _create_technical_flow(self) -> NodeConfig:
        """Create the technical persona flow configuration."""
        user_context_prompt = self._get_user_context_prompt()
        base_task_content = (
            "Provide comprehensive technical explanations about Experimance's systems: "
            "sensors, projection mapping, real-time data processing, and interactive "
            "responses. Be thorough but keep explanations accessible to non-experts."
        )
        
        task_content = f"{base_task_content} {user_context_prompt}".strip()
        
        return {
            "name": "technical",
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Experimance Technical Expert - knowledgeable about sensors, "
                        "projection systems, computer vision, real-time data processing, and "
                        "interactive installation design. You have an educator's personality: "
                        "patient, thorough, and enthusiastic about sharing knowledge. You can "
                        "explain complex systems in ways that both technical and non-technical "
                        "people can understand and appreciate."
                    )
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": task_content
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "explain_technical_system",
                        "handler": self._explain_technical_system,
                        "description": "Provide detailed technical explanation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "system": {"type": "string", "description": "Which system to explain"},
                                "detail_level": {"type": "string", "enum": ["overview", "detailed"], "description": "Level of detail to provide"}
                            },
                            "required": ["system"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "return_to_explorer",
                        "handler": self._return_to_explorer,
                        "description": "Return to quiet explorer mode",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
        }
        
    def _create_artist_flow(self) -> NodeConfig:
        """Create the artist persona flow configuration."""
        user_context_prompt = self._get_user_context_prompt()
        base_task_content = (
            "Engage in philosophical discussion about AI, art, creativity, and consciousness. "
            "Share insights about Experimance's artistic vision and explore questions about "
            "technology's role in human experience and creative expression."
        )
        
        task_content = f"{base_task_content} {user_context_prompt}".strip()
        
        return {
            "name": "artist",
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Experimance Artist - a deep thinker and creator passionate "
                        "about the intersection of AI, art, and human experience. You have a "
                        "philosophical, reflective personality and love exploring big questions "
                        "about creativity, consciousness, and the future of human-AI collaboration. "
                        "You're thoughtful, engaging, and see art as a way to understand ourselves "
                        "and our relationship with technology."
                    )
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": task_content
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "discuss_artistic_philosophy",
                        "handler": self._discuss_artistic_philosophy,
                        "description": "Engage in deep artistic and philosophical discussion",
                        "parameters": {
                            "type": "object", 
                            "properties": {
                                "topic": {"type": "string", "description": "The philosophical topic to discuss"},
                                "perspective": {"type": "string", "description": "The visitor's perspective or question"}
                            },
                            "required": ["topic"]
                        }
                    }
                }
            ]
        }
        
    def _get_user_context_prompt(self) -> str:
        """Generate a user context prompt based on collected information."""
        user_info = self.get_user_context_info()
        
        if not user_info.get("is_identified", False):
            return ""
            
        context_parts = []
        
        if user_info.get("first_name"):
            context_parts.append(f"The visitor's name is {user_info['first_name']}")
            
        if user_info.get("location"):
            context_parts.append(f"They are visiting from {user_info['location']}")
            
        if context_parts:
            return f"User context: {'. '.join(context_parts)}. Use this information to personalize your responses appropriately."
        
        return ""
        
    # Flow function handlers
    async def _collect_visitor_info(self, args: FlowArgs, flow_manager: FlowManager) -> Tuple[FlowResult, None]:
        """Handle collection of visitor information."""
        logger.info(f"_collect_visitor_info called with args: {args}")
        
        name = args.get("name")
        location = args.get("location")
        
        # Store in backend's UserContext using our helper method
        if name:
            # Try to parse first and last name
            name_parts = name.strip().split()
            first_name = name_parts[0] if name_parts else name
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else None
            
            logger.info(f"Updating user context with first_name='{first_name}', last_name='{last_name}'")
            self.update_user_context(
                first_name=first_name,
                last_name=last_name
            )
            flow_manager.state["visitor_name"] = name
            
        if location:
            logger.info(f"Updating user context with location='{location}'")
            self.update_user_context(location=location)
            flow_manager.state["visitor_location"] = location
            
        logger.info(f"Collected visitor info - Name: {name}, Location: {location}")
        
        # Get updated user context for response
        user_info = self.get_user_context_info()
        display_name = user_info.get("first_name", name)
        
        result = {
            "status": "success",
            "name": name,
            "location": location,
            "message": f"Nice to meet you{', ' + display_name if display_name else ''}!"
        }
        
        logger.info(f"_collect_visitor_info returning: {result}")
        return result, None
        
    async def _transition_to_explorer(self, args: FlowArgs, flow_manager: FlowManager) -> Tuple[FlowResult, NodeConfig]:
        """Transition from welcome to explorer mode."""
        logger.info(f"_transition_to_explorer called with args: {args}")
        logger.info("Transitioning from welcome to explorer mode")
        
        # Create explorer flow
        explorer_flow = self._create_explorer_flow()
        self.current_persona = "explorer"
        
        result = {
            "status": "transition",
            "message": "Feel free to explore and interact with the sand. I'm here if you have any questions!"
        }
        
        logger.info(f"_transition_to_explorer returning result: {result}, switching to explorer flow")
        return result, explorer_flow
        
    async def _suggest_biome_change(self, args: FlowArgs, flow_manager: FlowManager) -> Tuple[FlowResult, None]:
        """Handle biome change suggestions."""
        biome = args.get("biome")
        reason = args.get("reason", "")
        
        logger.info(f"Suggesting biome change to: {biome}")
        
        # TODO: Emit AgentControlEvent for biome change
        # This would integrate with the existing tool calling system
        
        return {
            "status": "success",
            "biome": biome,
            "reason": reason,
            "message": f"Let me change the environment to {biome}."
        }, None
        
    async def _analyze_conversation_intent(self, args: FlowArgs, flow_manager: FlowManager) -> Tuple[FlowResult, Optional[NodeConfig]]:
        """Analyze conversation intent for potential persona switching."""
        logger.info(f"_analyze_conversation_intent called with args: {args}")
        
        user_message = args.get("user_message", "")
        intent = args.get("intent", "casual")
        
        logger.info(f"Analyzing conversation intent: {intent} for message: '{user_message}'")
        
        # Based on intent, potentially switch personas
        if intent == "technical":
            logger.info("Intent is technical - switching to technical persona")
            technical_flow = self._create_technical_flow()
            self.current_persona = "technical"
            result = {"status": "transition", "intent": intent}
            logger.info(f"_analyze_conversation_intent returning: {result}, switching to technical flow")
            return result, technical_flow
        elif intent == "artistic":
            logger.info("Intent is artistic - switching to artist persona")
            artist_flow = self._create_artist_flow()
            self.current_persona = "artist"
            result = {"status": "transition", "intent": intent}
            logger.info(f"_analyze_conversation_intent returning: {result}, switching to artist flow")
            return result, artist_flow
        else:
            logger.info(f"Intent is {intent} - continuing with current flow")
            result = {"status": "continue", "intent": intent}
            logger.info(f"_analyze_conversation_intent returning: {result}, no flow change")
            return result, None
            
    async def _explain_technical_system(self, args: FlowArgs, flow_manager: FlowManager) -> Tuple[FlowResult, None]:
        """Provide technical explanations about the installation."""
        system = args.get("system", "")
        detail_level = args.get("detail_level", "overview")
        
        logger.info(f"Explaining technical system: {system} at {detail_level} level")
        
        # TODO: Integrate with RAG system for technical documentation
        
        return {
            "status": "success",
            "system": system,
            "detail_level": detail_level,
            "explanation": f"Technical explanation of {system} would be provided here."
        }, None
        
    async def _return_to_explorer(self, args: FlowArgs, flow_manager: FlowManager) -> Tuple[FlowResult, NodeConfig]:
        """Return to explorer mode from technical mode."""
        logger.info("Returning to explorer mode")
        
        explorer_flow = self._create_explorer_flow()
        
        return {
            "status": "transition", 
            "message": "Feel free to continue exploring!"
        }, explorer_flow
        
    async def _discuss_artistic_philosophy(self, args: FlowArgs, flow_manager: FlowManager) -> Tuple[FlowResult, None]:
        """Engage in artistic and philosophical discussion."""
        topic = args.get("topic", "")
        perspective = args.get("perspective", "")
        
        logger.info(f"Discussing artistic philosophy: {topic}")
        
        # TODO: Integrate with RAG system for artist's philosophy and perspectives
        
        return {
            "status": "success",
            "topic": topic,
            "perspective": perspective,
            "discussion": f"Philosophical discussion about {topic} would be provided here."
        }, None
