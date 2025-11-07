"""
Simple flow manager for testing and basic conversations.

This provides a minimal flow setup that can be used for testing the
PipecatBackend without the complexity of multiple personas.
"""

import logging
from typing import Dict, Any, Tuple

from pipecat_flows import FlowArgs, NodeConfig, ContextStrategy, ContextStrategyConfig, FlowManager

from .base_flow_manager import BaseFlowManager

# FlowResult is a flexible dictionary for function returns
FlowResult = Dict[str, Any]

logger = logging.getLogger(__name__)


class SimpleFlowManager(BaseFlowManager):
    """
    A simple flow manager with just basic conversation capability.
    
    This is useful for:
    - Testing the PipecatBackend integration
    - Simple conversational applications
    - Prototyping new flow ideas
    """
    
    def __init__(self, task, llm, context_aggregator, initial_persona: str = "assistant", user_context=None):
        """Initialize the simple flow manager."""
        super().__init__(task, llm, context_aggregator, initial_persona, user_context)
        
    def create_initial_flow(self) -> NodeConfig:
        """Create a simple conversational flow."""
        return self._create_assistant_flow()
        
    def get_available_personas(self) -> Dict[str, str]:
        """Get available simple personas."""
        return {
            "assistant": "A helpful AI assistant for general conversation"
        }
        
    def create_flow_for_persona(self, persona_name: str) -> NodeConfig:
        """Create flow for the specified persona."""
        if persona_name == "assistant":
            return self._create_assistant_flow()
        else:
            raise ValueError(f"Unknown simple persona: {persona_name}")
            
    def get_context_strategy_for_persona(self, persona_name: str) -> ContextStrategyConfig:
        """Get context strategy - simple flows just use APPEND."""
        return ContextStrategyConfig(strategy=ContextStrategy.APPEND)
        
    def _create_assistant_flow(self) -> NodeConfig:
        """Create a basic assistant flow configuration."""
        return {
            "name": "assistant",
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. Respond naturally to user questions "
                        "and provide helpful information. Keep responses conversational and "
                        "appropriate for speech output."
                    )
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": (
                        "Have a natural conversation with the user. Answer questions, "
                        "provide information, and be helpful and friendly."
                    )
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "end_conversation",
                        "handler": self._end_conversation,
                        "description": "End the conversation gracefully",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string", "description": "Reason for ending conversation"}
                            }
                        }
                    }
                }
            ]
        }
        
    async def _end_conversation(self, args: FlowArgs, flow_manager: FlowManager) -> Tuple[FlowResult, None]:
        """Handle conversation ending."""
        reason = args.get("reason", "User requested to end conversation")
        
        logger.info(f"Ending conversation: {reason}")
        
        return {
            "status": "ending",
            "reason": reason,
            "message": "Thank you for the conversation! Have a great day!"
        }, None
