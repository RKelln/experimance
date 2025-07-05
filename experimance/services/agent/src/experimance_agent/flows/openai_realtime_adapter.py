"""
OpenAI Realtime Beta adapter for Pipecat Flows.

This module provides a custom adapter that enables OpenAI Realtime Beta LLM Service
to work with pipecat-flows, allowing for flow-based conversational AI with real-time
speech-to-speech capabilities.

Based on the AWS Nova Sonic adapter pattern from pipecat-flows.
"""

import logging
from typing import Any, Dict, List, Optional

# Import required dependencies directly
from pipecat_flows.adapters import LLMAdapter
from pipecat_flows.types import FlowsFunctionSchema
from pipecat.services.openai_realtime_beta import OpenAIRealtimeBetaLLMService

logger = logging.getLogger(__name__)


class OpenAIRealtimeAdapter(LLMAdapter):
    """Format adapter for OpenAI Realtime Beta.
    
    Handles OpenAI Realtime Beta's real-time conversational format, converting between
    OpenAI's standard format and the realtime format as needed for flows.
    
    Note: OpenAI Realtime Beta is primarily designed for real-time speech-to-speech
    conversations. This adapter provides compatibility with pipecat-flows for
    text-based flow management, but some features may be limited compared to
    text-only LLM services.
    """
    
    def __init__(self):
        """Initialize the OpenAI Realtime adapter."""
        super().__init__()
        # Import here to avoid circular imports and ensure it's available
        try:
            from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
            self.provider_adapter = OpenAILLMAdapter()
        except ImportError:
            # Fallback - realtime service doesn't use standard provider adapter patterns
            # but we need to be compatible with the flow system
            self.provider_adapter = None
            logger.warning("Standard OpenAI adapter not available, using realtime-only implementation")
    
    def _get_function_name_from_dict(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from OpenAI Realtime function definition.
        
        Args:
            function_def: OpenAI Realtime-formatted function definition dictionary
            
        Returns:
            Function name from the definition
        """
        # OpenAI Realtime uses the same function calling format as standard OpenAI
        # Standard format: {"type": "function", "function": {"name": "...", ...}}
        if "function" in function_def:
            return function_def["function"]["name"]
        # Direct format fallback: {"name": "...", ...}
        return function_def.get("name", "")
    
    def format_summary_message(self, summary: str) -> dict:
        """Format summary as a system message for OpenAI Realtime.
        
        Uses the same format as standard OpenAI to maintain consistency.
        """
        return {"role": "system", "content": f"Here's a summary of the conversation:\n{summary}"}
    
    async def generate_summary(
        self, llm: Any, summary_prompt: str, messages: List[dict]
    ) -> Optional[str]:
        """Generate summary using OpenAI Realtime API.
        
        Note: OpenAI Realtime Beta is primarily designed for real-time conversations.
        For summary generation, we'll try to use the underlying OpenAI client directly
        following the same pattern as the standard OpenAI adapter.
        """
        try:
            # Try to access the underlying OpenAI client if available
            if hasattr(llm, '_client') and hasattr(llm._client, 'chat'):
                # Use the same format as the standard OpenAI adapter
                prompt_messages = [
                    {
                        "role": "system",
                        "content": summary_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Conversation history: {messages}",
                    },
                ]
                
                # Use the LLM's model name if available, otherwise fallback
                model_name = getattr(llm, 'model_name', 'gpt-4o-mini')
                
                response = await llm._client.chat.completions.create(
                    model=model_name,
                    messages=prompt_messages,
                    stream=False,
                )
                
                return response.choices[0].message.content
            
            # Fallback: simple concatenation of recent messages
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            summary_parts = []
            for msg in recent_messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    summary_parts.append(f"{role}: {content}")
            
            return "Recent conversation: " + " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"OpenAI Realtime summary generation failed: {e}", exc_info=True)
            return None
    
    def convert_to_function_schema(self, function_def: Dict[str, Any]) -> FlowsFunctionSchema:
        """Convert OpenAI Realtime function definition to FlowsFunctionSchema.
        
        OpenAI Realtime uses the same function calling format as standard OpenAI,
        so this follows the same pattern as the standard OpenAI adapter.
        
        Args:
            function_def: OpenAI Realtime function definition
            
        Returns:
            FlowsFunctionSchema equivalent with flow-specific fields
        """
        # Check for standard OpenAI function format: {"type": "function", "function": {...}}
        if "function" in function_def:
            func_data = function_def["function"]
            name = func_data["name"]
            description = func_data.get("description", "")
            parameters = func_data.get("parameters", {}) or {}
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])
            
            # Extract Flows-specific fields from the function data
            handler = func_data.get("handler")
            transition_to = func_data.get("transition_to")
            transition_callback = func_data.get("transition_callback")
        else:
            # Handle direct format fallback: {"name": "...", "description": "...", ...}
            name = function_def.get("name", "")
            description = function_def.get("description", "")
            
            # Handle parameters format
            if "parameters" in function_def:
                parameters = function_def["parameters"]
                properties = parameters.get("properties", {})
                required = parameters.get("required", [])
            else:
                properties = {}
                required = []
            
            # Extract Flows-specific fields
            handler = function_def.get("handler")
            transition_to = function_def.get("transition_to")
            transition_callback = function_def.get("transition_callback")
        
        return FlowsFunctionSchema(
            name=name,
            description=description,
            properties=properties,
            required=required,
            handler=handler,
            transition_to=transition_to,
            transition_callback=transition_callback,
        )


def is_openai_realtime_adapter_available() -> bool:
    """Check if the OpenAI Realtime adapter is available.
    
    Since we now import dependencies directly, this always returns True
    if the module imports successfully.
    """
    return True


def create_openai_realtime_adapter() -> OpenAIRealtimeAdapter:
    """Create an OpenAI Realtime adapter.
    
    Returns:
        OpenAIRealtimeAdapter instance
        
    Raises:
        Exception: If adapter creation fails
    """
    return OpenAIRealtimeAdapter()
