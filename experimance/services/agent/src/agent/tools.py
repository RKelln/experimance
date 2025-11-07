"""
Generic tools for agent backends.

This module contains utility tools that can be used across different projects.
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from experimance_common.transcript_manager import TranscriptManager, TranscriptMessageType

logger = logging.getLogger(__name__)


def extract_transcript_content(transcript_manager: TranscriptManager, message_types: Optional[List[TranscriptMessageType]] = None) -> str:
    """
    Extract and clean transcript content for sending to other services.
    
    Args:
        transcript_manager: The transcript manager instance
        message_types: List of message types to include (default: user and assistant only)
        
    Returns:
        Cleaned transcript content as a string
    """
    if message_types is None:
        message_types = [TranscriptMessageType.USER_SPEECH, TranscriptMessageType.AGENT_RESPONSE]
    
    messages = transcript_manager.get_messages(message_types=message_types)
    
    if not messages:
        return "No conversation content available"
    
    # Extract just the text content, clean it up
    content_parts = []
    for msg in messages:
        if hasattr(msg, 'content') and msg.content:
            # Clean up the content - remove excessive whitespace, etc.
            content = msg.content.strip()
            if content:
                content_parts.append(content)
    
    # Join with appropriate separators
    if content_parts:
        return " ".join(content_parts)
    else:
        return "No conversation content available"


def create_zmq_tool(
    tool_name: str,
    message_type: str,
    zmq_service,
    transcript_manager: TranscriptManager,
    content_transformer: Optional[Callable[[str], str]] = None
):
    """
    Factory function to create ZMQ-based tools.
    
    Args:
        tool_name: Name of the tool
        message_type: ZMQ message type to send
        zmq_service: ZMQ service instance for sending messages
        transcript_manager: Transcript manager for content extraction
        content_transformer: Optional function to transform the content
        
    Returns:
        Tool function that can be registered with a backend
    """
    async def tool_func(content: Optional[str] = None, update_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Send a ZMQ message with transcript content.
        
        Args:
            content: Optional custom content (if not provided, uses transcript)
            update_type: Optional update type for the message
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary
        """
        try:
            # Use provided content or extract from transcript
            if content is None:
                content = extract_transcript_content(transcript_manager)
            
            # Apply content transformer if provided
            if content_transformer:
                content = content_transformer(content)
            
            # Create message data
            message_data = {
                "content": content
            }
            
            if update_type:
                message_data["update_type"] = update_type
                
            # Add any additional kwargs
            message_data.update(kwargs)
            
            # Send ZMQ message
            await zmq_service.publish_message_async(message_type, message_data)
            
            logger.info(f"Tool '{tool_name}' sent {message_type} message")
            
            return {
                "success": True,
                "message": f"Successfully sent {message_type} message",
                "content_length": len(content)
            }
            
        except Exception as e:
            error_msg = f"Failed to send {message_type} message: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    # Set docstring for the tool
    tool_func.__doc__ = f"""
    {tool_name.replace('_', ' ').title()}
    
    Sends a {message_type} message with conversation transcript content.
    
    Args:
        content: Optional custom content (if not provided, uses current transcript)
        update_type: Optional type of update (e.g., 'clarification', 'addition')
        
    Returns:
        Dictionary with success status and details
    """
    
    return tool_func
