"""
Abstract base interface for agent backends.

This module defines the standardized interface that all agent backends must implement,
providing a consistent API for different conversation AI providers like LiveKit, Hume.ai, Ultravox, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator
import asyncio
import logging

logger = logging.getLogger(__name__)


class AgentBackendEvent(str, Enum):
    """Events that can be emitted by agent backends."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"
    SPEECH_DETECTED = "speech_detected"
    SPEECH_ENDED = "speech_ended"
    TRANSCRIPTION_RECEIVED = "transcription_received"
    RESPONSE_GENERATED = "response_generated"
    TOOL_CALLED = "tool_called"
    ERROR = "error"


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    speaker: str  # "human" or "agent"
    content: str  # The spoken/generated text
    timestamp: float  # Unix timestamp
    metadata: Optional[Dict[str, Any]] = None  # Backend-specific metadata


@dataclass
class ToolCall:
    """Represents a tool function call from the agent."""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None  # Backend-specific call identifier


class AgentBackend(ABC):
    """
    Abstract base class for agent conversation backends.
    
    This interface provides a standardized API for different agent providers,
    allowing the agent service to switch between backends while maintaining
    consistent functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent backend.
        
        Args:
            config: Backend-specific configuration parameters
        """
        self.config = config
        self.is_connected = False
        self.is_active = False
        self._event_callbacks: Dict[AgentBackendEvent, List[Callable]] = {}
        self._conversation_history: List[ConversationTurn] = []
        self._available_tools: Dict[str, Callable] = {}
        
    # =========================================================================
    # Lifecycle Management
    # =========================================================================
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the agent backend and initialize connections.
        
        Raises:
            Exception: If backend fails to start
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the agent backend and clean up resources.
        """
        pass
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the agent service (e.g., join room, establish session).
        
        Raises:
            Exception: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the agent service while keeping backend active.
        """
        pass
    
    # =========================================================================
    # Conversation Management
    # =========================================================================
    
    @abstractmethod
    async def send_message(self, message: str, speaker: str = "system") -> None:
        """
        Send a message to the conversation.
        
        Args:
            message: Text message to send
            speaker: Speaker identifier (e.g., "system", "user")
        """
        pass
    
    @abstractmethod
    async def get_conversation_history(self) -> List[ConversationTurn]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation turns
        """
        pass
    
    async def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        """
        self._conversation_history.clear()
    
    # =========================================================================
    # Tool Management
    # =========================================================================
    
    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        """
        Register a tool function that the agent can call.
        
        Args:
            name: Tool function name
            func: The callable function
            description: Human-readable description of the tool
        """
        self._available_tools[name] = func
        logger.info(f"Registered tool '{name}' with backend {self.__class__.__name__}")
    
    def unregister_tool(self, name: str) -> None:
        """
        Unregister a tool function.
        
        Args:
            name: Tool function name to remove
        """
        if name in self._available_tools:
            del self._available_tools[name]
            logger.info(f"Unregistered tool '{name}' from backend {self.__class__.__name__}")
    
    def get_available_tools(self) -> Dict[str, Callable]:
        """
        Get all available tool functions.
        
        Returns:
            Dictionary mapping tool names to functions
        """
        return self._available_tools.copy()
    
    @abstractmethod
    async def handle_tool_call(self, tool_call: ToolCall) -> Any:
        """
        Handle a tool call from the agent.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            Result of the tool call
            
        Raises:
            Exception: If tool call fails
        """
        pass
    
    # =========================================================================
    # Event System
    # =========================================================================
    
    def add_event_callback(self, event: AgentBackendEvent, callback: Callable) -> None:
        """
        Add a callback for a specific event.
        
        Args:
            event: Event type to listen for
            callback: Function to call when event occurs
        """
        if event not in self._event_callbacks:
            self._event_callbacks[event] = []
        self._event_callbacks[event].append(callback)
    
    def remove_event_callback(self, event: AgentBackendEvent, callback: Callable) -> None:
        """
        Remove a callback for a specific event.
        
        Args:
            event: Event type
            callback: Function to remove
        """
        if event in self._event_callbacks and callback in self._event_callbacks[event]:
            self._event_callbacks[event].remove(callback)
    
    async def emit_event(self, event: AgentBackendEvent, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit an event to all registered callbacks.
        
        Args:
            event: Event type to emit
            data: Optional event data
        """
        if event in self._event_callbacks:
            for callback in self._event_callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event, data)
                    else:
                        callback(event, data)
                except Exception as e:
                    logger.error(f"Error in event callback for {event}: {e}")
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    @property
    def backend_name(self) -> str:
        """Get the backend name (derived from class name)."""
        return self.__class__.__name__.replace("Backend", "").lower()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current backend status information.
        
        Returns:
            Dictionary with status information
        """
        return {
            "backend_name": self.backend_name,
            "is_connected": self.is_connected,
            "is_active": self.is_active,
            "conversation_turns": len(self._conversation_history),
            "available_tools": list(self._available_tools.keys()),
        }
    
    # =========================================================================
    # Advanced Features (Optional Implementation)
    # =========================================================================
    
    async def process_image(self, image_data: bytes, prompt: Optional[str] = None) -> Optional[str]:
        """
        Process an image through the agent (if supported).
        
        Args:
            image_data: Raw image bytes
            prompt: Optional text prompt to accompany the image
            
        Returns:
            Agent's response to the image, or None if not supported
        """
        logger.warning(f"Image processing not implemented for {self.backend_name}")
        return None
    
    async def set_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt for the agent.
        
        Args:
            prompt: New system prompt
        """
        logger.warning(f"System prompt setting not implemented for {self.backend_name}")
    
    async def get_transcript_stream(self) -> AsyncGenerator[ConversationTurn, None]:
        """
        Get a real-time stream of conversation turns.
        
        Yields:
            Conversation turns as they occur
        """
        # Default implementation - backends can override for real-time streaming
        for turn in self._conversation_history:
            yield turn
