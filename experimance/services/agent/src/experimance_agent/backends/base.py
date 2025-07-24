"""
Abstract base interface for agent backends.

This module defines the standardized interface that all agent backends must implement,
providing a consistent API for different conversation AI providers like Pipecat, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator, Union
import asyncio
import logging

from experimance_agent.config import AgentServiceConfig
from experimance_common.constants import AGENT_SERVICE_DIR
from experimance_common.transcript_manager import TranscriptManager, TranscriptMessage, TranscriptMessageType

logger = logging.getLogger(__name__)


class AgentBackendEvent(str, Enum):
    """Events that can be emitted by agent backends."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"
    SPEECH_DETECTED = "speech_detected"
    SPEECH_ENDED = "speech_ended"
    BOT_STARTED_SPEAKING = "bot_started_speaking"
    BOT_STOPPED_SPEAKING = "bot_stopped_speaking"
    TRANSCRIPTION_RECEIVED = "transcription_received"
    RESPONSE_GENERATED = "response_generated"
    TOOL_CALLED = "tool_called"
    PERSONA_SWITCHED = "persona_switched"
    ERROR = "error"
    CANCEL = "cancel"  # For graceful shutdowns or interruptions


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


@dataclass
class UserContext:
    """User context during a agent session."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    location: Optional[str] = None
    session_start: datetime = field(default_factory=datetime.now)
    custom_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> Optional[str]:
        """Get the full name if available."""
        if self.first_name:
            if self.last_name:
                return f"{self.first_name} {self.last_name}"
            return self.first_name
        return None

    @property
    def is_identified(self) -> bool:
        """Check if the user is identified."""
        return self.first_name is not None

    def summarize(self) -> str:
        """Return a summary of the user context."""
        if self.is_identified:
            userinfo = f"The user's name is {self.first_name}"
            if self.last_name:
                userinfo += f" {self.last_name}"
            if self.location:
                userinfo += f" from {self.location}"
            return userinfo
        return "User not yet identified."

    def reset(self) -> None:
        """Reset user information."""
        self.first_name = None
        self.last_name = None
        self.location = None
        self.custom_data.clear()


class AgentBackend(ABC):
    """
    Abstract base class for agent conversation backends.
    
    This interface provides a standardized API for different agent providers,
    allowing the agent service to switch between backends while maintaining
    consistent functionality.
    """
    
    def __init__(self, config: AgentServiceConfig):
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
        self.conversation_started = False  # Track if conversation has started
        self.user_context = UserContext()  # User context for the session
        
        # Initialize transcript manager
        self.transcript_manager = TranscriptManager(
            save_to_file=config.transcript.save_transcripts,
            output_directory=Path(config.transcript.transcript_directory),
            max_memory_messages=100,
            auto_flush_interval=30.0
        )
        
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
        # Start transcript session
        session_metadata = {
            "backend_type": self.__class__.__name__,
            "user_context": self.user_context.summarize()
        }
        await self.transcript_manager.start_session(session_metadata)
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the agent backend and clean up resources.
        """
        # Stop transcript session
        await self.transcript_manager.stop_session()

    async def graceful_shutdown(self, goodbye_message: Optional[str] = None) -> None:
        """
        Gracefully shutdown the conversation pipeline, allowing any final messages to be processed before terminating.
        This should be called when vision detects the user has left or the session should end naturally.
        By default, this is a no-op; override in backends that support graceful shutdown (e.g., Pipecat).
        Args:
            goodbye_message: Optional goodbye message to say before shutting down
        """
        logger.info(f"Graceful shutdown not implemented for {self.backend_name}")
        await self.disconnect()

    async def say_goodbye_and_shutdown(self, goodbye_message: str = "Thank you for visiting Experimance. Have a wonderful day!") -> None:
        """
        Say a goodbye message and then gracefully shutdown the conversation.
        Recommended for when users leave (detected by vision system).
        By default, this is a no-op; override in backends that support this feature.
        Args:
            goodbye_message: The goodbye message to speak before shutting down
        """
        logger.info(f"Polite farewell not implemented for {self.backend_name}")
        await self.graceful_shutdown(goodbye_message=goodbye_message)
    
    async def connect(self) -> None:
        """
        Connect to the agent service (e.g., join room, establish session).
        Default implementation just sets is_connected flag.
        Override in backends that need specific connection logic.
        
        Raises:
            Exception: If connection fails
        """
        self.is_connected = True
        logger.info(f"Connected to {self.backend_name} backend")
    
    async def disconnect(self) -> None:
        """
        Disconnect from the agent service while keeping backend active.
        Default implementation just clears is_connected flag.
        Override in backends that need specific disconnection logic.
        """
        self.is_connected = False
        logger.info(f"Disconnected from {self.backend_name} backend")

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
    
    # @abstractmethod
    # async def get_conversation_history(self) -> List[ConversationTurn]:
    #     """
    #     Get the current conversation history.
        
    #     Returns:
    #         List of conversation turns
    #     """
    #     pass
    
    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        """
        self._conversation_history.clear()
        # Also clear transcript manager
        self.transcript_manager._messages.clear()
    
    def add_conversation_turn(self, speaker: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a conversation turn to the history.
        
        Args:
            speaker: Speaker identifier ("human", "agent", etc.)
            content: The text content
            metadata: Optional metadata for the turn
        """
        import time
        turn = ConversationTurn(
            speaker=speaker,
            content=content,
            timestamp=time.time(),
            metadata=metadata
        )
        self._conversation_history.append(turn)
        
        # Note: This method is for backward compatibility only.
        # The transcript manager handles its own conversation history tracking.
        # Prefer using add_user_speech() and add_agent_response() directly.
    
    # =========================================================================
    # Transcript Management
    # =========================================================================
    
    async def add_user_speech(
        self, 
        content: str, 
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        is_partial: bool = False
    ) -> TranscriptMessage:
        """Add user speech to transcript and conversation history."""
        # Add to transcript manager
        message = await self.transcript_manager.add_user_speech(
            content=content,
            confidence=confidence,
            duration=duration,
            is_partial=is_partial
        )
        
        # Add to legacy conversation history (only for final transcripts)
        if not is_partial:
            import time
            turn = ConversationTurn(
                speaker="human",
                content=content,
                timestamp=time.time(),
                metadata={"confidence": confidence, "duration": duration}
            )
            self._conversation_history.append(turn)
        
        return message
    
    async def add_agent_response(self, content: str) -> TranscriptMessage:
        """Add agent response to transcript and conversation history."""
        agent_name = self.config.transcript.agent_speaker_name
        
        # Add to transcript manager
        message = await self.transcript_manager.add_agent_response(content, agent_name)
        
        # Add to legacy conversation history
        import time
        turn = ConversationTurn(
            speaker="agent",
            content=content,
            timestamp=time.time(),
            metadata={}
        )
        self._conversation_history.append(turn)
        
        return message
    
    async def add_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Optional[Any] = None
    ) -> TranscriptMessage:
        """Add tool call to transcript."""
        return await self.transcript_manager.add_tool_call(tool_name, parameters, result)
    
    def get_transcript_messages(
        self,
        limit: Optional[int] = None,
        message_types: Optional[List[TranscriptMessageType]] = None
    ) -> List[TranscriptMessage]:
        """Get transcript messages with optional filtering."""
        return self.transcript_manager.get_messages(limit=limit, message_types=message_types)
    
    def get_conversation_history(self) -> List[ConversationTurn]:
        """Get conversation history from transcript manager."""
        # Use transcript manager's conversation history which is more comprehensive
        history_dicts = self.transcript_manager.get_conversation_history()
        
        # Convert to ConversationTurn objects
        turns = []
        for hist in history_dicts:
            turn = ConversationTurn(
                speaker=hist["speaker"],
                content=hist["content"],
                timestamp=hist["timestamp"],
                metadata=hist.get("metadata")
            )
            turns.append(turn)
        
        return turns
    
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
    
    # @abstractmethod
    # async def handle_tool_call(self, tool_call: ToolCall) -> Any:
    #     """
    #     Handle a tool call from the agent.
        
    #     Args:
    #         tool_call: The tool call to execute
            
    #     Returns:
    #         Result of the tool call
            
    #     Raises:
    #         Exception: If tool call fails
    #     """
    #     pass
    
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
    
    # =========================================================================
    # Flow Management (for backends that support flows)
    # =========================================================================
    
    async def transition_to_node(self, node_name: str) -> bool:
        """
        Transition to a specific node in the conversation flow.
        
        Args:
            node_name: Name of the node to transition to
            
        Returns:
            True if transition was successful, False otherwise
            
        Note:
            Default implementation always returns False. Override in backends
            that support conversation flows (e.g., PipecatBackend).
        """
        logger.warning(f"Flow transitions not supported by {self.backend_name} backend")
        return False
    
    def get_current_node(self) -> Optional[str]:
        """
        Get the current active node name in the conversation flow.
        
        Returns:
            Current node name or None if not supported/available
            
        Note:
            Default implementation returns None. Override in backends
            that support conversation flows (e.g., PipecatBackend).
        """
        return None
    
    def is_conversation_active(self) -> bool:
        """
        Check if conversation is currently active (not in search or goodbye state).
        
        Returns:
            True if conversation is active, False otherwise
            
        Note:
            Default implementation returns True. Override in backends
            that support conversation flows for more specific logic.
        """
        return True
    
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
    
    def get_debug_status(self) -> Dict[str, Any]:
        """
        Get detailed debug status information.
        
        Returns:
            Dictionary with comprehensive debug information
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

# =========================================================================
# Utilities
# =========================================================================

def load_prompt(prompt_path: str | Path) -> str:
    """Load prompt instructions from prompt text file."""
    if isinstance(prompt_path, str):
        # if it doesn't end in an extension its probably just a string prompt
        if not any(prompt_path.endswith(ext) for ext in [".txt", ".md", ".json"]):
            return prompt_path

    prompt_path = Path(prompt_path)
    if not prompt_path.exists():
        # try in the prompt directory
        prompt_path = AGENT_SERVICE_DIR / "prompts" / prompt_path

    try:
        with open(str(prompt_path), "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt instructions: {e}")
        return "You are Experimance, an AI art installation assistant."