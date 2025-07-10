"""
Transcript management for conversation backends.

This module provides a centralized, backend-agnostic transcript manager that handles
real-time transcript accumulation, file persistence, and integration with display systems.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from .logger import configure_external_loggers

logger = logging.getLogger(__name__)


class TranscriptMessageType(str, Enum):
    """Types of transcript messages."""
    USER_SPEECH = "user_speech"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_MESSAGE = "system_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class TranscriptMessage:
    """A single transcript message with full metadata."""
    
    # Core message data
    content: str
    message_type: TranscriptMessageType
    timestamp: float
    
    # Speaker information
    speaker_id: str  # "user", "agent", "system", etc.
    speaker_display_name: Optional[str] = None
    
    # Message metadata
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    confidence: Optional[float] = None  # For STT confidence scores
    duration: Optional[float] = None    # For speech duration
    is_partial: bool = False           # For streaming/partial transcripts
    
    # Additional backend-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def display_name(self) -> str:
        """Get the display name for this message's speaker."""
        return self.speaker_display_name or self.speaker_id.title()
    
    @property
    def formatted_timestamp(self) -> str:
        """Get a human-readable timestamp."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'content': self.content,
            'message_type': self.message_type.value,
            'timestamp': self.timestamp,
            'speaker_id': self.speaker_id,
            'speaker_display_name': self.speaker_display_name,
            'session_id': self.session_id,
            'turn_id': self.turn_id,
            'confidence': self.confidence,
            'duration': self.duration,
            'is_partial': self.is_partial,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptMessage':
        """Create from dictionary."""
        return cls(
            content=data['content'],
            message_type=TranscriptMessageType(data['message_type']),
            timestamp=data['timestamp'],
            speaker_id=data['speaker_id'],
            speaker_display_name=data.get('speaker_display_name'),
            session_id=data.get('session_id'),
            turn_id=data.get('turn_id'),
            confidence=data.get('confidence'),
            duration=data.get('duration'),
            is_partial=data.get('is_partial', False),
            metadata=data.get('metadata', {})
        )


class TranscriptDisplayCallback(Protocol):
    """Protocol for transcript display callbacks."""
    
    async def __call__(self, message: TranscriptMessage) -> None:
        """Called when a new transcript message should be displayed."""
        ...


class TranscriptManager:
    """
    Centralized transcript manager for conversation backends.
    
    Handles real-time transcript accumulation, file persistence, display integration,
    and session management in a backend-agnostic way.
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        save_to_file: bool = True,
        output_directory: Optional[Path] = None,
        max_memory_messages: int = 1000,
        auto_flush_interval: float = 30.0
    ):
        """
        Initialize the transcript manager.
        
        Args:
            session_id: Unique identifier for this conversation session
            save_to_file: Whether to save transcripts to disk
            output_directory: Directory for transcript files (default: ./transcripts)
            max_memory_messages: Maximum messages to keep in memory
            auto_flush_interval: Interval to auto-flush to disk (seconds)
        """
        self.session_id = session_id or self._generate_session_id()
        self.save_to_file = save_to_file
        self.output_directory = output_directory or Path("./transcripts")
        self.max_memory_messages = max_memory_messages
        self.auto_flush_interval = auto_flush_interval
        
        # Message storage
        self._messages: List[TranscriptMessage] = []
        self._display_callbacks: List[TranscriptDisplayCallback] = []
        self._message_callbacks: List[Callable[[TranscriptMessage], None]] = []
        
        # File management
        self._file_path: Optional[Path] = None
        self._last_flush_time = time.time()
        self._flush_task: Optional[asyncio.Task] = None
        
        # Session metadata
        self.session_start_time = time.time()
        self.session_metadata: Dict[str, Any] = {}
        
        logger.info(f"TranscriptManager initialized for session {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    async def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start a new transcript session.
        
        Args:
            metadata: Optional session metadata (user info, settings, etc.)
        """
        if metadata:
            self.session_metadata.update(metadata)
        
        if self.save_to_file:
            await self._initialize_file()
        
        # Start auto-flush task
        if self.auto_flush_interval > 0:
            self._flush_task = asyncio.create_task(self._auto_flush_loop())
        
        logger.info(f"Started transcript session {self.session_id}")
    
    async def stop_session(self) -> None:
        """Stop the current transcript session and flush remaining data."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        await self._flush_to_disk()
        logger.info(f"Stopped transcript session {self.session_id}")
    
    async def add_message(
        self,
        content: str,
        message_type: TranscriptMessageType,
        speaker_id: str,
        speaker_display_name: Optional[str] = None,
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        is_partial: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TranscriptMessage:
        """
        Add a new transcript message.
        
        Args:
            content: The message content (text)
            message_type: Type of message
            speaker_id: Speaker identifier
            speaker_display_name: Human-readable speaker name
            confidence: Confidence score (for STT)
            duration: Duration of speech (seconds)
            is_partial: Whether this is a partial/streaming update
            metadata: Additional metadata
            
        Returns:
            The created TranscriptMessage
        """
        message = TranscriptMessage(
            content=content,
            message_type=message_type,
            timestamp=time.time(),
            speaker_id=speaker_id,
            speaker_display_name=speaker_display_name,
            session_id=self.session_id,
            confidence=confidence,
            duration=duration,
            is_partial=is_partial,
            metadata=metadata or {}
        )
        
        # Add to memory
        self._messages.append(message)
        
        # Trim memory if needed
        if len(self._messages) > self.max_memory_messages:
            self._messages = self._messages[-self.max_memory_messages:]
        
        # Notify callbacks
        await self._notify_callbacks(message)
        
        # Auto-flush if needed
        if (time.time() - self._last_flush_time) > self.auto_flush_interval:
            await self._flush_to_disk()
        
        return message
    
    async def add_user_speech(
        self,
        content: str,
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        is_partial: bool = False
    ) -> TranscriptMessage:
        """Convenience method for adding user speech."""
        return await self.add_message(
            content=content,
            message_type=TranscriptMessageType.USER_SPEECH,
            speaker_id="user",
            speaker_display_name="Visitor",
            confidence=confidence,
            duration=duration,
            is_partial=is_partial
        )
    
    async def add_agent_response(
        self,
        content: str,
        agent_name: str = "Experimance"
    ) -> TranscriptMessage:
        """Convenience method for adding agent responses."""
        return await self.add_message(
            content=content,
            message_type=TranscriptMessageType.AGENT_RESPONSE,
            speaker_id="agent",
            speaker_display_name=agent_name
        )
    
    async def add_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Optional[Any] = None
    ) -> TranscriptMessage:
        """Convenience method for adding tool calls."""
        content = f"Called {tool_name} with {parameters}"
        if result is not None:
            content += f" → {result}"
        
        return await self.add_message(
            content=content,
            message_type=TranscriptMessageType.TOOL_CALL,
            speaker_id="agent",
            metadata={"tool_name": tool_name, "parameters": parameters, "result": result}
        )
    
    def add_display_callback(self, callback: TranscriptDisplayCallback) -> None:
        """Add a callback for real-time transcript display."""
        self._display_callbacks.append(callback)
    
    def add_message_callback(self, callback: Callable[[TranscriptMessage], None]) -> None:
        """Add a callback for new messages."""
        self._message_callbacks.append(callback)
    
    def get_messages(
        self,
        limit: Optional[int] = None,
        message_types: Optional[List[TranscriptMessageType]] = None,
        since_timestamp: Optional[float] = None
    ) -> List[TranscriptMessage]:
        """
        Get transcript messages with optional filtering.
        
        Args:
            limit: Maximum number of messages to return
            message_types: Filter by message types
            since_timestamp: Only return messages after this timestamp
            
        Returns:
            List of filtered messages
        """
        messages = self._messages
        
        # Filter by timestamp
        if since_timestamp:
            messages = [m for m in messages if m.timestamp >= since_timestamp]
        
        # Filter by message type
        if message_types:
            messages = [m for m in messages if m.message_type in message_types]
        
        # Apply limit
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history in a standard format.
        
        Returns conversation turns without system messages and tool calls.
        """
        history = []
        for message in self._messages:
            if message.message_type in [TranscriptMessageType.USER_SPEECH, TranscriptMessageType.AGENT_RESPONSE]:
                history.append({
                    "speaker": "human" if message.message_type == TranscriptMessageType.USER_SPEECH else "agent",
                    "content": message.content,
                    "timestamp": message.timestamp,
                    "metadata": message.metadata
                })
        return history
    
    async def _notify_callbacks(self, message: TranscriptMessage) -> None:
        """Notify all registered callbacks about a new message."""
        # Display callbacks (async)
        for callback in self._display_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Error in display callback: {e}")
        
        # Message callbacks (sync)
        for callback in self._message_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in message callback: {e}")
    
    async def _initialize_file(self) -> None:
        """Initialize the transcript output file."""
        if not self.save_to_file:
            return
        
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.fromtimestamp(self.session_start_time).strftime("%Y%m%d_%H%M%S")
        self._file_path = self.output_directory / f"transcript_{timestamp}_{self.session_id}.jsonl"
        
        # Write session header
        session_info = {
            "session_id": self.session_id,
            "start_time": self.session_start_time,
            "metadata": self.session_metadata,
            "type": "session_start"
        }
        
        with open(self._file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(session_info) + "\n")
        
        logger.info(f"Initialized transcript file: {self._file_path}")
    
    async def _flush_to_disk(self) -> None:
        """Flush pending messages to disk."""
        if not self.save_to_file or not self._file_path:
            return
        
        # Get messages since last flush
        messages_to_write = [
            m for m in self._messages 
            if m.timestamp > self._last_flush_time
        ]
        
        if not messages_to_write:
            return
        
        try:
            with open(self._file_path, "a", encoding="utf-8") as f:
                for message in messages_to_write:
                    f.write(json.dumps(message.to_dict()) + "\n")
            
            self._last_flush_time = time.time()
            logger.debug(f"Flushed {len(messages_to_write)} messages to {self._file_path}")
            
        except Exception as e:
            logger.error(f"Error flushing transcript to disk: {e}")
    
    async def _auto_flush_loop(self) -> None:
        """Background task for automatic flushing."""
        try:
            while True:
                await asyncio.sleep(self.auto_flush_interval)
                await self._flush_to_disk()
        except asyncio.CancelledError:
            await self._flush_to_disk()  # Final flush
            raise
        except Exception as e:
            logger.error(f"Error in auto-flush loop: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session."""
        now = time.time()
        duration = now - self.session_start_time
        
        message_counts = {}
        for msg_type in TranscriptMessageType:
            count = len([m for m in self._messages if m.message_type == msg_type])
            message_counts[msg_type.value] = count
        
        return {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "total_messages": len(self._messages),
            "message_counts": message_counts,
            "file_path": str(self._file_path) if self._file_path else None,
            "session_metadata": self.session_metadata
        }


def load_transcript_session(file_path: Path) -> List[TranscriptMessage]:
    """
    Load a transcript session from a JSONL file.
    
    Args:
        file_path: Path to the transcript file
        
    Returns:
        List of transcript messages
    """
    messages = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Skip session metadata lines
                if data.get("type") == "session_start":
                    continue
                
                # Convert to TranscriptMessage
                message = TranscriptMessage.from_dict(data)
                messages.append(message)
                
    except Exception as e:
        logger.error(f"Error loading transcript from {file_path}: {e}")
        raise
    
    return messages
