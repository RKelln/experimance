"""MediaPrompt logging utilities for tracking prompt generation and usage.

This module provides structured logging of MediaPrompt objects to JSONL files,
enabling timeline tracking and analysis of prompt generation in the Fire installation.

The log format matches the transcript logging format for easy interleaving and viewing.
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from experimance_common.constants import LOGS_DIR

from .config import ImagePrompt, MediaPrompt

logger = logging.getLogger(__name__)

class PromptLogger:
    """Logger for MediaPrompt and ImagePrompt objects."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize the prompt logger.
        
        Args:
            log_dir: Directory for prompt logs. If None, uses environment variable
                    EXPERIMANCE_PROMPTS_DIR or defaults to /var/log/experimance/prompts
        """
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            env_dir = os.environ.get("EXPERIMANCE_PROMPTS_DIR")
            if env_dir:
                self.log_dir = Path(env_dir)
            else:
                # save to logs directory by default
                self.log_dir = Path(LOGS_DIR / "prompts")
        
        # Create directory if it doesn't exist
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create prompt log directory {self.log_dir}: {e}")
            # Fall back to current directory
            self.log_dir = Path.cwd() / "prompts"
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session_id: Optional[str] = None
        self.current_log_file: Optional[Path] = None
        
        logger.info(f"PromptLogger initialized with log directory: {self.log_dir}")
    
    def set_session(self, session_id: str) -> None:
        """Set the current session ID and create a new log file if needed.
        
        Args:
            session_id: Session identifier (usually timestamp-based)
        """
        if session_id != self.current_session_id:
            self.current_session_id = session_id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompts_{timestamp}_session_{session_id}.jsonl"
            self.current_log_file = self.log_dir / filename
            logger.info(f"📝 Started new prompt log session: {self.current_log_file}")
    
    def log_media_prompt(
        self,
        media_prompt: MediaPrompt,
        request_id: str,
        session_id: Optional[str] = None,
        event_type: str = "prompt_created",
        metadata: Optional[dict] = None
    ) -> None:
        """Log a MediaPrompt to the current session file.
        
        Args:
            media_prompt: The MediaPrompt object to log
            request_id: Unique request identifier
            session_id: Session ID (if None, uses current session)
            event_type: Type of event (prompt_created, prompt_queued, etc.)
            metadata: Additional metadata to include
        """
        if session_id:
            self.set_session(session_id)
        
        if not self.current_log_file:
            # Create a default session if none set
            default_session = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.set_session(default_session)
        
        # Create log entry
        timestamp = time.time()
        log_entry = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "session_id": self.current_session_id,
            "request_id": request_id,
            "event_type": event_type,
            "role": "system",  # For compatibility with transcript viewer
            "speaker_id": "prompt_generator",
            "speaker_display_name": "Prompt Generator",
            "visual_prompt": media_prompt.visual_prompt,
            "visual_negative_prompt": media_prompt.visual_negative_prompt,
            "audio_prompt": media_prompt.audio_prompt,
            "content": f"Visual: {media_prompt.visual_prompt[:100]}{'...' if len(media_prompt.visual_prompt) > 100 else ''}",  # Summary for viewing
        }
        
        # Add metadata if provided
        if metadata:
            log_entry.update(metadata)
        
        # Write to file
        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            logger.debug(f"📝 Logged MediaPrompt for request {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to write prompt log entry: {e}")
    
    def log_image_prompt(
        self,
        image_prompt: ImagePrompt,
        request_id: str,
        session_id: Optional[str] = None,
        event_type: str = "image_prompt_created",
        metadata: Optional[dict] = None
    ) -> None:
        """Log an ImagePrompt to the current session file.
        
        Args:
            image_prompt: The ImagePrompt object to log
            request_id: Unique request identifier
            session_id: Session ID (if None, uses current session)
            event_type: Type of event
            metadata: Additional metadata to include
        """
        # Convert ImagePrompt to MediaPrompt for consistent logging
        media_prompt = MediaPrompt(
            visual_prompt=image_prompt.prompt,
            visual_negative_prompt=image_prompt.negative_prompt,
            audio_prompt=None
        )
        
        self.log_media_prompt(
            media_prompt=media_prompt,
            request_id=request_id,
            session_id=session_id,
            event_type=event_type,
            metadata=metadata
        )
    
    def log_request_event(
        self,
        request_id: str,
        event_type: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Log a general request event (e.g., image_generated, request_completed).
        
        Args:
            request_id: Unique request identifier
            event_type: Type of event
            content: Event description
            session_id: Session ID (if None, uses current session)
            metadata: Additional metadata to include
        """
        if session_id:
            self.set_session(session_id)
        
        if not self.current_log_file:
            # Create a default session if none set
            default_session = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.set_session(default_session)
        
        # Create log entry
        timestamp = time.time()
        log_entry = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "session_id": self.current_session_id,
            "request_id": request_id,
            "event_type": event_type,
            "role": "system",
            "speaker_id": "fire_core",
            "speaker_display_name": "Fire Core",
            "content": content,
        }
        
        # Add metadata if provided
        if metadata:
            log_entry.update(metadata)
        
        # Write to file
        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            logger.debug(f"📝 Logged event {event_type} for request {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to write event log entry: {e}")


# Global instance
_prompt_logger: Optional[PromptLogger] = None

def get_prompt_logger() -> PromptLogger:
    """Get the global prompt logger instance."""
    global _prompt_logger
    if _prompt_logger is None:
        _prompt_logger = PromptLogger()
    return _prompt_logger
