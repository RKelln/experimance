"""
Presence detection and management for the Experimance Core Service.

This module handles the aggregation of presence inputs from various sources
(depth camera, agent vision, voice detection) and applies hysteresis logic
to make stable presence decisions.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from experimance_common.schemas_base import PresenceStatus
from experimance_core.config import PresenceConfig

logger = logging.getLogger(__name__)


class PresenceManager:
    """
    Manages presence detection state and applies hysteresis logic.
    
    Aggregates inputs from:
    - Hand detection (from depth camera)
    - Agent vision (people count, general presence)
    - Voice detection (user speaking)
    - Agent speaking status
    
    Applies debouncing/hysteresis to prevent rapid state changes.
    """
    
    def __init__(self, config: PresenceConfig):
        """
        Initialize the presence manager.
        
        Args:
            config: Presence configuration with thresholds and intervals
        """
        self.config = config
        
        # Current detection inputs (raw from sources)
        self._hand_detected: bool = False
        self._agent_people_count: int = 0
        self._voice_detected: bool = False
        self._agent_speaking: bool = False
        self._touch_triggered: bool = False  # One-shot trigger for audio SFX
        
        # Hysteresis state tracking
        self._presence_start_time: Optional[datetime] = None
        self._absence_start_time: Optional[datetime] = None
        
        # Last confirmed state change times for duration calculations
        self._last_present_time: Optional[datetime] = None
        self._last_hand_time: Optional[datetime] = None
        self._last_voice_time: Optional[datetime] = None
        self._last_touch_time: Optional[datetime] = None
        
        # Current stable decisions (after hysteresis)
        self._current_status = PresenceStatus(
            idle=True,
            present=False,
            hand=False,
            voice=False,
            touch=False,
            conversation=False,
            people_count=0
        )
        
        # Publishing control
        self._last_publish_time: Optional[datetime] = None
        self.updated : bool = False  # Flag to track if presence state has changed
        
        logger.info("PresenceManager initialized")
    
    # Property-based interface matching schema fields
    @property
    def people_count(self) -> int:
        """Number of people detected by vision system."""
        return self._agent_people_count
    
    @people_count.setter
    def people_count(self, value: int) -> None:
        if self._agent_people_count != value:
            logger.debug(f"People count changed: {value}")
            self._agent_people_count = value
            self._update_presence_state()
    
    @property
    def voice(self) -> bool:
        """Human voice detected."""
        return self._voice_detected
    
    @voice.setter
    def voice(self, value: bool) -> None:
        if self._voice_detected != value:
            logger.debug(f"Voice detection changed: {value}")
            self._voice_detected = value
            if value:
                self._last_voice_time = datetime.now()
            self._update_presence_state()
    
    @property
    def agent_speaking(self) -> bool:
        """Agent is currently speaking."""
        return self._agent_speaking
    
    @agent_speaking.setter
    def agent_speaking(self, value: bool) -> None:
        if self._agent_speaking != value:
            logger.debug(f"Agent speaking changed: {value}")
            self._agent_speaking = value
            self._update_presence_state()
    
    @property
    def hand(self) -> bool:
        """Hand detected over the bowl."""
        return self._hand_detected
    
    @hand.setter
    def hand(self, value: bool) -> None:
        if self._hand_detected != value:
            logger.debug(f"Hand detection changed: {value}")
            self._hand_detected = value
            if value:
                self._last_hand_time = datetime.now()
            self._update_presence_state()

    @property
    def touch(self) -> bool:
        """Touch/interaction detected in the sand (one-shot trigger for audio SFX).
        
        This is a momentary trigger that goes true when touch is detected,
        then immediately resets to false. Used to trigger audio effects.
        """
        return self._touch_triggered
    
    @touch.setter
    def touch(self, value: bool) -> None:
        if value:
            # Record touch event and trigger one-shot state
            logger.debug("Touch interaction detected - triggering audio SFX")
            self._last_touch_time = datetime.now()
            self._touch_triggered = True
            self._update_presence_state()
            # Immediately reset the trigger (one-shot behavior)
            self._touch_triggered = False

    def _update_presence_state(self) -> None:
        """Update the internal presence state based on current inputs."""
        now = datetime.now()
        self.updated = True  # Mark that state has changed
        
        # Determine if any presence indicators are active
        # Note: touch is one-shot and doesn't extend presence beyond the moment it occurs
        any_presence = (
            self._hand_detected or 
            self._agent_people_count > 0 or 
            self._voice_detected
            # Touch is NOT included here as it's a momentary trigger, not persistent presence
        )
        
        # Apply hysteresis logic
        if any_presence:
            # Start tracking presence if not already
            if self._presence_start_time is None:
                self._presence_start_time = now
                logger.debug("Started tracking presence detection")
            
            # Reset absence tracking
            self._absence_start_time = None
            
            # Check if presence has been stable long enough
            presence_duration = (now - self._presence_start_time).total_seconds()
            if presence_duration >= self.config.presence_threshold:
                if not self._current_status.present:
                    logger.info("Audience presence confirmed (hysteresis threshold met)")
                    self._last_present_time = now
                self._update_current_status(present=True)
        else:
            # Start tracking absence if not already
            if self._absence_start_time is None:
                self._absence_start_time = now
                logger.debug("Started tracking absence detection")
            
            # Reset presence tracking
            self._presence_start_time = None
            
            # Check if absence has been stable long enough
            absence_duration = (now - self._absence_start_time).total_seconds()
            if absence_duration >= self.config.idle_threshold:
                if self._current_status.present:
                    logger.info("Audience absence confirmed (hysteresis threshold met)")
                self._update_current_status(present=False)
    
    def _update_current_status(self, present: bool) -> None:
        """Update the current stable presence status."""
        now = datetime.now()
        
        # Calculate durations
        presence_duration = 0.0
        hand_duration = 0.0
        voice_duration = 0.0
        touch_duration = 0.0
        
        if present and self._last_present_time:
            presence_duration = (now - self._last_present_time).total_seconds()
        
        if self._hand_detected and self._last_hand_time:
            hand_duration = (now - self._last_hand_time).total_seconds()
        
        if self._voice_detected and self._last_voice_time:
            voice_duration = (now - self._last_voice_time).total_seconds()
        
        if self._last_touch_time is not None:
            touch_duration = (now - self._last_touch_time).total_seconds()
        
        # Calculate conversation status (either agent or human speaking)
        conversation_active = self._agent_speaking or self._voice_detected
        
        # Update status
        self._current_status = PresenceStatus(
            idle=not present,
            present=present,
            hand=self._hand_detected,
            voice=self._voice_detected,
            touch=self._touch_triggered,  # One-shot trigger state
            conversation=conversation_active,
            people_count=self._agent_people_count,
            last_present=self._last_present_time,
            last_hand=self._last_hand_time,
            last_voice=self._last_voice_time,
            last_touch=self._last_touch_time,
            presence_duration=presence_duration,
            hand_duration=hand_duration,
            voice_duration=voice_duration,
            touch_duration=touch_duration,
            timestamp=now
        )
    
    def get_current_status(self) -> PresenceStatus:
        """
        Get the current presence status.
        
        Returns:
            Current PresenceStatus with all detection states and durations
        """
        # Always update timestamps and durations before returning
        self._update_current_status(self._current_status.present)
        return self._current_status
    
    def should_publish(self) -> bool:
        """
        Check if enough time has passed to publish a presence status update.
        
        Returns:
            True if it's time to publish an update
        """
        if self.updated:
            self.updated = False
            return True  # Force publish if state was updated
        
        if self._last_publish_time is None:
            return True
        
        now = datetime.now()
        time_since_publish = (now - self._last_publish_time).total_seconds()
        return time_since_publish >= self.config.presence_publish_interval
    
    def mark_published(self) -> None:
        """Mark that a presence status has been published."""
        self._last_publish_time = datetime.now()
    
    def force_update(self) -> None:
        """Force an immediate update of the presence state (for testing)."""
        self._update_presence_state()
    
    def get_debug_info(self) -> dict:
        """Get debug information about the presence manager state."""
        now = datetime.now()
        
        presence_tracking_duration = 0.0
        if self._presence_start_time:
            presence_tracking_duration = (now - self._presence_start_time).total_seconds()
        
        absence_tracking_duration = 0.0
        if self._absence_start_time:
            absence_tracking_duration = (now - self._absence_start_time).total_seconds()
        
        return {
            "raw_inputs": {
                "hand_detected": self._hand_detected,
                "people_count": self._agent_people_count,
                "voice_detected": self._voice_detected,
                "agent_speaking": self._agent_speaking,
                "touch_triggered": self._touch_triggered,
            },
            "hysteresis_state": {
                "presence_tracking_duration": presence_tracking_duration,
                "absence_tracking_duration": absence_tracking_duration,
                "presence_threshold": self.config.presence_threshold,
                "idle_threshold": self.config.idle_threshold,
            },
            "current_status": {
                "idle": self._current_status.idle,
                "present": self._current_status.present,
                "hand": self._current_status.hand,
                "voice": self._current_status.voice,
                "touch": self._current_status.touch,
                "conversation": self._current_status.conversation,
                "people_count": self._current_status.people_count,
            }
        }
