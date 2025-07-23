"""
Comprehensive tests for PresenceManager.

Tests the full interaction flow from audience arrival to departure,
ensuring proper state transitions and idle detection.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from experimance_core.presence import PresenceManager
from experimance_core.config import PresenceConfig


@pytest.fixture
def presence_config():
    """Create a test presence configuration with short timeouts for faster testing."""
    return PresenceConfig(
        presence_threshold=1.0,  # 1 second to confirm presence
        idle_threshold=2.0,      # 2 seconds to confirm absence
        presence_publish_interval=0.5,  # Publish every 0.5 seconds
        touch_timeout=5.0,       # Not used anymore but kept for config
        conversation_timeout=3.0  # Not used anymore but kept for config
    )


@pytest.fixture
def presence_manager(presence_config):
    """Create a PresenceManager instance for testing."""
    return PresenceManager(presence_config)


class TestPresenceManagerInitialization:
    """Test PresenceManager initialization and default state."""
    
    def test_initial_state(self, presence_manager):
        """Test that PresenceManager starts in the correct initial state."""
        status = presence_manager.get_current_status()
        
        assert status.idle is True
        assert status.present is False
        assert status.hand is False
        assert status.voice is False
        assert status.touch is False
        assert status.conversation is False
        assert status.people_count == 0
        assert status.presence_duration == 0.0
        assert status.hand_duration == 0.0
        assert status.voice_duration == 0.0
        assert status.touch_duration == 0.0


class TestPropertyInterface:
    """Test the property-based interface for setting detection states."""
    
    def test_people_count_property(self, presence_manager):
        """Test people_count property setter and getter."""
        assert presence_manager.people_count == 0
        
        presence_manager.people_count = 2
        assert presence_manager.people_count == 2
        
        status = presence_manager.get_current_status()
        assert status.people_count == 2
    
    def test_voice_property(self, presence_manager):
        """Test voice property setter and getter."""
        assert presence_manager.voice is False
        
        presence_manager.voice = True
        assert presence_manager.voice is True
        
        status = presence_manager.get_current_status()
        assert status.voice is True
    
    def test_agent_speaking_property(self, presence_manager):
        """Test agent_speaking property setter and getter."""
        assert presence_manager.agent_speaking is False
        
        presence_manager.agent_speaking = True
        assert presence_manager.agent_speaking is True
        
        status = presence_manager.get_current_status()
        assert status.conversation is True  # Should activate conversation
    
    def test_hand_property(self, presence_manager):
        """Test hand property setter and getter."""
        assert presence_manager.hand is False
        
        presence_manager.hand = True
        assert presence_manager.hand is True
        
        status = presence_manager.get_current_status()
        assert status.hand is True
    
    def test_touch_property_one_shot(self, presence_manager):
        """Test touch property one-shot behavior."""
        assert presence_manager.touch is False
        
        # Trigger touch
        presence_manager.touch = True
        
        # Should immediately return to false (one-shot)
        assert presence_manager.touch is False
        
        # But the status should reflect the trigger during update
        status = presence_manager.get_current_status()
        # Touch should be false in status too since it's reset immediately
        assert status.touch is False
        
        # But last_touch should be recorded
        assert status.last_touch is not None


class TestConversationDetection:
    """Test conversation field logic."""
    
    def test_conversation_agent_speaking(self, presence_manager):
        """Test conversation is true when agent is speaking."""
        presence_manager.agent_speaking = True
        status = presence_manager.get_current_status()
        assert status.conversation is True
    
    def test_conversation_voice_detected(self, presence_manager):
        """Test conversation is true when human voice is detected."""
        presence_manager.voice = True
        status = presence_manager.get_current_status()
        assert status.conversation is True
    
    def test_conversation_both_speaking(self, presence_manager):
        """Test conversation is true when both agent and human are speaking."""
        presence_manager.agent_speaking = True
        presence_manager.voice = True
        status = presence_manager.get_current_status()
        assert status.conversation is True
    
    def test_conversation_neither_speaking(self, presence_manager):
        """Test conversation is false when neither is speaking."""
        presence_manager.agent_speaking = False
        presence_manager.voice = False
        status = presence_manager.get_current_status()
        assert status.conversation is False


class TestHysteresisLogic:
    """Test hysteresis/debouncing logic for presence detection."""
    
    def test_presence_requires_threshold_time(self, presence_manager):
        """Test that presence is not confirmed until threshold time passes."""
        # Set presence indicators
        presence_manager.people_count = 1
        
        # Should not be present immediately
        status = presence_manager.get_current_status()
        assert status.present is False
        assert status.idle is True
        
        # Fast-forward past threshold
        with patch('experimance_core.presence.datetime') as mock_datetime:
            future_time = datetime.now() + timedelta(seconds=2)
            mock_datetime.now.return_value = future_time
            presence_manager.force_update()
            
            status = presence_manager.get_current_status()
            assert status.present is True
            assert status.idle is False
    
    def test_absence_requires_threshold_time(self, presence_manager):
        """Test that absence is not confirmed until threshold time passes."""
        # First establish presence
        presence_manager.people_count = 1
        
        with patch('experimance_core.presence.datetime') as mock_datetime:
            # Fast-forward to establish presence
            future_time = datetime.now() + timedelta(seconds=2)
            mock_datetime.now.return_value = future_time
            presence_manager.force_update()
            
            status = presence_manager.get_current_status()
            assert status.present is True
            
            # Now remove presence indicators
            presence_manager.people_count = 0
            
            # Should still be present immediately after
            status = presence_manager.get_current_status()
            assert status.present is True
            assert status.idle is False
            
            # Fast-forward past idle threshold
            future_time = datetime.now() + timedelta(seconds=5)
            mock_datetime.now.return_value = future_time
            presence_manager.force_update()
            
            status = presence_manager.get_current_status()
            assert status.present is False
            assert status.idle is True


class TestTypicalInteractionFlow:
    """Test typical user interaction scenarios."""
    
    def test_complete_interaction_cycle(self, presence_manager):
        """Test a complete interaction from arrival to departure."""
        with patch('experimance_core.presence.datetime') as mock_datetime:
            base_time = datetime.now()
            mock_datetime.now.return_value = base_time
            
            # 1. Vision sees a person
            presence_manager.people_count = 1
            
            # Should not be present yet (hysteresis)
            status = presence_manager.get_current_status()
            assert status.present is False
            assert status.idle is True
            
            # 2. Wait for presence threshold
            mock_datetime.now.return_value = base_time + timedelta(seconds=1.5)
            presence_manager.force_update()
            
            status = presence_manager.get_current_status()
            assert status.present is True
            assert status.idle is False
            assert status.people_count == 1
            
            # 3. Bot speaks
            mock_datetime.now.return_value = base_time + timedelta(seconds=2)
            presence_manager.agent_speaking = True
            
            status = presence_manager.get_current_status()
            assert status.conversation is True
            assert status.present is True
            
            # 4. User speaks (bot stops)
            mock_datetime.now.return_value = base_time + timedelta(seconds=3)
            presence_manager.agent_speaking = False
            presence_manager.voice = True
            
            status = presence_manager.get_current_status()
            assert status.conversation is True  # Still in conversation
            assert status.voice is True
            
            # 5. User hand detected
            mock_datetime.now.return_value = base_time + timedelta(seconds=4)
            presence_manager.hand = True
            
            status = presence_manager.get_current_status()
            assert status.hand is True
            assert status.present is True
            
            # 6. Touch interaction
            mock_datetime.now.return_value = base_time + timedelta(seconds=5)
            presence_manager.touch = True
            
            status = presence_manager.get_current_status()
            assert status.last_touch is not None
            # Touch should be false (one-shot)
            assert status.touch is False
            
            # 7. User stops speaking
            mock_datetime.now.return_value = base_time + timedelta(seconds=6)
            presence_manager.voice = False
            
            status = presence_manager.get_current_status()
            assert status.conversation is False
            assert status.voice is False
            
            # 8. Hand removed
            mock_datetime.now.return_value = base_time + timedelta(seconds=7)
            presence_manager.hand = False
            
            status = presence_manager.get_current_status()
            assert status.hand is False
            assert status.present is True  # Still present due to people_count only
            
            # 9. Person leaves (vision loses them)
            mock_datetime.now.return_value = base_time + timedelta(seconds=8)
            presence_manager.people_count = 0
            
            # Should still be present (hysteresis)
            status = presence_manager.get_current_status()
            assert status.present is True
            assert status.people_count == 0
            
            # 10. Wait for idle threshold (touch doesn't extend presence)
            # Absence tracking should start at second 8 when people_count goes to 0
            # Idle threshold is 2s, so absence should be confirmed at second 10
            mock_datetime.now.return_value = base_time + timedelta(seconds=10)
            presence_manager.force_update()
            
            status = presence_manager.get_current_status()
            assert status.present is False
            assert status.idle is True
            assert status.people_count == 0
    
    def test_touch_triggers_audio_sfx(self, presence_manager):
        """Test that touch properly triggers for audio SFX."""
        # Simulate touch detection
        initial_status = presence_manager.get_current_status()
        assert initial_status.touch is False
        
        # Trigger touch
        presence_manager.touch = True
        
        # Touch should immediately reset (one-shot)
        assert presence_manager.touch is False
        
        # But timestamp should be recorded
        status = presence_manager.get_current_status()
        assert status.last_touch is not None
    
    def test_multiple_presence_indicators(self, presence_manager):
        """Test behavior with multiple simultaneous presence indicators."""
        with patch('experimance_core.presence.datetime') as mock_datetime:
            base_time = datetime.now()
            mock_datetime.now.return_value = base_time
            
            # Set multiple indicators
            presence_manager.people_count = 2
            presence_manager.hand = True
            presence_manager.voice = True
            
            # Wait for presence threshold
            mock_datetime.now.return_value = base_time + timedelta(seconds=1.5)
            presence_manager.force_update()
            
            status = presence_manager.get_current_status()
            assert status.present is True
            assert status.people_count == 2
            assert status.hand is True
            assert status.voice is True
            assert status.conversation is True
            
            # Remove one indicator at a time
            presence_manager.voice = False
            status = presence_manager.get_current_status()
            assert status.present is True  # Still present due to other indicators
            assert status.conversation is False
            
            presence_manager.hand = False
            status = presence_manager.get_current_status()
            assert status.present is True  # Still present due to people_count
            
            # Remove final indicator
            presence_manager.people_count = 0
            
            # Should still be present (hysteresis)
            status = presence_manager.get_current_status()
            assert status.present is True
            
            # Wait for idle threshold
            mock_datetime.now.return_value = base_time + timedelta(seconds=4)
            presence_manager.force_update()
            
            status = presence_manager.get_current_status()
            assert status.present is False
            assert status.idle is True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_rapid_state_changes(self, presence_manager):
        """Test rapid on/off state changes are handled correctly."""
        # Rapidly toggle presence indicators
        for i in range(5):
            presence_manager.people_count = 1 if i % 2 == 0 else 0
            status = presence_manager.get_current_status()
            # Should not become present due to hysteresis
            assert status.present is False
    
    def test_touch_multiple_triggers(self, presence_manager):
        """Test multiple touch triggers work correctly."""
        # First touch
        presence_manager.touch = True
        status1 = presence_manager.get_current_status()
        first_touch_time = status1.last_touch
        
        # Second touch (after small delay)
        with patch('experimance_core.presence.datetime') as mock_datetime:
            future_time = datetime.now() + timedelta(seconds=0.1)
            mock_datetime.now.return_value = future_time
            
            presence_manager.touch = True
            status2 = presence_manager.get_current_status()
            second_touch_time = status2.last_touch
            
            # Should have updated timestamp
            assert second_touch_time != first_touch_time
            assert second_touch_time > first_touch_time
    
    def test_durations_calculation(self, presence_manager):
        """Test that durations are calculated correctly."""
        with patch('experimance_core.presence.datetime') as mock_datetime:
            base_time = datetime.now()
            mock_datetime.now.return_value = base_time
            
            # Establish presence
            presence_manager.people_count = 1
            presence_manager.hand = True
            presence_manager.voice = True
            
            # Wait for presence threshold
            mock_datetime.now.return_value = base_time + timedelta(seconds=2)
            presence_manager.force_update()
            
            # Check durations after some time
            mock_datetime.now.return_value = base_time + timedelta(seconds=5)
            status = presence_manager.get_current_status()
            
            assert status.presence_duration > 0
            assert status.hand_duration > 0
            assert status.voice_duration > 0


class TestPublishingControl:
    """Test publishing control mechanisms."""
    
    def test_should_publish_initial(self, presence_manager):
        """Test that initial publish is allowed."""
        assert presence_manager.should_publish() is True
    
    def test_should_publish_after_interval(self, presence_manager):
        """Test publishing after interval."""
        # Mark as published
        presence_manager.mark_published()
        
        # Should not publish immediately
        assert presence_manager.should_publish() is False
        
        # Should publish after interval
        with patch('experimance_core.presence.datetime') as mock_datetime:
            future_time = datetime.now() + timedelta(seconds=1)
            mock_datetime.now.return_value = future_time
            
            assert presence_manager.should_publish() is True
    
    def test_debug_info(self, presence_manager):
        """Test debug info provides useful information."""
        presence_manager.people_count = 1
        presence_manager.hand = True
        
        debug_info = presence_manager.get_debug_info()
        
        assert "raw_inputs" in debug_info
        assert "hysteresis_state" in debug_info
        assert "current_status" in debug_info
        
        assert debug_info["raw_inputs"]["people_count"] == 1
        assert debug_info["raw_inputs"]["hand_detected"] is True
        # Note: current_status reflects the stable state after hysteresis
        # so it might not immediately reflect the raw inputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
