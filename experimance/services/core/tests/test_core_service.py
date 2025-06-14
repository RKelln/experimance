#!/usr/bin/env python3
"""
Unit tests for the Experimance Core Service.

This test suite covers:
- Service initialization and configuration
- State machine logic (era progression, biome selection)
- Interaction scoring and idle management
- Event publishing functionality
- Error handling and recovery
"""
import asyncio
import json
import logging
import pytest
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experimance_core.experimance_core import ExperimanceCoreService, ERA_PROGRESSION, ERA_BIOMES
from experimance_common.schemas import Era, Biome
from experimance_common.test_utils import active_service
from experimance_common.service_state import ServiceState

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def mock_config_file():
    """Create a temporary config file for testing."""
    config_content = """
[experimance_core]
name = "test_core"
heartbeat_interval = 1.0

[state_machine]
idle_timeout = 10.0
wilderness_reset = 60.0
interaction_threshold = 0.5
era_min_duration = 5.0

[depth_processing]
change_threshold = 25
min_depth = 0.4
max_depth = 0.6
resolution = [640, 480]
output_size = [512, 512]

[audio]
tag_config_path = "config/audio_tags.json"
interaction_sound_duration = 1.0

[prompting]
data_path = "data/"
locations_file = "locations.json"
developments_file = "anthropocene.json"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        f.flush()
        yield f.name
    
    # Cleanup
    Path(f.name).unlink()


@pytest.fixture
def core_service(mock_config_file):
    """Create a core service instance for testing."""
    # Mock ZMQ initialization to avoid network dependencies
    with patch('experimance_core.experimance_core.ZmqPublisherSubscriberService.__init__', return_value=None):
        service = ExperimanceCoreService(config_path=mock_config_file)
        
        # Mock parent class methods and attributes
        service.publish_message = AsyncMock()
        service.add_task = MagicMock()
        service._sleep_if_running = AsyncMock(return_value=False)
        service.record_error = MagicMock()
        
        return service


class TestServiceInitialization:
    """Test service initialization and configuration."""
    
    def test_service_creation(self, core_service):
        """Test that service can be created with proper configuration."""
        assert core_service.config.experimance_core.name == "test_core"
        assert core_service.current_era == Era.WILDERNESS
        assert core_service.current_biome == Biome.TEMPERATE_FOREST
        assert core_service.user_interaction_score == 0.0
        assert core_service.idle_timer == 0.0
        assert not core_service.audience_present
        assert not core_service.hand_detected
    
    def test_depth_processing_state_initialization(self, core_service):
        """Test depth processing state variables are properly initialized."""
        assert core_service.depth_generator is None
        assert core_service.previous_depth_image is None
        assert core_service.hand_detected == False
        assert core_service.depth_difference_score == 0.0
    
    def test_configuration_loading(self, core_service):
        """Test that configuration is loaded correctly."""
        assert core_service.config.state_machine.idle_timeout == 10.0
        assert core_service.config.state_machine.interaction_threshold == 0.5
        assert core_service.config.depth_processing.change_threshold == 25
        assert core_service.config.depth_processing.resolution == (640, 480)


class TestStateMachine:
    """Test state machine logic and era progression."""
    
    def test_era_progression_mapping(self):
        """Test that era progression mappings are correct."""
        assert Era.PRE_INDUSTRIAL in ERA_PROGRESSION[Era.WILDERNESS]
        assert Era.EARLY_INDUSTRIAL in ERA_PROGRESSION[Era.PRE_INDUSTRIAL]
        assert Era.WILDERNESS in ERA_PROGRESSION[Era.RUINS]
        assert len(ERA_PROGRESSION[Era.FUTURE]) == 2  # Can loop or progress to dystopia
    
    def test_biome_availability(self):
        """Test that biome availability by era is correct."""
        wilderness_biomes = ERA_BIOMES[Era.WILDERNESS]
        assert Biome.RAINFOREST in wilderness_biomes
        assert Biome.TEMPERATE_FOREST in wilderness_biomes
        
        dystopia_biomes = ERA_BIOMES[Era.DYSTOPIA]
        assert len(dystopia_biomes) < len(wilderness_biomes)  # Limited biomes
    
    def test_era_validation(self, core_service):
        """Test era validation methods."""
        assert core_service.is_valid_era("wilderness")
        assert core_service.is_valid_era("modern")
        assert not core_service.is_valid_era("invalid_era")
    
    def test_biome_validation(self, core_service):
        """Test biome validation methods."""
        assert core_service.is_valid_biome("temperate_forest")
        assert core_service.is_valid_biome("desert")
        assert not core_service.is_valid_biome("invalid_biome")
    
    def test_next_era_calculation(self, core_service):
        """Test next era calculation logic."""
        core_service.current_era = Era.WILDERNESS
        next_era = core_service.get_next_era()
        assert next_era == Era.PRE_INDUSTRIAL.value
        
        core_service.current_era = Era.RUINS
        next_era = core_service.get_next_era()
        assert next_era == Era.WILDERNESS.value
    
    async def test_era_transition(self, core_service):
        """Test era transition functionality."""
        initial_era = core_service.current_era
        success = await core_service.transition_to_era(Era.PRE_INDUSTRIAL.value)
        
        assert success
        assert core_service.current_era == Era.PRE_INDUSTRIAL
        assert core_service.era_progression_timer == 0.0
        core_service.publish_message.assert_called_once()
    
    def test_biome_selection(self, core_service):
        """Test biome selection for different eras."""
        # Test wilderness biome selection
        biome = core_service.select_biome_for_era(Era.WILDERNESS.value)
        assert biome in [b.value for b in ERA_BIOMES[Era.WILDERNESS]]
        
        # Test dystopia biome selection (limited options)
        biome = core_service.select_biome_for_era(Era.DYSTOPIA.value)
        assert biome in [b.value for b in ERA_BIOMES[Era.DYSTOPIA]]


class TestInteractionScoring:
    """Test user interaction scoring and idle management."""
    
    def test_interaction_score_calculation(self, core_service):
        """Test interaction score calculation and decay."""
        initial_score = core_service.user_interaction_score
        
        # Test score increase
        core_service.calculate_interaction_score(0.8)
        assert core_service.user_interaction_score > initial_score
        
        # Test score bounds [0, 1]
        core_service.calculate_interaction_score(2.0)  # Over limit
        assert core_service.user_interaction_score <= 1.0
        
        core_service.calculate_interaction_score(-1.0)  # Under limit
        assert core_service.user_interaction_score >= 0.0
    
    def test_idle_timer_management(self, core_service):
        """Test idle timer updates and reset functionality."""
        # Test idle timer increment
        core_service.update_idle_timer(5.0)
        assert core_service.idle_timer == 5.0
        
        # Test idle timer reset on interaction
        core_service.calculate_interaction_score(0.5)
        assert core_service.idle_timer == 0.0
    
    def test_wilderness_reset_condition(self, core_service):
        """Test wilderness reset condition logic."""
        # Set up non-wilderness state
        core_service.current_era = Era.MODERN
        core_service.idle_timer = 15.0  # Above threshold (10.0)
        
        assert core_service.should_reset_to_wilderness()
        
        # Test wilderness state doesn't reset
        core_service.current_era = Era.WILDERNESS
        assert not core_service.should_reset_to_wilderness()
    
    async def test_wilderness_reset(self, core_service):
        """Test wilderness reset functionality."""
        # Set up modern era state
        core_service.current_era = Era.MODERN
        core_service.user_interaction_score = 0.8
        core_service.idle_timer = 15.0
        
        await core_service.reset_to_wilderness()
        
        assert core_service.current_era == Era.WILDERNESS
        assert core_service.current_biome == Biome.TEMPERATE_FOREST
        assert core_service.idle_timer == 0.0
        assert core_service.user_interaction_score == 0.0
        assert not core_service.audience_present


class TestEventPublishing:
    """Test event publishing functionality."""
    
    async def test_era_changed_event(self, core_service):
        """Test era changed event publishing."""
        await core_service._publish_era_changed_event(Era.WILDERNESS.value, Era.PRE_INDUSTRIAL.value)
        
        core_service.publish_message.assert_called_once()
        call_args = core_service.publish_message.call_args[0][0]
        assert call_args["type"] == "EraChanged"
        assert call_args["old_era"] == Era.WILDERNESS.value
        assert call_args["new_era"] == Era.PRE_INDUSTRIAL.value
    
    async def test_interaction_sound_event(self, core_service):
        """Test interaction sound event publishing."""
        await core_service._publish_interaction_sound(True)
        
        core_service.publish_message.assert_called_once()
        call_args = core_service.publish_message.call_args[0][0]
        assert call_args["type"] == "AudioCommand"
        assert call_args["trigger"] == "interaction_start"
        assert call_args["hand_detected"] == True
    
    async def test_video_mask_event(self, core_service):
        """Test video mask event publishing."""
        core_service.user_interaction_score = 0.7
        core_service.depth_difference_score = 0.5
        core_service.hand_detected = True
        
        await core_service._publish_video_mask()
        
        core_service.publish_message.assert_called_once()
        call_args = core_service.publish_message.call_args[0][0]
        assert call_args["type"] == "VideoMask"
        assert call_args["interaction_score"] == 0.7
        assert call_args["hand_detected"] == True
    
    async def test_idle_state_event(self, core_service):
        """Test idle state event publishing."""
        core_service.idle_timer = 25.5
        
        await core_service._publish_idle_state_changed()
        
        core_service.publish_message.assert_called_once()
        call_args = core_service.publish_message.call_args[0][0]
        assert call_args["type"] == "IdleStatus"
        assert call_args["idle_duration"] == 25.5
    
    async def test_render_request_event(self, core_service):
        """Test render request event publishing."""
        core_service.current_era = Era.MODERN
        core_service.current_biome = Biome.COASTAL
        core_service.user_interaction_score = 0.8
        
        await core_service._publish_render_request()
        
        core_service.publish_message.assert_called_once()
        call_args = core_service.publish_message.call_args[0][0]
        assert call_args["type"] == "RenderRequest"
        assert call_args["current_era"] == Era.MODERN
        assert call_args["current_biome"] == Biome.COASTAL


class TestStatePersistence:
    """Test state saving and loading functionality."""
    
    def test_state_saving(self, core_service):
        """Test state serialization."""
        # Set up some state
        core_service.current_era = Era.MODERN
        core_service.current_biome = Biome.DESERT
        core_service.user_interaction_score = 0.6
        core_service.idle_timer = 12.5
        core_service.audience_present = True
        
        state = core_service.save_state()
        
        assert state["current_era"] == Era.MODERN
        assert state["current_biome"] == Biome.DESERT
        assert state["user_interaction_score"] == 0.6
        assert state["idle_timer"] == 12.5
        assert state["audience_present"] == True
        assert "session_start_time" in state
    
    def test_state_loading(self, core_service):
        """Test state deserialization."""
        state_data = {
            "current_era": "modern",
            "current_biome": "desert",
            "user_interaction_score": 0.7,
            "idle_timer": 8.0,
            "audience_present": True,
            "era_progression_timer": 15.0,
            "session_start_time": "2025-06-13T21:00:00"
        }
        
        core_service.load_state(state_data)
        
        assert core_service.current_era == Era.MODERN
        assert core_service.current_biome == Biome.DESERT
        assert core_service.user_interaction_score == 0.7
        assert core_service.idle_timer == 8.0
        assert core_service.audience_present == True


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    async def test_message_publishing_error_handling(self, core_service):
        """Test that message publishing errors are handled gracefully."""
        # Mock publish_message to raise an exception
        core_service.publish_message.side_effect = Exception("ZMQ Error")
        
        # Should not raise exception
        await core_service._publish_era_changed_event(Era.WILDERNESS.value, Era.PRE_INDUSTRIAL.value)
        
        # Error should be logged but not propagated
        core_service.publish_message.assert_called_once()
    
    def test_state_validation_and_correction(self, core_service):
        """Test state validation and correction."""
        # Set invalid state
        core_service.current_era = "invalid_era"
        core_service.current_biome = "invalid_biome"
        
        core_service.validate_and_correct_state()
        
        # Should be corrected to valid defaults
        assert core_service.is_valid_era(core_service.current_era)
        assert core_service.is_valid_biome(core_service.current_biome)


class TestDepthProcessingIntegration:
    """Test depth processing integration."""
    
    def test_depth_factory_creation(self, core_service):
        """Test depth generator factory creation."""
        factory = core_service._create_depth_generator_factory()
        assert callable(factory)
        
        # Test that factory function includes configuration
        # We can't actually call it without hardware, but we can test the structure
    
    async def test_process_depth_frame(self, core_service):
        """Test depth frame processing logic."""
        import numpy as np
        
        # Create mock depth images
        depth_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        core_service.previous_depth_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Mock detect_difference to return a known value
        with patch('experimance_core.experimance_core.detect_difference', return_value=(1000, None)):
            await core_service._process_depth_frame(depth_image, True)
        
        # Check that hand detection state was updated
        assert core_service.hand_detected == True
        assert core_service.previous_depth_image is not None
        assert core_service.last_depth_map is not None
        
        # Should have published interaction sound
        core_service.publish_message.assert_called()


@pytest.mark.asyncio
async def test_service_lifecycle():
    """Test complete service lifecycle with proper patterns."""
    # Mock ZMQ and depth initialization to avoid hardware dependencies
    with patch('experimance_core.experimance_core.ZmqPublisherSubscriberService.__init__', return_value=None), \
         patch('experimance_core.experimance_core.ZmqPublisherSubscriberService.start') as mock_start, \
         patch('experimance_core.experimance_core.ZmqPublisherSubscriberService.stop') as mock_stop:
        
        service = ExperimanceCoreService()
        
        # Mock required methods
        service.publish_message = AsyncMock()
        service.add_task = MagicMock()
        service._sleep_if_running = AsyncMock(return_value=False)
        service.record_error = MagicMock()
        
        # Mock depth initialization to avoid hardware dependency
        with patch.object(service, '_initialize_depth_processing', return_value=None):
            # Use the active_service context manager for proper lifecycle management
            async with active_service(service, target_state=ServiceState.INITIALIZED) as active:
                # Verify the service is properly initialized
                assert active.config.experimance_core.name == "experimance_core_dev"
                assert active.current_era == Era.WILDERNESS
                assert active.current_biome == Biome.TEMPERATE_FOREST
                
                # Verify mock methods were called
                mock_start.assert_called_once()
        
        # Service should be stopped after context exit
        mock_stop.assert_called_once()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
