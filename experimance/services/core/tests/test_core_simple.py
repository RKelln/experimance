#!/usr/bin/env python3
"""
Simple unit tests for the Experimance Core Service functionality.

This test suite focuses on testing the core logic without service lifecycle complexity.
"""
import pytest
import sys
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experimance_core.experimance_core import ExperimanceCoreService, ERA_PROGRESSION, ERA_BIOMES
from experimance_common.schemas import Era, Biome


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
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        f.flush()
        yield f.name
    
    Path(f.name).unlink()


@pytest.fixture
def mock_service(mock_config_file):
    """Create a mock service for testing."""
    with patch('experimance_core.experimance_core.ZmqPublisherSubscriberService.__init__', return_value=None):
        service = ExperimanceCoreService(config_path=mock_config_file)
        service.publish_message = AsyncMock()
        return service


class TestStateMachineLogic:
    """Test state machine logic without service lifecycle."""
    
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
    
    def test_era_validation(self, mock_service):
        """Test era validation methods."""
        assert mock_service.is_valid_era("wilderness")
        assert mock_service.is_valid_era("modern")
        assert not mock_service.is_valid_era("invalid_era")
    
    def test_biome_validation(self, mock_service):
        """Test biome validation methods."""
        assert mock_service.is_valid_biome("temperate_forest")
        assert mock_service.is_valid_biome("desert")
        assert not mock_service.is_valid_biome("invalid_biome")
    
    def test_next_era_calculation(self, mock_service):
        """Test next era calculation logic."""
        mock_service.current_era = Era.WILDERNESS
        next_era = mock_service.get_next_era()
        assert next_era == Era.PRE_INDUSTRIAL.value
        
        mock_service.current_era = Era.RUINS
        next_era = mock_service.get_next_era()
        assert next_era == Era.WILDERNESS.value
    
    def test_biome_selection(self, mock_service):
        """Test biome selection for different eras."""
        # Test wilderness biome selection
        biome = mock_service.select_biome_for_era(Era.WILDERNESS.value)
        assert biome in [b.value for b in ERA_BIOMES[Era.WILDERNESS]]
        
        # Test dystopia biome selection (limited options)
        biome = mock_service.select_biome_for_era(Era.DYSTOPIA.value)
        assert biome in [b.value for b in ERA_BIOMES[Era.DYSTOPIA]]


class TestInteractionScoring:
    """Test user interaction scoring and idle management."""
    
    def test_interaction_score_calculation(self, mock_service):
        """Test interaction score calculation and decay."""
        initial_score = mock_service.user_interaction_score
        
        # Test score increase
        mock_service.calculate_interaction_score(0.8)
        assert mock_service.user_interaction_score > initial_score
        
        # Test score bounds [0, 1]
        mock_service.calculate_interaction_score(2.0)  # Over limit
        assert mock_service.user_interaction_score <= 1.0
        
        mock_service.calculate_interaction_score(-1.0)  # Under limit
        assert mock_service.user_interaction_score >= 0.0
    
    def test_idle_timer_management(self, mock_service):
        """Test idle timer updates and reset functionality."""
        # Test idle timer increment
        mock_service.update_idle_timer(5.0)
        assert mock_service.idle_timer == 5.0
        
        # Test idle timer reset on interaction
        mock_service.calculate_interaction_score(0.5)
        assert mock_service.idle_timer == 0.0
    
    def test_wilderness_reset_condition(self, mock_service):
        """Test wilderness reset condition logic."""
        # Set up non-wilderness state
        mock_service.current_era = Era.MODERN
        mock_service.idle_timer = 15.0  # Above threshold (10.0)
        
        assert mock_service.should_reset_to_wilderness()
        
        # Test wilderness state doesn't reset
        mock_service.current_era = Era.WILDERNESS
        assert not mock_service.should_reset_to_wilderness()


class TestStatePersistence:
    """Test state saving and loading functionality."""
    
    def test_state_saving(self, mock_service):
        """Test state serialization."""
        # Set up some state
        mock_service.current_era = Era.MODERN
        mock_service.current_biome = Biome.DESERT
        mock_service.user_interaction_score = 0.6
        mock_service.idle_timer = 12.5
        mock_service.audience_present = True
        
        state = mock_service.save_state()
        
        assert state["current_era"] == Era.MODERN
        assert state["current_biome"] == Biome.DESERT
        assert state["user_interaction_score"] == 0.6
        assert state["idle_timer"] == 12.5
        assert state["audience_present"] == True
        assert "session_start_time" in state
    
    def test_state_loading(self, mock_service):
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
        
        mock_service.load_state(state_data)
        
        assert mock_service.current_era == Era.MODERN
        assert mock_service.current_biome == Biome.DESERT
        assert mock_service.user_interaction_score == 0.7
        assert mock_service.idle_timer == 8.0
        assert mock_service.audience_present == True


class TestEventPublishing:
    """Test event publishing functionality."""
    
    @pytest.mark.asyncio
    async def test_era_changed_event(self, mock_service):
        """Test era changed event publishing."""
        await mock_service._publish_era_changed_event(Era.WILDERNESS.value, Era.PRE_INDUSTRIAL.value)
        
        mock_service.publish_message.assert_called_once()
        call_args = mock_service.publish_message.call_args[0][0]
        assert call_args["type"] == "EraChanged"
        assert call_args["old_era"] == Era.WILDERNESS.value
        assert call_args["new_era"] == Era.PRE_INDUSTRIAL.value
    
    @pytest.mark.asyncio
    async def test_interaction_sound_event(self, mock_service):
        """Test interaction sound event publishing."""
        await mock_service._publish_interaction_sound(True)
        
        mock_service.publish_message.assert_called_once()
        call_args = mock_service.publish_message.call_args[0][0]
        assert call_args["type"] == "AudioCommand"
        assert call_args["trigger"] == "interaction_start"
        assert call_args["hand_detected"] == True
    
    @pytest.mark.asyncio
    async def test_video_mask_event(self, mock_service):
        """Test video mask event publishing."""
        mock_service.user_interaction_score = 0.7
        mock_service.depth_difference_score = 0.5
        mock_service.hand_detected = True
        
        await mock_service._publish_video_mask()
        
        mock_service.publish_message.assert_called_once()
        call_args = mock_service.publish_message.call_args[0][0]
        assert call_args["type"] == "VideoMask"
        assert call_args["interaction_score"] == 0.7
        assert call_args["hand_detected"] == True
    
    @pytest.mark.asyncio
    async def test_render_request_event(self, mock_service):
        """Test render request event publishing."""
        mock_service.current_era = Era.MODERN
        mock_service.current_biome = Biome.COASTAL
        mock_service.user_interaction_score = 0.8
        
        await mock_service._publish_render_request()
        
        mock_service.publish_message.assert_called_once()
        call_args = mock_service.publish_message.call_args[0][0]
        assert call_args["type"] == "RenderRequest"
        assert call_args["current_era"] == Era.MODERN
        assert call_args["current_biome"] == Biome.COASTAL


class TestDepthProcessingIntegration:
    """Test depth processing integration."""
    
    def test_depth_factory_creation(self, mock_service):
        """Test depth generator factory creation."""
        factory = mock_service._create_depth_generator_factory()
        assert callable(factory)
    
    @pytest.mark.asyncio
    async def test_process_depth_frame(self, mock_service):
        """Test depth frame processing logic."""
        # Create mock depth images
        depth_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_service.previous_depth_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Mock detect_difference to return a known value
        with patch('experimance_core.experimance_core.detect_difference', return_value=(1000, None)):
            await mock_service._process_depth_frame(depth_image, True)
        
        # Check that hand detection state was updated
        assert mock_service.hand_detected == True
        assert mock_service.previous_depth_image is not None
        assert mock_service.last_depth_map is not None
        
        # Should have published interaction sound
        mock_service.publish_message.assert_called()


class TestConfigurationValidation:
    """Test configuration loading and validation."""
    
    def test_service_initialization(self, mock_service):
        """Test that service initializes with correct configuration."""
        assert mock_service.config.experimance_core.name == "test_core"
        assert mock_service.current_era == Era.WILDERNESS
        assert mock_service.current_biome == Biome.TEMPERATE_FOREST
        assert mock_service.user_interaction_score == 0.0
        assert mock_service.idle_timer == 0.0
        assert not mock_service.audience_present
        assert not mock_service.hand_detected
    
    def test_depth_processing_configuration(self, mock_service):
        """Test depth processing configuration."""
        assert mock_service.config.state_machine.idle_timeout == 10.0
        assert mock_service.config.state_machine.interaction_threshold == 0.5
        assert mock_service.config.depth_processing.change_threshold == 25
        assert mock_service.config.depth_processing.resolution == (640, 480)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
