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

import numpy as np

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experimance_core.experimance_core import ExperimanceCoreService, ERA_PROGRESSION, ERA_BIOMES
from experimance_core.config import CoreServiceConfig
from experimance_common.schemas import Era, Biome
from experimance_common.test_utils import active_service
from experimance_common.service_state import ServiceState

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_config():
    """Create a test configuration for the core service."""
    # Use config overrides instead of TOML files for testing
    config_overrides = {
        "service_name": "test_core",
        "experimance_core": {
            "name": "test_core",
        },
        "state_machine": {
            "idle_timeout": 10.0,
            "wilderness_reset": 60.0,
            "interaction_threshold": 0.5,
            "era_min_duration": 5.0
        },
        "depth_processing": {
            "change_threshold": 25,
            "min_depth": 0.4,
            "max_depth": 0.6,
            "resolution": [640, 480],
            "output_size": [512, 512]
        },
        "audio": {
            "tag_config_path": "config/audio_tags.json",
            "interaction_sound_duration": 1.0
        },
        "prompting": {
            "data_path": "data/",
            "locations_file": "locations.json",
            "developments_file": "anthropocene.json"
        },
        "zmq": {
            "name": "test_zmq",
            "log_level": "DEBUG",
            "timeout": 1.0,
            "publisher": {
                "address": "tcp://*",
                "port": 5555,
                "bind": True
            },
            "subscriber": {
                "address": "tcp://localhost",
                "port": 5556,
                "bind": False,
                "topics": []
            },
            "workers": {}
        },
        "visualize": False  # Disable visualization for tests
    }
    
    return CoreServiceConfig.from_overrides(override_config=config_overrides)


@pytest.fixture
def core_service(test_config):
    """Create a core service instance for testing."""
    from unittest.mock import Mock, AsyncMock, patch
    from experimance_core.experimance_core import ExperimanceCoreService
    
    # Create a mock instance with AsyncMock methods
    mock_zmq_service = Mock()
    mock_zmq_service.start = AsyncMock()
    mock_zmq_service.stop = AsyncMock()
    mock_zmq_service.publish = AsyncMock()
    mock_zmq_service.send_work_to_worker = AsyncMock()
    mock_zmq_service.add_message_handler = Mock()
    mock_zmq_service.add_response_handler = Mock()
    
    # Patch the ControllerService class to return our mock
    patcher = patch('experimance_core.experimance_core.ControllerService', return_value=mock_zmq_service)
    patcher.start()
    
    service = ExperimanceCoreService(config=test_config)
    
    # Mock essential methods
    service.add_task = Mock()
    service.record_error = Mock()
    service._sleep_if_running = AsyncMock(return_value=False)
    
    service.zmq_service = mock_zmq_service

    # Store patcher for cleanup
    service._test_patcher = patcher
    
    yield service
    
    # Cleanup
    patcher.stop()


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
        assert core_service._depth_processor is None
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
        core_service.zmq_service.publish.assert_called_once()
    
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
        """Test interaction score calculation."""
        initial_score = core_service.user_interaction_score
        
        # Test score increase
        core_service.calculate_interaction_score(0.8)
        assert core_service.user_interaction_score > initial_score
        
        # Test that scores accumulate (current implementation for testing)
        score_before = core_service.user_interaction_score
        core_service.calculate_interaction_score(0.5)
        assert core_service.user_interaction_score == score_before + 0.5
        
        # Test that negative values still get added (current implementation)
        score_before = core_service.user_interaction_score
        core_service.calculate_interaction_score(-0.2)
        assert core_service.user_interaction_score == score_before - 0.2
    
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
        
        core_service.zmq_service.publish.assert_called_once()
        call_args = core_service.zmq_service.publish.call_args[0][0]
        assert call_args["type"] == "EraChanged"
        assert call_args["old_era"] == Era.WILDERNESS.value
        assert call_args["new_era"] == Era.PRE_INDUSTRIAL.value
    
    async def test_interaction_sound_event(self, core_service):
        """Test interaction sound event publishing."""
        await core_service._publish_interaction_sound(True)
        
        core_service.zmq_service.publish.assert_called_once()
        call_args = core_service.zmq_service.publish.call_args[0][0]
        assert call_args["type"] == "AudioCommand"
        assert call_args["trigger"] == "interaction_start"
        assert call_args["hand_detected"] == True
    
    async def test_change_map_event(self, core_service):
        """Test change map event publishing."""
        core_service.user_interaction_score = 0.7
        core_service.depth_difference_score = 0.5
        core_service.hand_detected = False

        change_map = np.zeros_like(np.random.randint(0, 255, (512, 512), dtype=np.uint8))
        await core_service._publish_change_map(change_map, change_score=0.7)
        
        core_service.zmq_service.publish.assert_called_once()
        call_args = core_service.zmq_service.publish.call_args[0][0]
        assert call_args["type"] == "ChangeMap"
        assert call_args["change_score"] == 0.7
        assert call_args["has_change_map"] == True
    
    async def test_idle_state_event(self, core_service):
        """Test idle state event publishing."""
        core_service.idle_timer = 25.5
        
        await core_service._publish_idle_state_changed()
        
        core_service.zmq_service.publish.assert_called_once()
        call_args = core_service.zmq_service.publish.call_args[0][0]
        assert call_args["type"] == "IdleStatus"
        assert call_args["idle_duration"] == 25.5
    
    async def test_render_request_event(self, core_service):
        """Test render request event publishing."""
        core_service.current_era = Era.MODERN
        core_service.current_biome = Biome.COASTAL
        core_service.user_interaction_score = 0.8
        
        await core_service._publish_render_request()
        
        # This method uses send_work_to_worker, not publish
        core_service.zmq_service.send_work_to_worker.assert_called_once()
        call_args = core_service.zmq_service.send_work_to_worker.call_args[0]
        worker_name = call_args[0]
        request_data = call_args[1]
        
        assert worker_name == "image_server"
        assert request_data.era == Era.MODERN
        assert request_data.biome == Biome.COASTAL


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
        core_service.zmq_service.publish.side_effect = Exception("ZMQ Error")
        
        # Should not raise exception
        await core_service._publish_era_changed_event(Era.WILDERNESS.value, Era.PRE_INDUSTRIAL.value)
        
        # Error should be logged but not propagated
        core_service.zmq_service.publish.assert_called_once()
    
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
    
    def test_depth_processor_initialization(self, core_service):
        """Test depth processor initialization state."""
        assert core_service._depth_processor is None
        assert core_service._camera_state is not None
        
        # Test that camera config method exists
        camera_config = core_service._create_camera_config()
        assert camera_config is not None
    
    async def test_process_depth_frame(self, core_service):
        """Test depth frame processing logic."""
        from experimance_core.config import DepthFrame
        
        # Create mock depth frame
        depth_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        depth_frame = DepthFrame(
            depth_image=depth_image,
            hand_detected=True,
            timestamp=time.time()
        )
        
        # Set up initial state
        core_service.previous_depth_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Process the frame
        await core_service._process_depth_frame(depth_frame)
        
        # Check that hand detection state was updated
        assert core_service.hand_detected == True
        assert core_service.previous_depth_image is not None
        
        # Should have published interaction sound
        core_service.zmq_service.publish.assert_called()


class TestConfigurationValidation:
    """Test configuration loading and validation (additional tests)."""
    
    def test_depth_processing_configuration(self, core_service):
        """Test depth processing configuration details."""
        assert core_service.config.depth_processing.change_threshold == 25
        assert core_service.config.depth_processing.resolution == (640, 480)
        assert core_service.config.depth_processing.output_size == (512, 512)
        assert core_service.config.depth_processing.min_depth == 0.4
        assert core_service.config.depth_processing.max_depth == 0.6
    
    def test_service_attributes_initialization(self, core_service):
        """Test that all required service attributes are properly initialized."""
        assert hasattr(core_service, 'current_era')
        assert hasattr(core_service, 'current_biome')
        assert hasattr(core_service, 'user_interaction_score')
        assert hasattr(core_service, 'idle_timer')
        assert hasattr(core_service, 'audience_present')
        assert hasattr(core_service, 'hand_detected')
        assert hasattr(core_service, 'depth_difference_score')
        assert hasattr(core_service, '_depth_processor')
        assert hasattr(core_service, 'previous_depth_image')


@pytest.mark.asyncio
async def test_service_basic_functionality(core_service):
    """Test basic service functionality without full lifecycle complexity."""
    # Test basic service properties
    assert core_service.config.experimance_core.name == "test_core"
    assert core_service.current_era == Era.WILDERNESS
    assert core_service.current_biome == Biome.TEMPERATE_FOREST
    
    # Test basic era transition
    success = await core_service.transition_to_era(Era.PRE_INDUSTRIAL.value)
    assert success
    assert core_service.current_era == Era.PRE_INDUSTRIAL
    
    # Test state persistence
    state = core_service.save_state()
    assert state["current_era"] == Era.PRE_INDUSTRIAL
    
    # Test wilderness reset
    await core_service.reset_to_wilderness()
    assert core_service.current_era == Era.WILDERNESS


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
