"""
Tests for Experimance Core Service state management functionality.

Following TDD approach for Phase 2: State Management
"""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig
from experimance_common.schemas import Era, Biome


class TestExperimanceCoreServiceStateManagement:
    """Test state management functionality."""

    @pytest.fixture
    def test_config(self):
        """Create a test configuration for the core service."""
        config_overrides = {
            "experimance_core": {
                "name": "test_experimance_core_state",
            },
            "state_machine": {
                "idle_timeout": 45.0,
                "wilderness_reset": 300.0,
                "interaction_threshold": 0.3,
                "era_min_duration": 10.0
            },
            "depth_processing": {
                "change_threshold": 50,
                "min_depth": 0.49,
                "max_depth": 0.56,
                "resolution": [1280, 720],
                "output_size": [1024, 1024]
            },
            "audio": {
                "tag_config_path": "config/audio_tags.json",
                "interaction_sound_duration": 2.0
            },
            "prompting": {
                "data_path": "data/",
                "locations_file": "locations.json",
                "developments_file": "anthropocene.json"
            },
            "visualize": False
        }
        
        return CoreServiceConfig.from_overrides(override_config=config_overrides)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(config_data, f)
            return f.name

    def test_service_has_era_state_data_structures(self, test_config):
        """Test that service defines proper era state data structures."""
        service = ExperimanceCoreService(config=test_config)
        
        # Should have Era enumeration or constants
        assert hasattr(service, 'AVAILABLE_ERAS') or hasattr(service, 'Era')
        
        # Should have era transition logic
        assert hasattr(service, 'transition_to_era')
        assert hasattr(service, 'can_transition_to_era')

    def test_service_has_biome_state_data_structures(self, test_config):
        """Test that service defines proper biome state data structures."""
        service = ExperimanceCoreService(config=test_config)
        
        # Should have Biome enumeration or constants
        assert hasattr(service, 'AVAILABLE_BIOMES') or hasattr(service, 'Biome')
        
        # Should have biome selection logic
        assert hasattr(service, 'select_biome_for_era')

    @pytest.mark.asyncio
    async def test_era_progression_logic(self, test_config):
        """Test era progression state machine logic."""
        service = ExperimanceCoreService(config=test_config)
        
        # Test initial state
        assert service.current_era.value == "wilderness"
        
        # Test progression through eras
        await service.progress_era()
        assert service.current_era.value in ["pre_industrial", "early_industrial", "late_industrial", "modern", "current", "future"]
        
        # Test era can progress to future
        while service.current_era.value != "future":
            await service.progress_era()
        assert service.current_era.value == "future"
        
        # Test future era can loop or progress to dystopia
        next_era = service.get_next_era()
        assert next_era in ["future", "dystopia"]

    @pytest.mark.asyncio
    async def test_idle_timeout_functionality(self, test_config):
        """Test idle timeout and wilderness reset functionality."""
        service = ExperimanceCoreService(config=test_config)
        
        # Set service to a non-wilderness era
        service.current_era = Era.MODERN
        service.idle_timer = 0.0
        
        # Simulate idle time passing
        service.update_idle_timer(50.0)  # Beyond idle_timeout of 45s
        
        # Should trigger wilderness drift
        assert service.should_reset_to_wilderness()
        
        # Apply reset
        await service.reset_to_wilderness()
        assert service.current_era.value == "wilderness"
        assert service.idle_timer == 0.0

    def test_user_interaction_scoring(self, test_config):
        """Test user interaction score calculation."""
        service = ExperimanceCoreService(config=test_config)
        
        # Test initial score
        assert service.user_interaction_score == 0.0
        
        # Test score calculation
        interaction_intensity = 0.8
        service.calculate_interaction_score(interaction_intensity)
        
        # Score should be updated based on intensity
        assert service.user_interaction_score > 0.0
        assert service.user_interaction_score <= 1.0

    @pytest.mark.asyncio
    async def test_era_change_triggers_events(self, test_config):
        """Test that era changes trigger appropriate events."""
        with patch('experimance_core.experimance_core.ControllerService') as mock_zmq_service:
            mock_instance = Mock()
            mock_instance.publish = AsyncMock()
            mock_zmq_service.return_value = mock_instance
            
            service = ExperimanceCoreService(config=test_config)
            
            # Change era
            old_era = service.current_era
            await service.transition_to_era("pre_industrial")
            
            # Should publish EraChanged event
            mock_instance.publish.assert_called()
            call_args = mock_instance.publish.call_args[0][0]
            assert call_args["type"] == "EraChanged"
        assert call_args["old_era"] == old_era.value
        assert call_args["new_era"] == "pre_industrial"

    @pytest.mark.asyncio
    async def test_state_persistence(self, test_config):
        """Test state can be saved and loaded."""
        service = ExperimanceCoreService(config=test_config)
        
        # Change state
        service.current_era = Era.MODERN
        service.current_biome = Biome.DESERT
        service.user_interaction_score = 0.7
        
        # Save state
        state_data = service.save_state()
        assert "current_era" in state_data
        assert "current_biome" in state_data
        assert "user_interaction_score" in state_data
        
        # Reset service state
        service.current_era = Era.WILDERNESS
        service.current_biome = Biome.TEMPERATE_FOREST
        service.user_interaction_score = 0.0
        
        # Load state
        service.load_state(state_data)
        assert service.current_era.value == Era.MODERN
        assert service.current_biome.value == Biome.DESERT
        assert service.user_interaction_score == 0.7

    def test_state_validation_and_recovery(self, test_config):
        """Test state validation and error recovery."""
        service = ExperimanceCoreService(config=test_config)
        
        # Test invalid era
        assert not service.is_valid_era("invalid_era")
        
        # Test state correction - set era directly for testing
        service.current_era = Era.WILDERNESS  # Start with valid state
        # Simulate invalid state by calling validate with invalid data
        state_data = {"current_era": "invalid_era", "current_biome": "temperate_forest"}
        with pytest.raises(ValueError):
            service.load_state(state_data)
        # After load_state, it should be corrected to wilderness
        assert service.current_era.value == "wilderness"  # No longer needed, exception is expected
        
        # Test invalid biome
        assert not service.is_valid_biome("invalid_biome")
        
        # Test biome correction
        state_data = {"current_era": "wilderness", "current_biome": "invalid_biome"}
        with pytest.raises(ValueError):
            service.load_state(state_data)
        # After load_state, biome should be corrected to temperate_forest
        assert service.current_biome.value == "temperate_forest"  # No longer needed, exception is expected


class TestExperimanceCoreServiceStateData:
    """Test state data structures and constants."""
    
    def test_era_definitions(self):
        """Test that eras are properly defined."""
        # This test helps define what eras we need
        expected_eras = ["wilderness", "agricultural", "industrial", "modern", "ai", "post_apocalyptic", "ruins"]
        
        # The service should define these eras
        # This will fail until we implement the Era constants
        pass
    
    def test_biome_definitions(self):
        """Test that biomes are properly defined."""
        # This test helps define what biomes we need
        expected_biomes = ["forest", "desert", "ocean", "mountain", "tundra", "grassland"]
        
        # The service should define these biomes
        # This will fail until we implement the Biome constants
        pass
    
    def test_era_biome_compatibility(self):
        """Test era-biome compatibility matrix."""
        # This test helps define which biomes are available for each era
        # Some biomes might not be available in certain eras (e.g., post-apocalyptic)
        pass


# Cleanup function
def cleanup_temp_file(filepath):
    """Clean up temporary files after tests."""
    try:
        Path(filepath).unlink()
    except FileNotFoundError:
        pass
