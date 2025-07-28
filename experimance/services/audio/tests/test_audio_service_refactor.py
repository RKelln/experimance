"""
Test suite for the refactored Audio Service using the new ZMQ architecture.

This tests the new BaseService + PubSubService composition pattern.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from experimance_common.zmq.mocks import MockPubSubService, mock_message_bus
from experimance_common.schemas import MessageType, SpeechDetected
from experimance_common.service_state import ServiceState

from experimance_audio.audio_service import AudioService
from experimance_audio.config import AudioServiceConfig, SuperColliderConfig


@pytest.fixture
def mock_osc_bridge():
    """Mock OSC bridge to avoid SuperCollider dependencies in tests."""
    with patch('experimance_audio.audio_service.OscBridge') as mock_osc:
        mock_instance = Mock()
        mock_instance.start_supercollider.return_value = True
        mock_instance.stop_supercollider.return_value = True
        mock_osc.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_config_loader():
    """Mock config loader to avoid file dependencies in tests."""
    with patch('experimance_audio.audio_service.AudioConfigLoader') as mock_loader:
        mock_instance = Mock()
        mock_instance.load_configs.return_value = True
        mock_loader.return_value = mock_instance
        yield mock_instance


@pytest.fixture
async def audio_service(mock_osc_bridge, mock_config_loader):
    """Create an audio service for testing."""
    # Clear global message bus
    mock_message_bus.clear()
    
    # Create test config with SuperCollider disabled
    test_config = AudioServiceConfig(
        service_name="test-audio-service",
        supercollider=SuperColliderConfig(auto_start=False)
    )
    
    service = AudioService(config=test_config)
    yield service
    
    # Clean up
    if service.state not in [ServiceState.STOPPED, ServiceState.INITIALIZED]:
        await service.stop()


class TestAudioServiceRefactor:
    """Test the refactored audio service."""
    
    async def test_service_initialization(self, audio_service):
        """Test that the service initializes correctly with new architecture."""
        assert audio_service.service_name == "test-audio-service"
        assert audio_service.service_type == "audio"
        assert audio_service.state == ServiceState.INITIALIZED
        assert hasattr(audio_service, 'zmq_service')
        assert hasattr(audio_service, 'zmq_config')
    
    async def test_service_lifecycle(self, audio_service):
        """Test basic service lifecycle."""
        # Test start
        await audio_service.start()
        assert audio_service.state == ServiceState.STARTED  # BaseService uses STARTED, not RUNNING
        
        # Test stop
        await audio_service.stop()
        assert audio_service.state == ServiceState.STOPPED
    
    async def test_zmq_configuration(self, audio_service):
        """Test that ZMQ is configured correctly."""
        config = audio_service.zmq_config
        
        # Should be subscriber only (no publisher)
        assert config.publisher is None
        assert config.subscriber is not None
        
        # Should subscribe to correct topics
        expected_topics = [
            MessageType.SPACE_TIME_UPDATE,
            MessageType.IDLE_STATUS,
            MessageType.SPEECH_DETECTED,
        ]
        assert set(config.subscriber.topics) == set(expected_topics)
    
    async def test_era_changed_handler(self, audio_service, mock_osc_bridge):
        """Test ERA_CHANGED message handling."""
        await audio_service.start()
        
        # Simulate ERA_CHANGED message
        era_message = {
            "era": "CURRENT",
            "biome": "RAINFOREST"
        }
        
        await audio_service._handle_era_changed(era_message)
        
        # Verify state was updated
        assert audio_service.current_era == "CURRENT"
        assert audio_service.current_biome == "RAINFOREST"
        
        # Verify OSC calls were made
        mock_osc_bridge.send_spacetime.assert_called_once_with("RAINFOREST", "CURRENT")
        mock_osc_bridge.transition.assert_called()
        
        await audio_service.stop()
    
    async def test_idle_status_handler(self, audio_service):
        """Test IDLE_STATUS message handling."""
        await audio_service.start()
        
        # Simulate IDLE_STATUS message
        idle_message = {"status": True}
        
        await audio_service._handle_idle_status(idle_message)
        
        # Handler should complete without error
        # (Currently no specific action for idle status)
        
        await audio_service.stop()

    async def test_speech_detected_handler(self, audio_service, mock_osc_bridge):
        """Test SPEECH_DETECTED message handling."""
        await audio_service.start()
        
        # Simulate SPEECH_DETECTED message
        message = SpeechDetected(
            type=MessageType.SPEECH_DETECTED,
            is_speaking=True,
            speaker="agent"
        )

        await audio_service._handle_speech_detected(message)

        # Verify OSC call was made
        mock_osc_bridge.speaking.assert_called_once_with(True)
        
        await audio_service.stop()
    
    async def test_message_data_type_handling(self, audio_service):
        """Test that handlers work with both dict and MessageBase types."""
        await audio_service.start()
        
        # Test with dict
        dict_message = {"era": "FUTURE", "biome": "ARCTIC"}
        await audio_service._handle_era_changed(dict_message)
        assert audio_service.current_era == "FUTURE"
        
        # Test with MessageBase-like object - inherit from MessageBase for proper type checking
        from experimance_common.schemas import MessageBase
        
        class MockEraChanged(MessageBase):
            type: str = "ERA_CHANGED"  # Required field
            
            def model_dump(self):
                return {"era": "DYSTOPIA", "biome": "DESERT"}
        
        message_base = MockEraChanged()
        await audio_service._handle_era_changed(message_base)
        assert audio_service.current_era == "DYSTOPIA"
        
        await audio_service.stop()
    
    async def test_invalid_message_handling(self, audio_service):
        """Test handling of invalid or incomplete messages."""
        await audio_service.start()
        
        # Test incomplete ERA_CHANGED message
        incomplete_message = {"era": "CURRENT"}  # Missing biome
        await audio_service._handle_era_changed(incomplete_message)
        
        # Should not update state
        assert audio_service.current_era != "CURRENT"
        
        # Test invalid message type
        invalid_message = "not a dict or messagebase"
        await audio_service._handle_era_changed(invalid_message)
        
        await audio_service.stop()


@pytest.mark.integration
class TestAudioServiceIntegration:
    """Integration tests using the mock ZMQ system."""
    
    async def test_end_to_end_message_flow(self, mock_osc_bridge, mock_config_loader):
        """Test full message flow using mock ZMQ bus."""
        mock_message_bus.clear()
        
        # Create service with proper config
        test_config = AudioServiceConfig(
            service_name="integration-test-audio",
            supercollider=SuperColliderConfig(auto_start=False)
        )
        service = AudioService(config=test_config)
        
        try:
            await service.start()
            
            # Simulate publishing an ERA_CHANGED message to the mock bus
            era_message = {
                "era": "MODERN",
                "biome": "COASTAL"
            }
            
            # Directly call handler to simulate message reception
            await service._handle_space_time_update(era_message)
            
            # Verify the service processed the message
            assert service.current_era == "MODERN"
            assert service.current_biome == "COASTAL"
            
            # Verify OSC bridge was called
            mock_osc_bridge.send_spacetime.assert_called_with("COASTAL", "MODERN")
            
        finally:
            await service.stop()
