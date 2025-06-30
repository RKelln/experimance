#!/usr/bin/env python3
"""
Tests for Core Service IMAGE_READY message handling and DISPLAY_MEDIA generation.

Tests the new flow:
1. Core receives IMAGE_READY from image server
2. Core decides whether transitions are needed
3. Core sends DISPLAY_MEDIA to display service
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, Mock
from datetime import datetime

from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig
from experimance_common.schemas import Era, Biome, ContentType, MessageType
from experimance_common.zmq.config import ControllerServiceConfig, PublisherConfig, SubscriberConfig


class TestCoreImageReadyHandling:
    """Test Core Service handling of IMAGE_READY messages."""
    
    @pytest.fixture
    async def service(self):
        """Create a test core service instance."""
        # Create minimal config for testing
        mock_config = Mock(spec=CoreServiceConfig)
        mock_config.service_name = "test_core"
        mock_config.experimance_core = Mock()
        mock_config.experimance_core.name = "test_core"
        mock_config.experimance_core.heartbeat_interval = 5.0
        mock_config.experimance_core.change_smoothing_queue_size = 5
        mock_config.state_machine = Mock()
        mock_config.state_machine.idle_timeout = 300.0
        mock_config.state_machine.interaction_threshold = 0.5
        mock_config.state_machine.era_min_duration = 60.0
        mock_config.visualize = False
        mock_config.camera = Mock()
        mock_config.camera.debug_mode = False
        
        # Add ZMQ config for new architecture
        mock_config.zmq = ControllerServiceConfig(
            name="test_zmq",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=[]),
            workers={}
        )
        
        # Create mock ZMQ service
        mock_zmq_service = Mock()
        mock_zmq_service.start = AsyncMock()
        mock_zmq_service.stop = AsyncMock()
        mock_zmq_service.publish = AsyncMock()
        mock_zmq_service.send_work_to_worker = AsyncMock()
        mock_zmq_service.add_message_handler = Mock()
        mock_zmq_service.add_response_handler = Mock()
        
        # Patch ControllerService 
        with patch('experimance_core.experimance_core.ControllerService', return_value=mock_zmq_service):
            service = ExperimanceCoreService(config=mock_config)
            
            # Mock essential methods
            service.add_task = Mock()
            service.record_error = Mock()
            service._sleep_if_running = AsyncMock(return_value=False)
            
            # Store mock for test access
            service._mock_zmq_service = mock_zmq_service # type: ignore
            
            return service
    
    @pytest.mark.asyncio
    async def test_handle_image_ready_no_transition_needed(self, service):
        """Test IMAGE_READY handling when no transition is needed."""
        # Create mock IMAGE_READY message
        image_ready_message = {
            "type": "ImageReady",
            "request_id": "test_render_123",
            "uri": "file:///tmp/generated_image.png"
        }
        
        # Mock should_request_transition to return False
        with patch.object(service, '_should_request_transition', return_value=False):
            # Call the handler
            await service._handle_image_ready(image_ready_message)
        
        # Verify DISPLAY_MEDIA message was published using new ZMQ service
        service._mock_zmq_service.publish.assert_called_once()
        
        # Get the published message from keyword arguments
        call_args = service._mock_zmq_service.publish.call_args
        published_message = call_args.kwargs['data'] if call_args.kwargs else call_args[0][0]
        
        # Check DISPLAY_MEDIA message structure
        assert published_message["type"] == MessageType.DISPLAY_MEDIA
        assert published_message["content_type"] == ContentType.IMAGE
        assert published_message["request_id"] == "test_render_123"
        assert published_message["uri"] == "file:///tmp/generated_image.png"
        assert "transition_type" not in published_message  # No transition
    
    @pytest.mark.asyncio
    async def test_handle_image_ready_with_transition(self, service):
        """Test IMAGE_READY handling when transition is needed."""
        # Simulate era change to trigger transition
        service._last_era = Era.WILDERNESS
        service.current_era = Era.PRE_INDUSTRIAL
        
        image_ready_message = {
            "type": "ImageReady",
            "request_id": "test_render_456",
            "uri": "file:///tmp/generated_image.png"
        }
        
        # Mock should_request_transition to return False
        with patch.object(service, '_should_request_transition', return_value=True):
            # Call the handler
            await service._handle_image_ready(image_ready_message)
        
        # Verify DISPLAY_MEDIA message was published with transition
        service._mock_zmq_service.publish.assert_called_once()
        
        # Get the published message from keyword arguments
        call_args = service._mock_zmq_service.publish.call_args
        published_message = call_args.kwargs['data'] if call_args.kwargs else call_args[0][0]
        
        assert published_message["type"] == MessageType.DISPLAY_MEDIA.value
        assert published_message["content_type"] == ContentType.IMAGE.value
        assert "transition_type" in published_message
        assert "transition_duration" in published_message
        assert published_message["era"] == Era.PRE_INDUSTRIAL.value
    
    @pytest.mark.asyncio
    async def test_get_transition_type_era_based(self, service):
        """Test transition type selection based on era."""
        # Test different eras get different transition types
        era_transitions = [
            (Era.WILDERNESS, "fade"),
            (Era.MODERN, "slide"),
            (Era.FUTURE, "morph"),
            (Era.DYSTOPIA, "fade")
        ]
        
        for era, expected_transition in era_transitions:
            service._last_era = Era.WILDERNESS  # Different from current
            service.current_era = era
            
            transition_type = service._get_transition_type()
            assert transition_type == expected_transition
    
    @pytest.mark.asyncio
    async def test_send_display_media_copies_image_fields(self, service):
        """Test that _send_display_media copies all relevant image fields."""
        from experimance_common.schemas import ImageReady
        
        image_message = ImageReady(
            request_id="test_123",
            uri="file:///tmp/test.png"
        )
        
        await service._send_display_media(image_message, transition_type="fade")
        
        # Verify all image fields were copied
        service._mock_zmq_service.publish.assert_called_once()
        
        # Get the published message from keyword arguments
        call_args = service._mock_zmq_service.publish.call_args
        published_message = call_args.kwargs['data'] if call_args.kwargs else call_args[0][0]
        
        assert published_message["uri"] == image_message.uri
        assert published_message["request_id"] == image_message.request_id
        assert published_message["transition_type"] == "fade"
    
    @pytest.mark.asyncio
    async def test_handle_image_ready_missing_request_id(self, service):
        """Test graceful handling of malformed IMAGE_READY messages."""
        # Message without request_id
        bad_message = {
            "type": "ImageReady",
            "uri": "file:///tmp/test.png"
        }
        
        # Should not raise exception, but also should not publish
        await service._handle_image_ready(bad_message)
        
        # Should not have published anything
        service._mock_zmq_service.publish.assert_not_called()
    
