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
from experimance_common.schemas import Era, Biome, ContentType
from experimance_common.zmq.zmq_utils import MessageType


class TestCoreImageReadyHandling:
    """Test Core Service handling of IMAGE_READY messages."""
    
    @pytest.fixture
    async def service(self):
        """Create a test core service instance."""
        # Use the reusable mock from mocks.py
        from .mocks import create_mock_core_service
        
        # Create service with custom config overrides for this test
        service = create_mock_core_service(config_overrides={
            "experimance_core": {
                "name": "test_core",
                "heartbeat_interval": 5.0,
                "change_smoothing_queue_size": 5
            },
            "state_machine": {
                "idle_timeout": 300.0,
                "interaction_threshold": 0.5,
                "era_min_duration": 60.0
            },
            "camera": {
                "debug_mode": False
            },
            "visualize": False
        })
        
        return service
    
    @pytest.mark.asyncio
    async def test_handle_image_ready_no_transition_needed(self, service):
        """Test IMAGE_READY handling when no transition is needed."""
        # Create mock IMAGE_READY message
        image_ready_message = {
            "type": MessageType.IMAGE_READY.value,
            "request_id": "test_render_123",
            "uri": "file:///tmp/generated_image.png",
            "image_format": "PNG",
            "timestamp": datetime.now().isoformat()
        }
        
        # Mock should_request_transition to return False
        with patch.object(service, '_should_request_transition', return_value=False):
            # Call the handler
            await service._handle_image_ready(image_ready_message)
        
        # Verify DISPLAY_MEDIA message was published
        service.publish_message.assert_called_once()
        published_message = service.publish_message.call_args[0][0]
        
        # Check DISPLAY_MEDIA message structure
        assert published_message["type"] == MessageType.DISPLAY_MEDIA.value
        assert published_message["content_type"] == ContentType.IMAGE.value
        assert published_message["request_id"] == "test_render_123"
        assert published_message["uri"] == "file:///tmp/generated_image.png"
        assert published_message["era"] == service.current_era.value
        assert published_message["biome"] == service.current_biome.value
        assert "transition_type" not in published_message  # No transition
    
    @pytest.mark.asyncio
    async def test_handle_image_ready_with_transition(self, service):
        """Test IMAGE_READY handling when transition is needed."""
        # Simulate era change to trigger transition
        service._last_era = Era.WILDERNESS
        service.current_era = Era.PRE_INDUSTRIAL
        
        image_ready_message = {
            "type": MessageType.IMAGE_READY.value,
            "request_id": "test_render_456",
            "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "image_format": "PNG"
        }
        
        # Call the handler - should detect era change and request transition
        await service._handle_image_ready(image_ready_message)
        
        # Verify DISPLAY_MEDIA message was published with transition
        service.publish_message.assert_called_once()
        published_message = service.publish_message.call_args[0][0]
        
        assert published_message["type"] == MessageType.DISPLAY_MEDIA.value
        assert published_message["content_type"] == ContentType.IMAGE.value
        assert "transition_type" in published_message
        assert "transition_duration" in published_message
        assert published_message["era"] == Era.PRE_INDUSTRIAL.value
    
    @pytest.mark.asyncio
    async def test_should_request_transition_era_change(self, service):
        """Test transition detection for era changes."""
        # Set up era change scenario
        service._last_era = Era.WILDERNESS
        service.current_era = Era.MODERN
        
        dummy_message = {"request_id": "test"}
        
        result = await service._should_request_transition(dummy_message)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_should_request_transition_high_interaction(self, service):
        """Test transition detection for high interaction scores."""
        # Set high interaction score (above 2x threshold)
        service.user_interaction_score = service.config.state_machine.interaction_threshold * 2.5
        service._last_era = service.current_era  # No era change
        
        dummy_message = {"request_id": "test"}
        
        result = await service._should_request_transition(dummy_message)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_should_request_transition_no_change(self, service):
        """Test transition detection when no significant change."""
        # No era change, low interaction
        service._last_era = service.current_era
        service.user_interaction_score = 0.1
        
        dummy_message = {"request_id": "test"}
        
        result = await service._should_request_transition(dummy_message)
        assert result is False
    
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
        image_message = {
            "request_id": "test_123",
            "uri": "file:///tmp/test.png",
            "image_data": "base64_encoded_data",
            "image_format": "PNG",
            "image_id": "generated_image_456",
            "mask_id": "mask_789"
        }
        
        await service._send_display_media(image_message, transition_type="fade")
        
        # Verify all image fields were copied
        service.publish_message.assert_called_once()
        published_message = service.publish_message.call_args[0][0]
        
        assert published_message["uri"] == image_message["uri"]
        assert published_message["image_data"] == image_message["image_data"]
        assert published_message["image_format"] == image_message["image_format"]
        assert published_message["image_id"] == image_message["image_id"]
        assert published_message["mask_id"] == image_message["mask_id"]
        assert published_message["transition_type"] == "fade"
    
    @pytest.mark.asyncio
    async def test_handle_image_ready_missing_request_id(self, service):
        """Test graceful handling of malformed IMAGE_READY messages."""
        # Message without request_id
        bad_message = {
            "type": MessageType.IMAGE_READY.value,
            "uri": "file:///tmp/test.png"
        }
        
        # Should not raise exception, but also should not publish
        await service._handle_image_ready(bad_message)
        
        # Should not have published anything
        service.publish_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_handle_image_ready_updates_last_era(self, service):
        """Test that handling IMAGE_READY updates era tracking."""
        service.current_era = Era.MODERN
        service._last_era = Era.WILDERNESS  # Different era
        
        image_message = {
            "type": MessageType.IMAGE_READY.value,
            "request_id": "test_era_update",
            "uri": "file:///tmp/test.png"
        }
        
        await service._handle_image_ready(image_message)
        
        # After handling, _last_era should be updated
        assert service._last_era == Era.MODERN
        assert service._last_era == service.current_era
