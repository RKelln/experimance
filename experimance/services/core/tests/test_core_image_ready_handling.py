#!/usr/bin/env python3
"""
Test Core Service IMAGE_READY handling and DISPLAY_MEDIA functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from experimance_common.schemas import Era, Biome, ContentType, DisplayMedia
from experimance_common.zmq.zmq_utils import MessageType
from experimance_common.zmq.config import ControllerServiceConfig, PublisherConfig, SubscriberConfig
from experimance_common.test_utils import active_service
from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig


class TestCoreServiceImageReadyHandling:
    """Test Core Service handling of IMAGE_READY messages and DISPLAY_MEDIA publishing."""
    
    def create_test_service(self):
        """Create a properly mocked test service."""
        # Create minimal config for testing
        mock_config = Mock(spec=CoreServiceConfig)
        mock_config.service_name = "test_core"
        mock_config.experimance_core = Mock()
        mock_config.experimance_core.name = "test_core"
        mock_config.experimance_core.heartbeat_interval = 1.0
        mock_config.experimance_core.change_smoothing_queue_size = 5
        mock_config.state_machine = Mock()
        mock_config.state_machine.idle_timeout = 300.0
        mock_config.state_machine.interaction_threshold = 0.5
        mock_config.state_machine.era_min_duration = 60.0
        mock_config.visualize = False
        mock_config.camera = Mock()
        mock_config.camera.debug_mode = False
        
        # Add ZMQ config
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
        
        # Patch and create service
        patcher = patch('experimance_core.experimance_core.ControllerService')
        mock_controller_class = patcher.start()
        mock_controller_class.return_value = mock_zmq_service
        
        service = ExperimanceCoreService(config=mock_config)
        
        # Mock essential methods
        service.add_task = Mock()
        service.record_error = Mock()
        service._sleep_if_running = AsyncMock(return_value=False)
        
        # Store patcher and mock for test access
        service._mock_patcher = patcher
        service._mock_zmq_service = mock_zmq_service
        
        return service
    
    @pytest.mark.asyncio
    async def test_image_ready_same_era_direct_display(self):
        """Test IMAGE_READY handling when no transition is needed (same era)."""
        service = self.create_test_service()
        
        try:
            async with active_service(service) as active:
                # Set initial era
                active.current_era = Era.MODERN
                active._last_era = Era.MODERN  # Same era, no transition needed
                
                # Create test IMAGE_READY message
                image_ready_message = {
                    "type": MessageType.IMAGE_READY.value,
                    "uri": "file:///tmp/test_image.png",
                    "request_id": "render_request_456",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Handle the message
                await active._handle_image_ready(image_ready_message)
                
                # Verify DISPLAY_MEDIA was published directly (no transition)
                # Service may publish multiple messages (heartbeats, etc), so check for DISPLAY_MEDIA specifically
                display_media_calls = [
                    call for call in service._mock_zmq_service.publish.call_args_list 
                    if call[0][0].get("type") == MessageType.DISPLAY_MEDIA.value
                ]
                
                assert len(display_media_calls) == 1, f"Expected 1 DISPLAY_MEDIA call, got {len(display_media_calls)}"
                call_args = display_media_calls[0][0][0]  # Get the message argument
                
                assert call_args["type"] == MessageType.DISPLAY_MEDIA.value
                assert call_args["content_type"] == ContentType.IMAGE.value
                assert call_args.get("transition_type") is None  # No transition
                assert call_args["request_id"] == "render_request_456"
                assert "uri" in call_args
        finally:
            service._mock_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_image_ready_era_change_requests_transition(self):
        """Test IMAGE_READY handling when era changes and transition is needed."""
        service = self.create_test_service()
        
        try:
            async with active_service(service) as active:
                # Set up era change scenario
                active.current_era = Era.CURRENT
                active._last_era = Era.MODERN  # Different era, transition needed
                
                # Create test IMAGE_READY message
                image_ready_message = {
                    "type": MessageType.IMAGE_READY.value,
                    "uri": "file:///tmp/era_change.png",
                    "request_id": "render_request_789",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Handle the message
                await active._handle_image_ready(image_ready_message)
                
                # Check for specific DISPLAY_MEDIA calls, ignoring other service messages
                display_media_calls = [
                    call for call in service._mock_zmq_service.publish.call_args_list 
                    if call[0][0].get("type") == MessageType.DISPLAY_MEDIA.value
                ]
                
                assert len(display_media_calls) == 1, f"Expected 1 DISPLAY_MEDIA call, got {len(display_media_calls)}"
                call_args = display_media_calls[0][0][0]
                
                assert call_args["type"] == MessageType.DISPLAY_MEDIA.value
                assert call_args["content_type"] == ContentType.IMAGE.value
                assert call_args.get("transition_type") == "fade"  # Should have transition
                assert call_args["request_id"] == "render_request_789"
        finally:
            service._mock_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_transition_ready_publishes_display_media(self):
        """Test TRANSITION_READY handling publishes DISPLAY_MEDIA with transition."""
        service = self.create_test_service()
        
        try:
            async with active_service(service) as active:
                # Set up pending transition state
                active._pending_transition = {
                    "target_image_id": "final_image",
                    "target_image_uri": "file:///tmp/final.png",
                    "from_era": Era.MODERN.value,
                    "to_era": Era.CURRENT.value
                }
                
                # Create test TRANSITION_READY message
                transition_ready_message = {
                    "type": MessageType.TRANSITION_READY.value,
                    "transition_id": "fade_transition_123",
                    "transition_type": "fade",
                    "sequence_path": "/tmp/transition_sequence/",
                    "duration": 2.5,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Handle the message
                await active._handle_transition_ready(transition_ready_message)
                
                # Verify DISPLAY_MEDIA was published with transition
                service._mock_zmq_service.publish.assert_called_once()
                call_args = service._mock_zmq_service.publish.call_args[0][0]
                
                assert call_args["type"] == MessageType.DISPLAY_MEDIA.value
                assert call_args["content_type"] == ContentType.IMAGE_SEQUENCE.value
                assert call_args["transition_type"] == "fade"
                assert call_args["sequence_path"] == "/tmp/transition_sequence/"
                assert call_args["duration"] == 2.5
                assert call_args["final_image_id"] == "final_image"
                
                # Verify pending transition was cleared
                assert active._pending_transition is None
        finally:
            service._mock_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_display_media_creation_with_single_image(self):
        """Test _create_display_media method for single image content."""
        service = self.create_test_service()
        
        try:
            async with active_service(service) as active:
                # Test single image display media
                display_media = active._create_display_media(
                    content_type=ContentType.IMAGE,
                    request_id="single_test_image",
                    uri="file:///tmp/single.png"
                )
                
                assert display_media["type"] == MessageType.DISPLAY_MEDIA.value
                assert display_media["content_type"] == ContentType.IMAGE.value
                assert display_media["request_id"] == "single_test_image"
                assert display_media["uri"] == "file:///tmp/single.png"
                assert "timestamp" in display_media
        finally:
            if hasattr(service, '_mock_patcher'):
                service._mock_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_display_media_creation_with_image_sequence(self):
        """Test _create_display_media method for image sequence content."""
        service = self.create_test_service()
        
        try:
            async with active_service(service) as active:
                # Test image sequence display media
                display_media = active._create_display_media(
                    content_type=ContentType.IMAGE_SEQUENCE,
                    transition_type="morph",
                    sequence_path="/tmp/morph_sequence/",
                    duration=3.0,
                    final_image_id="morphed_result"
                )
                
                assert display_media["type"] == MessageType.DISPLAY_MEDIA.value
                assert display_media["content_type"] == ContentType.IMAGE_SEQUENCE.value
                assert display_media["transition_type"] == "morph"
                assert display_media["sequence_path"] == "/tmp/morph_sequence/"
                assert display_media["duration"] == 3.0
                assert display_media["final_image_id"] == "morphed_result"
        finally:
            if hasattr(service, '_mock_patcher'):
                service._mock_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_era_change_detection(self):
        """Test _needs_transition method for era change detection."""
        service = self.create_test_service()
        
        try:
            async with active_service(service) as active:
                # Test case 1: Same era, no transition needed
                active.current_era = Era.MODERN
                active._last_era = Era.MODERN
                assert not active._needs_transition()
                
                # Test case 2: Era change, transition needed
                active.current_era = Era.CURRENT
                active._last_era = Era.MODERN
                assert active._needs_transition()
                
                # Test case 3: First image (no last era), no transition needed
                active.current_era = Era.WILDERNESS
                active._last_era = None
                assert not active._needs_transition()
        finally:
            if hasattr(service, '_mock_patcher'):
                service._mock_patcher.stop()
