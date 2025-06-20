#!/usr/bin/env python3
"""
Test for Core Service image publishing with new enum-based utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from experimance_common.zmq.zmq_utils import IMAGE_TRANSPORT_MODES
from experimance_common.zmq.config import ControllerServiceConfig, PublisherConfig, SubscriberConfig
from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig


class TestCoreServiceImagePublishing:
    """Test that core service properly uses the new image utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create minimal config for testing
        self.mock_config = Mock(spec=CoreServiceConfig)
        self.mock_config.service_name = "test_core"  # Required by new architecture
        self.mock_config.experimance_core = Mock()
        self.mock_config.experimance_core.name = "test_core"
        self.mock_config.experimance_core.heartbeat_interval = 5.0
        self.mock_config.experimance_core.change_smoothing_queue_size = 5
        self.mock_config.state_machine = Mock()
        self.mock_config.state_machine.idle_timeout = 300.0
        self.mock_config.state_machine.interaction_threshold = 0.5
        self.mock_config.state_machine.era_min_duration = 60.0
        self.mock_config.visualize = False
        self.mock_config.camera = Mock()
        self.mock_config.camera.debug_mode = False
        
        # Add ZMQ config for new architecture
        self.mock_config.zmq = ControllerServiceConfig(
            name="test_zmq",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=[]),
            workers={}
        )
        
        # Create a mock instance with AsyncMock methods
        self.mock_zmq_service = Mock()
        self.mock_zmq_service.start = AsyncMock()
        self.mock_zmq_service.stop = AsyncMock()
        self.mock_zmq_service.publish = AsyncMock()
        self.mock_zmq_service.send_work_to_worker = AsyncMock()
        self.mock_zmq_service.add_message_handler = Mock()
        self.mock_zmq_service.add_response_handler = Mock()
        
        # Start the patch
        self.controller_patcher = patch('experimance_core.experimance_core.ControllerService')
        mock_controller_class = self.controller_patcher.start()
        mock_controller_class.return_value = self.mock_zmq_service
        
        self.service = ExperimanceCoreService(config=self.mock_config)
        
        # Mock essential methods
        self.service.add_task = Mock()
        self.service.record_error = Mock()
        self.service._sleep_if_running = AsyncMock(return_value=False)
        # Don't mock _publish_change_map since we want to test it
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self, 'controller_patcher'):
            self.controller_patcher.stop()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Stop any mock patchers
        if hasattr(self.service, '_mock_patcher'):
            self.service._mock_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_publish_change_map_uses_enum_transport_modes(self):
        """Test that _publish_change_map uses the new enum-based transport modes."""
        # Create test change map (binary image)
        change_map = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255
        change_score = 0.75
        
        # Mock the prepare_image_message function to capture its arguments
        with patch('experimance_core.experimance_core.prepare_image_message') as mock_prepare:
            mock_prepare.return_value = {
                "type": "CHANGE_MAP",
                "image_data": "mock_base64_data",
                "change_score": change_score,
                "timestamp": datetime.now().isoformat()
            }
            
            # Call the method
            await self.service._publish_change_map(change_map, change_score)
            
            # Verify prepare_image_message was called with correct arguments
            mock_prepare.assert_called_once()
            call_args = mock_prepare.call_args
            
            # Check that it uses numpy array
            assert isinstance(call_args[1]['image_data'], np.ndarray)
            assert np.array_equal(call_args[1]['image_data'], change_map)
            
            # Check that it uses the enum-based transport mode
            assert call_args[1]['transport_mode'] == IMAGE_TRANSPORT_MODES["AUTO"]
            
            # Check that it has proper message fields as kwargs (not nested in metadata)
            assert call_args[1]['change_score'] == change_score
            assert call_args[1]['type'] == "ChangeMap"  # This is the actual MessageType.CHANGE_MAP.value
            assert 'timestamp' in call_args[1]
            assert call_args[1]['has_change_map'] == True
            assert call_args[1]['mask_id'].startswith('change_map_')
            
            # Check that target_address is properly set for local transport
            assert 'localhost' in call_args[1]['target_address']
            
            # Check that mask_id is generated
            assert 'mask_id' in call_args[1]
            assert 'change_map_' in call_args[1]['mask_id']
        
        # Verify the message was published
        self.mock_zmq_service.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_change_map_handles_numpy_arrays_correctly(self):
        """Test that different numpy array formats are handled correctly."""
        test_cases = [
            # Grayscale binary image
            np.random.randint(0, 2, (240, 320), dtype=np.uint8) * 255,
            # Different sized binary image  
            np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255,
            # Edge case: very small image
            np.random.randint(0, 2, (32, 32), dtype=np.uint8) * 255,
        ]
        
        for i, change_map in enumerate(test_cases):
            change_score = 0.1 * (i + 1)  # Different scores for each test
            
            with patch('experimance_core.experimance_core.prepare_image_message') as mock_prepare:
                mock_prepare.return_value = {"type": "CHANGE_MAP", "image_data": "mock_data"}
                
                # Should not raise an exception
                await self.service._publish_change_map(change_map, change_score)
                
                # Verify the function was called
                mock_prepare.assert_called_once()
                
                # Verify numpy array shape is preserved
                call_args = mock_prepare.call_args
                passed_array = call_args[1]['image_data']
                assert passed_array.shape == change_map.shape
                assert passed_array.dtype == change_map.dtype
    
    @pytest.mark.asyncio
    async def test_publish_change_map_error_handling(self):
        """Test that errors in publishing are handled gracefully."""
        change_map = np.zeros((100, 100), dtype=np.uint8)
        change_score = 0.5
        
        # Mock prepare_image_message to raise an exception
        with patch('experimance_core.experimance_core.prepare_image_message', 
                  side_effect=Exception("Test error")), \
             patch('experimance_core.experimance_core.logger') as mock_logger:
            
            # Should not raise an exception (errors should be caught)
            await self.service._publish_change_map(change_map, change_score)
            
            # Should log the error
            mock_logger.error.assert_called_once()
            error_message = str(mock_logger.error.call_args[0][0])
            assert "Error publishing change map" in error_message
    
    @pytest.mark.asyncio
    async def test_publish_change_map_message_format_compatibility(self):
        """Test that the published message is compatible with display service expectations."""
        change_map = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255
        change_score = 0.42
        
        with patch('experimance_core.experimance_core.prepare_image_message') as mock_prepare:
            # Mock a realistic message format that prepare_image_message would return
            mock_message = {
                "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
                "image_id": "change_map_1234567890",
                "type": "ChangeMap",  # Correct MessageType enum value
                "change_score": change_score,
                "has_change_map": True,
                "timestamp": "2025-06-18T10:30:00"
            }
            mock_prepare.return_value = mock_message
            
            await self.service._publish_change_map(change_map, change_score)
            
            # Verify the message that was published has the expected format
            published_message = self.mock_zmq_service.publish.call_args[0][0]
            
            # Should contain image data (base64 or URI)
            assert "image_data" in published_message or "uri" in published_message
            
            # Should contain the metadata we expect
            assert published_message["change_score"] == change_score
            assert published_message["type"] == "ChangeMap"  # Correct MessageType enum value
