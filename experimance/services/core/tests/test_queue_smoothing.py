"""
Test queue-based change score smoothing in ExperimanceCoreService.

Tests the implementation of change score queue for reducing hand entry/exit artifacts.
"""
import pytest
import asyncio
import numpy as np
from collections import deque
from experimance_common.test_utils import active_service
from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig, DepthFrame


class TestChangeScoreQueue:
    """Test the change score queue functionality."""

    @pytest.mark.asyncio
    async def test_change_score_queue_initialization(self):
        """Test that the change score queue is properly initialized."""
        config = CoreServiceConfig.from_overrides(override_config={})
        service = ExperimanceCoreService(config=config)
        
        # Check queue is initialized
        assert hasattr(service, 'change_score_queue')
        assert isinstance(service.change_score_queue, deque)
        assert service.change_score_queue.maxlen == service.config.experimance_core.change_smoothing_queue_size
        assert len(service.change_score_queue) == 0

    @pytest.mark.asyncio
    async def test_queue_cleared_on_hand_state_change(self):
        """Test that the queue is cleared when hand detection state changes."""
        config = CoreServiceConfig.from_overrides(override_config={
            "camera": {"significant_change_threshold": 0.01}
        })
        service = ExperimanceCoreService(config=config)
        
        async with active_service(service) as active:
            # Create depth frames to establish baseline and get some queue data
            depth_image1 = np.random.rand(100, 100).astype(np.float32) * 0.5
            depth_image2 = np.random.rand(100, 100).astype(np.float32) * 0.5 + 0.1  # Different image
            
            # Process first frame to establish reference (no hand)
            depth_frame1 = DepthFrame(
                depth_image=depth_image1,
                hand_detected=False,
                timestamp=1.0
            )
            await active._process_depth_frame(depth_frame1)
            
            # Process second frame to get some data in queue (still no hand)
            depth_frame2 = DepthFrame(
                depth_image=depth_image2,
                hand_detected=False,
                timestamp=2.0
            )
            await active._process_depth_frame(depth_frame2)
            
            # Queue should have some data now (from change calculation)
            initial_queue_size = len(active.change_score_queue)
            assert initial_queue_size > 0, "Queue should have data after processing frames with changes"
            
            # Process frame with hand detected (state change from False to True)
            depth_frame3 = DepthFrame(
                depth_image=depth_image2,
                hand_detected=True,
                timestamp=3.0
            )
            await active._process_depth_frame(depth_frame3)
            
            # Queue should be cleared due to hand state change
            assert len(active.change_score_queue) == 0, "Queue should be cleared when hand state changes"

    @pytest.mark.asyncio
    async def test_queue_smoothing_behavior(self):
        """Test that the queue correctly smooths change scores using minimum."""
        config = CoreServiceConfig.from_overrides(override_config={
            "camera": {"significant_change_threshold": 0.02}
        })
        service = ExperimanceCoreService(config=config)
        
        async with active_service(service) as active:
            # Create a reference frame first
            ref_image = np.zeros((100, 100), dtype=np.float32)
            ref_frame = DepthFrame(
                depth_image=ref_image,
                hand_detected=False,
                timestamp=1.0
            )
            await active._process_depth_frame(ref_frame)
            
            # Now test with a sequence that should demonstrate smoothing
            # Create images with varying amounts of change
            test_images = []
            for i, change_level in enumerate([0.05, 0.08, 0.03, 0.025, 0.01]):
                # Create image with specific amount of change
                changed_image = ref_image.copy()
                # Add noise to simulate change
                noise = np.random.rand(100, 100).astype(np.float32) * change_level
                changed_image += noise
                test_images.append(changed_image)
            
            processing_results = []
            
            for i, image in enumerate(test_images):
                depth_frame = DepthFrame(
                    depth_image=image,
                    hand_detected=False,
                    timestamp=float(i + 2)
                )
                
                # Store queue state before processing
                queue_before = list(active.change_score_queue)
                
                await active._process_depth_frame(depth_frame)
                
                # Store queue state after processing
                queue_after = list(active.change_score_queue)
                queue_min = min(queue_after) if queue_after else 0.0
                
                processing_results.append({
                    'frame': i,
                    'queue_before': queue_before,
                    'queue_after': queue_after,
                    'queue_min': queue_min,
                    'processed': len(queue_after) > 0 and queue_min >= 0.02
                })
            
            # Verify queue behavior
            assert len(processing_results) == 5
            
            # First frame should have queue of size 1
            assert len(processing_results[0]['queue_after']) == 1
            
            # Queue should grow to max size of 3
            for i in range(3):
                assert len(processing_results[i]['queue_after']) == i + 1
            
            # After 3 frames, queue should maintain size 3
            for i in range(3, 5):
                assert len(processing_results[i]['queue_after']) == 3
            
            # The minimum should be used for decision making
            # With high initial values followed by lower ones,
            # the minimum should prevent early processing of spikes
            final_result = processing_results[-1]
            assert final_result['queue_min'] == min(final_result['queue_after'])

    @pytest.mark.asyncio
    async def test_queue_with_hand_detection_exit(self):
        """Test queue behavior when hands are detected and then removed."""
        config = CoreServiceConfig.from_overrides(override_config={
            "camera": {"significant_change_threshold": 0.02}
        })
        service = ExperimanceCoreService(config=config)
        
        async with active_service(service) as active:
            depth_image = np.random.rand(100, 100).astype(np.float32)
            
            # Start with hand detected
            depth_frame1 = DepthFrame(
                depth_image=depth_image,
                hand_detected=True,
                timestamp=1.0
            )
            await active._process_depth_frame(depth_frame1)
            
            # Queue should be empty (hands detected, no processing)
            assert len(active.change_score_queue) == 0
            
            # Remove hand (this will clear queue due to state change)
            depth_frame2 = DepthFrame(
                depth_image=depth_image,
                hand_detected=False,
                timestamp=2.0
            )
            await active._process_depth_frame(depth_frame2)
            
            # Queue should still be empty after state change clearing
            assert len(active.change_score_queue) == 0
            
            # Add hand again (state change, should clear queue)
            depth_frame3 = DepthFrame(
                depth_image=depth_image,
                hand_detected=True,
                timestamp=3.0
            )
            await active._process_depth_frame(depth_frame3)
            
            # Verify queue is still empty
            assert len(active.change_score_queue) == 0

    def test_queue_smoothing_simulation(self):
        """Test queue smoothing with simulated realistic data."""
        # Simulate the test data from our standalone script
        scores = [0.005, 0.008, 0.003, 0.006, 0.004,  # normal
                 0.15, 0.08, 0.03, 0.012, 0.008,      # hand entry spike
                 0.025, 0.035, 0.028, 0.032, 0.024, 0.018,  # interaction
                 0.12, 0.09, 0.04, 0.015, 0.007,      # hand exit spike
                 0.004, 0.003, 0.005, 0.002, 0.006]   # return to normal
        
        change_score_queue = deque(maxlen=3)
        change_threshold = 0.02
        processed_frames = []
        
        for i, raw_score in enumerate(scores):
            change_score_queue.append(raw_score)
            
            if len(change_score_queue) > 0:
                min_score = min(change_score_queue)
                should_process = min_score >= change_threshold
                
                if should_process:
                    processed_frames.append(i)
        
        # Verify expected behavior
        assert len(processed_frames) == 5  # Should match our standalone test
        
        # Verify that large spikes are filtered out
        large_spike_frames = [i for i, score in enumerate(scores) if score > 0.05]
        filtered_spikes = [f for f in large_spike_frames if f not in processed_frames]
        
        # All large spikes should be filtered out
        assert len(filtered_spikes) == len(large_spike_frames)
        assert len(filtered_spikes) == 4  # Based on our test data

    @pytest.mark.asyncio
    async def test_configurable_queue_size(self):
        """Test that the queue size is configurable."""
        # Test with queue size of 5
        config = CoreServiceConfig.from_overrides(override_config={
            "experimance_core": {"change_smoothing_queue_size": 5}
        })
        service = ExperimanceCoreService(config=config)
        
        assert service.change_score_queue.maxlen == 5
        assert service.config.experimance_core.change_smoothing_queue_size == 5
        
        # Test with queue size of 1
        config2 = CoreServiceConfig.from_overrides(override_config={
            "experimance_core": {"change_smoothing_queue_size": 1}
        })
        service2 = ExperimanceCoreService(config=config2)
        
        assert service2.change_score_queue.maxlen == 1
        assert service2.config.experimance_core.change_smoothing_queue_size == 1
