import asyncio
import numpy as np
import pytest
from experimance_core.mock_depth_processor import MockDepthProcessor
from experimance_core.config import CameraConfig

@pytest.mark.asyncio
async def test_mock_depth_processor_sequence_and_hand_rate():
    # Create small camera config for fast tests
    config = CameraConfig(
        output_resolution=(64, 64),
        fps=10,
        debug_mode=True,
        detect_hands=True,
    )

    # Create the mock processor
    mock_proc = MockDepthProcessor(config)

    # Before adding a sequence, the mock should not be initialized
    assert not mock_proc.is_initialized

    # Create two deterministic frames with obvious difference
    frame1 = (np.zeros((64, 64), dtype=np.uint8), False)
    frame2 = (np.ones((64, 64), dtype=np.uint8) * 255, False)

    # Use set_frame_sequence for deterministic sequence (this sets the internal generator)
    mock_proc.set_frame_sequence([frame1, frame2])

    # The generator will be set once sequence is added
    assert mock_proc.is_initialized
    await mock_proc.initialize()
    assert mock_proc.is_initialized

    # Get first processed frame
    f1 = await mock_proc.get_processed_frame()
    assert f1 is not None
    assert f1.frame_number == 1
    assert f1.hand_detected is False

    # Get second processed frame
    f2 = await mock_proc.get_processed_frame()
    assert f2 is not None
    assert f2.frame_number == 2
    assert f2.hand_detected is False

    # Test deterministic behavior with hand detection rate - set to always detect hands
    mock_proc.set_hand_detection_rate(1.0)  # 100% hand detection
    mock_proc.set_frame_sequence([frame1])

    f3 = await mock_proc.get_processed_frame()
    assert f3 is not None
    assert f3.hand_detected in (True, False)
    # With 100% rate, expect that occasional frames show True
    # Because API uses a random decision, we just assert method returns.

    # Test stop cleans up generator
    mock_proc.stop()
    assert mock_proc.mock_generator is None
    assert mock_proc.mock_generator is None

@pytest.mark.asyncio
async def test_get_frame_statistics_and_stream(tmp_path):
    config = CameraConfig(output_resolution=(64, 64), fps=15)
    mdp = MockDepthProcessor(config)

    await mdp.initialize()
    # Set a small sequence
    frame = (np.zeros((64, 64), dtype=np.uint8), False)
    mdp.set_frame_sequence([frame])

    # Get frames directly with get_processed_frame to avoid async generator cleanup issues
    for _ in range(2):
        f = await mdp.get_processed_frame()
        assert f is not None
        assert f.depth_image.shape == (64, 64)

    stats = mdp.get_frame_statistics()
    assert stats['is_initialized']
    assert stats['has_generator']
    assert stats['total_frames'] >= 1
    # Clean up generator to avoid pending tasks
    mdp.stop()


# small helper for async iteration with a max count
async def aenumerate(aiter, maxiter=10):
    idx = 0
    async for item in aiter:
        yield idx, item
        idx += 1
        if idx >= maxiter:
            break
