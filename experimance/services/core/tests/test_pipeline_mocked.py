import asyncio
import numpy as np
import pytest
from experimance_core.config import CoreServiceConfig, CameraConfig, CameraState
from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.mock_depth_processor import MockDepthProcessor
from experimance_common.zmq.mocks import MockControllerService
from experimance_common.zmq.config import PublisherConfig, SubscriberConfig, ControllerServiceConfig

@pytest.mark.asyncio
async def test_core_pipeline_with_mocked_depth_and_zmq():
    # Create a test config with deterministic values and quick thresholds
    override = {
        "service_name": "test_core",
        "visualize": False,
        "experimance_core": {"change_smoothing_queue_size": 1},
        "state_machine": {"era_min_duration": 5.0},
        "presence": {"always_present": True},
        "depth_processing": {
            "mock_depth_images_path": None,
        },
    }

    config = CoreServiceConfig.from_overrides(override_config=override)

    # Configure ZMQ controller config to create MockControllerService
    zmq_cfg = config.zmq
    pub_cfg = PublisherConfig(address="tcp://*", port=5555, default_topic="core.events")
    sub_cfg = SubscriberConfig(address="tcp://localhost", port=5556, topics=[])
    controller_cfg = ControllerServiceConfig(name="test_controller", publisher=pub_cfg, subscriber=sub_cfg, workers={})

    # Instantiate service and override its zmq_service with the mock
    service = ExperimanceCoreService(config=config)
    mock_controller = MockControllerService(controller_cfg)
    await mock_controller.start()
    service.zmq_service = mock_controller #type: ignore

    # Create a simple mock depth processor with deterministic frames
    camera_cfg = CameraConfig(output_resolution=(64, 64), fps=10, debug_mode=False)
    mdp = MockDepthProcessor(camera_cfg)
    mdp.set_frame_sequence([
        (np.zeros((64,64), dtype=np.uint8), False),
        (np.ones((64,64), dtype=np.uint8) * 255, False)
    ])
    await mdp.initialize()

    # Attach mock depth processor to service
    service._depth_processor = mdp
    service._camera_state = CameraState.READY

    # Process first frame - should set last_significant_depth_map but not publish
    f1 = await mdp.get_processed_frame()
    assert f1 is not None
    await service._process_depth_frame(f1)
    assert len(mock_controller.published_messages) == 0

    # Process second frame - should detect significant change and publish change_map
    f2 = await mdp.get_processed_frame()
    assert f2 is not None
    await service._process_depth_frame(f2)

    # Allow time for async publishing to complete
    await asyncio.sleep(0.01)

    assert len(mock_controller.published_messages) >= 1
    found = any(m.topic == "ChangeMap" or m.topic == "CHANGE_MAP" or m.topic == "core.events" for m in mock_controller.published_messages)
    assert found, f"No ChangeMap event found in messages: {mock_controller.published_messages}"

    await mock_controller.stop()
