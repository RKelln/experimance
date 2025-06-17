"""
Mock utilities for testing the Experimance Core Service.

This module provides mock classes and factory functions specifically for testing
the core service without requiring real hardware or network dependencies.
"""
from unittest.mock import Mock, AsyncMock, patch
from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig


def create_mock_core_service(config_overrides=None):
    """
    Create a properly mocked ExperimanceCoreService for testing.
    
    This utility handles all the common mocking patterns needed to test the core service
    without initializing ZMQ connections or hardware dependencies.
    
    Args:
        config_overrides: Optional dict of config overrides
        
    Returns:
        A mocked ExperimanceCoreService ready for testing
    """
    # Default config overrides for testing
    default_overrides = {
        "experimance_core": {
            "name": "test_core",
            "heartbeat_interval": 1.0,
            "change_smoothing_queue_size": 1  # Small queue for faster test setup
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
            "output_size": [512, 512],
            "significant_change_threshold": 0.01  # Low threshold for test reliability
        },
        "visualize": False  # Disable visualization for tests
    }
    
    # Merge user overrides with defaults
    if config_overrides:
        # Simple merge - could be made more sophisticated if needed
        for section, values in config_overrides.items():
            if section in default_overrides:
                default_overrides[section].update(values)
            else:
                default_overrides[section] = values
    
    config = CoreServiceConfig.from_overrides(override_config=default_overrides)
    
    # Mock ZMQ initialization to avoid network dependencies
    with patch('experimance_core.experimance_core.ZmqPublisherSubscriberService.__init__', return_value=None):
        service = ExperimanceCoreService(config=config)
        
        # Mock essential methods
        service.publish_message = AsyncMock()
        service.add_task = Mock()
        service.record_error = Mock()
        service._sleep_if_running = AsyncMock(return_value=False)
        service._publish_change_map = AsyncMock()  # Mock the change map publishing method
        
        return service


def mock_zmq_for_core_service():
    """
    Context manager that mocks ZMQ components commonly used by the core service.
    
    Usage:
        with mock_zmq_for_core_service():
            service = ExperimanceCoreService(config)
            # Service now has mocked ZMQ components
    """
    return patch('experimance_core.experimance_core.ZmqPublisherSubscriberService.__init__', return_value=None)
