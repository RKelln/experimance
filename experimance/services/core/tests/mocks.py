"""
Mock utilities for testing the Experimance Core Service.

This module provides mock classes and factory functions specifically for testing
the core service without requiring real hardware or network dependencies.
Updated for the new ZMQ composition architecture with ControllerService.
"""
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from experimance_core.experimance_core import ExperimanceCoreService
from experimance_core.config import CoreServiceConfig
from experimance_common.zmq.mocks import MockControllerService
from experimance_common.zmq.config import ControllerServiceConfig, PublisherConfig, SubscriberConfig


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
        "service_name": "test_core",
        "experimance_core": {
            "name": "test_core",
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
        "zmq": {
            "name": "test_core_zmq",
            "log_level": "DEBUG",
            "timeout": 1.0,
            "heartbeat_interval": 1.0,
            "publisher": {
                "address": "tcp://*",
                "port": 5555,
                "bind": True
            },
            "subscriber": {
                "address": "tcp://localhost",
                "port": 5556,
                "bind": False,
                "topics": []
            },
            "workers": {}
        },
        "visualize": False  # Disable visualization for tests
    }
    
    # Merge user overrides with defaults
    if config_overrides:
        # Improved merge - handle both dict and non-dict values
        for section, values in config_overrides.items():
            if section in default_overrides:
                if isinstance(values, dict) and isinstance(default_overrides[section], dict):
                    default_overrides[section].update(values)
                else:
                    # Direct assignment for non-dict values
                    default_overrides[section] = values
            else:
                default_overrides[section] = values
    
    config = CoreServiceConfig.from_overrides(override_config=default_overrides)
    
    # Use the real MockControllerService from experimance_common
    mock_zmq_service = MockControllerService(config.zmq)
    
    # Replace key methods with AsyncMocks for easier testing
    mock_zmq_service.publish = AsyncMock()
    mock_zmq_service.send_work_to_worker = AsyncMock()
    
    # Mark the mock service as running so it can handle calls
    mock_zmq_service.running = True
    
    # Patch the ControllerService class to return our mock
    # Note: This patching will be global for the test - make sure tests clean up
    patcher = patch('experimance_core.experimance_core.ControllerService', return_value=mock_zmq_service)
    patcher.start()
    
    try:
        service = ExperimanceCoreService(config=config)
        
        # Mock essential methods that would normally be inherited from BaseService
        service.add_task = Mock()
        service.record_error = Mock()
        service._sleep_if_running = AsyncMock(return_value=False)
        service._publish_change_map = AsyncMock()  # Mock the change map publishing method
        
        # Add compatibility attributes for legacy tests
        service.publish_message = AsyncMock()  # Legacy method name, delegates to zmq_service.publish
        service.service_name = config.service_name if hasattr(config, 'service_name') else "test_core"
        
        # Store the patcher on the service for cleanup
        service._mock_patcher = patcher
        
        return service
    except Exception:
        patcher.stop()
        raise


def mock_zmq_for_core_service():
    """
    Context manager that mocks ZMQ components commonly used by the core service.
    
    Usage:
        with mock_zmq_for_core_service():
            service = ExperimanceCoreService(config)
            # Service now has mocked ZMQ components
    """
    return patch('experimance_core.experimance_core.ControllerService')


def create_mock_core_service_with_custom_config(config):
    """
    Create a properly mocked ExperimanceCoreService with a custom config object.
    
    This is useful when tests need to provide their own specific config mock
    rather than using the default overrides.
    
    Args:
        config: A config object (real or mock) to use for the service
        
    Returns:
        A mocked ExperimanceCoreService ready for testing
    """
    # Use the real MockControllerService from experimance_common
    # Need to ensure config.zmq exists
    if hasattr(config, 'zmq'):
        mock_zmq_service = MockControllerService(config.zmq)
    else:
        # Create a minimal zmq config for testing
        zmq_config = ControllerServiceConfig(
            name="test_zmq",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=[]),
            workers={}
        )
        mock_zmq_service = MockControllerService(zmq_config)
    
    # Patch the ControllerService class to return our mock
    with patch('experimance_core.experimance_core.ControllerService', return_value=mock_zmq_service):
        service = ExperimanceCoreService(config=config)
        
        # Mock essential methods that would normally be inherited from BaseService
        service.add_task = Mock()
        service.record_error = Mock()
        service._sleep_if_running = AsyncMock(return_value=False)
        service._publish_change_map = AsyncMock()  # Mock the change map publishing method
        
        # Add compatibility attributes for legacy tests
        service.publish_message = AsyncMock()  # Legacy method name, delegates to zmq_service.publish
        service.service_name = config.service_name if hasattr(config, 'service_name') else "test_core"
        
        return service


def comprehensive_core_service_mock():
    """
    Comprehensive context manager that mocks all the necessary ZMQ and service components.
    
    This can be used as a context manager in any test that needs to create a core service
    without dealing with ZMQ initialization issues.
    
    Usage:
        with comprehensive_core_service_mock():
            service = ExperimanceCoreService(config=my_config)
            # Service now has all necessary mocks applied
    """
    return patch('experimance_core.experimance_core.ControllerService')


def mock_core_service_for_testing(config=None):
    """
    Factory function that creates a fully mocked core service for testing.
    
    This is the recommended way to create core services in tests as it handles
    all the necessary mocking automatically.
    
    Args:
        config: Optional config object. If None, creates a default test config.
        
    Returns:
        Tuple of (service, context_manager) where context_manager should be used
        to ensure proper cleanup
    """
    if config is None:
        # Create default test config
        config = CoreServiceConfig.from_overrides(override_config={
            "service_name": "test_core",
            "experimance_core": {
                "name": "test_core",
                "change_smoothing_queue_size": 1
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
                "significant_change_threshold": 0.01
            },
            "zmq": {
                "name": "test_core_zmq",
                "log_level": "DEBUG", 
                "timeout": 1.0,
                "heartbeat_interval": 1.0
            },
            "visualize": False
        })
    
    # Create patches for ZMQ components
    patches = [
        patch('experimance_core.experimance_core.ControllerService')
    ]
    
    # Start all patches and get the mock controller service
    mock_controller_service = patches[0].start()  # The ControllerService patch is the first one
    
    try:
        # Create a mock controller service instance
        mock_zmq_service = Mock()
        mock_zmq_service.start = AsyncMock()
        mock_zmq_service.stop = AsyncMock()
        mock_zmq_service.publish = AsyncMock()
        mock_zmq_service.send_work_to_worker = AsyncMock()
        mock_zmq_service.add_message_handler = Mock()
        mock_zmq_service.add_response_handler = Mock()
        mock_controller_service.return_value = mock_zmq_service
        
        service = ExperimanceCoreService(config=config)
        
        # Mock essential methods
        service.add_task = Mock()
        service.record_error = Mock()
        service._sleep_if_running = AsyncMock(return_value=False)
        service._publish_change_map = AsyncMock()  # Mock the change map publishing method
        
        # Add compatibility attributes for legacy tests
        service.publish_message = AsyncMock()  # Legacy method name, delegates to zmq_service.publish
        service.service_name = config.service_name if hasattr(config, 'service_name') else "test_core"
        
        # Create cleanup function
        def cleanup():
            for p in patches:
                p.stop()
        
        return service, cleanup
        
    except Exception:
        # Clean up patches if service creation fails
        for p in patches:
            p.stop()
        raise
