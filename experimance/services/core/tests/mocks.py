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
    
    # Mock ZMQ initialization to avoid network dependencies
    with patch('experimance_core.experimance_core.ZmqControllerMultiWorkerService.__init__', return_value=None), \
         patch.object(ExperimanceCoreService, 'setup_workers_from_config_provider', return_value=None):
        
        service = ExperimanceCoreService(config=config)
        
        # Initialize the essential attributes that would normally be set by the base class
        # Using setattr to avoid type checker issues with mocked objects
        setattr(service, 'worker_connections', {})
        setattr(service, 'service_name', config.experimance_core.name)
        setattr(service, 'tasks', [])
        setattr(service, '_running', False)
        
        # Mock essential methods
        service.publish_message = AsyncMock()
        service.add_task = Mock()
        service.record_error = Mock()
        service._sleep_if_running = AsyncMock(return_value=False)
        service._publish_change_map = AsyncMock()  # Mock the change map publishing method
        service.setup_workers_from_config_provider = Mock()  # Mock worker setup
        
        return service


def mock_zmq_for_core_service():
    """
    Context manager that mocks ZMQ components commonly used by the core service.
    
    Usage:
        with mock_zmq_for_core_service():
            service = ExperimanceCoreService(config)
            # Service now has mocked ZMQ components
    """
    return patch('experimance_core.experimance_core.ZmqControllerMultiWorkerService.__init__', return_value=None)


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
    # Mock ZMQ initialization to avoid network dependencies
    with patch('experimance_core.experimance_core.ZmqControllerMultiWorkerService.__init__', return_value=None), \
         patch.object(ExperimanceCoreService, 'setup_workers_from_config_provider', return_value=None):
        
        service = ExperimanceCoreService(config=config)
        
        # Initialize the essential attributes that would normally be set by the base class
        # Using setattr to avoid type checker issues with mocked objects
        setattr(service, 'worker_connections', {})
        setattr(service, 'service_name', getattr(config.experimance_core, 'name', 'test_core'))
        setattr(service, 'tasks', [])
        setattr(service, '_running', False)
        
        # Mock essential methods
        service.publish_message = AsyncMock()
        service.add_task = Mock()
        service.record_error = Mock()
        service._sleep_if_running = AsyncMock(return_value=False)
        service._publish_change_map = AsyncMock()  # Mock the change map publishing method
        service.setup_workers_from_config_provider = Mock()  # Mock worker setup
        
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
    return patch.multiple(
        'experimance_core.experimance_core',
        ZmqControllerMultiWorkerService=Mock,
        # Mock the base class constructor
        **{'ZmqControllerMultiWorkerService.__init__': Mock(return_value=None)}
    )


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
            "experimance_core": {
                "name": "test_core",
                "heartbeat_interval": 1.0,
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
            "visualize": False
        })
    
    # Create patches for ZMQ components
    patches = [
        patch('experimance_core.experimance_core.ZmqControllerMultiWorkerService.__init__', return_value=None),
        patch.object(ExperimanceCoreService, 'setup_workers_from_config_provider', return_value=None)
    ]
    
    # Start all patches
    for p in patches:
        p.start()
    
    try:
        service = ExperimanceCoreService(config=config)
        
        # Initialize essential attributes
        setattr(service, 'worker_connections', {})
        setattr(service, 'service_name', config.experimance_core.name)
        setattr(service, 'tasks', [])
        setattr(service, '_running', False)
        
        # Mock essential methods
        service.publish_message = AsyncMock()
        service.add_task = Mock()
        service.record_error = Mock()
        service._sleep_if_running = AsyncMock(return_value=False)
        service._publish_change_map = AsyncMock()
        service.setup_workers_from_config_provider = Mock()
        
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
