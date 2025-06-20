"""
Mock utilities for testing the Experimance Display Service.

This module provides mock classes and factory functions specifically for testing
the display service without requiring actual Pyglet windows, ZMQ connections,
or hardware dependencies.
"""
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional
from pathlib import Path

from experimance_display.display_service import DisplayService
from experimance_display.config import DisplayServiceConfig
from experimance_common.zmq.mocks import MockPubSubService


def create_mock_display_service(config_overrides: Optional[Dict[str, Any]] = None) -> DisplayService:
    """
    Create a properly mocked DisplayService for testing.
    
    This utility handles all the common mocking patterns needed to test the display service
    without initializing Pyglet windows, ZMQ connections, or renderer dependencies.
    
    Args:
        config_overrides: Optional dict of config overrides
        
    Returns:
        A mocked DisplayService ready for testing
    """
    # Default config overrides for testing
    default_overrides = {
        "service_name": "test_display",
        "display": {
            "headless": True,  # Always headless for tests
            "fullscreen": False,
            "resolution": [800, 600],
            "debug_overlay": False,
            "debug_text": False,
            "profile": False
        },
        "title_screen": {
            "enabled": False  # Disable title screen for tests
        },
        "zmq": {
            "name": "test_display_zmq",
            "subscriber": {
                "address": "tcp://localhost",
                "port": 15555,
                "topics": ["image.ready", "text.overlay", "transition.ready"]
            }
        }
    }
    
    # Merge user overrides with defaults
    if config_overrides:
        _deep_merge_dict(default_overrides, config_overrides)
    
    config = DisplayServiceConfig.from_overrides(override_config=default_overrides)
    
    # Create the service instance first
    service = DisplayService.__new__(DisplayService)
    assert service is not None, "Failed to create DisplayService instance"
    
    # Initialize base attributes manually
    service.config = config
    service.target_fps = 30
    service.frame_timer = 0.0
    service.frame_count = 0
    service.fps_display_timer = 0.0
    service._direct_handlers = {}
    
    # Mock the ZMQ service with MockPubSubService
    service.zmq_service = MockPubSubService(config.zmq) # type: ignore
    
    # Mock essential Pyglet components
    service.window = create_mock_window()
    service.layer_manager = create_mock_layer_manager()
    service.image_renderer = create_mock_image_renderer()
    service.video_overlay_renderer = create_mock_video_overlay_renderer()
    service.text_overlay_manager = create_mock_text_overlay_manager()
    service.debug_overlay_renderer = create_mock_debug_overlay_renderer()
    
    # Mock base service methods
    service.record_error = Mock()
    service.request_stop = Mock()
    service.add_task = Mock()
    # Note: Don't set running property as it's read-only
    
    # Register the direct handlers (call the actual method)
    service._register_direct_handlers()
    
    return service


def create_mock_display_config(config_overrides: Optional[Dict[str, Any]] = None) -> DisplayServiceConfig:
    """
    Create a test-friendly DisplayServiceConfig.
    
    Args:
        config_overrides: Optional dict of config overrides
        
    Returns:
        A DisplayServiceConfig configured for testing
    """
    default_overrides = {
        "service_name": "test_display",
        "display": {
            "headless": True,
            "fullscreen": False,
            "resolution": [800, 600],
            "debug_overlay": False,
            "debug_text": False
        },
        "title_screen": {
            "enabled": False
        }
    }
    
    if config_overrides:
        _deep_merge_dict(default_overrides, config_overrides)
    
    return DisplayServiceConfig.from_overrides(override_config=default_overrides)


def create_mock_window():
    """Create a mock Pyglet window."""
    window = Mock()
    window.width = 800
    window.height = 600
    window.fullscreen = False
    window.has_exit = False
    window.clear = Mock()
    window.close = Mock()
    window.flip = Mock()
    window.switch_to = Mock()
    window.dispatch_events = Mock()
    window.dispatch_event = Mock()
    window.set_fullscreen = Mock()
    return window


def create_mock_layer_manager():
    """Create a mock LayerManager."""
    layer_manager = AsyncMock()
    layer_manager.register_renderer = Mock()
    layer_manager.update = Mock()
    layer_manager.render = Mock()
    layer_manager.cleanup = AsyncMock()
    return layer_manager


def create_mock_image_renderer():
    """Create a mock ImageRenderer."""
    renderer = AsyncMock()
    renderer.handle_image_ready = AsyncMock()
    renderer.handle_transition_ready = AsyncMock()
    return renderer


def create_mock_video_overlay_renderer():
    """Create a mock VideoOverlayRenderer."""
    renderer = AsyncMock()
    renderer.handle_video_mask = AsyncMock()
    return renderer


def create_mock_text_overlay_manager():
    """Create a mock TextOverlayManager."""
    manager = AsyncMock()
    manager.handle_text_overlay = AsyncMock()
    manager.handle_remove_text = AsyncMock()
    return manager


def create_mock_debug_overlay_renderer():
    """Create a mock DebugOverlayRenderer."""
    renderer = AsyncMock()
    renderer.update = Mock()
    renderer.render = Mock()
    return renderer


def mock_pyglet_components():
    """
    Context manager that mocks all Pyglet components used by the display service.
    
    Usage:
        with mock_pyglet_components():
            service = DisplayService(config)
            # Service now has mocked Pyglet components
    """
    return patch.multiple(
        'experimance_display.display_service',
        pyglet=Mock(),
        LayerManager=Mock(),
        ImageRenderer=Mock(),
        VideoOverlayRenderer=Mock(),
        TextOverlayManager=Mock(),
        DebugOverlayRenderer=Mock()
    )


def create_test_message(message_type: str, **kwargs) -> Dict[str, Any]:
    """
    Create test messages in the format expected by the display service.
    
    Args:
        message_type: Type of message (e.g., "image_ready", "text_overlay")
        **kwargs: Additional message fields
        
    Returns:
        A properly formatted test message
    """
    base_messages = {
        "image_ready": {
            "image_id": "test_image_001",
            "uri": "file:///tmp/test_image.png",
            "request_id": "test_request_001"
        },
        "text_overlay": {
            "text_id": "test_text_001",
            "content": "Test message content",
            "speaker": "agent",
            "duration": 5.0
        },
        "remove_text": {
            "text_id": "test_text_001"
        },
        "transition_ready": {
            "transition_id": "test_transition_001",
            "uri": "file:///tmp/test_transition.mp4",
            "request_id": "test_request_002"
        },
        "video_mask": {
            "mask_data": [0.0] * 100,  # Mock mask data
            "timestamp": 1234567890.0
        },
        "era_changed": {
            "era": "anthropocene",
            "timestamp": 1234567890.0
        }
    }
    
    if message_type not in base_messages:
        raise ValueError(f"Unknown message type: {message_type}")
    
    message = base_messages[message_type].copy()
    message.update(kwargs)
    return message


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """
    Deep merge override dict into base dict.
    
    Args:
        base: Base dictionary to merge into
        override: Override dictionary with new values
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
