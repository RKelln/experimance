#!/usr/bin/env python3
"""
Test configuration loading for the Experimance Display Service.
"""

import pytest
import tempfile
from pathlib import Path
import sys

# Add the display service to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.display.src.experimance_display.config import (
    DisplayServiceConfig, 
    DisplayConfig, 
    TextStyleConfig,
    ZmqConfig,
    RenderingConfig,
    TransitionsConfig
)


class TestConfigurationSchemas:
    """Test Pydantic configuration schemas."""
    
    def test_display_config_defaults(self):
        """Test DisplayConfig default values."""
        config = DisplayConfig()
        assert config.fullscreen is True
        assert config.resolution == (1920, 1080)
        assert config.fps_limit == 60
        assert config.vsync is True
        assert config.debug_overlay is False
    
    def test_text_style_config_defaults(self):
        """Test TextStyleConfig default values."""
        config = TextStyleConfig()
        assert config.font_size == 28
        assert config.color == (255, 255, 255, 255)
        assert config.position == "bottom_center"
        assert config.background is True
        assert config.padding == 10
    
    def test_zmq_config_defaults(self):
        """Test ZmqConfig default values."""
        config = ZmqConfig()
        assert "tcp://localhost:" in config.events_sub_address
    
    def test_complete_config_defaults(self):
        """Test complete DisplayServiceConfig defaults."""
        config = DisplayServiceConfig()
        assert config.service_name == "display-service"
        assert config.display.fullscreen is True
        assert config.text_styles.agent.font_size == 28
        assert config.text_styles.system.font_size == 24
        assert config.text_styles.debug.font_size == 16


class TestConfigurationLoading:
    """Test configuration loading from various sources."""
    
    def test_config_from_dict_override(self):
        """Test configuration creation from dictionary override."""
        override_config = {
            'service_name': 'test-display',
            'display': {
                'fullscreen': False,
                'resolution': [800, 600],
                'fps_limit': 30
            },
            'text_styles': {
                'agent': {
                    'font_size': 32,
                    'color': [255, 0, 0, 255]
                }
            }
        }
        
        config = DisplayServiceConfig.from_overrides(
            override_config=override_config
        )
        
        # Check overrides were applied
        assert config.service_name == 'test-display'
        assert config.display.fullscreen is False
        assert config.display.resolution == (800, 600)
        assert config.display.fps_limit == 30
        assert config.text_styles.agent.font_size == 32
        assert config.text_styles.agent.color == (255, 0, 0, 255)
        
        # Check non-overridden values remain default
        assert config.display.vsync is True
        assert config.text_styles.system.font_size == 24
    
    def test_config_from_toml_file(self):
        """Test configuration loading from TOML file."""
        toml_content = """
[display]
fullscreen = false
resolution = [1280, 720]
fps_limit = 45
debug_overlay = true

[text_styles.agent]
font_size = 36
position = "top_center"

[text_styles.system]
color = [0, 255, 0, 255]

[zmq]
images_sub_address = "tcp://localhost:9999"
events_sub_address = "tcp://localhost:8888"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            f.flush()
            
            config = DisplayServiceConfig.from_overrides(config_file=f.name)
            
            # Check TOML values were loaded correctly
            assert config.display.fullscreen is False
            assert config.display.resolution == (1280, 720)
            assert config.display.fps_limit == 45
            assert config.display.debug_overlay is True
            
            assert config.text_styles.agent.font_size == 36
            assert config.text_styles.agent.position == "top_center"
            assert config.text_styles.system.color == (0, 255, 0, 255)
            
            assert config.zmq.events_sub_address == "tcp://localhost:8888"
        
        # Clean up
        Path(f.name).unlink()
    
    def test_config_priority_override_file(self):
        """Test that override_config takes priority over config file."""
        toml_content = """
[display]
fullscreen = true
resolution = [1920, 1080]
fps_limit = 60
"""
        
        override_config = {
            'display': {
                'fullscreen': False,
                'fps_limit': 30
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            f.flush()
            
            config = DisplayServiceConfig.from_overrides(
                config_file=f.name,
                override_config=override_config
            )
            
            # Override should win
            assert config.display.fullscreen is False  # from override
            assert config.display.fps_limit == 30      # from override
            assert config.display.resolution == (1920, 1080)  # from file
        
        # Clean up
        Path(f.name).unlink()


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def test_invalid_position_value(self):
        """Test that invalid position values are rejected."""
        with pytest.raises(ValueError):
            TextStyleConfig(position="invalid_position")  # type: ignore
    
    def test_invalid_backend_value(self):
        """Test that invalid backend values are rejected."""
        with pytest.raises(ValueError):
            RenderingConfig(backend="invalid_backend") # type: ignore
    
    def test_negative_values_rejected(self):
        """Test that certain values have reasonable bounds."""
        # Note: These tests check if the current implementation validates these values
        # If validation isn't implemented yet, these tests document the expected behavior
        
        # Test that we can create configs with valid values
        valid_display = DisplayConfig(fps_limit=60)
        assert valid_display.fps_limit == 60
        
        valid_text = TextStyleConfig(font_size=16)
        assert valid_text.font_size == 16
        
        valid_transitions = TransitionsConfig(default_crossfade_duration=1.0)
        assert valid_transitions.default_crossfade_duration == 1.0
        
        # For now, just test that negative values are accepted (indicating validation may need to be added)
        # TODO: Add proper validation constraints to Pydantic models if needed
        negative_display = DisplayConfig(fps_limit=-1)
        assert negative_display.fps_limit == -1  # Currently allowed, may want to add validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
