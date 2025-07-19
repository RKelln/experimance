#!/usr/bin/env python3
"""
Tests for configuration loading and validation.
"""
import pytest
import sys
import tempfile
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experimance_core.config import CoreServiceConfig


class TestConfiguration:
    """Test configuration loading and validation."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = CoreServiceConfig()
        
        # Test default values
        assert config.experimance_core.name == "experimance_core"
        assert config.state_machine.idle_timeout == 45.0
        assert config.depth_processing.change_threshold == 50
        assert config.depth_processing.resolution == (1280, 720)
    
    def test_config_from_file(self):
        """Test loading configuration from TOML file."""
        config_content = """
[experimance_core]
name = "test_core_config"

[state_machine]
idle_timeout = 30.0
interaction_threshold = 0.4

[depth_processing]
change_threshold = 75
min_depth = 0.3
max_depth = 0.7
resolution = [1920, 1080]
output_size = [2048, 2048]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            config = CoreServiceConfig.from_overrides(config_file=f.name)
            
            # Test loaded values
            assert config.experimance_core.name == "test_core_config"
            assert config.state_machine.idle_timeout == 30.0
            assert config.state_machine.interaction_threshold == 0.4
            assert config.depth_processing.change_threshold == 75
            assert config.depth_processing.resolution == (1920, 1080)
            assert config.depth_processing.output_size == (2048, 2048)
        
        # Cleanup
        Path(f.name).unlink()
    
    def test_partial_config_override(self):
        """Test partial configuration override (defaults + overrides)."""
        config_content = """
[experimance_core]
name = "partial_test"

[depth_processing]
change_threshold = 100
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            config = CoreServiceConfig.from_overrides(config_file=f.name)
            
            # Test overridden values
            assert config.experimance_core.name == "partial_test"
            assert config.depth_processing.change_threshold == 100
            
            # Test default values are preserved
            assert config.state_machine.idle_timeout == 45.0  # Default
            assert config.depth_processing.resolution == (1280, 720)  # Default
        
        # Cleanup
        Path(f.name).unlink()
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = CoreServiceConfig()
        assert config.state_machine.idle_timeout > 0
        assert config.depth_processing.change_threshold >= 0
        assert len(config.depth_processing.resolution) == 2
        assert len(config.depth_processing.output_size) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
