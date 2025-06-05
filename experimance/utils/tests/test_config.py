#!/usr/bin/env python3
"""
Tests for the configuration utilities in experimance_common.config.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import pytest
import toml
from pydantic import BaseModel, Field

from experimance_common.config import (
    load_config_with_overrides,
    deep_merge,
    namespace_to_dict,
    Config,
    ConfigError
)


def test_deep_merge():
    """Test that deep_merge correctly merges nested dictionaries."""
    base = {
        "a": 1,
        "b": {
            "c": 2,
            "d": 3
        },
        "e": [1, 2, 3]
    }
    
    override = {
        "a": 10,
        "b": {
            "c": 20,
            "f": 30
        },
        "g": "new"
    }
    
    expected = {
        "a": 10,  # Overridden
        "b": {
            "c": 20,  # Overridden
            "d": 3,   # Kept from base
            "f": 30   # Added from override
        },
        "e": [1, 2, 3],  # Kept from base
        "g": "new"       # Added from override
    }
    
    result = deep_merge(base, override)
    assert result == expected
    
    # Original dictionaries should not be modified
    assert base["a"] == 1
    assert base["b"]["c"] == 2
    assert "g" not in base
    
    assert override["a"] == 10
    assert "d" not in override["b"]
    assert "e" not in override


def test_namespace_to_dict():
    """Test conversion of argparse.Namespace to nested dictionary."""
    # Create a namespace with flat and nested keys
    namespace = argparse.Namespace(
        name="test",
        log_level="INFO",
        timeout=30,
        none_value=None  # This should be excluded
    )
    setattr(namespace, "zmq.port", 5555)
    setattr(namespace, "database.host", "localhost")
    setattr(namespace, "database.port", 5432)
    setattr(namespace, "database.credentials.username", "user")
    
    expected = {
        "name": "test",
        "log_level": "INFO",
        "timeout": 30,
        "zmq": {
            "port": 5555
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "credentials": {
                "username": "user"
            }
        }
        # none_value should be excluded
    }
    
    result = namespace_to_dict(namespace)
    assert result == expected


def test_load_config_with_overrides():
    """Test loading configuration with various override combinations."""
    # Setup
    default_config = {
        "service": {
            "name": "test-service",
            "log_level": "INFO"
        },
        "zmq": {
            "port": 5555,
            "host": "localhost"
        }
    }
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.toml', mode='w+', delete=False) as temp_file:
        file_config = {
            "service": {
                "name": "file-service",
                "timeout": 30
            },
            "zmq": {
                "port": 6666
            }
        }
        toml.dump(file_config, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Test with all options
        override_config = {
            "service": {
                "name": "override-service"
            },
            "database": {
                "url": "postgres://localhost/db"
            }
        }
        
        args = argparse.Namespace()
        setattr(args, "service.log_level", "DEBUG")
        setattr(args, "zmq.host", "127.0.0.1")
        
        # Load with all options
        result = load_config_with_overrides(
            override_config=override_config,
            config_file=temp_file_path,
            default_config=default_config,
            args=args
        )
        
        # Check result combines all sources with correct priority
        assert result["service"]["name"] == "override-service"  # From override_config
        assert result["service"]["log_level"] == "DEBUG"        # From args
        assert result["service"]["timeout"] == 30              # From file
        assert result["zmq"]["port"] == 6666                   # From file
        assert result["zmq"]["host"] == "127.0.0.1"            # From args
        assert result["database"]["url"] == "postgres://localhost/db"  # From override_config
        
        # Test with just default config
        result = load_config_with_overrides(default_config=default_config)
        assert result == default_config
        
        # Test with non-existent file
        result = load_config_with_overrides(
            config_file="non_existent_file.toml",
            default_config=default_config
        )
        assert result == default_config
        
        # Test with empty args
        empty_args = argparse.Namespace()
        result = load_config_with_overrides(
            default_config=default_config,
            args=empty_args
        )
        assert result == default_config
        
    finally:
        # Clean up
        os.unlink(temp_file_path)



# Test the Pydantic Config class and from_overrides method
class MockZmqConfig(BaseModel):
    port: int = 5555
    host: str = "localhost"

class MockServiceConfig(Config):
    name: str = "test-service"
    zmq: MockZmqConfig = Field(default_factory=MockZmqConfig)
    log_level: str = "INFO"
    debug: bool = False


def test_pydantic_config_from_overrides():
    """Test creating a Pydantic config using from_overrides method."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.toml', mode='w+', delete=False) as temp_file:
        file_config = {
            "name": "file-service",
            "zmq": {
                "port": 6666
            }
        }
        toml.dump(file_config, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Test with various combinations
        
        # 1. Config from file
        config1 = MockServiceConfig.from_overrides(config_file=temp_file_path)
        assert config1.name == "file-service"  # From file
        assert config1.zmq.port == 6666        # From file
        assert config1.zmq.host == "localhost" # From default
        assert config1.log_level == "INFO"     # From default
        
        # 2. Config with overrides
        override_config = {"name": "override-service", "debug": True}
        config2 = MockServiceConfig.from_overrides(
            override_config=override_config,
            config_file=temp_file_path
        )
        assert config2.name == "override-service"  # From override
        assert config2.zmq.port == 6666            # From file
        assert config2.debug is True               # From override
        
        # 3. Config with args
        args = argparse.Namespace()
        setattr(args, "log_level", "DEBUG")
        setattr(args, "zmq.host", "127.0.0.1")
        
        config3 = MockServiceConfig.from_overrides(
            config_file=temp_file_path,
            args=args
        )
        assert config3.name == "file-service"   # From file
        assert config3.zmq.port == 6666         # From file
        assert config3.zmq.host == "127.0.0.1"  # From args
        assert config3.log_level == "DEBUG"     # From args
        
        # 4. Full priority chain
        config4 = MockServiceConfig.from_overrides(
            override_config=override_config,
            config_file=temp_file_path,
            args=args
        )
        assert config4.name == "override-service"  # From override (highest priority)
        assert config4.zmq.port == 6666            # From file
        assert config4.zmq.host == "127.0.0.1"     # From args
        assert config4.log_level == "DEBUG"        # From args
        assert config4.debug is True               # From override
        
    finally:
        # Clean up
        os.unlink(temp_file_path)


if __name__ == "__main__":
    pytest.main()
