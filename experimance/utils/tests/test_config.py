#!/usr/bin/env python3
"""
Tests for the configuration utilities in experimance_common.config and CLI integration.
"""

import argparse
import os
import tempfile
from typing import Optional

import pytest
import toml
from pydantic import BaseModel, Field

from experimance_common.config import (
    load_config_with_overrides,
    deep_merge,
    namespace_to_dict,
    BaseConfig,
    ConfigError
)
from experimance_common.cli import (
    extract_cli_args_from_config,
    create_service_parser,
    TrackedAction,
    TrackedStoreTrueAction,
    TrackedStoreFalseAction
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
    
    # Manually set the tracking attribute to simulate all these were explicitly set
    namespace._explicitly_set = {'name', 'log_level', 'timeout', 'zmq.port', 'database.host', 'database.port', 'database.credentials.username'}
    
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
        # Manually set tracking attribute for these explicitly set args
        args._explicitly_set = {'service.log_level', 'zmq.host'}
        
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

class MockServiceConfig(BaseConfig):
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
        # Manually set tracking attribute for these explicitly set args
        args._explicitly_set = {'log_level', 'zmq.host'}
        
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


# ============================================================================
# CLI INTEGRATION TESTS
# ============================================================================

def test_extract_cli_args_from_config_basic():
    """Test CLI argument extraction from simple Pydantic models."""
    
    class SimpleConfig(BaseModel):
        name: str = Field(default="test", description="Service name")
        port: int = Field(default=5555, description="Port number")
        debug: bool = Field(default=False, description="Enable debug mode")
        timeout: float = Field(default=30.0, description="Timeout in seconds")
        optional_field: Optional[str] = Field(default=None, description="Optional field")
    
    cli_args = extract_cli_args_from_config(SimpleConfig)
    
    # Check that correct number of arguments are generated (excluding optional None field)
    assert len(cli_args) == 4
    
    # Check string field
    assert "--name" in cli_args
    assert cli_args["--name"]["type"] == str
    assert "Service name" in cli_args["--name"]["help"]
    assert cli_args["--name"]["dest"] == "name"
    
    # Check int field
    assert "--port" in cli_args
    assert cli_args["--port"]["type"] == int
    assert "Port number" in cli_args["--port"]["help"]
    assert cli_args["--port"]["dest"] == "port"
    
    # Check boolean field (default False -> TrackedStoreTrueAction)
    assert "--debug" in cli_args
    assert cli_args["--debug"]["action"] == TrackedStoreTrueAction
    assert "Enable debug mode" in cli_args["--debug"]["help"]
    assert cli_args["--debug"]["dest"] == "debug"
    
    # Check float field
    assert "--timeout" in cli_args
    assert cli_args["--timeout"]["type"] == float
    assert "Timeout in seconds" in cli_args["--timeout"]["help"]
    assert cli_args["--timeout"]["dest"] == "timeout"


def test_extract_cli_args_from_config_boolean_defaults():
    """Test boolean field handling with different default values."""
    
    class BoolConfig(BaseModel):
        enabled: bool = Field(default=True, description="Feature enabled")
        disabled: bool = Field(default=False, description="Feature disabled")
    
    cli_args = extract_cli_args_from_config(BoolConfig)
    
    # True default should create --no-enabled flag with TrackedStoreFalseAction
    assert "--no-enabled" in cli_args
    assert cli_args["--no-enabled"]["action"] == TrackedStoreFalseAction
    assert cli_args["--no-enabled"]["dest"] == "enabled"
    
    # False default should create --disabled flag with TrackedStoreTrueAction
    assert "--disabled" in cli_args
    assert cli_args["--disabled"]["action"] == TrackedStoreTrueAction
    assert cli_args["--disabled"]["dest"] == "disabled"


def test_extract_cli_args_from_config_nested():
    """Test CLI argument extraction from nested Pydantic models."""
    
    class DatabaseConfig(BaseModel):
        host: str = Field(default="localhost", description="Database host")
        port: int = Field(default=5432, description="Database port")
        ssl: bool = Field(default=True, description="Use SSL connection")
    
    class ServiceConfig(BaseModel):
        name: str = Field(default="service", description="Service name")
        database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
        debug: bool = Field(default=False, description="Debug mode")
    
    cli_args = extract_cli_args_from_config(ServiceConfig)
    
    # Check top-level fields
    assert "--name" in cli_args
    assert cli_args["--name"]["dest"] == "name"
    assert "--debug" in cli_args
    assert cli_args["--debug"]["dest"] == "debug"
    
    # Check nested fields use dotted notation for dest
    assert "--database-host" in cli_args
    assert cli_args["--database-host"]["dest"] == "database.host"
    assert "[Database] Database host" in cli_args["--database-host"]["help"]
    
    assert "--database-port" in cli_args
    assert cli_args["--database-port"]["dest"] == "database.port"
    assert "[Database] Database port" in cli_args["--database-port"]["help"]
    
    # Check nested boolean with True default
    assert "--no-database-ssl" in cli_args
    assert cli_args["--no-database-ssl"]["action"] == TrackedStoreFalseAction
    assert cli_args["--no-database-ssl"]["dest"] == "database.ssl"


def test_extract_cli_args_from_config_deep_nesting():
    """Test CLI argument extraction with multiple levels of nesting."""
    
    class AuthConfig(BaseModel):
        username: str = Field(default="user", description="Username")
        password: str = Field(default="pass", description="Password")
    
    class DatabaseConfig(BaseModel):
        host: str = Field(default="localhost", description="Host")
        auth: AuthConfig = Field(default_factory=AuthConfig, description="Authentication")
    
    class AppConfig(BaseModel):
        name: str = Field(default="app", description="App name")
        database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database config")
    
    cli_args = extract_cli_args_from_config(AppConfig)
    
    # Check deeply nested fields
    assert "--database-auth-username" in cli_args
    assert cli_args["--database-auth-username"]["dest"] == "database.auth.username"
    assert "[Database Auth] Username" in cli_args["--database-auth-username"]["help"]
    
    assert "--database-auth-password" in cli_args
    assert cli_args["--database-auth-password"]["dest"] == "database.auth.password"
    assert "[Database Auth] Password" in cli_args["--database-auth-password"]["help"]


def test_create_service_parser_with_config_class():
    """Test service parser creation with automatic CLI argument generation."""
    
    class TestConfig(BaseModel):
        name: str = Field(default="test-service", description="Service name")
        port: int = Field(default=8080, description="Port number")
        debug: bool = Field(default=False, description="Debug mode")
    
    parser = create_service_parser(
        service_name="Test",
        description="Test service",
        default_config_path="/tmp/config.toml",
        config_class=TestConfig
    )
    
    # Test that parser can parse generated arguments
    args = parser.parse_args(["--name", "custom-service", "--port", "9090", "--debug"])
    
    assert args.name == "custom-service"
    assert args.port == 9090
    assert args.debug is True
    assert args.config == "/tmp/config.toml"
    assert args.log_level == "INFO"  # Default value


def test_create_service_parser_nested_config_integration():
    """Test parser with nested config and argument parsing."""
    
    class DatabaseConfig(BaseModel):
        host: str = Field(default="localhost", description="Database host")
        port: int = Field(default=5432, description="Database port")
        ssl: bool = Field(default=True, description="Use SSL")
    
    class ServiceConfig(BaseModel):
        name: str = Field(default="service", description="Service name")
        database: DatabaseConfig = Field(default_factory=DatabaseConfig)
        timeout: float = Field(default=30.0, description="Request timeout")
    
    parser = create_service_parser(
        service_name="Test",
        description="Test service",
        config_class=ServiceConfig
    )
    
    # Test parsing nested arguments
    args = parser.parse_args([
        "--name", "my-service",
        "--database-host", "db.example.com",
        "--database-port", "3306",
        "--no-database-ssl",
        "--timeout", "60.0"
    ])
    
    # Verify namespace contains dotted notation
    assert args.name == "my-service"
    assert getattr(args, "database.host") == "db.example.com"
    assert getattr(args, "database.port") == 3306
    assert getattr(args, "database.ssl") is False  # --no-database-ssl
    assert args.timeout == 60.0


def test_namespace_to_dict_with_cli_args():
    """Test that CLI-generated namespaces convert correctly to nested dicts."""
    
    # Simulate a namespace created by our CLI system
    namespace = argparse.Namespace()
    namespace.name = "test-service"
    namespace.debug = True
    setattr(namespace, "database.host", "localhost")
    setattr(namespace, "database.port", 5432)
    setattr(namespace, "database.auth.username", "user")
    setattr(namespace, "cache.redis.host", "redis.example.com")
    # Simulate that all these were explicitly set
    namespace._explicitly_set = {'name', 'debug', 'database.host', 'database.port', 'database.auth.username', 'cache.redis.host'}
    
    result = namespace_to_dict(namespace)
    
    expected = {
        "name": "test-service",
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
            "auth": {
                "username": "user"
            }
        },
        "cache": {
            "redis": {
                "host": "redis.example.com"
            }
        }
    }
    
    assert result == expected


def test_config_from_overrides_with_cli_integration():
    """Test complete integration of CLI args with Config.from_overrides."""
    
    class DatabaseConfig(BaseModel):
        host: str = Field(default="localhost", description="Database host")
        port: int = Field(default=5432, description="Database port")
        ssl: bool = Field(default=True, description="Use SSL")
    
    class TestServiceConfig(BaseConfig):
        name: str = Field(default="test-service", description="Service name")
        database: DatabaseConfig = Field(default_factory=DatabaseConfig)
        debug: bool = Field(default=False, description="Debug mode")
        timeout: float = Field(default=30.0, description="Timeout")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.toml', mode='w+', delete=False) as temp_file:
        file_config = {
            "name": "file-service",
            "database": {
                "host": "file-db.com",
                "port": 3306
            },
            "timeout": 45.0
        }
        toml.dump(file_config, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Use actual CLI parser to create proper namespace with tracking
        parser = create_service_parser(
            service_name="Test",
            description="CLI integration test",
            config_class=TestServiceConfig
        )
        
        # Parse CLI args that should override file values
        args = parser.parse_args(['--debug', '--database-host', 'cli-db.com', '--no-database-ssl'])
        
        # Create config with all sources
        config = TestServiceConfig.from_overrides(
            config_file=temp_file_path,
            args=args
        )
        
        # Verify priority order
        assert config.name == "file-service"      # From file
        assert config.debug is True               # From CLI (highest priority)
        assert config.database.host == "cli-db.com"  # From CLI (overrides file)
        assert config.database.port == 3306      # From file
        assert config.database.ssl is False      # From CLI (overrides default)
        assert config.timeout == 45.0            # From file
        
    finally:
        os.unlink(temp_file_path)


def test_end_to_end_cli_workflow():
    """Test complete workflow from config class to CLI parsing to config creation."""
    
    class CacheConfig(BaseModel):
        redis_host: str = Field(default="localhost", description="Redis host")
        redis_port: int = Field(default=6379, description="Redis port")
        enabled: bool = Field(default=True, description="Enable caching")
    
    class ServiceConfig(BaseConfig):
        service_name: str = Field(default="my-service", description="Service name")
        port: int = Field(default=8080, description="Service port")
        cache: CacheConfig = Field(default_factory=CacheConfig)
        verbose: bool = Field(default=False, description="Verbose logging")
    
    # 1. Create parser with auto-generated args
    parser = create_service_parser(
        service_name="TestService",
        description="End-to-end test service",
        config_class=ServiceConfig
    )
    
    # 2. Parse CLI arguments
    cli_args = [
        "--service-name", "production-service",
        "--port", "9000",
        "--cache-redis-host", "prod-redis.com",
        "--no-cache-enabled",
        "--verbose"
    ]
    args = parser.parse_args(cli_args)
    
    # 3. Create config from CLI args
    config = ServiceConfig.from_overrides(args=args)
    
    # 4. Verify final configuration
    assert config.service_name == "production-service"
    assert config.port == 9000
    assert config.cache.redis_host == "prod-redis.com"
    assert config.cache.redis_port == 6379  # Default value
    assert config.cache.enabled is False    # --no-cache-enabled
    assert config.verbose is True           # --verbose
    
    # 5. Verify help text contains proper information
    help_text = parser.format_help()
    assert "--service-name" in help_text
    assert "--cache-redis-host" in help_text
    assert "--no-cache-enabled" in help_text


def test_cli_args_with_special_characters():
    """Test CLI arg generation with fields that have special characters."""
    
    class SpecialConfig(BaseModel):
        field_with_underscores: str = Field(default="test", description="Field with underscores")
        field123: int = Field(default=42, description="Field with numbers")
        # Note: Pydantic field names can't have hyphens, so we test underscore conversion
        
    args = extract_cli_args_from_config(SpecialConfig)
    
    # Should convert underscores to hyphens in CLI arg names
    assert '--field-with-underscores' in args
    assert '--field123' in args
    
    # Check that dest uses original field names (with underscores)
    assert args['--field-with-underscores']['dest'] == 'field_with_underscores'
    assert args['--field123']['dest'] == 'field123'


def test_cli_args_ignore_unsupported_types():
    """Test that CLI arg generation ignores complex types we can't handle."""
    
    class ComplexConfig(BaseModel):
        simple_str: str = Field(default="test")
        complex_list: list = Field(default_factory=list, description="This should be ignored")
        complex_dict: dict = Field(default_factory=dict, description="This should be ignored")
        tuple_field: tuple = Field(default=(), description="This should be ignored")
        
    args = extract_cli_args_from_config(ComplexConfig)
    
    # Should only include the simple string field
    assert '--simple-str' in args
    assert '--complex-list' not in args
    assert '--complex-dict' not in args
    assert '--tuple-field' not in args


def test_boolean_field_variations():
    """Test different boolean field default combinations."""
    
    class BoolConfig(BaseModel):
        default_false: bool = Field(default=False, description="Defaults to False")
        default_true: bool = Field(default=True, description="Defaults to True")
        no_default: bool = Field(description="No explicit default")
        
    args = extract_cli_args_from_config(BoolConfig)
    
    # False defaults should create --flag (TrackedStoreTrueAction)
    assert '--default-false' in args
    assert args['--default-false']['action'] == TrackedStoreTrueAction
    
    # True defaults should create --no-flag (TrackedStoreFalseAction)
    assert '--no-default-true' in args
    assert args['--no-default-true']['action'] == TrackedStoreFalseAction
    
    # No explicit default should be treated as False (TrackedStoreTrueAction)
    assert '--no-default' in args
    assert args['--no-default']['action'] == TrackedStoreTrueAction


def test_cli_integration_with_real_parser():
    """Test full integration with argparse parser."""
    
    class TestConfig(BaseModel):
        debug: bool = Field(default=False, description="Enable debug mode")
        port: int = Field(default=8080, description="Server port")
        host: str = Field(default="localhost", description="Server host")
        timeout: float = Field(default=30.0, description="Request timeout")
        
    class NestedConfig(BaseModel):
        test: TestConfig = Field(default_factory=TestConfig)
        app_name: str = Field(default="test-app", description="Application name")
        
    parser = create_service_parser(
        service_name="Test",
        description="Test service",
        config_class=NestedConfig
    )
    
    # Test parsing various argument combinations
    args1 = parser.parse_args(['--test-debug', '--test-port', '9000'])
    assert getattr(args1, 'test.debug') is True
    assert getattr(args1, 'test.port') == 9000
    
    args2 = parser.parse_args(['--test-host', 'example.com', '--app-name', 'my-app'])
    assert getattr(args2, 'test.host') == 'example.com'
    assert getattr(args2, 'app_name') == 'my-app'


def test_namespace_to_dict_edge_cases():
    """Test namespace_to_dict with edge cases."""
    
    # Test with None values (should be ignored)
    ns = argparse.Namespace()
    ns.some_field = "value"
    ns.none_field = None
    setattr(ns, 'nested.field', 'nested_value')
    # Manually set tracking attribute to include the non-None values
    ns._explicitly_set = {'some_field', 'nested.field'}
    
    result = namespace_to_dict(ns)
    
    assert result == {
        'some_field': 'value',
        'nested': {
            'field': 'nested_value'
        }
    }
    # None values should be omitted
    assert 'none_field' not in result


def test_config_from_overrides_priority():
    """Test that CLI args have highest priority in Config.from_overrides."""
    
    class PriorityConfig(BaseConfig):
        value: str = Field(default="default")
        number: int = Field(default=100)
        
    # Create test TOML content
    toml_content = {
        'value': 'from_toml',
        'number': 200
    }
    
    # Create args namespace with CLI overrides
    args = argparse.Namespace()
    args.value = 'from_cli'
    # Manually set tracking attribute to indicate this was explicitly set
    args._explicitly_set = {'value'}
    # Don't set number in CLI, should come from TOML
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(toml_content, f)
        temp_file = f.name
    
    try:
        config = PriorityConfig.from_overrides(
            config_file=temp_file,
            args=args
        )
        
        # CLI should override TOML
        assert config.value == 'from_cli'
        # TOML should override default
        assert config.number == 200
        
    finally:
        os.unlink(temp_file)


def test_empty_config_class():
    """Test CLI generation with empty config class."""
    
    class EmptyConfig(BaseModel):
        pass
        
    args = extract_cli_args_from_config(EmptyConfig)
    assert args == {}


def test_cli_metavar_types():
    """Test that metavar types are set correctly for different field types."""
    
    class MetavarConfig(BaseModel):
        int_field: int = Field(default=42)
        float_field: float = Field(default=3.14)
        str_field: str = Field(default="test")
        bool_field: bool = Field(default=False)
        
    args = extract_cli_args_from_config(MetavarConfig)
    
    assert args['--int-field']['metavar'] == 'N'
    assert args['--float-field']['metavar'] == 'VALUE'
    assert args['--str-field']['metavar'] == 'TEXT'
    # Boolean fields shouldn't have metavar
    assert 'metavar' not in args['--bool-field']


def test_cli_args_dont_override_config_with_defaults():
    """
    CRITICAL TEST: Ensure CLI boolean defaults don't override config file values.
    
    This test catches the bug where argparse sets boolean flags to their default
    values even when not specified by the user, which then overrides config file
    values inappropriately.
    """
    
    class BooleanConfig(BaseConfig):
        # These booleans have opposite defaults in the class vs config file
        feature_a: bool = Field(default=True, description="Feature A")
        feature_b: bool = Field(default=False, description="Feature B") 
        feature_c: bool = Field(default=True, description="Feature C")
        non_bool_field: str = Field(default="default", description="Non-boolean field")
    
    # Create config file with values opposite to the class defaults
    toml_content = {
        'feature_a': False,  # Class default is True
        'feature_b': True,   # Class default is False
        'feature_c': False,  # Class default is True
        'non_bool_field': 'from_config'
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(toml_content, f)
        temp_file = f.name
    
    try:
        # Create a parser and parse EMPTY CLI args (simulating user didn't specify any flags)
        parser = create_service_parser(
            service_name="Test",
            description="Test boolean override bug",
            config_class=BooleanConfig
        )
        
        # Parse empty args - this would set all booleans to their defaults
        args = parser.parse_args([])
        
        # Now create config using both file and args
        config = BooleanConfig.from_overrides(
            config_file=temp_file,
            args=args
        )
        
        # CRITICAL: Config file values should NOT be overridden by CLI defaults
        assert config.feature_a is False, f"Expected False from config file, got {config.feature_a} (CLI default override bug!)"
        assert config.feature_b is True, f"Expected True from config file, got {config.feature_b} (CLI default override bug!)"
        assert config.feature_c is False, f"Expected False from config file, got {config.feature_c} (CLI default override bug!)"
        assert config.non_bool_field == 'from_config', f"Expected 'from_config', got {config.non_bool_field}"
        
        # Now test that EXPLICIT CLI args DO override config file
        explicit_args = parser.parse_args(['--no-feature-a', '--feature-b'])  # Explicitly set some flags
        
        explicit_config = BooleanConfig.from_overrides(
            config_file=temp_file,
            args=explicit_args
        )
        
        # Explicitly set CLI args should override config file
        assert explicit_config.feature_a is False, "Explicit --no-feature-a should override config file" 
        assert explicit_config.feature_b is True, "Explicit --feature-b should override config file" 
        # feature_c wasn't specified on CLI, should come from config file
        assert explicit_config.feature_c is False, "Unspecified feature_c should come from config file"
        
    finally:
        os.unlink(temp_file)


def test_namespace_to_dict_only_includes_explicitly_set_values():
    """
    Test that namespace_to_dict only includes values that were explicitly provided,
    not argparse defaults.
    """
    
    class TestConfig(BaseModel):
        explicit_field: str = Field(default="default_value")
        bool_field: bool = Field(default=False)
    
    parser = create_service_parser(
        service_name="Test", 
        description="Test explicit args only",
        config_class=TestConfig
    )
    
    # Parse with no arguments - should result in empty dict after filtering
    empty_args = parser.parse_args([])
    
    # The key fix: namespace_to_dict should only include explicitly set values
    args_dict = namespace_to_dict(empty_args)
    
    # Should NOT include boolean defaults that weren't explicitly set
    assert 'bool_field' not in args_dict, "Boolean defaults should not be included when not explicitly set"
    assert 'explicit_field' not in args_dict, "String defaults should not be included when not explicitly set"
    
    # Should only include the built-in args like log_level, config, visualize
    expected_builtin_args = {'log_level', 'config', 'visualize'}  # Standard args from create_service_parser
    actual_keys = set(args_dict.keys())
    
    # All keys should be built-in args, not config-generated defaults
    unexpected_keys = actual_keys - expected_builtin_args
    assert len(unexpected_keys) == 0, f"Unexpected default values found: {unexpected_keys}"
    
    # Now test with explicit arguments
    explicit_args = parser.parse_args(['--explicit-field', 'user_value', '--bool-field'])
    explicit_dict = namespace_to_dict(explicit_args)
    
    # These should be included because they were explicitly set
    assert 'explicit_field' in explicit_dict, "Explicitly set string field should be included"
    assert explicit_dict['explicit_field'] == 'user_value'
    assert 'bool_field' in explicit_dict, "Explicitly set boolean field should be included"
    assert explicit_dict['bool_field'] is True


def test_config_priority_with_mixed_sources():
    """
    Test the complete priority chain: CLI (explicit) > override_config > file > defaults.
    This ensures the priority order is maintained and defaults don't interfere.
    """
    
    class PriorityTestConfig(BaseConfig):
        field1: str = Field(default="class_default")
        field2: bool = Field(default=False) 
        field3: int = Field(default=100)
        field4: str = Field(default="another_default")
    
    # Config file values
    toml_content = {
        'field1': 'from_file',
        'field2': True,
        'field3': 200
        # field4 not in file, should use class default
    }
    
    # Override config values  
    override_config = {
        'field1': 'from_override',
        # field2 not in override, should use file value
        'field3': 300,
        'field4': 'from_override'
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(toml_content, f)
        temp_file = f.name
    
    try:
        parser = create_service_parser(
            service_name="Priority",
            description="Test priority",
            config_class=PriorityTestConfig
        )
        
        # CLI args - only set field1 explicitly
        cli_args = parser.parse_args(['--field1', 'from_cli'])
        
        config = PriorityTestConfig.from_overrides(
            override_config=override_config,
            config_file=temp_file,
            args=cli_args
        )
        
        # Verify priority order
        assert config.field1 == 'from_cli',      "CLI should have highest priority"
        assert config.field2 is True,            "File should override class default"  
        assert config.field3 == 300,             "Override should beat file"
        assert config.field4 == 'from_override', "Override should beat class default"
        
    finally:
        os.unlink(temp_file)
