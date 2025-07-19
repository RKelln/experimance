"""
Test suite for ZMQ configuration schemas.

Tests Pydantic configuration models including validation, frozen config pattern,
and integration with BaseConfig system.
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from experimance_common.zmq.config import (
    ZmqSocketConfig, PublisherConfig, SubscriberConfig, PushConfig, PullConfig,
    ControllerPushConfig, ControllerPullConfig, WorkerPushConfig, WorkerPullConfig,
    WorkerConfig, PubSubServiceConfig, WorkerServiceConfig, ControllerServiceConfig,
    MessageType
)


class TestZmqSocketConfig:
    """Test core ZMQ socket configuration."""
    
    def test_valid_tcp_config(self):
        """Test valid TCP configuration."""
        config = ZmqSocketConfig(
            address="tcp://*",
            port=5555
        )
        assert config.address == "tcp://*"
        assert config.port == 5555
        assert config.bind is True  # default
        assert config.full_address == "tcp://*:5555"
    
    def test_address_validation(self):
        """Test address protocol validation."""
        # Valid protocols
        for protocol in ["tcp://", "ipc://", "inproc://"]:
            config = ZmqSocketConfig(address=f"{protocol}test", port=5555)
            assert config.address.startswith(protocol)
        
        # Invalid protocol
        with pytest.raises(ValidationError):
            ZmqSocketConfig(address="invalid://test", port=5555)
    
    def test_port_validation(self):
        """Test port range validation."""
        # Valid ports
        ZmqSocketConfig(address="tcp://*", port=1024)
        ZmqSocketConfig(address="tcp://*", port=65535)
        
        # Invalid ports
        with pytest.raises(ValidationError):
            ZmqSocketConfig(address="tcp://*", port=1023)  # Too low
        
        with pytest.raises(ValidationError):
            ZmqSocketConfig(address="tcp://*", port=65536)  # Too high
    
    def test_config_immutability(self):
        """Test that socket configs are frozen."""
        config = ZmqSocketConfig(address="tcp://*", port=5555)
        
        with pytest.raises(ValidationError):
            config.port = 6666  # Should fail - config is frozen


class TestSocketSpecificConfigs:
    """Test socket-specific configuration classes."""
    
    def test_publisher_config_defaults(self):
        """Test PublisherConfig defaults."""
        config = PublisherConfig(address="tcp://*", port=5555)
        assert config.bind is True  # Publishers typically bind
        assert config.default_topic is None
    
    def test_subscriber_config_defaults(self):
        """Test SubscriberConfig defaults."""
        config = SubscriberConfig(address="tcp://localhost", port=5555)
        assert config.bind is False  # Subscribers typically connect
        assert config.topics == []
    
    def test_subscriber_with_topics(self):
        """Test SubscriberConfig with topics."""
        topics = ["topic1", "topic2"]
        config = SubscriberConfig(
            address="tcp://localhost", 
            port=5555, 
            topics=topics
        )
        assert config.topics == topics
    
    def test_push_pull_defaults(self):
        """Test Push/Pull config defaults."""
        push_config = PushConfig(address="tcp://localhost", port=5555)
        assert push_config.bind is False  # Push typically connects
        
        pull_config = PullConfig(address="tcp://*", port=5556)
        assert pull_config.bind is True  # Pull typically binds


class TestWorkerConfig:
    """Test WorkerConfig validation and defaults."""
    
    def test_valid_worker_config(self):
        """Test valid worker configuration."""
        push_config = ControllerPushConfig(port=5555)
        pull_config = ControllerPullConfig(port=5556)
        
        config = WorkerConfig(
            name="test_worker",
            push_config=push_config,
            pull_config=pull_config,
            message_types=["image.process", "text.analyze"]
        )
        
        assert config.name == "test_worker"
        assert config.push_config == push_config
        assert config.pull_config == pull_config
        assert config.message_types == ["image.process", "text.analyze"]
        assert config.max_queue_size == 1000  # default
    
    def test_worker_name_validation(self):
        """Test worker name validation."""
        push_config = ControllerPushConfig(port=5555)
        pull_config = ControllerPullConfig(port=5556)
        
        # Empty name should fail
        with pytest.raises(ValidationError):
            WorkerConfig(
                name="",
                push_config=push_config,
                pull_config=pull_config
            )
        
        # Whitespace-only name should fail
        with pytest.raises(ValidationError):
            WorkerConfig(
                name="   ",
                push_config=push_config,
                pull_config=pull_config
            )
    
    def test_worker_config_immutability(self):
        """Test that worker configs are frozen."""
        push_config = ControllerPushConfig(port=5555)
        pull_config = ControllerPullConfig(port=5556)
        
        config = WorkerConfig(
            name="test_worker",
            push_config=push_config,
            pull_config=pull_config
        )
        
        with pytest.raises(ValidationError):
            config.name = "changed"  # Should fail - config is frozen


class TestServiceConfigs:
    """Test service-level configuration classes."""
    
    def test_pubsub_service_config(self):
        """Test PubSubServiceConfig."""
        pub_config = PublisherConfig(address="tcp://*", port=5555)
        sub_config = SubscriberConfig(
            address="tcp://localhost", 
            port=5556, 
            topics=["test"]
        )
        
        config = PubSubServiceConfig(
            name="test_pubsub",
            publisher=pub_config,
            subscriber=sub_config,
            timeout=5.0
        )
        
        assert config.name == "test_pubsub"
        assert config.publisher == pub_config
        assert config.subscriber == sub_config
        assert config.timeout == 5.0
    
    def test_pubsub_optional_components(self):
        """Test PubSubServiceConfig with optional components."""
        # Publisher only
        config = PubSubServiceConfig(
            name="pub_only",
            publisher=PublisherConfig(address="tcp://*", port=5555)
        )
        assert config.publisher is not None
        assert config.subscriber is None
        
        # Subscriber only
        config = PubSubServiceConfig(
            name="sub_only",
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=["test"])
        )
        assert config.publisher is None
        assert config.subscriber is not None
    
    def test_controller_service_config(self):
        """Test ControllerServiceConfig with workers."""
        pub_config = PublisherConfig(address="tcp://*", port=5555)
        sub_config = SubscriberConfig(address="tcp://localhost", port=5556, topics=["status"])
        
        worker1 = WorkerConfig(
            name="worker1",
            push_config=ControllerPushConfig(port=5557),
            pull_config=ControllerPullConfig(port=5558)
        )
        
        worker2 = WorkerConfig(
            name="worker2",
            push_config=ControllerPushConfig(port=5559),
            pull_config=ControllerPullConfig(port=5560)
        )
        
        config = ControllerServiceConfig(
            name="test_controller",
            publisher=pub_config,
            subscriber=sub_config,
            workers={"worker1": worker1, "worker2": worker2}
        )
        
        assert config.name == "test_controller"
        assert len(config.workers) == 2
        assert "worker1" in config.workers
        assert "worker2" in config.workers
    
    def test_controller_port_conflict_validation(self):
        """Test that ControllerServiceConfig detects port conflicts."""
        pub_config = PublisherConfig(address="tcp://*", port=5555)
        sub_config = SubscriberConfig(address="tcp://localhost", port=5556, topics=["status"])
        
        # Create workers with conflicting ports
        worker1 = WorkerConfig(
            name="worker1",
            push_config=ControllerPushConfig(port=5557),
            pull_config=ControllerPullConfig(port=5558)
        )
        
        worker2 = WorkerConfig(
            name="worker2",
            push_config=ControllerPushConfig(port=5557),  # Same port!
            pull_config=ControllerPullConfig(port=5559)
        )
        
        with pytest.raises(ValidationError, match="Port conflict"):
            ControllerServiceConfig(
                name="test_controller",
                publisher=pub_config,
                subscriber=sub_config,
                workers={"worker1": worker1, "worker2": worker2}
            )


class TestMessageTypes:
    """Test MessageType enum."""
    
    def test_message_type_values(self):
        """Test that MessageType enum has expected values."""
        assert MessageType.IMAGE_READY == "ImageReady"
        assert MessageType.SPACE_TIME_UPDATE == "EraChanged"
        
        # Test string conversion
        assert str(MessageType.IMAGE_READY) == "ImageReady"
    
    def test_message_type_in_config(self):
        """Test using MessageType in configuration."""
        config = PublisherConfig(
            address="tcp://*", 
            port=5555,
            default_topic=MessageType.IMAGE_READY
        )
        # The validator should convert enum to string
        assert isinstance(config.default_topic, str)
        assert config.default_topic == "ImageReady"


class TestConfigIntegration:
    """Test integration with BaseConfig system."""
    
    def test_baseconfig_inheritance(self):
        """Test that service configs inherit from BaseConfig."""
        config = PubSubServiceConfig(name="test")
        
        # Should have BaseConfig methods
        assert hasattr(config, 'from_overrides')
        assert hasattr(config, 'model_copy')
        assert hasattr(config, 'model_dump')
    
    def test_from_overrides_integration(self):
        """Test from_overrides functionality."""
        base_config = PubSubServiceConfig(
            name="base",
            timeout=1.0
        )
        
        # Override with new values using model_copy
        override_config = base_config.model_copy(
            update={"timeout": 5.0, "name": "overridden"}
        )
        
        assert override_config.name == "overridden"
        assert override_config.timeout == 5.0
    
    def test_config_serialization(self):
        """Test config serialization/deserialization."""
        pub_config = PublisherConfig(address="tcp://*", port=5555)
        
        # Serialize to dict
        config_dict = pub_config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["address"] == "tcp://*"
        assert config_dict["port"] == 5555
        
        # Deserialize from dict
        new_config = PublisherConfig(**config_dict)
        assert new_config == pub_config


class TestFactoryFunctions:
    """Test factory functions for quick setup."""
    
    def test_local_pubsub_factory(self):
        """Test create_local_pubsub_config factory function."""
        from experimance_common.zmq.config import create_local_pubsub_config
        
        config = create_local_pubsub_config(
            name="test-factory",
            pub_port=5555,
            sub_port=5556,
            sub_topics=["test", "factory"],
            default_pub_topic="general"
        )
        
        assert config.name == "test-factory"
        assert config.publisher is not None
        assert config.publisher.port == 5555
        assert config.subscriber is not None
        assert config.subscriber.port == 5556
        assert config.subscriber.topics == ["test", "factory"]
        assert config.publisher.default_topic == "general"
    
    def test_local_controller_factory(self):
        """Test create_local_controller_config factory function."""
        from experimance_common.zmq.config import create_local_controller_config
        
        config = create_local_controller_config(
            name="test-controller",
            pub_port=5555,
            sub_port=5556,
            worker_configs={
                "worker1": {"push": 5557, "pull": 5558},
                "worker2": {"push": 5559, "pull": 5560}
            }
        )
        
        assert config.name == "test-controller"
        assert config.publisher.port == 5555
        assert config.subscriber.port == 5556
        assert len(config.workers) == 2
        assert "worker1" in config.workers
        assert "worker2" in config.workers
        assert config.workers["worker1"].push_config.port == 5557
        assert config.workers["worker1"].pull_config.port == 5558
