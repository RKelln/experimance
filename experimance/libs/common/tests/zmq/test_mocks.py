"""
Test suite for ZMQ mock services.

Tests mock implementations for PubSubService, WorkerService, and ControllerService
using the frozen config pattern and message bus architecture.
"""

import asyncio
import pytest
from experimance_common.zmq.mocks import (
    MockPubSubService, MockWorkerService, MockControllerService,
    mock_environment, mock_message_bus
)
from experimance_common.zmq.config import (
    PubSubServiceConfig, WorkerServiceConfig, ControllerServiceConfig,
    PublisherConfig, SubscriberConfig, PushConfig, PullConfig, WorkerConfig
)


class TestMockPubSubService:
    """Test MockPubSubService functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_publishing(self):
        """Test basic message publishing."""
        config = PubSubServiceConfig(
            name="test_publisher",
            publisher=PublisherConfig(address="tcp://*", port=5555)
        )
        
        async with mock_environment():
            async with MockPubSubService(config) as service:
                # Test publishing
                await service.publish("test.topic", {"message": "hello"})
                await service.publish("test.topic", {"message": "world"})
                
                # Check published messages
                published = service.get_published_messages()
                assert len(published) == 2
                assert published[0].topic == "test.topic"
                assert published[0].content == {"message": "hello"}
                assert published[1].content == {"message": "world"}
    
    @pytest.mark.asyncio
    async def test_basic_subscribing(self):
        """Test basic message subscribing."""
        config = PubSubServiceConfig(
            name="test_subscriber",
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5556,
                topics=["events", "alerts"]
            )
        )
        
        received_messages = []
        
        async with mock_environment():
            async with MockPubSubService(config) as service:
                # Set up message handler
                def handler(topic, message):
                    received_messages.append((topic, message))
                
                service.set_message_handler(handler)
                
                # Simulate external messages
                await mock_message_bus.publish("events", {"event": "started"}, "external")
                await mock_message_bus.publish("alerts", {"level": "warning"}, "external")
                await mock_message_bus.publish("other", {"ignored": True}, "external")  # Not subscribed
                
                await asyncio.sleep(0.1)  # Allow processing
                
                # Check received messages
                assert len(received_messages) == 2
                assert received_messages[0] == ("events", {"event": "started"})
                assert received_messages[1] == ("alerts", {"level": "warning"})
                
                # Check service storage
                received = service.get_received_messages()
                assert len(received) == 2
    
    @pytest.mark.asyncio
    async def test_pubsub_bidirectional(self):
        """Test bidirectional pub/sub communication."""
        config = PubSubServiceConfig(
            name="bidirectional_service",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5556,
                topics=["commands"]
            )
        )
        
        received_commands = []
        
        async with mock_environment():
            async with MockPubSubService(config) as service:
                # Set up command handler
                def command_handler(topic, message):
                    received_commands.append(message)
                
                service.set_message_handler(command_handler)
                
                # Simulate receiving a command
                await mock_message_bus.publish("commands", {"action": "start"}, "external")
                await asyncio.sleep(0.1)
                
                # Process command and publish response
                if received_commands:
                    await service.publish("responses", {"status": "started"})
                
                # Verify
                assert len(received_commands) == 1
                assert received_commands[0] == {"action": "start"}
                
                published = service.get_published_messages()
                assert len(published) == 1
                assert published[0].topic == "responses"
    
    @pytest.mark.asyncio
    async def test_service_lifecycle(self):
        """Test service lifecycle management."""
        config = PubSubServiceConfig(
            name="lifecycle_test",
            publisher=PublisherConfig(address="tcp://*", port=5555)
        )
        
        async with mock_environment():
            service = MockPubSubService(config)
            
            # Initially not running
            assert not service.is_running
            assert service.uptime is None
            
            # Start service
            await service.start()
            assert service.is_running
            assert service.uptime is not None
            
            # Can publish when running
            await service.publish("test", {"running": True})
            assert len(service.get_published_messages()) == 1
            
            # Stop service
            await service.stop()
            assert not service.is_running
            
            # Cannot publish when stopped
            with pytest.raises(RuntimeError):
                await service.publish("test", {"stopped": True})


class TestMockWorkerService:
    """Test MockWorkerService functionality."""
    
    @pytest.mark.asyncio
    async def test_worker_service_components(self):
        """Test WorkerService with all components."""
        config = WorkerServiceConfig(
            name="test_worker",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=["tasks"]),
            push=PushConfig(address="tcp://localhost", port=5557),
            pull=PullConfig(address="tcp://*", port=5558)
        )
        
        received_tasks = []
        pulled_work = []
        
        async with mock_environment():
            async with MockWorkerService(config) as service:
                # Set up handlers
                def task_handler(topic, message):
                    received_tasks.append(message)
                
                def work_handler(message):
                    pulled_work.append(message)
                
                service.set_message_handler(task_handler)
                service.set_work_handler(work_handler)
                
                # Test pub/sub functionality
                await mock_message_bus.publish("tasks", {"task": "process"}, "external")
                await asyncio.sleep(0.1)
                
                # Test push functionality
                await service.push({"result": "completed"})
                
                # Test work pulling (simulate external work)
                await mock_message_bus.push_to_worker("test_worker", {"work": "analyze"}, "controller")
                await asyncio.sleep(0.1)
                
                # Verify
                assert len(received_tasks) == 1
                assert received_tasks[0] == {"task": "process"}
                
                assert len(service.pushed_work) == 1
                assert service.pushed_work[0].content == {"result": "completed"}
                
                assert len(pulled_work) == 1
                assert pulled_work[0] == {"work": "analyze"}


class TestMockControllerService:
    """Test MockControllerService functionality."""
    
    @pytest.mark.asyncio
    async def test_controller_with_workers(self):
        """Test ControllerService managing multiple workers."""
        config = ControllerServiceConfig(
            name="test_controller",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=["status"]),
            workers={
                "image_worker": WorkerConfig(
                    name="image_worker",
                    push_config=PushConfig(address="tcp://localhost", port=5557),
                    pull_config=PullConfig(address="tcp://*", port=5558),
                    message_types=["image.process"]
                ),
                "text_worker": WorkerConfig(
                    name="text_worker",
                    push_config=PushConfig(address="tcp://localhost", port=5559),
                    pull_config=PullConfig(address="tcp://*", port=5560),
                    message_types=["text.analyze"]
                )
            }
        )
        
        received_status = []
        image_work = []
        text_work = []
        
        async with mock_environment():
            async with MockControllerService(config) as controller:
                # Set up handlers
                def status_handler(topic, message):
                    received_status.append(message)
                
                def image_handler(message):
                    image_work.append(message)
                
                def text_handler(message):
                    text_work.append(message)
                
                controller.set_message_handler(status_handler)
                controller.set_worker_handler("image_worker", image_handler)
                controller.set_worker_handler("text_worker", text_handler)
                
                # Test publishing status
                await controller.publish("heartbeat", {"controller": "alive"})
                
                # Test worker task distribution
                await controller.push_to_worker("image_worker", {"image": "process.jpg"})
                await controller.push_to_worker("text_worker", {"text": "analyze this"})
                
                # Test status reception
                await mock_message_bus.publish("status", {"worker": "ready"}, "external")
                await asyncio.sleep(0.1)
                
                # Verify
                published = controller.published_messages
                assert len(published) == 1
                assert published[0].topic == "heartbeat"
                
                assert len(received_status) == 1
                assert received_status[0] == {"worker": "ready"}
                
                assert len(image_work) == 1
                assert image_work[0] == {"image": "process.jpg"}
                
                assert len(text_work) == 1
                assert text_work[0] == {"text": "analyze this"}
    
    @pytest.mark.asyncio
    async def test_controller_worker_messages(self):
        """Test controller worker message tracking."""
        config = ControllerServiceConfig(
            name="tracking_controller",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=[]),
            workers={
                "test_worker": WorkerConfig(
                    name="test_worker",
                    push_config=PushConfig(address="tcp://localhost", port=5557),
                    pull_config=PullConfig(address="tcp://*", port=5558)
                )
            }
        )
        
        async with mock_environment():
            async with MockControllerService(config) as controller:
                # Send multiple messages to worker
                await controller.push_to_worker("test_worker", {"task": 1})
                await controller.push_to_worker("test_worker", {"task": 2})
                await asyncio.sleep(0.1)
                
                # Check worker messages
                worker_messages = controller.get_worker_messages("test_worker")
                assert len(worker_messages) == 2
                assert worker_messages[0].content == {"task": 1}
                assert worker_messages[1].content == {"task": 2}
                
                # Test invalid worker
                with pytest.raises(ValueError):
                    controller.get_worker_messages("nonexistent_worker")


class TestMessageBusIntegration:
    """Test global message bus functionality."""
    
    @pytest.mark.asyncio
    async def test_cross_service_communication(self):
        """Test communication between multiple mock services."""
        pub_config = PubSubServiceConfig(
            name="publisher_service",
            publisher=PublisherConfig(address="tcp://*", port=5555)
        )
        
        sub_config = PubSubServiceConfig(
            name="subscriber_service",
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5556,
                topics=["notifications"]
            )
        )
        
        received_notifications = []
        
        async with mock_environment():
            async with MockPubSubService(pub_config) as publisher:
                async with MockPubSubService(sub_config) as subscriber:
                    # Set up subscriber
                    def notification_handler(topic, message):
                        received_notifications.append(message)
                    
                    subscriber.set_message_handler(notification_handler)
                    
                    # Publisher sends notification
                    await publisher.publish("notifications", {"alert": "system ready"})
                    await asyncio.sleep(0.1)
                    
                    # Verify cross-service communication
                    assert len(received_notifications) == 1
                    assert received_notifications[0] == {"alert": "system ready"}
                    
                    # Check both services have the message
                    pub_messages = publisher.get_published_messages()
                    sub_messages = subscriber.get_received_messages()
                    
                    assert len(pub_messages) == 1
                    assert len(sub_messages) == 1
                    assert pub_messages[0].topic == sub_messages[0].topic
    
    @pytest.mark.asyncio
    async def test_message_bus_isolation(self):
        """Test that mock_environment provides clean isolation."""
        # First test run
        async with mock_environment():
            await mock_message_bus.publish("test", {"run": 1}, "test1")
            messages = mock_message_bus.get_messages()
            assert len(messages) == 1
        
        # Second test run - should be clean
        async with mock_environment():
            messages = mock_message_bus.get_messages()
            assert len(messages) == 0  # Clean slate
            
            await mock_message_bus.publish("test", {"run": 2}, "test2")
            messages = mock_message_bus.get_messages()
            assert len(messages) == 1
            assert messages[0].content == {"run": 2}


class TestErrorHandling:
    """Test error handling in mock services."""
    
    @pytest.mark.asyncio
    async def test_service_not_running_errors(self):
        """Test that services raise errors when not running."""
        config = PubSubServiceConfig(
            name="error_test",
            publisher=PublisherConfig(address="tcp://*", port=5555)
        )
        
        async with mock_environment():
            service = MockPubSubService(config)
            
            # Should fail when not running
            with pytest.raises(RuntimeError):
                await service.publish("test", {"should": "fail"})
    
    @pytest.mark.asyncio
    async def test_missing_components_errors(self):
        """Test errors when required components are missing."""
        # Service without publisher
        config = PubSubServiceConfig(
            name="no_publisher",
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=["test"])
        )
        
        async with mock_environment():
            async with MockPubSubService(config) as service:
                # Should fail - no publisher configured
                with pytest.raises(RuntimeError):
                    await service.publish("test", {"no": "publisher"})
    
    @pytest.mark.asyncio
    async def test_handler_error_recovery(self):
        """Test that handler errors don't crash the service."""
        config = PubSubServiceConfig(
            name="error_handler_test",
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=["test"])
        )
        
        async with mock_environment():
            async with MockPubSubService(config) as service:
                # Set up a handler that raises an error
                def failing_handler(topic, message):
                    raise ValueError("Handler error!")
                
                service.set_message_handler(failing_handler)
                
                # Send message - should not crash service
                await mock_message_bus.publish("test", {"should": "survive"}, "external")
                await asyncio.sleep(0.1)
                
                # Service should still be running and have incremented error count
                assert service.is_running
                assert service.error_count > 0
                
                # Message should still be recorded
                received = service.get_received_messages()
                assert len(received) == 1
