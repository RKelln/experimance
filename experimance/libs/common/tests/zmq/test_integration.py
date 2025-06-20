"""
Test suite for ZMQ service integration patterns.

Tests various communication patterns, binding configurations, and real-world
integration scenarios using mock services.
"""

import asyncio
import pytest
from experimance_common.zmq.mocks import (
    MockPubSubService, MockControllerService, mock_environment
)
from experimance_common.zmq.config import (
    PubSubServiceConfig, ControllerServiceConfig,
    PublisherConfig, SubscriberConfig, WorkerConfig,
    PushConfig, PullConfig, MessageType,
    ControllerPushConfig, ControllerPullConfig, WorkerPushConfig, WorkerPullConfig
)


class TestCommunicationPatterns:
    """Test various ZMQ communication patterns."""
    
    @pytest.mark.asyncio
    async def test_push_pull_binding_patterns(self):
        """Test correct PUSH/PULL binding patterns for controller/worker setup."""
        # Controller config - controller binds both PUSH and PULL (acts as server)
        controller_config = ControllerServiceConfig(
            name="pattern_controller",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(address="tcp://localhost", port=5556, topics=["status"]),
            workers={
                "test_worker": WorkerConfig(
                    name="test_worker",
                    push_config=ControllerPushConfig(address="tcp://*", port=5557),  # Bind to distribute work
                    pull_config=ControllerPullConfig(address="tcp://*", port=5558),  # Bind for results
                    message_types=["test.work"]
                )
            }
        )
        
        # Verify binding patterns
        worker_config = controller_config.workers["test_worker"]
        
        # Controller PUSH should bind (bind=True) to distribute work
        assert worker_config.push_config.bind is True
        assert worker_config.push_config.address == "tcp://*"
        
        # Controller PULL should bind (bind=True) to collect results
        assert worker_config.pull_config.bind is True
        assert worker_config.pull_config.address == "tcp://*"
        
        # Test the pattern works in practice
        work_received = []
        
        async with mock_environment():
            async with MockControllerService(controller_config) as controller:
                def work_handler(message):
                    work_received.append(message)
                
                controller.set_worker_handler("test_worker", work_handler)
                
                # Controller distributes work
                await controller.push_to_worker("test_worker", {"task": "process_data"})
                await asyncio.sleep(0.1)
                
                assert len(work_received) == 1
                assert work_received[0] == {"task": "process_data"}
    
    @pytest.mark.asyncio
    async def test_pubsub_topic_filtering(self):
        """Test pub/sub topic filtering and multiple subscribers."""
        # Create publisher
        pub_config = PubSubServiceConfig(
            name="topic_publisher",
            publisher=PublisherConfig(address="tcp://*", port=5555)
        )
        
        # Create subscribers with different topic filters
        events_sub_config = PubSubServiceConfig(
            name="events_subscriber",
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5556,
                topics=["events.user", "events.system"]
            )
        )
        
        alerts_sub_config = PubSubServiceConfig(
            name="alerts_subscriber",
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5557,
                topics=["alerts.error", "alerts.warning"]
            )
        )
        
        events_received = []
        alerts_received = []
        
        async with mock_environment():
            async with MockPubSubService(pub_config) as publisher:
                async with MockPubSubService(events_sub_config) as events_sub:
                    async with MockPubSubService(alerts_sub_config) as alerts_sub:
                        # Set up handlers
                        def events_handler(topic, message):
                            events_received.append((topic, message))
                        
                        def alerts_handler(topic, message):
                            alerts_received.append((topic, message))
                        
                        events_sub.set_message_handler(events_handler)
                        alerts_sub.set_message_handler(alerts_handler)
                        
                        # Publish various messages
                        await publisher.publish({"action": "login"}, "events.user")
                        await publisher.publish({"status": "startup"}, "events.system")
                        await publisher.publish({"error": "database_down"}, "alerts.error")
                        await publisher.publish({"warning": "high_memory"}, "alerts.warning")
                        await publisher.publish({"ignored": True}, "other.topic")
                        
                        await asyncio.sleep(0.1)
                        
                        # Verify topic filtering
                        assert len(events_received) == 2
                        assert events_received[0] == ("events.user", {"action": "login"})
                        assert events_received[1] == ("events.system", {"status": "startup"})
                        
                        assert len(alerts_received) == 2
                        assert alerts_received[0] == ("alerts.error", {"error": "database_down"})
                        assert alerts_received[1] == ("alerts.warning", {"warning": "high_memory"})
    
    @pytest.mark.asyncio
    async def test_message_type_enum_integration(self):
        """Test using MessageType enum in real scenarios."""
        config = PubSubServiceConfig(
            name="typed_publisher",
            publisher=PublisherConfig(
                address="tcp://*",
                port=5555,
                default_topic=MessageType.HEARTBEAT
            ),
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5556,
                topics=[MessageType.HEARTBEAT, MessageType.IMAGE_READY, MessageType.ERA_CHANGED]
            )
        )
        
        received_messages = []
        
        async with mock_environment():
            async with MockPubSubService(config) as service:
                def handler(topic, message):
                    received_messages.append((topic, message))
                
                service.set_message_handler(handler)
                
                # Publish using enum values
                await service.publish({"timestamp": 1234567890}, MessageType.HEARTBEAT)
                await service.publish({"image_id": "img_001"}, MessageType.IMAGE_READY)
                await service.publish({"new_era": "digital"}, MessageType.ERA_CHANGED)
                
                await asyncio.sleep(0.1)
                
                # Verify enum values work correctly
                assert len(received_messages) == 3
                assert received_messages[0][0] == "Heartbeat"
                assert received_messages[1][0] == "ImageReady"
                assert received_messages[2][0] == "EraChanged"


class TestRealWorldScenarios:
    """Test realistic service integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_image_processing_pipeline(self):
        """Test an image processing pipeline scenario."""
        # Image controller manages image and transition workers
        controller_config = ControllerServiceConfig(
            name="image_controller",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5556,
                topics=["core.era_changed", "display.image_request"]
            ),
            workers={
                "image_generator": WorkerConfig(
                    name="image_generator",
                    push_config=ControllerPushConfig(address="tcp://localhost", port=5557),
                    pull_config=ControllerPullConfig(address="tcp://*", port=5558),
                    message_types=["image.generate"]
                ),
                "transition_creator": WorkerConfig(
                    name="transition_creator",
                    push_config=ControllerPushConfig(address="tcp://localhost", port=5559),
                    pull_config=ControllerPullConfig(address="tcp://*", port=5560),
                    message_types=["transition.create"]
                )
            }
        )
        
        # Track the pipeline
        received_requests = []
        image_tasks = []
        transition_tasks = []
        
        async with mock_environment():
            async with MockControllerService(controller_config) as controller:
                # Set up handlers
                def request_handler(topic, message):
                    received_requests.append((topic, message))
                
                def image_handler(message):
                    image_tasks.append(message)
                
                def transition_handler(message):
                    transition_tasks.append(message)
                
                controller.set_message_handler(request_handler)
                controller.set_worker_handler("image_generator", image_handler)
                controller.set_worker_handler("transition_creator", transition_handler)
                
                # Simulate core service requesting era change
                from experimance_common.zmq.mocks import mock_message_bus
                await mock_message_bus.publish(
                    "core.era_changed",
                    {"era": "renaissance", "seed": 12345},
                    "core_service"
                )
                
                # Simulate display requesting new image
                await mock_message_bus.publish(
                    "display.image_request",
                    {"request_id": "req_001", "priority": "high"},
                    "display_service"
                )
                
                await asyncio.sleep(0.1)
                
                # Controller processes requests and delegates work
                for topic, message in received_requests:
                    if topic == "core.era_changed":
                        await controller.push_to_worker("image_generator", {
                            "action": "generate_base_image",
                            "era": message["era"],
                            "seed": message["seed"]
                        })
                    elif topic == "display.image_request":
                        await controller.push_to_worker("transition_creator", {
                            "action": "create_transition",
                            "request_id": message["request_id"]
                        })
                
                await asyncio.sleep(0.1)
                
                # Verify pipeline execution
                assert len(received_requests) == 2
                assert len(image_tasks) == 1
                assert len(transition_tasks) == 1
                
                assert image_tasks[0]["action"] == "generate_base_image"
                assert image_tasks[0]["era"] == "renaissance"
                
                assert transition_tasks[0]["action"] == "create_transition"
                assert transition_tasks[0]["request_id"] == "req_001"
                
                # Simulate worker completing tasks and publishing results
                await controller.publish({
                    "image_id": "img_renaissance_001",
                    "path": "/tmp/renaissance.jpg"
                }, "image.ready")
                
                await controller.publish({
                    "transition_id": "trans_001",
                    "duration": 2.5
                }, "transition.ready")
                
                # Verify results published
                published = controller.published_messages
                assert len(published) == 2
                
                result_topics = [msg.topic for msg in published]
                assert "image.ready" in result_topics
                assert "transition.ready" in result_topics
    
    @pytest.mark.asyncio
    async def test_heartbeat_monitoring_system(self):
        """Test a heartbeat monitoring system."""
        # Core service publishes heartbeats
        core_config = PubSubServiceConfig(
            name="core_service",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5556,
                topics=["system.shutdown"]
            )
        )
        
        # Monitor service watches heartbeats
        monitor_config = PubSubServiceConfig(
            name="monitor_service",
            publisher=PublisherConfig(address="tcp://*", port=5557),
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5558,
                topics=["heartbeat"]
            )
        )
        
        received_heartbeats = []
        shutdown_commands = []
        alerts = []
        
        async with mock_environment():
            async with MockPubSubService(core_config) as core:
                async with MockPubSubService(monitor_config) as monitor:
                    # Set up handlers
                    def heartbeat_handler(topic, message):
                        received_heartbeats.append(message)
                        # Monitor responds to heartbeats
                        if message.get("status") == "unhealthy":
                            asyncio.create_task(monitor.publish({
                                "level": "critical",
                                "message": f"Service {message['service']} unhealthy"
                            }, "alerts"))
                    
                    def shutdown_handler(topic, message):
                        shutdown_commands.append(message)
                    
                    monitor.set_message_handler(heartbeat_handler)
                    core.set_message_handler(shutdown_handler)
                    
                    # Simulate heartbeat sequence
                    await core.publish({
                        "service": "core",
                        "status": "healthy",
                        "timestamp": 1000
                    }, "heartbeat")
                    
                    await core.publish({
                        "service": "core",
                        "status": "unhealthy",
                        "timestamp": 2000,
                        "error": "database_connection_lost"
                    }, "heartbeat")
                    
                    await asyncio.sleep(0.1)
                    
                    # Monitor publishes shutdown command
                    await monitor.publish({
                        "reason": "core_service_unhealthy",
                        "timestamp": 3000
                    }, "system.shutdown")
                    
                    await asyncio.sleep(0.1)
                    
                    # Verify monitoring sequence
                    assert len(received_heartbeats) == 2
                    assert received_heartbeats[0]["status"] == "healthy"
                    assert received_heartbeats[1]["status"] == "unhealthy"
                    
                    assert len(shutdown_commands) == 1
                    assert shutdown_commands[0]["reason"] == "core_service_unhealthy"
                    
                    # Check published alerts from monitor
                    monitor_published = monitor.get_published_messages()
                    alert_messages = [msg for msg in monitor_published if msg.topic == "alerts"]
                    assert len(alert_messages) >= 1


class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_service_restart_simulation(self):
        """Test service restart behavior."""
        config = PubSubServiceConfig(
            name="resilient_service",
            publisher=PublisherConfig(address="tcp://*", port=5555),
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5556,
                topics=["commands"]
            )
        )
        
        async with mock_environment():
            # First lifecycle
            service1 = MockPubSubService(config)
            await service1.start()
            
            await service1.publish({"phase": "startup"}, "status")
            assert len(service1.get_published_messages()) == 1
            
            # Simulate crash/stop
            await service1.stop()
            assert not service1.is_running
            
            # Restart with new instance (common pattern)
            service2 = MockPubSubService(config)
            await service2.start()
            
            # New instance starts fresh
            assert service2.is_running
            assert len(service2.get_published_messages()) == 0  # Fresh state
            
            await service2.publish({"phase": "restart"}, "status")
            assert len(service2.get_published_messages()) == 1
            
            await service2.stop()
    
    @pytest.mark.asyncio
    async def test_message_handler_resilience(self):
        """Test that services remain stable with failing handlers."""
        config = PubSubServiceConfig(
            name="resilient_handlers",
            subscriber=SubscriberConfig(
                address="tcp://localhost",
                port=5556,
                topics=["test"]
            )
        )
        
        handler_call_count = 0
        
        async with mock_environment():
            async with MockPubSubService(config) as service:
                def unreliable_handler(topic, message):
                    nonlocal handler_call_count
                    handler_call_count += 1
                    if handler_call_count <= 2:
                        raise RuntimeError(f"Handler error #{handler_call_count}")
                    # Succeeds on third call
                
                service.set_message_handler(unreliable_handler)
                
                # Send multiple messages
                from experimance_common.zmq.mocks import mock_message_bus
                await mock_message_bus.publish("test", {"attempt": 1}, "external")
                await mock_message_bus.publish("test", {"attempt": 2}, "external")
                await mock_message_bus.publish("test", {"attempt": 3}, "external")
                
                await asyncio.sleep(0.1)
                
                # Service should still be running despite handler errors
                assert service.is_running
                assert service.error_count == 2  # Two failed handler calls
                assert handler_call_count == 3  # All messages processed
                
                # All messages should still be recorded
                received = service.get_received_messages()
                assert len(received) == 3
