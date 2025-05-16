"""
Tests for using the ZeroMQ utilities with the connection retry mechanism.
"""

import asyncio
import json
import logging
import pytest
import socket
import time
import zmq
from typing import Any, Dict, List, Tuple, Optional

from experimance_common.connection_retry import retry_with_backoff, async_retry_with_backoff
from experimance_common.zmq_utils import (
    ZmqPublisher, 
    ZmqSubscriber, 
    ZmqPushSocket, 
    ZmqPullSocket,
    MessageType,
    ZmqTimeoutError
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
TEST_RETRY_PORT_PUB = 5570
TEST_RETRY_PORT_PUSH = 5571


class DelayedPublisher:
    """
    A publisher that binds after a specified delay to simulate a service starting late.
    """
    
    def __init__(self, port: int, topic: str = "test", delay_seconds: float = 1.0):
        self.port = port
        self.topic = topic
        self.delay = delay_seconds
        self.ready = asyncio.Event()
        self.message_sent = asyncio.Event()
        self.publisher = None
        self.task = None
    
    async def start(self):
        """Start the delayed publisher."""
        self.task = asyncio.create_task(self._run())
        return self
    
    async def _run(self):
        """Run the publisher after a delay."""
        logger.info(f"Publisher will bind to port {self.port} after {self.delay}s")
        try:
            await asyncio.sleep(self.delay)
            
            # Create and bind the publisher
            logger.info(f"Creating publisher on port {self.port}")
            self.publisher = ZmqPublisher(f"tcp://*:{self.port}", self.topic)
            
            # Signal that the publisher is ready
            self.ready.set()
            logger.info(f"Publisher bound to port {self.port}")
            
            # Keep the publisher running
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Publisher task cancelled")
        except Exception as e:
            logger.error(f"Publisher error: {e}")
        finally:
            self.close()
            logger.info("Publisher cleaned up")
    
    def close(self):
        """Close the publisher."""
        if self.publisher:
            try:
                self.publisher.close()
                self.publisher = None
                logger.info("Publisher socket closed")
            except Exception as e:
                logger.error(f"Error closing publisher: {e}")
                
        if self.task and not self.task.done():
            self.task.cancel()
    
    async def wait_until_ready(self):
        """Wait until the publisher is ready."""
        await self.ready.wait()
    
    async def publish(self, message: Dict[str, Any]) -> bool:
        """Publish a message."""
        if not self.publisher:
            raise RuntimeError("Publisher not initialized")
            
        logger.info(f"Publishing message: {message}")
        result = await self.publisher.publish_async(message)
        if result:
            self.message_sent.set()
            logger.info("Message sent successfully")
        else:
            logger.error("Failed to send message")
        return result


class DelayedPushSocket:
    """
    A push socket that binds after a specified delay to simulate a service starting late.
    """
    
    def __init__(self, port: int, delay_seconds: float = 1.0):
        self.port = port
        self.delay = delay_seconds
        self.ready = asyncio.Event()
        self.message_sent = asyncio.Event()
        self.push_socket = None
        self.task = None
    
    async def start(self):
        """Start the delayed push socket."""
        self.task = asyncio.create_task(self._run())
        return self
    
    async def _run(self):
        """Run the push socket after a delay."""
        logger.info(f"Push socket will bind to port {self.port} after {self.delay}s")
        try:
            await asyncio.sleep(self.delay)
            
            # Create and bind the push socket
            logger.info(f"Creating push socket on port {self.port}")
            self.push_socket = ZmqPushSocket(f"tcp://*:{self.port}")
            
            # Signal that the push socket is ready
            self.ready.set()
            logger.info(f"Push socket bound to port {self.port}")
            
            # Keep the push socket running
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Push socket task cancelled")
        except Exception as e:
            logger.error(f"Push socket error: {e}")
        finally:
            self.close()
            logger.info("Push socket cleaned up")
    
    def close(self):
        """Close the push socket."""
        if self.push_socket:
            try:
                self.push_socket.close()
                self.push_socket = None
                logger.info("Push socket closed")
            except Exception as e:
                logger.error(f"Error closing push socket: {e}")
                
        if self.task and not self.task.done():
            self.task.cancel()
    
    async def wait_until_ready(self):
        """Wait until the push socket is ready."""
        await self.ready.wait()
    
    async def push(self, message: Dict[str, Any]) -> bool:
        """Push a message."""
        if not self.push_socket:
            raise RuntimeError("Push socket not initialized")
            
        logger.info(f"Pushing message: {message}")
        result = await self.push_socket.push_async(message)
        if result:
            self.message_sent.set()
            logger.info("Message pushed successfully")
        else:
            logger.error("Failed to push message")
        return result


class TestZmqWithRetry:
    """Tests for using ZeroMQ with retry mechanism."""
    
    @pytest.mark.asyncio
    async def test_subscriber_with_retry(self):
        """Test creating a subscriber that retries connecting to a publisher."""
        # Create a delayed publisher in the background
        delayed_publisher = await DelayedPublisher(TEST_RETRY_PORT_PUB, "test-topic", delay_seconds=1.0).start()
        subscriber = None
        test_complete = asyncio.Event()
        received_message = None
        received_topic = None
        
        try:
            # Define a function to create a subscriber with retries
            connection_attempts = 0
            
            async def create_subscriber_with_retry():
                nonlocal connection_attempts
                connection_attempts += 1
                logger.info(f"Attempting to connect subscriber (attempt {connection_attempts})")
                
                # Simulate connection failures for the first 2 attempts
                if connection_attempts <= 2:
                    logger.warning(f"Simulating connection failure (attempt {connection_attempts})")
                    raise ConnectionError(f"Simulated connection failure on attempt {connection_attempts}")
                
                try:
                    # This might fail if the publisher isn't ready yet
                    logger.info("Creating ZMQ subscriber")
                    subscriber = ZmqSubscriber(f"tcp://localhost:{TEST_RETRY_PORT_PUB}", ["test-topic"])
                    logger.info("ZMQ subscriber created successfully")
                    return subscriber
                except Exception as e:
                    logger.warning(f"Subscriber connection failed: {e}")
                    raise ConnectionError(f"Failed to connect subscriber: {e}")
            
            # Use the retry mechanism to create the subscriber
            try:
                logger.info("Creating subscriber with retry")
                subscriber = await async_retry_with_backoff(
                    create_subscriber_with_retry,
                    max_retries=5,
                    initial_backoff=0.2,
                    max_backoff=1.0,
                    backoff_factor=1.5,
                    exceptions=(ConnectionError, zmq.error.ZMQError, ZmqTimeoutError)
                )
                logger.info("Subscriber created successfully after retries")
                
                # Wait for the publisher to be ready
                logger.info("Waiting for publisher to be ready")
                await delayed_publisher.wait_until_ready()
                logger.info("Publisher is ready")
                
                # Add a delay to allow ZMQ subscription to be established
                logger.info("Allowing time for subscription to be established")
                await asyncio.sleep(1.0)
                
                # Create a task to receive the message
                async def receive_message_task():
                    nonlocal received_message, received_topic
                    try:
                        logger.info("Waiting to receive message...")
                        topic, message = await asyncio.wait_for(subscriber.receive_async(), timeout=5.0)
                        logger.info(f"Received message on topic {topic}: {message}")
                        received_topic = topic
                        received_message = message
                        test_complete.set()
                    except (asyncio.TimeoutError, ZmqTimeoutError) as e:
                        logger.error(f"Timed out waiting for message: {e}")
                        test_complete.set()
                    except Exception as e:
                        logger.error(f"Error receiving message: {e}")
                        test_complete.set()
                
                # Start the receive task
                receive_task = asyncio.create_task(receive_message_task())
                
                # Test communication
                test_message = {"type": MessageType.HEARTBEAT, "timestamp": time.time()}
                logger.info(f"Publishing test message: {test_message}")
                await delayed_publisher.publish(test_message)
                
                # Wait for the test to complete or timeout
                try:
                    await asyncio.wait_for(test_complete.wait(), timeout=7.0)
                except asyncio.TimeoutError:
                    logger.error("Test timed out after waiting")
                    pytest.fail("Test timed out")
                
                # Verify the received message
                assert received_topic == "test-topic", f"Expected topic 'test-topic', got '{received_topic}'"
                assert received_message is not None, "No message received"
                assert received_message["type"] == MessageType.HEARTBEAT, f"Expected HEARTBEAT message type, got {received_message.get('type')}"
                assert "timestamp" in received_message, "Timestamp not in message"
                
                # Verify that we had to retry at least once
                assert connection_attempts > 1, "Expected multiple connection attempts"
                
                # Cancel the receive task if it's still running
                if not receive_task.done():
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass
                
            except Exception as e:
                logger.error(f"Test failed with error: {e}")
                raise
                
        finally:
            # Clean up subscriber
            if subscriber:
                logger.info("Cleaning up subscriber")
                subscriber.close()
            
            # Clean up publisher
            logger.info("Cleaning up publisher")
            delayed_publisher.close()
            
            # Ensure the test is marked as complete
            test_complete.set()
    
    @pytest.mark.asyncio
    async def test_pull_socket_with_retry(self):
        """Test creating a pull socket that retries connecting to a push socket."""
        # Create a delayed push socket in the background
        delayed_push_socket = await DelayedPushSocket(TEST_RETRY_PORT_PUSH, delay_seconds=1.0).start()
        pull_socket = None
        test_complete = asyncio.Event()
        received_message = None
        
        try:
            # Define a function to create a pull socket with retries
            connection_attempts = 0
            
            async def create_pull_socket_with_retry():
                nonlocal connection_attempts
                connection_attempts += 1
                logger.info(f"Attempting to connect pull socket (attempt {connection_attempts})")
                
                # Simulate connection failures for the first 2 attempts
                if connection_attempts <= 2:
                    logger.warning(f"Simulating connection failure (attempt {connection_attempts})")
                    raise ConnectionError(f"Simulated connection failure on attempt {connection_attempts}")
                
                try:
                    # This might fail if the push socket isn't ready yet
                    logger.info("Creating ZMQ pull socket")
                    pull_socket = ZmqPullSocket(f"tcp://localhost:{TEST_RETRY_PORT_PUSH}")
                    logger.info("ZMQ pull socket created successfully")
                    return pull_socket
                except Exception as e:
                    logger.warning(f"Pull socket connection failed: {e}")
                    raise ConnectionError(f"Failed to connect pull socket: {e}")
            
            # Use the retry mechanism to create the pull socket
            try:
                logger.info("Creating pull socket with retry")
                pull_socket = await async_retry_with_backoff(
                    create_pull_socket_with_retry,
                    max_retries=5,
                    initial_backoff=0.2,
                    max_backoff=1.0,
                    backoff_factor=1.5,
                    exceptions=(ConnectionError, zmq.error.ZMQError, ZmqTimeoutError)
                )
                logger.info("Pull socket created successfully after retries")
                
                # Wait for the push socket to be ready
                logger.info("Waiting for push socket to be ready")
                await delayed_push_socket.wait_until_ready()
                logger.info("Push socket is ready")
                
                # Add a delay to allow ZMQ connection to be established
                logger.info("Allowing time for connection to be established")
                await asyncio.sleep(1.0)
                
                # Create a task to receive the message
                async def receive_message_task():
                    nonlocal received_message
                    try:
                        logger.info("Waiting to receive message...")
                        message = await asyncio.wait_for(pull_socket.pull_async(), timeout=5.0)
                        logger.info(f"Received message: {message}")
                        received_message = message
                        test_complete.set()
                    except (asyncio.TimeoutError, ZmqTimeoutError) as e:
                        logger.error(f"Timed out waiting for message: {e}")
                        test_complete.set()
                    except Exception as e:
                        logger.error(f"Error receiving message: {e}")
                        test_complete.set()
                
                # Start the receive task
                receive_task = asyncio.create_task(receive_message_task())
                
                # Test communication
                test_message = {"type": MessageType.HEARTBEAT, "timestamp": time.time()}
                logger.info(f"Pushing test message: {test_message}")
                await delayed_push_socket.push(test_message)
                
                # Wait for the test to complete or timeout
                try:
                    await asyncio.wait_for(test_complete.wait(), timeout=7.0)
                except asyncio.TimeoutError:
                    logger.error("Test timed out after waiting")
                    pytest.fail("Test timed out")
                
                # Verify the received message
                assert received_message is not None, "No message received"
                assert received_message["type"] == MessageType.HEARTBEAT, f"Expected HEARTBEAT message type, got {received_message.get('type')}"
                assert "timestamp" in received_message, "Timestamp not in message"
                
                # Verify that we had to retry at least once
                assert connection_attempts > 1, "Expected multiple connection attempts"
                
                # Cancel the receive task if it's still running
                if not receive_task.done():
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass
                
            except Exception as e:
                logger.error(f"Test failed with error: {e}")
                raise
                
        finally:
            # Clean up pull socket
            if pull_socket:
                logger.info("Cleaning up pull socket")
                pull_socket.close()
            
            # Clean up push socket
            logger.info("Cleaning up push socket")
            delayed_push_socket.close()
            
            # Ensure the test is marked as complete
            test_complete.set()


if __name__ == "__main__":
    pytest.main(["-v", "test_zmq_with_retry.py"])
