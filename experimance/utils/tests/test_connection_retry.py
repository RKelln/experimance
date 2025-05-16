"""
Tests for the connection retry functionality with ZeroMQ.
"""

import asyncio
import json
import logging
import pytest
import time
import zmq
import zmq.asyncio
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List, Tuple, Optional, Callable

from experimance_common.connection_retry import retry_with_backoff, async_retry_with_backoff
from experimance_common.constants import DEFAULT_PORTS

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
TEST_RETRY_PORT = 5599


class TestConnectionRetry:
    """Tests for connection retry functionality."""
    
    def test_sync_retry_with_backoff_success(self):
        """Test synchronous retry with backoff that eventually succeeds."""
        # Mock function that fails twice then succeeds
        call_count = 0
        
        def mock_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated connection error")
            return "success"
        
        # Call with retry
        result = retry_with_backoff(
            mock_function,
            max_retries=5,
            initial_backoff=0.1,  # Use small values for faster tests
            max_backoff=0.5,
            backoff_factor=1.5,
            exceptions=(ConnectionError,)
        )
        
        # Verify
        assert result == "success"
        assert call_count == 3
    
    def test_sync_retry_with_backoff_failure(self):
        """Test synchronous retry with backoff that fails all attempts."""
        # Mock function that always fails
        call_count = 0
        
        def mock_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Simulated connection error")
        
        # Call with retry and expect failure
        with pytest.raises(ConnectionError):
            retry_with_backoff(
                mock_function,
                max_retries=3,
                initial_backoff=0.1,
                max_backoff=0.5,
                backoff_factor=1.5,
                exceptions=(ConnectionError,)
            )
        
        # Verify
        assert call_count == 4  # Initial try + 3 retries
    
    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_success(self):
        """Test asynchronous retry with backoff that eventually succeeds."""
        # Mock async function that fails twice then succeeds
        call_count = 0
        
        async def mock_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated connection error")
            return "success"
        
        # Call with retry
        result = await async_retry_with_backoff(
            mock_async_function,
            max_retries=5,
            initial_backoff=0.1,
            max_backoff=0.5,
            backoff_factor=1.5,
            exceptions=(ConnectionError,)
        )
        
        # Verify
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_with_backoff_failure(self):
        """Test asynchronous retry with backoff that fails all attempts."""
        # Mock async function that always fails
        call_count = 0
        
        async def mock_async_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Simulated connection error")
        
        # Call with retry and expect failure
        with pytest.raises(ConnectionError):
            await async_retry_with_backoff(
                mock_async_function,
                max_retries=3,
                initial_backoff=0.1,
                max_backoff=0.5,
                backoff_factor=1.5,
                exceptions=(ConnectionError,)
            )
        
        # Verify
        assert call_count == 4  # Initial try + 3 retries


class TestZmqRetry:
    """Tests for ZeroMQ operations with connection retry."""
    
    @pytest.mark.asyncio
    async def test_zmq_pub_sub_retry(self):
        """Test ZeroMQ PUB-SUB pattern with connection retry."""
        # We'll use these to coordinate between publisher and subscriber
        publisher_ready = asyncio.Event()
        message_sent = asyncio.Event()
        publisher_task = None
        subscriber_task = None
        connect_attempts = 0
        received_message = None
        test_complete = asyncio.Event()
        
        # Define the message that will be sent
        test_message = "test Hello, World!"
        
        # Create the publisher task
        async def run_publisher():
            logger.info("Publisher task starting")
            # Create context and socket inside the task
            pub_context = zmq.Context()
            pub_socket = pub_context.socket(zmq.PUB)
            
            try:
                # Delay binding to simulate late binding
                logger.info("Publisher will be ready in 1.0 second")
                await asyncio.sleep(1.0)
                
                # Bind the publisher
                pub_socket.bind(f"tcp://*:{TEST_RETRY_PORT}")
                logger.info(f"Publisher bound to port {TEST_RETRY_PORT}")
                
                # Signal that the publisher is ready
                publisher_ready.set()
                
                # Wait until we're told to send the message
                while not message_sent.is_set():
                    # Add a small sleep to avoid burning CPU
                    await asyncio.sleep(0.1)
                    
                    # If the test is complete, exit early
                    if test_complete.is_set():
                        return
                
                # Send the test message
                logger.info(f"Sending message: {test_message}")
                pub_socket.send_string(test_message)
                logger.info("Message sent")
                
                # Keep the publisher running until the test is complete
                while not test_complete.is_set():
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                logger.info("Publisher task was cancelled")
            except Exception as e:
                logger.error(f"Publisher error: {e}")
                raise
            finally:
                # Clean up resources
                logger.info("Cleaning up publisher resources")
                pub_socket.close()
                pub_context.term()
                logger.info("Publisher cleaned up")
        
        # Create the subscriber task
        async def run_subscriber():
            nonlocal connect_attempts, received_message
            logger.info("Subscriber task starting")
            
            # Create the subscriber context and socket
            sub_context = zmq.asyncio.Context()
            sub_socket = sub_context.socket(zmq.SUB)
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, "test")
            
            try:
                # Define the connection function with retries
                async def connect_subscriber():
                    nonlocal connect_attempts
                    connect_attempts += 1
                    logger.info(f"Attempting to connect subscriber (attempt {connect_attempts})")
                    
                    # Simulate connection failures for the first 2 attempts
                    if connect_attempts <= 2:
                        logger.warning(f"Simulating connection failure (attempt {connect_attempts})")
                        raise ConnectionError(f"Simulated connection failure on attempt {connect_attempts}")
                    
                    try:
                        sub_socket.connect(f"tcp://localhost:{TEST_RETRY_PORT}")
                        logger.info(f"Subscriber connected to port {TEST_RETRY_PORT}")
                        
                        # We're connected (but we might not be subscribed yet)
                        return True
                    except zmq.error.ZMQError as e:
                        logger.warning(f"Connection attempt failed: {e}")
                        raise ConnectionError(f"Failed to connect: {e}")
                
                # Use our retry mechanism to connect
                await async_retry_with_backoff(
                    connect_subscriber,
                    max_retries=5,
                    initial_backoff=0.2,
                    max_backoff=1.0,
                    backoff_factor=1.5,
                    exceptions=(ConnectionError, zmq.error.ZMQError),
                )
                
                # Wait for the publisher to be ready
                logger.info("Waiting for publisher to be ready")
                await publisher_ready.wait()
                logger.info("Publisher is ready")
                
                # ZMQ PUB-SUB needs time to establish the subscription after connection
                logger.info("Waiting for subscription to settle")
                await asyncio.sleep(1.0)  # Increased from 0.5 to 1.0 seconds
                
                # Signal that we're ready to receive the message
                message_sent.set()
                
                # Receive with a timeout
                try:
                    logger.info("Waiting to receive message")
                    received_message = await asyncio.wait_for(sub_socket.recv_string(), timeout=5.0)  # Increased timeout
                    logger.info(f"Received message: {received_message}")
                except asyncio.TimeoutError:
                    logger.error("Timed out waiting for message")
                    raise
                
                # Signal test completion
                test_complete.set()
                
            except asyncio.CancelledError:
                logger.info("Subscriber task was cancelled")
            except Exception as e:
                logger.error(f"Subscriber error: {e}")
                test_complete.set()  # Signal test completion even on error
                raise
            finally:
                # Clean up resources
                logger.info("Cleaning up subscriber resources")
                sub_socket.close(linger=0)  # Don't linger on close
                sub_context.term()
                logger.info("Subscriber cleaned up")
        
        try:
            # Start both tasks concurrently
            publisher_task = asyncio.create_task(run_publisher())
            subscriber_task = asyncio.create_task(run_subscriber())
            
            # Wait for the test to complete or timeout
            try:
                await asyncio.wait_for(test_complete.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("Test timed out after 10 seconds")
                pytest.fail("Test timed out after 10 seconds")
            
            # Verify that we received the expected message
            assert received_message == test_message, f"Expected '{test_message}', got '{received_message}'"
            
            # Verify that we had to retry at least once
            assert connect_attempts > 1, "Expected multiple connection attempts"
            
        finally:
            # Ensure test is marked as complete
            test_complete.set()
            
            # Cancel tasks if they're still running
            if publisher_task and not publisher_task.done():
                logger.info("Cancelling publisher task")
                publisher_task.cancel()
            
            if subscriber_task and not subscriber_task.done():
                logger.info("Cancelling subscriber task")
                subscriber_task.cancel()
            
            # Wait for tasks to complete
            if publisher_task:
                try:
                    await asyncio.wait_for(publisher_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("Publisher task cancellation timed out")
            
            if subscriber_task:
                try:
                    await asyncio.wait_for(subscriber_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("Subscriber task cancellation timed out")


if __name__ == "__main__":
    pytest.main(["-v", "test_connection_retry.py"])
