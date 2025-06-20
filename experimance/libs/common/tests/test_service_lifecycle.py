#!/usr/bin/env python3
"""
Comprehensive test suite for service lifecycle management.

This test suite specifically focuses on identifying and fixing issues with
service stopping, state management, and resource cleanup.
"""

import asyncio
import pytest
import logging
import time
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch, AsyncMock

# Configure logging for detailed debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experimance_common.base_service import BaseService
from experimance_common.service_state import ServiceState
from experimance_common.test_utils import active_service, wait_for_service_state
from experimance_common.zmq.services import PubSubService
from experimance_common.zmq.config import PubSubServiceConfig


class MinimalTestService(BaseService):
    """Minimal test service for lifecycle testing."""
    
    def __init__(self, service_name: str = "test-service", delay_stop: bool = False):
        super().__init__(service_name=service_name, service_type="test")
        self.delay_stop = delay_stop
        self._run_called = False
        self._stop_called = False
        # Register a simple task to run
        self.add_task(self._run_loop())
        
    async def _run_loop(self):
        """Simple run loop for testing."""
        self._run_called = True
        logger.info(f"Running {self.service_name}")
        try:
            while self.state == ServiceState.RUNNING:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info(f"Run loop cancelled for {self.service_name}")
            raise
            
    async def stop(self):
        """Stop the service."""
        self._stop_called = True
        logger.info(f"Stopping {self.service_name}")
        if self.delay_stop:
            # Simulate slow cleanup
            await asyncio.sleep(0.5)
        # Call parent stop to get base cleanup
        await super().stop()


class ZmqTestService(BaseService):
    """Test service with ZMQ components for testing ZMQ cleanup."""
    
    def __init__(self, service_name: str = "zmq-test-service"):
        super().__init__(service_name=service_name, service_type="zmq-test")
        self._run_called = False
        self._stop_called = False
        # Add some mock ZMQ components to simulate cleanup issues
        self._zmq_components = []
        # Register a simple task to run
        self.add_task(self._run_loop())
        
    async def _run_loop(self):
        """Run with ZMQ components."""
        self._run_called = True
        logger.info(f"Running ZMQ service {self.service_name}")
        try:
            while self.state == ServiceState.RUNNING:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info(f"ZMQ service run cancelled for {self.service_name}")
            raise
            
    async def stop(self):
        """Stop with ZMQ cleanup."""
        self._stop_called = True
        logger.info(f"Stopping ZMQ service {self.service_name}")
        # Simulate ZMQ component cleanup that might hang
        for i in range(len(self._zmq_components)):
            await asyncio.sleep(0.01)  # Small delay per component
        # Call parent stop to get base cleanup
        await super().stop()


class TestServiceLifecycleBasics:
    """Test basic service lifecycle operations."""
    
    @pytest.mark.asyncio
    async def test_service_basic_lifecycle(self):
        """Test basic start/stop lifecycle."""
        service = MinimalTestService("basic-test")
        
        # Initial state
        assert service.state == ServiceState.INITIALIZED
        
        # Start
        await service.start()
        assert service.state == ServiceState.STARTED
        
        # Stop
        await service.stop()
        assert service.state == ServiceState.STOPPED
        assert service._stop_called
        
    @pytest.mark.asyncio
    async def test_service_with_run_task(self):
        """Test service lifecycle with run task."""
        service = MinimalTestService("run-task-test")
        
        # Start and create run task
        await service.start()
        run_task = asyncio.create_task(service.run())
        
        # Wait briefly for run to start
        await asyncio.sleep(0.1)
        assert service.state == ServiceState.RUNNING
        assert service._run_called
        
        # Stop service
        await service.stop()
        
        # Wait for run task to complete
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.CancelledError:
            pass  # Expected
        except asyncio.TimeoutError:
            pytest.fail("Run task did not complete within timeout")
            
        assert service.state == ServiceState.STOPPED
        assert service._stop_called
        
    @pytest.mark.asyncio
    async def test_active_service_context_manager(self):
        """Test the active_service context manager."""
        service = MinimalTestService("context-test")
        
        async with active_service(service) as active:
            assert active.state == ServiceState.RUNNING
            assert active._run_called
            
        # Should be stopped after context
        assert service.state == ServiceState.STOPPED
        assert service._stop_called
        
    @pytest.mark.asyncio
    async def test_multiple_stop_calls(self):
        """Test that multiple stop calls don't cause issues."""
        service = MinimalTestService("multi-stop-test")
        
        await service.start()
        
        # Multiple stop calls should be safe
        await service.stop()
        await service.stop()
        await service.stop()
        
        assert service.state == ServiceState.STOPPED


class TestServiceLifecycleEdgeCases:
    """Test edge cases in service lifecycle."""
    
    @pytest.mark.asyncio
    async def test_slow_stopping_service(self):
        """Test service that takes time to stop."""
        service = MinimalTestService("slow-stop-test", delay_stop=True)
        
        async with active_service(service) as active:
            assert active.state == ServiceState.RUNNING
            
        # Should eventually stop despite delay
        assert service.state == ServiceState.STOPPED
        
    @pytest.mark.asyncio
    async def test_service_stop_timeout(self):
        """Test behavior when service stop times out."""
        service = MinimalTestService("timeout-test")
        
        # Mock the service stop to hang
        original_stop = service.stop
        async def hanging_stop():
            await asyncio.sleep(10)  # Hang for a long time
            
        service.stop = hanging_stop
        
        await service.start()
        
        # Stop should timeout but not hang the test
        start_time = time.monotonic()
        await service.stop()
        elapsed = time.monotonic() - start_time
        
        # Should not take the full 10 seconds
        assert elapsed < 8.0, f"Stop took too long: {elapsed}s"
        
    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test all valid state transitions."""
        service = MinimalTestService("state-test")
        
        # INITIALIZED -> STARTING -> STARTED
        assert service.state == ServiceState.INITIALIZED
        
        await service.start()
        assert service.state == ServiceState.STARTED
        
        # STARTED -> RUNNING (via run())
        run_task = asyncio.create_task(service.run())
        await asyncio.sleep(0.1)
        assert service.state == ServiceState.RUNNING
        
        # RUNNING -> STOPPING -> STOPPED
        await service.stop()
        
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.CancelledError:
            pass
            
        assert service.state == ServiceState.STOPPED


class TestServiceLifecycleSequence:
    """Test multiple services in sequence to identify state leakage."""
    
    @pytest.mark.asyncio
    async def test_sequential_services(self):
        """Test multiple services created and destroyed in sequence."""
        for i in range(3):
            service = MinimalTestService(f"seq-test-{i}")
            
            async with active_service(service) as active:
                assert active.state == ServiceState.RUNNING
                assert active._run_called
                
            assert service.state == ServiceState.STOPPED
            assert service._stop_called
            
            # Small delay to allow cleanup
            await asyncio.sleep(0.1)
            
    @pytest.mark.asyncio
    async def test_parallel_services(self):
        """Test multiple services running in parallel."""
        services = [MinimalTestService(f"parallel-test-{i}") for i in range(3)]
        
        # Start all services
        for service in services:
            await service.start()
            
        # Create run tasks
        run_tasks = [asyncio.create_task(service.run()) for service in services]
        
        # Wait for all to be running
        await asyncio.sleep(0.1)
        for service in services:
            assert service.state == ServiceState.RUNNING
            
        # Stop all services
        for service in services:
            await service.stop()
            
        # Wait for all run tasks to complete
        for task in run_tasks:
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.CancelledError:
                pass
                
        # Verify all stopped
        for service in services:
            assert service.state == ServiceState.STOPPED


class TestZmqServiceLifecycle:
    """Test ZMQ service lifecycle issues."""
    
    @pytest.mark.asyncio
    async def test_zmq_service_basic_lifecycle(self):
        """Test ZMQ service basic lifecycle."""
        service = ZmqTestService("zmq-basic-test")
        
        async with active_service(service) as active:
            assert active.state == ServiceState.RUNNING
            assert active._run_called
            
        assert service.state == ServiceState.STOPPED
        assert service._stop_called
        
    @pytest.mark.asyncio
    async def test_zmq_service_sequence(self):
        """Test multiple ZMQ services in sequence."""
        for i in range(3):
            service = ZmqTestService(f"zmq-seq-test-{i}")
            
            start_time = time.monotonic()
            async with active_service(service) as active:
                assert active.state == ServiceState.RUNNING
                
            elapsed = time.monotonic() - start_time
            logger.info(f"ZMQ service {i} lifecycle took {elapsed:.2f}s")
            
            assert service.state == ServiceState.STOPPED
            
            # Allow extra time for ZMQ cleanup
            await asyncio.sleep(0.2)


class TestServiceLifecycleDebugging:
    """Test suite for debugging specific lifecycle issues."""
    
    @pytest.mark.asyncio
    async def test_wait_for_service_state_timeout(self):
        """Test wait_for_service_state with various scenarios."""
        service = MinimalTestService("wait-test")
        
        # Test waiting for STOPPED when already stopped
        start_time = time.monotonic()
        await wait_for_service_state(service, ServiceState.STOPPED, timeout=1.0)
        elapsed = time.monotonic() - start_time
        assert elapsed < 0.1, "Should return immediately when already in target state"
        
        # Test waiting for state transition
        await service.start()
        
        # Stop in background
        async def stop_after_delay():
            await asyncio.sleep(0.1)
            await service.stop()
            
        stop_task = asyncio.create_task(stop_after_delay())
        
        # Wait for stopped state
        start_time = time.monotonic()
        await wait_for_service_state(service, ServiceState.STOPPED, timeout=2.0)
        elapsed = time.monotonic() - start_time
        
        await stop_task
        assert service.state == ServiceState.STOPPED
        assert elapsed < 1.0, f"Should transition quickly, took {elapsed:.2f}s"
        
    @pytest.mark.asyncio
    async def test_service_state_consistency(self):
        """Test service state consistency during lifecycle."""
        service = MinimalTestService("consistency-test")
        
        # Track state changes
        state_changes = []
        
        def state_callback():
            state_changes.append(service.state)
            
        # Register callback (if available)
        if hasattr(service, 'register_state_callback'):
            service.register_state_callback(ServiceState.STARTED, state_callback)
            service.register_state_callback(ServiceState.RUNNING, state_callback)
            service.register_state_callback(ServiceState.STOPPED, state_callback)
        
        async with active_service(service) as active:
            assert active.state == ServiceState.RUNNING
            
        assert service.state == ServiceState.STOPPED
        
        # Log state transitions for debugging
        logger.info(f"State changes: {state_changes}")


class TestServiceStoppingDiagnostics:
    """Comprehensive tests for service stopping issues and diagnostics."""
    
    @pytest.mark.asyncio
    async def test_sequential_service_start_stop(self):
        """Test multiple services started and stopped in sequence."""
        services = []
        
        for i in range(3):
            service = MinimalTestService(f"sequential-{i}")
            services.append(service)
            
            # Start and stop each service
            await service.start()
            assert service.state == ServiceState.STARTED
            
            await service.stop()
            assert service.state == ServiceState.STOPPED
            assert service._stop_called
            
        # All services should be cleanly stopped
        for service in services:
            assert service.state == ServiceState.STOPPED
    
    @pytest.mark.asyncio
    async def test_parallel_service_start_stop(self):
        """Test multiple services started and stopped in parallel."""
        services = [MinimalTestService(f"parallel-{i}") for i in range(3)]
        
        # Start all services in parallel
        start_tasks = [service.start() for service in services]
        await asyncio.gather(*start_tasks)
        
        for service in services:
            assert service.state == ServiceState.STARTED
            
        # Stop all services in parallel
        stop_tasks = [service.stop() for service in services]
        await asyncio.gather(*stop_tasks)
        
        for service in services:
            assert service.state == ServiceState.STOPPED
            assert service._stop_called
    
    @pytest.mark.asyncio
    async def test_service_with_running_task_stop(self):
        """Test stopping a service while it has a running task."""
        service = MinimalTestService("running-task-test")
        
        # Start service and run task
        await service.start()
        run_task = asyncio.create_task(service.run())
        
        # Wait for RUNNING state
        await wait_for_service_state(service, ServiceState.RUNNING, timeout=2.0)
        assert service.state == ServiceState.RUNNING
        assert service._run_called
        
        # Stop service while running
        await service.stop()
        
        # Run task should complete (be cancelled)
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.CancelledError:
            pass  # Expected
        except asyncio.TimeoutError:
            pytest.fail("Run task did not complete within timeout")
            
        assert service.state == ServiceState.STOPPED
        assert service._stop_called
    
    @pytest.mark.asyncio
    async def test_service_stop_during_startup(self):
        """Test stopping a service during startup phase."""
        service = MinimalTestService("startup-stop-test")
        
        # Start service but don't wait for completion
        start_task = asyncio.create_task(service.start())
        
        # Immediately try to stop (while potentially still starting)
        await asyncio.sleep(0.01)  # Small delay to let start begin
        stop_task = asyncio.create_task(service.stop())
        
        # Wait for both to complete
        await asyncio.gather(start_task, stop_task, return_exceptions=True)
        
        # Service should end up stopped
        assert service.state == ServiceState.STOPPED
    
    @pytest.mark.asyncio
    async def test_service_rapid_start_stop_cycles(self):
        """Test rapid start/stop cycles that might cause race conditions."""
        service = MinimalTestService("rapid-cycle-test")
        
        for cycle in range(5):
            logger.info(f"Cycle {cycle}")
            
            await service.start()
            assert service.state == ServiceState.STARTED
            
            await service.stop()
            assert service.state == ServiceState.STOPPED
            
            # Reset the service for next cycle
            service._stop_called = False
            service._run_called = False
    
    @pytest.mark.asyncio
    async def test_service_with_slow_stop(self):
        """Test service with deliberately slow stop process."""
        service = MinimalTestService("slow-stop-test", delay_stop=True)
        
        await service.start()
        
        start_time = time.monotonic()
        await service.stop()
        elapsed = time.monotonic() - start_time
        
        # Should take at least the delay time
        assert elapsed >= 0.4  # Allow some margin under the 0.5s delay
        assert service.state == ServiceState.STOPPED
        assert service._stop_called
    
    @pytest.mark.asyncio
    async def test_service_stop_state_transitions(self):
        """Test that stop state transitions happen in correct order."""
        service = MinimalTestService("state-transition-test")
        
        # Track state changes
        state_changes = []
        
        def state_callback(old_state, new_state):
            state_changes.append((old_state, new_state))
            
        service._state_manager.add_transition_callback(state_callback)
        
        # Start and stop service
        await service.start()
        await service.stop()
        
        # Check that we see the expected transitions
        transitions = [(old.value, new.value) for old, new in state_changes]
        logger.info(f"State transitions: {transitions}")
        
        # Should see transitions to stopping and stopped
        assert ('started', 'stopping') in transitions
        assert ('stopping', 'stopped') in transitions
    
    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        """Test active_service context manager when exception occurs."""
        service = MinimalTestService("exception-test")
        
        try:
            async with active_service(service) as active:
                assert active.state == ServiceState.RUNNING
                # Simulate an exception in the test
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        # Service should still be stopped cleanly
        assert service.state == ServiceState.STOPPED
        assert service._stop_called
    
    @pytest.mark.asyncio
    async def test_multiple_context_managers_sequential(self):
        """Test multiple context managers used sequentially."""
        for i in range(3):
            service = MinimalTestService(f"ctx-sequential-{i}")
            
            async with active_service(service) as active:
                assert active.state == ServiceState.RUNNING
                assert active._run_called
                
            # Should be stopped after context
            assert service.state == ServiceState.STOPPED
            assert service._stop_called
    
    @pytest.mark.asyncio 
    async def test_service_cleanup_after_cancelled_run(self):
        """Test service cleanup when run task is cancelled externally."""
        service = MinimalTestService("cancelled-run-test")
        
        await service.start()
        run_task = asyncio.create_task(service.run())
        
        # Wait for running state
        await wait_for_service_state(service, ServiceState.RUNNING, timeout=2.0)
        
        # Cancel the run task directly
        run_task.cancel()
        
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        
        # Service should still be able to stop cleanly
        await service.stop()
        assert service.state == ServiceState.STOPPED


if __name__ == "__main__":
    # Run specific tests for debugging
    import sys
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        pytest.main([f"-xvs", f"test_service_lifecycle.py::{test_name}"])
    else:
        pytest.main(["-xvs", "test_service_lifecycle.py"])
