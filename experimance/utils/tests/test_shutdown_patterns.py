#!/usr/bin/env python3
"""
Tests for service shutdown and error handling patterns in experimance_common.

This test suite validates the best practices outlined in the README_SERVICE.md:
1. Different shutdown methods (stop, request_stop, error-triggered)
2. Error categorization and automatic shutdown for fatal errors
3. Error recovery patterns
4. Task naming for debugging
5. Real-world usage patterns

Run with:
    uv run -m pytest utils/tests/test_shutdown_patterns.py -v
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import pytest

from experimance_common.base_service import BaseService, ServiceState, ServiceStatus
from utils.tests.test_utils import wait_for_service_shutdown, wait_for_service_state

# Configure test logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkerService(BaseService):
    """A realistic worker service for testing shutdown patterns."""
    
    def __init__(self, name: str = "worker-service"):
        super().__init__(service_name=name, service_type="worker")
        self.work_completed = 0
        self.requests_processed = 0
        self.error_conditions = {}
        self.shutdown_reason = None
        
    async def start(self):
        """Start the worker service with background tasks."""
        self.add_task(self.main_work_loop())
        self.add_task(self.health_monitor())
        await super().start()
    
    async def main_work_loop(self):
        """Main work loop demonstrating different shutdown scenarios."""
        while self.running:
            try:
                # Simulate work
                await asyncio.sleep(0.1)
                self.work_completed += 1
                
                # Check for configured error conditions
                if 'fatal_at_work' in self.error_conditions:
                    if self.work_completed >= self.error_conditions['fatal_at_work']:
                        self.record_error(RuntimeError("Fatal work error"), is_fatal=True)
                        break
                
                if 'recoverable_at_work' in self.error_conditions:
                    if self.work_completed == self.error_conditions['recoverable_at_work']:
                        self.record_error(ValueError("Recoverable work error"), is_fatal=False)
                
                if 'stop_at_work' in self.error_conditions:
                    if self.work_completed >= self.error_conditions['stop_at_work']:
                        self.shutdown_reason = "work_completed"
                        self.request_stop()
                        break
                        
            except Exception as e:
                logger.error(f"Unexpected error in work loop: {e}")
                self.record_error(e, is_fatal=True)
                break
    
    async def health_monitor(self):
        """Monitor service health and handle error conditions."""
        while self.running:
            try:
                await asyncio.sleep(0.1)  # Check more frequently
                
                # Simulate health checks
                if self.errors > 3:
                    logger.warning("Too many errors detected, requesting shutdown")
                    self.shutdown_reason = "too_many_errors"
                    self.request_stop()
                    break
                    
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                self.record_error(e, is_fatal=False)
    
    async def process_request(self, request_data: Dict[str, Any]):
        """Process a request and handle potential errors."""
        try:
            self.requests_processed += 1
            
            # Simulate processing
            await asyncio.sleep(0.05)
            
            # Check for error injection
            if request_data.get('cause_error'):
                error_type = request_data.get('error_type', 'ValueError')
                is_fatal = request_data.get('is_fatal', False)
                
                if error_type == 'ValueError':
                    error = ValueError(f"Simulated error in request {self.requests_processed}")
                elif error_type == 'RuntimeError':
                    error = RuntimeError(f"Simulated runtime error in request {self.requests_processed}")
                else:
                    error = Exception(f"Simulated generic error in request {self.requests_processed}")
                
                self.record_error(error, is_fatal=is_fatal)
                if is_fatal:
                    raise error
                
            return {"status": "success", "request_id": self.requests_processed}
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise


class CircuitBreakerService(BaseService):
    """A service demonstrating circuit breaker pattern with error handling."""
    
    def __init__(self, name: str = "circuit-breaker-service"):
        super().__init__(service_name=name, service_type="circuit-breaker")
        self.failure_count = 0
        self.failure_threshold = 3
        self.circuit_open = False
        self.external_calls = 0
        self.successful_calls = 0
        
    async def start(self):
        """Start the circuit breaker service."""
        self.add_task(self.external_service_monitor())
        await super().start()
    
    async def external_service_monitor(self):
        """Monitor external service calls with circuit breaker."""
        while self.running:
            try:
                result = await self.call_external_service()
                if result:
                    self.successful_calls += 1
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.record_error(e, is_fatal=False)
                await asyncio.sleep(0.2)  # Backoff on error
    
    async def call_external_service(self):
        """Call external service with circuit breaker logic."""
        self.external_calls += 1
        
        # Check circuit breaker state
        if self.circuit_open:
            if self.failure_count > 5:  # Too many failures
                self.record_error(
                    RuntimeError("Circuit breaker: too many failures"), 
                    is_fatal=True
                )
                return False
            
            # Try to recover
            self.failure_count = max(0, self.failure_count - 1)
            if self.failure_count == 0:
                self.circuit_open = False
                logger.info("Circuit breaker reset")
                return True
            return False
        
        # Simulate external service call
        if self.external_calls % 4 == 0:  # Simulate 25% failure rate
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.circuit_open = True
                logger.warning("Circuit breaker opened")
            raise ConnectionError(f"External service unavailable (call {self.external_calls})")
        
        # Success
        self.failure_count = max(0, self.failure_count - 1)
        return True


@asynccontextmanager
async def run_service_until_stopped(service: BaseService, timeout: float = 5.0):
    """Context manager to run a service until it stops naturally or times out."""
    await service.start()
    
    run_task = asyncio.create_task(service.run())
    
    try:
        yield service
        
        # Wait for natural shutdown or timeout
        try:
            await asyncio.wait_for(run_task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Service {service.service_name} did not stop within {timeout}s")
            
    finally:
        # Ensure cleanup
        if service.state not in [ServiceState.STOPPING, ServiceState.STOPPED]:
            await service.stop()
            
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass


class TestShutdownPatterns:
    """Test different shutdown patterns and best practices."""
    
    @pytest.mark.asyncio
    async def test_manual_shutdown_request(self):
        """Test manual shutdown using request_stop()."""
        service = WorkerService("manual-shutdown")
        service.error_conditions['stop_at_work'] = 5
        
        async with run_service_until_stopped(service, timeout=2.0):
            # Wait for service to reach running state
            await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Service should stop itself after completing work
            await service.wait_for_state(ServiceState.STOPPED, timeout=2.0)
        
        assert service.state == ServiceState.STOPPED
        assert service.shutdown_reason == "work_completed"
        assert service.work_completed >= 5
    
    @pytest.mark.asyncio
    async def test_fatal_error_auto_shutdown(self):
        """Test automatic shutdown triggered by fatal error."""
        service = WorkerService("fatal-error-shutdown")
        service.error_conditions['fatal_at_work'] = 3
        
        async with run_service_until_stopped(service, timeout=2.0):
            # Wait for service to reach running state
            await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Service should stop itself after fatal error
            await service.wait_for_state(ServiceState.STOPPED, timeout=2.0)
        
        assert service.state == ServiceState.STOPPED
        assert service.status == ServiceStatus.FATAL
        assert service.errors >= 1
        assert service.work_completed >= 3
    
    @pytest.mark.asyncio
    async def test_recoverable_error_continues_running(self):
        """Test that recoverable errors don't stop the service."""
        service = WorkerService("recoverable-error")
        service.error_conditions['recoverable_at_work'] = 2
        service.error_conditions['stop_at_work'] = 8  # Stop after more work
        
        async with run_service_until_stopped(service, timeout=3.0):
            # Wait for service to reach running state
            await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Wait for completion
            await service.wait_for_state(ServiceState.STOPPED, timeout=3.0)
        
        assert service.state == ServiceState.STOPPED
        assert service.status == ServiceStatus.ERROR  # Had recoverable error
        assert service.errors >= 1
        assert service.work_completed >= 8  # Continued working after error
    
    @pytest.mark.asyncio
    async def test_external_stop_call(self):
        """Test external stop() call during operation."""
        service = WorkerService("external-stop")
        
        await service.start()
        run_task = asyncio.create_task(service.run())
        
        try:
            # Wait for service to be running
            await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # External stop call
            await service.stop()
            
            assert service.state == ServiceState.STOPPED
            
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
    
    @pytest.mark.asyncio
    async def test_health_monitor_triggered_shutdown(self):
        """Test shutdown triggered by health monitoring."""
        service = WorkerService("health-monitor-shutdown")
        
        # Don't configure automatic work errors to avoid interference
        
        await service.start()
        run_task = asyncio.create_task(service.run())
        
        try:
            # Wait for service to be running
            await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Manually trigger multiple errors to exceed threshold (>3)
            for i in range(5):  # Add 5 errors to be well above threshold
                service.record_error(ValueError(f"Test error {i}"), is_fatal=False)
                await asyncio.sleep(0.05)  # Allow health monitor to check between errors
            
            # Wait for health monitor to detect too many errors and stop service
            await service.wait_for_state(ServiceState.STOPPED, timeout=3.0)
            
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
        
        assert service.state == ServiceState.STOPPED
        assert service.shutdown_reason == "too_many_errors"
        assert service.errors >= 4


class TestErrorRecoveryPatterns:
    """Test error recovery and resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker pattern with error recovery."""
        service = CircuitBreakerService("circuit-breaker")
        
        async with run_service_until_stopped(service, timeout=3.0):
            await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Let it run and handle circuit breaker logic
            await asyncio.sleep(1.0)
            
            # Should have some successful calls despite failures
            assert service.successful_calls > 0
            assert service.external_calls > service.successful_calls
    
    @pytest.mark.asyncio
    async def test_error_status_reset(self):
        """Test error status reset after recovery."""
        service = WorkerService("error-reset")
        
        await service.start()
        
        # Inject a recoverable error
        service.record_error(ValueError("Test error"), is_fatal=False)
        assert service.status == ServiceStatus.ERROR
        assert service.errors == 1
        
        # Reset error status (simulating recovery)
        service.reset_error_status()
        assert service.status == ServiceStatus.HEALTHY
        assert service.errors == 1  # Count remains
    
    @pytest.mark.asyncio
    async def test_request_processing_with_errors(self):
        """Test request processing with different error types."""
        service = WorkerService("request-processing")
        
        await service.start()
        
        # Process successful request
        result = await service.process_request({"data": "normal"})
        assert result["status"] == "success"
        assert service.requests_processed == 1
        
        # Process request with recoverable error
        result = await service.process_request({
            "cause_error": True,
            "error_type": "ValueError",
            "is_fatal": False
        })
        # Should complete successfully but record error
        assert service.errors == 1
        assert service.status == ServiceStatus.ERROR
        
        # Process request with fatal error
        with pytest.raises(RuntimeError):
            await service.process_request({
                "cause_error": True,
                "error_type": "RuntimeError", 
                "is_fatal": True
            })
        
        assert service.errors == 2
        assert service.status == ServiceStatus.FATAL


class TestShutdownTaskNaming:
    """Test task naming for debugging and monitoring."""
    
    @pytest.mark.asyncio
    async def test_shutdown_task_naming_patterns(self):
        """Test that shutdown tasks have proper naming for debugging."""
        service = WorkerService("task-naming-test")
        
        # Mock asyncio.create_task to capture task names
        original_create_task = asyncio.create_task
        created_tasks = []
        
        def mock_create_task(coro, **kwargs):
            task = original_create_task(coro, **kwargs)
            if 'name' in kwargs:
                created_tasks.append(kwargs['name'])
            return task
        
        await service.start()
        run_task = asyncio.create_task(service.run())
        
        try:
            # Wait for service to be running
            await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            with patch('asyncio.create_task', side_effect=mock_create_task):
                # Test manual shutdown trigger
                service.request_stop()
                
                # Brief delay for task to be created
                await asyncio.sleep(0.1)
            
            # Wait for shutdown to complete
            await service.wait_for_state(ServiceState.STOPPED, timeout=2.0)
            
            # Create a new service to test fatal error shutdown
            service2 = WorkerService("task-naming-test-2")
            await service2.start()
            run_task2 = asyncio.create_task(service2.run())
            
            try:
                await service2.wait_for_state(ServiceState.RUNNING, timeout=1.0)
                
                with patch('asyncio.create_task', side_effect=mock_create_task):
                    # Test fatal error trigger
                    service2.record_error(RuntimeError("fatal"), is_fatal=True)
                    
                    # Brief delay for task to be created
                    await asyncio.sleep(0.1)
                
                await service2.wait_for_state(ServiceState.STOPPED, timeout=2.0)
                
            finally:
                if not run_task2.done():
                    run_task2.cancel()
                    try:
                        await run_task2
                    except asyncio.CancelledError:
                        pass
        
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
        
        # Verify task naming patterns
        expected_patterns = [
            "task-naming-test-requested-stop",
            "task-naming-test-2-fatal-error-stop"
        ]
        
        for pattern in expected_patterns:
            assert any(pattern in name for name in created_tasks), \
                f"Expected task name pattern '{pattern}' not found in {created_tasks}"


class TestRealWorldUsagePatterns:
    """Test patterns that simulate real-world service usage."""
    
    @pytest.mark.asyncio
    async def test_service_lifecycle_with_multiple_errors(self):
        """Test complete service lifecycle with various error scenarios."""
        service = WorkerService("lifecycle-test")
        
        await service.start()
        assert service.state == ServiceState.STARTED
        
        run_task = asyncio.create_task(service.run())
        
        try:
            # Wait for running state
            await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Process some requests with mixed success/failure
            requests = [
                {"data": "success1"},
                {"cause_error": True, "error_type": "ValueError", "is_fatal": False},
                {"data": "success2"},
                {"cause_error": True, "error_type": "ConnectionError", "is_fatal": False},
                {"data": "success3"},
            ]
            
            for req in requests:
                try:
                    result = await service.process_request(req)
                    logger.info(f"Request processed: {result}")
                except Exception as e:
                    logger.info(f"Request failed as expected: {e}")
            
            # Service should still be running after recoverable errors
            assert service.running
            assert service.requests_processed == len(requests)
            assert service.errors == 2  # Two recoverable errors
            assert service.status == ServiceStatus.ERROR
            
            # Reset error status (simulating recovery/mitigation)
            service.reset_error_status()
            assert service.status == ServiceStatus.HEALTHY
            
            # Process a fatal error
            with pytest.raises(RuntimeError):
                await service.process_request({
                    "cause_error": True, 
                    "error_type": "RuntimeError", 
                    "is_fatal": True
                })
            
            # Service should auto-stop due to fatal error
            await service.wait_for_state(ServiceState.STOPPED, timeout=2.0)
            assert service.status == ServiceStatus.FATAL
            
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
    
    @pytest.mark.asyncio
    async def test_concurrent_shutdown_requests(self):
        """Test handling of concurrent shutdown requests."""
        service = WorkerService("concurrent-shutdown")
        
        await service.start()
        run_task = asyncio.create_task(service.run())
        
        try:
            await service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
            
            # Multiple concurrent shutdown requests
            service.request_stop()
            service.request_stop()
            service.request_stop()
            
            # Small delay to let shutdown tasks be scheduled
            await asyncio.sleep(0.1)
            
            # Service should stop cleanly
            await service.wait_for_state(ServiceState.STOPPED, timeout=2.0)
            assert service.state == ServiceState.STOPPED
            
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
