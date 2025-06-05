#!/usr/bin/env python3
"""
Tests for the service state management in experimance_common.

This test suite validates:
1. State transitions in service lifecycle methods (start, stop, run)
2. State transitions across class inheritance hierarchies
3. StateManager validation and transition methods
4. State observation and waiting utilities
5. Service state decorators and custom transitions

Run with:
    uv run -m pytest utils/tests/test_service_state.py -v
"""

import asyncio
import logging
import signal
from typing import Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from experimance_common.service import BaseService, ServiceState
from experimance_common.service_state import StateManager
from experimance_common.service_decorators import lifecycle_service
from utils.tests.test_utils import wait_for_service_shutdown, wait_for_service_state

# Configure test logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleService(BaseService):
    """A simple service implementation for testing."""
    
    def __init__(self, name: str = "simple-service"):
        super().__init__(service_name=name, service_type="test")
        self.start_called = False
        self.stop_called = False
        self.run_called = False
    
    async def start(self):
        self.start_called = True
        await super().start()
    
    async def stop(self):
        self.stop_called = True
        await super().stop()
    
    async def run(self):
        self.run_called = True
        try:
            # Register a periodic task to ensure run() doesn't exit immediately
            async def periodic_check():
                while self.state == ServiceState.RUNNING:
                    await asyncio.sleep(0.1)  # Short sleep to simulate work
                logger.debug(f"periodic_check exiting in {self.service_name}")
                    
            # Store reference to allow proper cleanup
            self.periodic_task = periodic_check()
            self._register_task(self.periodic_task)
            await super().run()
        except asyncio.CancelledError:
            logger.debug(f"SimpleService.run() was cancelled for {self.service_name}")
            raise  # Re-raise CancelledError to ensure proper propagation


class ChildService(SimpleService):
    """A child service for testing inheritance chains."""
    
    def __init__(self, name: str = "child-service"):
        super().__init__(name=name)
        self.child_start_called = False
        self.child_stop_called = False
        self.child_run_called = False
    
    async def start(self):
        self.child_start_called = True
        await super().start()
    
    async def stop(self):
        self.child_stop_called = True
        await super().stop()
    
    async def run(self):
        self.child_run_called = True
        try:
            await super().run()
        except asyncio.CancelledError:
            logger.debug(f"ChildService.run() was cancelled for {self.service_name}")
            raise  # Re-raise CancelledError to ensure proper propagation


class CustomStateService(BaseService):
    """A service with custom state validation and transition methods."""
    
    def __init__(self, name: str = "custom-state-service"):
        super().__init__(service_name=name, service_type="test")
        self.custom_method_called = False
        self.custom_state_reached = False
    
    async def custom_method(self, valid_states: Set[ServiceState], target_state: ServiceState):
        """A custom method that uses state validation and transition."""
        # Validate and begin transition
        self._state_manager.validate_and_begin_transition(
            'custom_method',
            valid_states,
            ServiceState.STOPPING  # Use STOPPING as temporary state
        )
        
        try:
            # Perform work
            self.custom_method_called = True
            await asyncio.sleep(0.1)
            
            # Simulate reaching custom state
            self.custom_state_reached = True
        finally:
            # Complete transition
            self._state_manager.complete_transition(
                'custom_method',
                ServiceState.STOPPING,
                target_state
            )


class TestStateManager:
    """Tests for the StateManager class directly."""
    
    @pytest.fixture
    def state_manager(self):
        """Create a StateManager instance for testing."""
        return StateManager("test-manager", ServiceState.INITIALIZED)
    
    def test_initial_state(self, state_manager):
        """Test that StateManager initializes with the correct state."""
        assert state_manager.state == ServiceState.INITIALIZED
    
    def test_state_transitions(self, state_manager):
        """Test direct state transitions."""
        state_manager.state = ServiceState.STARTING
        assert state_manager.state == ServiceState.STARTING
        
        state_manager.state = ServiceState.STARTED
        assert state_manager.state == ServiceState.STARTED
    
    def test_state_history(self, state_manager):
        """Test that state history is recorded."""
        # Initial state is already in history
        assert len(state_manager.get_state_history()) == 1
        
        state_manager.state = ServiceState.STARTING
        state_manager.state = ServiceState.STARTED
        state_manager.state = ServiceState.RUNNING
        
        history = state_manager.get_state_history()
        assert len(history) == 4  # Initial + 3 transitions
        
        # Check the states in the history
        states = [state for state, _ in history]
        assert states == [
            ServiceState.INITIALIZED, 
            ServiceState.STARTING, 
            ServiceState.STARTED, 
            ServiceState.RUNNING
        ]
    
    def test_validate_and_begin_transition_valid(self, state_manager):
        """Test validate_and_begin_transition with valid state."""
        state_manager.state = ServiceState.INITIALIZED
        
        # Should not raise an exception
        state_manager.validate_and_begin_transition(
            'test_method',
            {ServiceState.INITIALIZED, ServiceState.STOPPED},
            ServiceState.STARTING
        )
        
        assert state_manager.state == ServiceState.STARTING
    
    def test_validate_and_begin_transition_invalid(self, state_manager):
        """Test validate_and_begin_transition with invalid state."""
        state_manager.state = ServiceState.RUNNING
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError):
            state_manager.validate_and_begin_transition(
                'test_method',
                {ServiceState.INITIALIZED, ServiceState.STOPPED},
                ServiceState.STARTING
            )
        
        # State should remain unchanged
        assert state_manager.state == ServiceState.RUNNING
    
    def test_complete_transition(self, state_manager):
        """Test complete_transition."""
        state_manager.state = ServiceState.STARTING
        
        state_manager.complete_transition(
            'test_method',
            ServiceState.STARTING,
            ServiceState.STARTED
        )
        
        assert state_manager.state == ServiceState.STARTED
    
    def test_complete_transition_with_state_change(self, state_manager):
        """Test complete_transition when state was changed during the method execution."""
        state_manager.state = ServiceState.STARTING
        
        # Simulate state changing during method execution
        state_manager.state = ServiceState.RUNNING
        
        # Should not change the state
        state_manager.complete_transition(
            'test_method',
            ServiceState.STARTING,
            ServiceState.STARTED
        )
        
        # State should remain what it was changed to
        assert state_manager.state == ServiceState.RUNNING
    
    @pytest.mark.asyncio
    async def test_wait_for_state(self, state_manager):
        """Test waiting for a state."""
        # Schedule a state change
        asyncio.get_event_loop().call_later(
            0.1, 
            lambda: setattr(state_manager, 'state', ServiceState.RUNNING)
        )
        
        # Wait for the state change
        result = await state_manager.wait_for_state(ServiceState.RUNNING, timeout=0.5)
        
        assert result is True
        assert state_manager.state == ServiceState.RUNNING
    
    @pytest.mark.asyncio
    async def test_wait_for_state_timeout(self, state_manager):
        """Test waiting for a state with timeout."""
        # Wait for a state that won't be set
        result = await state_manager.wait_for_state(ServiceState.RUNNING, timeout=0.1)
        
        assert result is False
        assert state_manager.state == ServiceState.INITIALIZED
    
    @pytest.mark.asyncio
    async def test_observe_state_change(self, state_manager):
        """Test observing a state change."""
        # Schedule a state change
        asyncio.get_event_loop().call_later(
            0.1, 
            lambda: setattr(state_manager, 'state', ServiceState.RUNNING)
        )
        
        # Use the context manager to observe the state change
        async with state_manager.observe_state_change(ServiceState.RUNNING, timeout=0.5):
            pass
        
        assert state_manager.state == ServiceState.RUNNING
    
    @pytest.mark.asyncio
    async def test_observe_state_change_timeout(self, state_manager):
        """Test observing a state change with timeout."""
        with pytest.raises(asyncio.TimeoutError):
            # Use the context manager to observe a state change that won't happen
            async with state_manager.observe_state_change(ServiceState.RUNNING, timeout=0.1):
                pass
        
        assert state_manager.state == ServiceState.INITIALIZED


class TestServiceStateLifecycle:
    """Tests for service state lifecycle transitions."""
    
    @pytest.fixture
    async def simple_service(self):
        """Create a SimpleService instance for testing."""
        service = SimpleService(name="state-test-service")
        yield service
        # Clean up
        if service.state != ServiceState.STOPPED:
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_initialization_state(self, simple_service):
        """Test that service initializes with correct state."""
        assert simple_service.state == ServiceState.INITIALIZED
    
    @pytest.mark.asyncio
    async def test_start_state_transition(self, simple_service):
        """Test state transition during start."""
        # Start should transition INITIALIZED -> STARTING -> STARTED
        async with simple_service.observe_state_change(ServiceState.STARTED):
            await simple_service.start()
        
        # Final state after start() should be STARTED
        assert simple_service.state == ServiceState.STARTED
        assert simple_service.start_called is True
    
    @pytest.mark.asyncio
    async def test_run_state_transition(self, simple_service):
        """Test state transition during run."""
        # Start the service first
        await simple_service.start()
        
        # Run should transition STARTED -> RUNNING
        run_task = asyncio.create_task(simple_service.run(), name="simple_service_run_task")
        
        # Wait for RUNNING state
        await simple_service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
        
        # Verify service is running correctly
        assert simple_service.state == ServiceState.RUNNING
        assert simple_service.run_called is True
        
        # Now stop the service
        logger.debug("Stopping service after verifying it's RUNNING")
        await simple_service.stop()
        
        # Verify the service has stopped
        assert simple_service.state == ServiceState.STOPPED
        
        # Check if the task is done - it should be after the service is stopped
        if not run_task.done():
            logger.warning("Task is not yet done after service stopped, cancelling directly")
            run_task.cancel()
            try:
                await asyncio.wait_for(run_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # The task should be done now
        assert run_task.done(), "Run task should be completed or cancelled"
    
    @pytest.mark.asyncio
    async def test_stop_state_transition(self, simple_service):
        """Test state transition during stop."""
        # Start and run the service first
        await simple_service.start()
        run_task = asyncio.create_task(simple_service.run())
        
        # Wait for RUNNING state
        await simple_service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
        
        # Stop should transition RUNNING -> STOPPING -> STOPPED
        async with simple_service.observe_state_change(ServiceState.STOPPED):
            await simple_service.stop()
        
        assert simple_service.state == ServiceState.STOPPED
        assert simple_service.stop_called is True
        
        # Clean up run task
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_call_stop_without_start(self):
        """Test calling stop without starting."""
        service = SimpleService(name="no-start-service")
        
        # Stop without start should be handled gracefully
        await service.stop()
        assert service.state == ServiceState.STOPPED
    
    @pytest.mark.asyncio
    async def test_call_run_without_start(self):
        """Test calling run without starting."""
        service = SimpleService(name="no-start-service")
        
        # Run without start should raise a RuntimeError due to state validation
        with pytest.raises(RuntimeError):
            await service.run()
        
        # State should remain INITIALIZED
        assert service.state == ServiceState.INITIALIZED


class TestServiceStateInheritance:
    """Tests for service state transitions across inheritance chains."""
    
    @pytest.fixture
    async def child_service(self):
        """Create a ChildService instance for testing."""
        service = ChildService(name="child-test-service")
        yield service
        # Clean up
        if service.state != ServiceState.STOPPED:
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_start_inheritance_chain(self, child_service):
        """Test state transition during start in inheritance chain."""
        # Start should transition INITIALIZED -> STARTING -> STARTED
        async with child_service.observe_state_change(ServiceState.STARTED):
            await child_service.start()
        
        # Both child and parent start methods should be called
        assert child_service.state == ServiceState.STARTED
        assert child_service.child_start_called is True
        assert child_service.start_called is True
    
    @pytest.mark.asyncio
    async def test_run_inheritance_chain(self, child_service: ChildService):
        """Test state transition during run in inheritance chain."""
        # Start the service first
        await child_service.start()
        
        # Run should transition STARTED -> RUNNING
        run_task = asyncio.create_task(child_service.run(), name="child_service_run_task")
        
        # Wait for RUNNING state
        await child_service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
        
        # Both child and parent run methods should be called
        assert child_service.state == ServiceState.RUNNING
        assert child_service.child_run_called is True
        assert child_service.run_called is True
        
        # Now stop the service
        logger.debug("Stopping service after verifying it's RUNNING")
        await child_service.stop()
        
        # Verify the service has stopped
        assert child_service.state == ServiceState.STOPPED

        # Check if the task is done - it should be after the service is stopped
        if not run_task.done():
            logger.warning("Task is not yet done after service stopped, cancelling directly")
            run_task.cancel()
            try:
                await asyncio.wait_for(run_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        child_service._clear_tasks()

        # The task should be done now
        assert run_task.done(), "Run task should be completed or cancelled"
    
    @pytest.mark.asyncio
    async def test_stop_inheritance_chain(self, child_service):
        """Test state transition during stop in inheritance chain."""
        # Start and run the service first
        await child_service.start()
        run_task = asyncio.create_task(child_service.run())
        
        # Wait for RUNNING state
        await child_service.wait_for_state(ServiceState.RUNNING, timeout=1.0)
        
        # Stop should transition RUNNING -> STOPPING -> STOPPED
        async with child_service.observe_state_change(ServiceState.STOPPED):
            await child_service.stop()
        
        # Both child and parent stop methods should be called
        assert child_service.state == ServiceState.STOPPED
        assert child_service.child_stop_called is True
        assert child_service.stop_called is True
        
        # Clean up run task
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass


class TestCustomStateTransitions:
    """Tests for custom state transitions using StateManager directly."""
    
    @pytest.fixture
    async def custom_service(self):
        """Create a CustomStateService instance for testing."""
        service = CustomStateService(name="custom-test-service")
        yield service
        # Clean up
        try:
            if service.state != ServiceState.STOPPED:
                # If we're in STOPPING state, just force to STOPPED rather than calling stop()
                if service.state == ServiceState.STOPPING:
                    logger.debug("Service is in STOPPING state during fixture teardown, forcing to STOPPED")
                    service._state_manager.state = ServiceState.STOPPED
                else:
                    await service.stop()
        except Exception as e:
            logger.warning(f"Error during custom_service fixture teardown: {e}")
            service._state_manager.state = ServiceState.STOPPED
    
    @pytest.mark.asyncio
    async def test_custom_state_transition_valid(self, custom_service):
        """Test custom state transition with valid state."""
        # Start the service first
        await custom_service.start()
        
        # Call custom method
        await custom_service.custom_method(
            valid_states={ServiceState.STARTED},
            target_state=ServiceState.RUNNING
        )
        
        # Should transition through the states
        assert custom_service.state == ServiceState.RUNNING
        assert custom_service.custom_method_called is True
        assert custom_service.custom_state_reached is True
    
    @pytest.mark.asyncio
    async def test_custom_state_transition_invalid(self, custom_service):
        """Test custom state transition with invalid state."""
        # Service is in INITIALIZED state
        
        # Call custom method with wrong valid_states
        with pytest.raises(RuntimeError):
            await custom_service.custom_method(
                valid_states={ServiceState.RUNNING},
                target_state=ServiceState.STOPPED
            )
        
        # Method should not execute due to state validation
        assert custom_service.state == ServiceState.INITIALIZED
        assert custom_service.custom_method_called is False
        assert custom_service.custom_state_reached is False
    
    @pytest.mark.asyncio
    async def test_custom_state_transition_with_exception(self, custom_service):
        """Test custom state transition with exception."""
        # Override custom_method to raise an exception
        original_method = custom_service.custom_method
        
        async def failing_method(*args, **kwargs):
            # Begin state transition
            custom_service._state_manager.validate_and_begin_transition(
                'failing_method',
                {ServiceState.INITIALIZED},
                ServiceState.STOPPING
            )
            
            # Raise an exception before completing transition
            raise ValueError("Test exception")
        
        # Replace the method
        custom_service.custom_method = failing_method
        
        try:
            # Call the failing method
            with pytest.raises(ValueError):
                await custom_service.custom_method()
            
            # State should be left at STOPPING
            assert custom_service.state == ServiceState.STOPPING
            
            # Manually set state to STOPPED so the fixture teardown doesn't try to call stop()
            # This simulates what would happen in a real service where error handling would
            # ensure proper state transition even after an exception
            custom_service._state_manager.state = ServiceState.STOPPED
        finally:
            # Restore original method
            custom_service.custom_method = original_method
    
