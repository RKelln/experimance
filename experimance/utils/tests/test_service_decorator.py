#!/usr/bin/env python3
"""
Tests for decorator-based state management with BaseService.

This test validates how the @lifecycle_service decorator handles service state transitions,
particularly with custom state management methods.

Run with:
    uv run -m pytest utils/tests/test_service_decorator.py -v
"""

import asyncio
import logging
from typing import Set

from experimance_common.constants import TICK
import pytest

from experimance_common.service import BaseService, ServiceState
from experimance_common.service_state import StateManager

# Configure test logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DecoratedStateService(BaseService):
    """A service to test state validation and transition methods."""
    
    def __init__(self, name: str = "decorated-service"):
        super().__init__(service_name=name, service_type="test")
        self.custom_method_calls = []
    
    async def custom_validation(self, valid_states: Set[ServiceState]) -> bool:
        """A method that validates the current state using StateManager directly."""
        method_name = 'custom_validation'
        try:
            # Manually validate state
            self._state_manager.validate_and_begin_transition(
                method_name, valid_states, self.state
            )
            # Record the call
            self.custom_method_calls.append((method_name, self.state))
            return True
        except RuntimeError:
            # Record the failure
            self.custom_method_calls.append((f"{method_name}_failed", self.state))
            return False
    
    async def custom_transition(self, valid_states: Set[ServiceState], target_state: ServiceState) -> bool:
        """A method that performs a full state transition."""
        method_name = 'custom_transition'
        
        # Validate and begin transition
        try:
            self._state_manager.validate_and_begin_transition(
                method_name, valid_states, ServiceState.STOPPING
            )
        except RuntimeError:
            # Record failed validation
            self.custom_method_calls.append((f"{method_name}_failed", self.state))
            return False
            
        # Record the call
        self.custom_method_calls.append((method_name, self.state))
        
        # Run some "work"
        await asyncio.sleep(TICK)
        
        # Complete the transition
        self._state_manager.complete_transition(
            method_name, ServiceState.STOPPING, target_state
        )
        
        # Record after transition
        self.custom_method_calls.append((f"{method_name}_completed", self.state))
        return True


class TestServiceDecorators:
    """Tests for service state decorators."""
    
    @pytest.fixture
    async def decorated_service(self):
        """Create a DecoratedStateService instance for testing."""
        service = DecoratedStateService()
        yield service
        # Clean up after test
        if service.state != ServiceState.STOPPED:
            await service.stop()
    
    @pytest.mark.asyncio
    async def test_state_validation_success(self, decorated_service):
        """Test state validation with valid state."""
        # Service starts in INITIALIZED state
        valid_states = {ServiceState.INITIALIZED, ServiceState.STOPPED}
        
        # Should succeed with valid state
        result = await decorated_service.custom_validation(valid_states)
        assert result is True
        
        # Verify the call was recorded
        assert ('custom_validation', ServiceState.INITIALIZED) in decorated_service.custom_method_calls
    
    @pytest.mark.asyncio
    async def test_state_validation_failure(self, decorated_service):
        """Test state validation with invalid state."""
        # Service starts in INITIALIZED state
        valid_states = {ServiceState.RUNNING, ServiceState.STOPPED}
        
        # Should fail with invalid state
        result = await decorated_service.custom_validation(valid_states)
        assert result is False
        
        # Verify the failure was recorded
        assert ('custom_validation_failed', ServiceState.INITIALIZED) in decorated_service.custom_method_calls
    
    @pytest.mark.asyncio
    async def test_custom_transition_success(self, decorated_service):
        """Test custom transition from a valid state."""
        # Service starts in INITIALIZED state
        valid_states = {ServiceState.INITIALIZED}
        target_state = ServiceState.RUNNING
        
        # Should succeed with valid transition
        result = await decorated_service.custom_transition(valid_states, target_state)
        assert result is True
        
        # Verify the state changed
        assert decorated_service.state == ServiceState.RUNNING
        
        # Verify the transition was recorded
        assert ('custom_transition', ServiceState.STOPPING) in decorated_service.custom_method_calls
        assert ('custom_transition_completed', ServiceState.RUNNING) in decorated_service.custom_method_calls
    
    @pytest.mark.asyncio  
    async def test_custom_transition_failure(self, decorated_service):
        """Test custom transition from an invalid state."""
        # Start the service to change state
        await decorated_service.start()
        assert decorated_service.state == ServiceState.STARTED
        
        # Try transition with wrong valid_states
        valid_states = {ServiceState.INITIALIZED}
        target_state = ServiceState.RUNNING
        
        # Should fail with invalid state
        result = await decorated_service.custom_transition(valid_states, target_state)
        assert result is False
        
        # State should remain unchanged
        assert decorated_service.state == ServiceState.STARTED
        
        # Verify the failure was recorded
        assert ('custom_transition_failed', ServiceState.STARTED) in decorated_service.custom_method_calls
    
    @pytest.mark.asyncio
    async def test_state_history_tracking(self, decorated_service):
        """Test that state history is properly tracked during transitions."""
        # Execute a complete lifecycle to check history
        await decorated_service.start()
        run_task = asyncio.create_task(decorated_service.run())
        
        # Wait a bit for run to change state
        await asyncio.sleep(TICK)
        
        await decorated_service.stop()
        
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
        
        # Get state history
        history = decorated_service._state_manager.get_state_history()
        states = [state for state, _ in history]
        
        # Check that all lifecycle states were recorded
        assert ServiceState.INITIALIZING in states
        assert ServiceState.INITIALIZED in states
        assert ServiceState.STARTING in states 
        assert ServiceState.STARTED in states
        assert ServiceState.RUNNING in states
        assert ServiceState.STOPPING in states
        assert ServiceState.STOPPED in states
        
        # Check state ordering - key transitions
        init_idx = states.index(ServiceState.INITIALIZED)
        stopped_idx = states.index(ServiceState.STOPPED)
        assert init_idx < stopped_idx, "INITIALIZED should come before STOPPED in history"
