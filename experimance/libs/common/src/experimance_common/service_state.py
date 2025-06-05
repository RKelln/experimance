"""
Service state management for Experimance services.

This module provides a state management class that handles service lifecycle states,
state transitions, events, and callbacks.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
from enum import Enum
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

class ServiceState(str, Enum):
    """Service lifecycle states."""
    INITIALIZING = "initializing"  # Service is in the process of initialization
    INITIALIZED = "initialized"    # Service has been fully instantiated
    STARTING = "starting"         # Service is in the process of starting up
    STARTED = "started"           # Service has completed startup but not yet running
    RUNNING = "running"           # Service is fully operational
    STOPPING = "stopping"         # Service is in the process of shutting down
    STOPPED = "stopped"           # Service has been fully stopped


class StateManager:
    """Manages service state transitions with events and callbacks."""
    
    def __init__(self, service_name: str, initial_state: ServiceState = ServiceState.INITIALIZING):
        """Initialize the state manager.
        
        Args:
            service_name: Name of the service for logging purposes
            initial_state: Initial state to start in
        """
        self.service_name = service_name
        self._state = initial_state
        
        # Create events for each state
        self._state_events = {state: asyncio.Event() for state in ServiceState}
        self._state_events[initial_state].set()  # Set initial state event
        
        # Callbacks for state transitions
        self._state_transition_callbacks = {}
        
        # Track transition history for debugging
        self._state_history = [(initial_state, asyncio.get_event_loop().time())]
        
        logger.debug(f"StateManager initialized for {service_name} in state {initial_state}")
    
    @property
    def state(self) -> ServiceState:
        """Get the current service state."""
        return self._state
    
    @state.setter
    def state(self, new_state: ServiceState):
        """Set the service state and trigger the corresponding event."""
        old_state = self._state
        if new_state != old_state:
            timestamp = asyncio.get_event_loop().time()
            self._state_history.append((new_state, timestamp))
            
            logger.debug(f"Service {self.service_name} state changing from {old_state} to {new_state}")
            
            # Clear the old state event and set the new one
            self._state_events[old_state].clear()
            self._state_events[new_state].set()
            self._state = new_state
            
            # Execute callbacks for the new state
            if new_state in self._state_transition_callbacks:
                for callback in self._state_transition_callbacks[new_state]:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error in state transition callback: {e}")
            
            # Log the state change
            logger.info(f"Service {self.service_name} state changed to {new_state}")
    
    def register_state_callback(self, state: ServiceState, callback: Callable[[], None]):
        """Register a callback to be called when the service enters a specific state.
        
        Args:
            state: State to trigger the callback
            callback: Function to call when the state is entered
        """
        if state not in self._state_transition_callbacks:
            self._state_transition_callbacks[state] = []
        
        self._state_transition_callbacks[state].append(callback)
    
    async def wait_for_state(self, state: ServiceState, timeout: Optional[float] = None) -> bool:
        """Wait until the service reaches the specified state.
        
        Args:
            state: State to wait for
            timeout: Maximum time to wait in seconds, or None to wait indefinitely
            
        Returns:
            True if the state was reached, False if timeout occurred
        """
        if self._state == state:
            return True
            
        try:
            await asyncio.wait_for(self._state_events[state].wait(), timeout)
            return True
        except asyncio.TimeoutError:
            logger.debug(f"Timeout waiting for service {self.service_name} to reach state {state}")
            return False
    
    @asynccontextmanager
    async def observe_state_change(self, expected_state: ServiceState, timeout: float = 5.0):
        """Context manager to observe state changes.
        
        This allows for elegant testing of state transitions.
        
        Example usage:
            async with state_manager.observe_state_change(ServiceState.RUNNING):
                # Code that should trigger the state change
                
        Args:
            expected_state: The state to wait for
            timeout: Maximum time to wait in seconds
            
        Yields:
            None
            
        Raises:
            asyncio.TimeoutError: If the expected state is not reached within the timeout
        """
        # Create a future that will be resolved when the state changes to expected_state
        future = asyncio.get_running_loop().create_future()
        
        def on_state_change():
            if not future.done():
                future.set_result(True)
        
        # Register the callback
        self.register_state_callback(expected_state, on_state_change)
        
        try:
            # Yield to let the caller execute code that should trigger the state change
            yield
            
            # Wait for the state change with timeout
            try:
                if self.state != expected_state:
                    await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"Timeout waiting for service {self.service_name} to reach state {expected_state}. "
                    f"Current state is {self.state}."
                )
        finally:
            # Clean up the callback to avoid memory leaks
            if expected_state in self._state_transition_callbacks:
                callbacks = self._state_transition_callbacks[expected_state]
                if on_state_change in callbacks:
                    callbacks.remove(on_state_change)
    
    def get_state_history(self):
        """Get the history of state transitions.
        
        Returns:
            List of (state, timestamp) tuples
        """
        return self._state_history
    
    def validate_and_begin_transition(self, method_name: str, valid_states: Set[ServiceState], 
                                     progress_state: ServiceState) -> None:
        """Validate current state and begin a lifecycle method transition.
        
        This method should be called at the BEGINNING of lifecycle methods in the leaf class
        to validate the state and set it to the "in progress" state (e.g., STARTING).
        
        Args:
            method_name: Name of the lifecycle method (for logging)
            valid_states: Set of valid states for this transition
            progress_state: The "in progress" state to set (e.g., STARTING)
            
        Raises:
            RuntimeError: If current state is not valid for this transition
        """
        current = self._state
        
        # First validate the transition
        if current not in valid_states:
            raise RuntimeError(
                f"Cannot call {method_name}() when service {self.service_name} is in {current} state. "
                f"Valid states are: {', '.join(str(s) for s in valid_states)}"
            )
        
        # Set the in-progress state
        if current != progress_state:
            logger.debug(f"Service {self.service_name} beginning {method_name}: {current} -> {progress_state}")
            self.state = progress_state
    
    def complete_transition(self, method_name: str, progress_state: ServiceState,
                           completed_state: ServiceState) -> None:
        """Complete a lifecycle method transition.
        
        This method should be called at the END of lifecycle methods in the leaf class
        to set the state to the "completed" state (e.g., STARTED).
        
        Args:
            method_name: Name of the lifecycle method (for logging)
            progress_state: The expected "in progress" state (e.g., STARTING)
            completed_state: The "completed" state to set (e.g., STARTED)
        """
        current = self._state
        
        # Only change state if we're still in the expected in-progress state
        if current == progress_state:
            logger.debug(f"Service {self.service_name} completing {method_name}: {current} -> {completed_state}")
            self.state = completed_state
        else:
            logger.debug(f"Service {self.service_name} state changed from {progress_state} to {current} during {method_name}. Not changing to {completed_state}.")