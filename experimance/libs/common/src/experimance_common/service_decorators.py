"""
Decorators for Experimance service methods to handle state transitions.

This module provides decorators that can be used to mark service methods
that should set states at entry and exit points, with proper handling of
inheritance chains.
"""

import functools
import logging
from typing import Any, Callable, Optional, TypeVar, cast

from .service_state import ServiceState

logger = logging.getLogger(__name__)

# Type variable for the decorated method
F = TypeVar('F', bound=Callable)

def lifecycle_service(cls):
    """Class decorator to add service lifecycle state management to methods.
    
    This decorator finds and wraps start(), stop(), and run() methods to handle proper 
    state transitions. The decoration happens at class definition time.
    
    Example:
        @lifecycle_service
        class MyService(BaseService):
            async def start(self):
                # Custom initialization
                await super().start()
    """
    original_init = cls.__init__
    
    # Find original methods in the class
    original_start = getattr(cls, 'start', None)
    original_stop = getattr(cls, 'stop', None)
    original_run = getattr(cls, 'run', None)
    
    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        # Call original __init__
        original_init(self, *args, **kwargs)
        
        # Set up validation rules for each lifecycle method
        self._lifecycle_rules = {
            'start': {
                'valid_states': {ServiceState.INITIALIZED, ServiceState.STOPPED},
                'progress_state': ServiceState.STARTING,
                'completed_state': ServiceState.STARTED
            },
            'stop': {
                'valid_states': {ServiceState.INITIALIZED, ServiceState.STARTING, 
                               ServiceState.STARTED, ServiceState.RUNNING},
                'progress_state': ServiceState.STOPPING,
                'completed_state': ServiceState.STOPPED
            },
            'run': {
                'valid_states': {ServiceState.STARTED},
                'progress_state': ServiceState.RUNNING,
                'completed_state': ServiceState.RUNNING  # Remains in RUNNING until stop
            }
        }
    
    if original_start:
        @functools.wraps(original_start)
        async def wrapped_start(self, *args, **kwargs):
            # Validate and set state at the beginning of the call chain
            self._state_manager.validate_and_begin_transition(
                'start',
                self._lifecycle_rules['start']['valid_states'], 
                self._lifecycle_rules['start']['progress_state']
            )
            
            # Call the original method
            result = await original_start(self, *args, **kwargs)
            
            # Set the completed state at the end
            self._state_manager.complete_transition(
                'start',
                self._lifecycle_rules['start']['progress_state'],
                self._lifecycle_rules['start']['completed_state']
            )
            
            return result
        
        cls.start = wrapped_start
    
    if original_stop:
        @functools.wraps(original_stop)
        async def wrapped_stop(self, *args, **kwargs):
            # Skip if already stopped
            if self.state == ServiceState.STOPPED:
                return
                
            # Validate and set state at the beginning
            self._state_manager.validate_and_begin_transition(
                'stop',
                self._lifecycle_rules['stop']['valid_states'], 
                self._lifecycle_rules['stop']['progress_state']
            )
            
            # Call the original method
            result = await original_stop(self, *args, **kwargs)
            
            # Set the completed state at the end
            self._state_manager.complete_transition(
                'stop',
                self._lifecycle_rules['stop']['progress_state'],
                self._lifecycle_rules['stop']['completed_state']
            )
            
            return result
        
        cls.stop = wrapped_stop
    
    if original_run:
        @functools.wraps(original_run)
        async def wrapped_run(self, *args, **kwargs):
            # Validate and set state at the beginning
            self._state_manager.validate_and_begin_transition(
                'run',
                self._lifecycle_rules['run']['valid_states'], 
                self._lifecycle_rules['run']['progress_state']
            )
            
            # Call the original method
            return await original_run(self, *args, **kwargs)
        
        cls.run = wrapped_run
    
    # Replace __init__ to set up lifecycle rules
    cls.__init__ = wrapped_init
    
    return cls

