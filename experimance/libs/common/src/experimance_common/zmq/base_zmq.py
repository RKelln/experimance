import asyncio
import logging
from contextlib import asynccontextmanager

import zmq

from experimance_common.base_service import BaseService
from experimance_common.service_state import ServiceState

# Configure logging
logger = logging.getLogger(__name__)


class BaseZmqService(BaseService):
    """Base class for ZeroMQ-based services in the Experimance system.
    
    This class extends BaseService with ZeroMQ-specific functionality:
    - ZMQ socket registration and cleanup
    - Common ZMQ communication patterns
    
    Subclasses should implement their specific communication patterns
    by extending this class and implementing the necessary methods.
    """
    
    
    def __init__(self, service_name: str, service_type: str = "zmq-service"):
        """Initialize the base ZMQ service.
        
        Args:
            service_name: Unique name for this service instance
            service_type: Type of service (for logging and monitoring)
        """
        super().__init__(service_name, service_type)
        
        # ZMQ sockets - to be initialized by subclasses
        self._sockets = []
        self._zmq_sockets_closed = False # Initialize flag
    
    def register_socket(self, socket):
        """Register a ZMQ socket for automatic cleanup.
        
        Args:
            socket: ZMQ socket wrapper to register
        """
        self._sockets.append(socket)
    
    
    async def stop(self):
        """Stop the service and clean up ZMQ resources.
        
        This method ensures all ZMQ sockets are properly closed
        in addition to the standard service cleanup.
        """
        logger.debug(f"Entering BaseZmqService.stop() for {self.service_name}. Current state: {self.state}")

        # Close ZMQ sockets first. This should happen before tasks that might use them 
        # are cancelled by super().stop(). This needs to be idempotent.
        if not self._zmq_sockets_closed:
            logger.info(f"Closing ZMQ sockets for {self.service_name}...")
            
            # Give pending operations a chance to complete or be cancelled
            # This helps prevent CancelledError exceptions in callbacks
            try:
                # Short sleep to allow any pending async operations to complete
                # or at least reach a cancellable state
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.debug(f"Sleep interrupted during ZMQ socket cleanup for {self.service_name}")
            
            socket_errors = 0
            # Iterate over a copy if closing modifies the list, or ensure socket.close() is safe
            for socket_obj in reversed(list(self._sockets)): # Iterate over a copy
                if socket_obj: # Check if socket_obj is not None
                    try:
                        logger.debug(f"Closing socket: {type(socket_obj).__name__}")
                        # Make sure linger is set to 0 for immediate close
                        if hasattr(socket_obj, 'socket') and hasattr(socket_obj.socket, 'set'):
                            try:
                                socket_obj.socket.set(zmq.LINGER, 0)
                            except Exception as e:
                                logger.debug(f"Could not set LINGER on socket: {e}")
                        
                        socket_obj.close() # Assuming this is synchronous and idempotent
                    except asyncio.CancelledError:
                        # Ignore CancelledError, as this might happen if the socket is already closed
                        logger.debug(f"CancelledError while closing socket {type(socket_obj).__name__} in {self.service_name}")
                    except Exception as e:
                        logger.warning(f"Error closing ZMQ socket {type(socket_obj).__name__} in {self.service_name}: {e}")
                        socket_errors += 1
            
            if socket_errors > 0:
                logger.warning(f"Encountered {socket_errors} errors while closing ZMQ sockets for {self.service_name}")
            
            self._zmq_sockets_closed = True
            # Clear the original list after closing all sockets from the copy
            self._sockets.clear()
        else:
            logger.debug(f"ZMQ sockets for {self.service_name} already marked as closed.")

        # Delegate to the base class stop method for general task cancellation and state management.
        logger.debug(f"Calling super().stop() from BaseZmqService.stop() for {self.service_name}")
        await super().stop()
        
        logger.debug(f"Exiting BaseZmqService.stop() for {self.service_name}. Final state from super: {self.state}")

