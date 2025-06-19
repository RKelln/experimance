"""
Worker configuration schemas for ZMQ Multi-Controller.

This module defines Pydantic models for validating worker configurations
in a type-safe way, similar to the service configurations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator

from experimance_common.constants import DEFAULT_PORTS
from experimance_common.zmq.zmq_utils import MessageType


class WorkerConnectionConfig(BaseModel):
    """Configuration for a single worker connection."""
    
    worker_type: str = Field(
        description="Type of worker (e.g., 'image', 'transition')"
    )
    
    push_address: str = Field(
        description="Address for controller to bind PUSH socket (tasks to workers)"
    )
    
    pull_address: str = Field(
        description="Address for controller to bind PULL socket (results from workers)"
    )
    
    push_message_types: List[MessageType] = Field(
        default_factory=list,
        description="Message types this worker accepts via PUSH (tasks)"
    )
    
    pull_message_types: List[MessageType] = Field(
        default_factory=list, 
        description="Message types this worker sends via PULL (results)"
    )
    
    @validator('push_address', 'pull_address')
    def validate_address_format(cls, v):
        """Validate ZMQ address format."""
        if not v.startswith('tcp://'):
            raise ValueError(f"Address must start with 'tcp://': {v}")
        return v
    
    @validator('push_address')
    def validate_push_address_binding(cls, v):
        """Validate that push address is for binding (controller side)."""
        if not ('*' in v or '0.0.0.0' in v):
            raise ValueError(f"Push address should bind to all interfaces (use '*' or '0.0.0.0'): {v}")
        return v
    
    @validator('pull_address') 
    def validate_pull_address_binding(cls, v):
        """Validate that pull address is for binding (controller side)."""
        if not ('*' in v or '0.0.0.0' in v):
            raise ValueError(f"Pull address should bind to all interfaces (use '*' or '0.0.0.0'): {v}")
        return v
    
    def get_push_port(self) -> int:
        """Extract port number from push address."""
        try:
            return int(self.push_address.split(":")[-1])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot extract port from push address: {self.push_address}")
    
    def get_pull_port(self) -> int:
        """Extract port number from pull address.""" 
        try:
            return int(self.pull_address.split(":")[-1])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot extract port from pull address: {self.pull_address}")
    
    @validator('pull_address')
    def validate_no_port_conflict(cls, v, values):
        """Validate that push and pull ports are different."""
        if 'push_address' not in values:
            return v
            
        try:
            push_port = int(values['push_address'].split(":")[-1])
            pull_port = int(v.split(":")[-1])
            if push_port == pull_port:
                raise ValueError(f"Push and pull ports cannot be the same: {push_port}")
        except (ValueError, IndexError):
            pass  # Let other validators handle address format errors
        return v


class MultiControllerWorkerConfig(BaseModel):
    """Configuration for all workers in a multi-controller setup."""
    
    workers: Dict[str, WorkerConnectionConfig] = Field(
        description="Worker configurations keyed by worker type"
    )
    
    @validator('workers')
    def validate_no_port_conflicts(cls, v):
        """Validate that no workers have conflicting ports."""
        used_ports = set()
        
        for worker_type, config in v.items():
            push_port = config.get_push_port()
            pull_port = config.get_pull_port()
            
            if push_port in used_ports:
                raise ValueError(f"Port conflict: {worker_type} push port {push_port} already in use")
            if pull_port in used_ports:
                raise ValueError(f"Port conflict: {worker_type} pull port {pull_port} already in use")
                
            used_ports.update([push_port, pull_port])
        
        return v
    
    @validator('workers')
    def validate_message_type_uniqueness(cls, v):
        """Validate that each push message type is handled by exactly one worker."""
        push_message_map = {}
        
        for worker_type, config in v.items():
            for msg_type in config.push_message_types:
                if msg_type in push_message_map:
                    raise ValueError(
                        f"Message type {msg_type} handled by multiple workers: "
                        f"{push_message_map[msg_type]} and {worker_type}"
                    )
                push_message_map[msg_type] = worker_type
        
        return v
    
    def get_worker_for_message_type(self, message_type: MessageType) -> Optional[str]:
        """Get the worker type that handles a specific message type."""
        for worker_type, config in self.workers.items():
            if message_type in config.push_message_types:
                return worker_type
        return None
    
    def get_routing_map(self) -> Dict[MessageType, str]:
        """Get a mapping of message types to worker types."""
        routing_map = {}
        for worker_type, config in self.workers.items():
            for msg_type in config.push_message_types:
                routing_map[msg_type] = worker_type
        return routing_map


# Predefined configurations for common setups
DEFAULT_IMAGE_WORKER_CONFIG = WorkerConnectionConfig(
    worker_type="image",
    push_address=f"tcp://*:{DEFAULT_PORTS['images']}",
    pull_address=f"tcp://*:{DEFAULT_PORTS['image_results']}",
    push_message_types=[MessageType.RENDER_REQUEST],
    pull_message_types=[MessageType.IMAGE_READY]
)

DEFAULT_TRANSITION_WORKER_CONFIG = WorkerConnectionConfig(
    worker_type="transition",
    push_address=f"tcp://*:{DEFAULT_PORTS['transitions']}",
    pull_address=f"tcp://*:{DEFAULT_PORTS['transition_results']}",
    push_message_types=[MessageType.TRANSITION_REQUEST],
    pull_message_types=[MessageType.TRANSITION_READY]
)

DEFAULT_LOOP_WORKER_CONFIG = WorkerConnectionConfig(
    worker_type="loop",
    push_address=f"tcp://*:{DEFAULT_PORTS['videos']}",
    pull_address=f"tcp://*:{DEFAULT_PORTS['video_results']}",
    push_message_types=[MessageType.LOOP_REQUEST],
    pull_message_types=[MessageType.LOOP_READY]
)

DEFAULT_MULTI_CONTROLLER_CONFIG = MultiControllerWorkerConfig(
    workers={
        "image": DEFAULT_IMAGE_WORKER_CONFIG,
        "transition": DEFAULT_TRANSITION_WORKER_CONFIG,
        "loop": DEFAULT_LOOP_WORKER_CONFIG
    }
)
