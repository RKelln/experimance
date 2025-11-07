"""
Backend package for agent implementations.
"""

from .base import AgentBackend, AgentBackendEvent, ConversationTurn, ToolCall, UserContext

# PipecatBackend is imported lazily to avoid pipecat dependency when not needed

__all__ = [
    "AgentBackend", 
    "AgentBackendEvent", 
    "ConversationTurn",
    "ToolCall",
    "UserContext",
    # "PipecatBackend"  # Available via lazy import
]
