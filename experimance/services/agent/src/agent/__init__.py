"""
Agent service base package.

Provides AgentServiceBase for per-project agent services to subclass.
"""

from .service import AgentServiceBase, SERVICE_TYPE

__all__ = ["AgentServiceBase", "SERVICE_TYPE"]
