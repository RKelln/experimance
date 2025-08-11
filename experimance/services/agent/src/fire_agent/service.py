from __future__ import annotations

from experimance_common.logger import setup_logging
from agent import AgentServiceBase, SERVICE_TYPE
from agent.config import AgentServiceConfig

logger = setup_logging(__name__, log_filename=f"{SERVICE_TYPE}.log")

class FireAgentService(AgentServiceBase):
    """Minimal voice-only agent for the Fire project."""

    def register_project_handlers(self) -> None:
        # No project-specific handlers yet
        return

    async def _initialize_background_tasks(self):
        # TODO: wait for presence before starting a conversation
        await self._start_backend_for_conversation()