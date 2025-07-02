"""
CLI entry point for the Experimance Agent Service.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from experimance_common.cli import create_simple_main
from .agent import AgentService
from .config import AgentServiceConfig

logger = logging.getLogger(__name__)


async def run_agent_service(
    config_path: Optional[str] = None, 
    args: Optional[argparse.Namespace] = None
):
    """
    Run the Experimance Agent Service.
    
    Args:
        config_path: Path to configuration file
        args: CLI arguments from argparse (if using new CLI system)
    """
    # Create config with CLI overrides
    config = AgentServiceConfig.from_overrides(
        config_file=config_path,
        args=args
    )
    
    service = AgentService(config=config)
    
    await service.start()
    await service.run()


def main():
    """Main entry point for the agent service."""
    
    # Create the main function using the common CLI pattern
    main_func = create_simple_main(
        service_name="Agent",
        description="Experimance Agent Service - AI conversation and audience interaction",
        service_runner=run_agent_service,
        config_class=AgentServiceConfig
    )
    
    # Run the main function
    return main_func()


if __name__ == "__main__":
    sys.exit(main())
