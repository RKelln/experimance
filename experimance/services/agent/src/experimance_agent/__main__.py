"""CLI entry point for the Experimance Agent Service."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from experimance_common.cli import create_simple_main
from .service import ExperimanceAgentService
from .config import ExperimanceAgentServiceConfig, DEFAULT_CONFIG_PATH

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
    config = ExperimanceAgentServiceConfig.from_overrides(
        config_file=config_path,
        args=args
    )
    
    service = ExperimanceAgentService(config=config)
    
    await service.start()
    await service.run()


main = create_simple_main(
    service_name="Experimance Agent",
    description="Experimance Agent Service - AI conversation and audience interaction",
    service_runner=run_agent_service,
    default_config_path=DEFAULT_CONFIG_PATH,
    config_class=ExperimanceAgentServiceConfig
)

if __name__ == "__main__":
    main()
