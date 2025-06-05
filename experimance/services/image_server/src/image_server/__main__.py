"""
Module entry point for the Experimance Image Server Service.

Allows running the service with `python -m services.image_server`.
"""

import asyncio
from .main import main

if __name__ == "__main__":
    asyncio.run(main())