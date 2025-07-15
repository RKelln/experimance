#!/usr/bin/env python3
"""
Simple launcher for the interactive detector tuning tool.
"""

import sys
from pathlib import Path

# Add agent src to path so we can import the tuning tool
agent_src = Path(__file__).parent / "services" / "agent" / "src"
sys.path.insert(0, str(agent_src))

from experimance_agent.vision.interactive_detector_tuning import main

if __name__ == "__main__":
    main()
