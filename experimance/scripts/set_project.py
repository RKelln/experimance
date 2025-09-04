#!/usr/bin/env python3
"""
Script to set the current project for the Experimance system.

This script manages the .project file that tells services which project
configuration to use automatically.
"""

from experimance_common.project_utils import cli_main

if __name__ == "__main__":
    cli_main()
