#!/bin/bash

# LaunchAgent wrapper for fire_agent
# This script bypasses macOS TCC restrictions by running from a shell script

cd /Users/fireproject/Documents/experimance/experimance

# Set project environment  
export PROJECT_ENV=fire

exec /Users/fireproject/.local/bin/uv run -m experimance_agent
