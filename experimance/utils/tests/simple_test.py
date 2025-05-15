#!/usr/bin/env python3
"""Simple test script that only tests basic imports without requiring extras."""

import sys
import os
import site

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Site packages: {site.getsitepackages()}")

# Try direct import first
print("\nDirect imports:")
try:
    import experimance_core
    print(f"✓ Successfully imported experimance_core")
except ImportError as e:
    print(f"✗ Error importing experimance_core: {e}")

try:
    import experimance_display
    print(f"✓ Successfully imported experimance_display")
except ImportError as e:
    print(f"✗ Error importing experimance_display: {e}")

try:
    from experimance_common import constants
    print(f"✓ Successfully imported experimance_common.constants")
except ImportError as e:
    print(f"✗ Error importing experimance_common.constants: {e}")

# Check what's installed via pip
print("\nInstalled packages:")
import subprocess
import re

def get_installed_packages():
    try:
        output = subprocess.check_output(["uv", "pip", "list"]).decode('utf-8')
        return output
    except subprocess.CalledProcessError:
        return "Error running pip list"

pip_list = get_installed_packages()
print("Checking for experimance packages...")

packages = [
    "experimance-project",
    "experimance-common",
    "experimance-core",
    "experimance-display",
    "experimance-audio",
    "experimance-agent",
    "experimance-image-server",
    "experimance-transition"
]

for package in packages:
    if re.search(rf"{package}\s+", pip_list):
        print(f"✓ {package} is installed")
    else:
        print(f"✗ {package} is NOT installed")

print("\nImport test complete")
