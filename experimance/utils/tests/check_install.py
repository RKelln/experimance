#!/usr/bin/env python3
"""Script to verify package installation via pip list."""

import sys
import subprocess
import os

def run_cmd(cmd):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print("\nChecking installed packages:")
    
    # Check if experimance is installed
    pip_list = run_cmd("pip list")
    print(pip_list)
    
    if "experimance " in pip_list:
        print("\n✓ experimance package is installed.")
    else:
        print("\n✗ experimance package is NOT installed.")
        
    if "experimance-common " in pip_list:
        print("✓ experimance-common package is installed.")
    else:
        print("✗ experimance-common package is NOT installed.")
    
    print("\nInstallation verification complete.")
