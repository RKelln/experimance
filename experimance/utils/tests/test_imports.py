#!/usr/bin/env python3
"""Test script to verify that the experimance package and experimance-common can be imported correctly."""

import sys
import os
import site

# Add the package paths manually to ensure they're found
site_packages = site.getsitepackages()[0]
sys.path.insert(0, os.path.abspath('.'))  # Add current directory
sys.path.insert(0, os.path.abspath('./libs/common'))  # Add common library

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")
print(f"Environment PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"Site packages: {site_packages}")
print("\n" + "-"*60 + "\n")

try:
    print("Attempting to import experimance...")
    import experimance
    print(f"Successfully imported experimance v{experimance.__version__}")
    
    print("\nAttempting to import experimance.libs...")
    import experimance.libs
    print("Successfully imported experimance.libs")
    
    print("\nAttempting to import experimance.services...")
    import experimance.services
    print("Successfully imported experimance.services")
    
    print("\nAttempting to import experimance.infra...")
    import experimance.infra
    print("Successfully imported experimance.infra")
    
    try:
        print("\nAttempting to import experimance_common.constants...")
        from experimance_common import constants
        print(f"Successfully imported experimance_common.constants with DEFAULT_PORTS: {constants.DEFAULT_PORTS}")
    except ImportError as e:
        print(f"Error importing experimance_common.constants: {e}")
        
    try:
        print("\nAttempting to import experimance_common.zmq_utils...")
        from experimance_common.zmq_utils import ZmqPublisher, ZmqSubscriber
        print("Successfully imported experimance_common.zmq_utils")
    except ImportError as e:
        print(f"Error importing experimance_common.zmq_utils: {e}")
        
except ImportError as e:
    print(f"Error importing experimance: {e}")

print("\nPackage verification complete!")
