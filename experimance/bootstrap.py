"""Bootstrapping file to help with imports for experimance package.

This file ensures that the experimance package is importable.
"""

import os
import sys

# Add the current directory to the path to make experimance importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"Added {current_dir} to Python path")
print(f"Current Python path: {sys.path}")

# Try to import the experimance package
try:
    import experimance
    print(f"Successfully imported experimance")
except ImportError as e:
    print(f"Failed to import experimance: {e}")
    # If that fails, try to find where experimance might be installed
    import glob
    import site
    
    site_packages = site.getsitepackages()[0]
    print(f"Checking site-packages: {site_packages}")
    experimance_files = glob.glob(f"{site_packages}/experimance*")
    print(f"Found: {experimance_files}")
    
    # If we find an egg link, read it and add that path
    for file in experimance_files:
        if file.endswith('.egg-link'):
            with open(file, 'r') as f:
                egg_path = f.readline().strip()
                print(f"Found egg link pointing to: {egg_path}")
                if egg_path and egg_path not in sys.path:
                    sys.path.insert(0, egg_path)
                    print(f"Added egg path to Python path: {egg_path}")
                    try:
                        import experimance
                        print(f"Successfully imported experimance after adding egg path")
                        break
                    except ImportError as e2:
                        print(f"Still failed to import: {e2}")
    
    # Last resort: add parent directories
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added parent directory to Python path: {parent_dir}")
    
    try:
        import experimance
        print(f"Successfully imported experimance after adding parent directory")
    except ImportError as e:
        print(f"All import attempts failed: {e}")
        sys.exit(1)
