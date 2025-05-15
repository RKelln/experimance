"""Installation verification and package path fixing script.

This script checks if the experimance package is correctly importable and
creates a .pth file in the site-packages directory if needed.
"""

import os
import site
import sys


def create_pth_file():
    """Create a .pth file in the site-packages directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    site_packages = site.getsitepackages()[0]
    
    pth_path = os.path.join(site_packages, 'experimance.pth')
    
    print(f"Creating .pth file at: {pth_path}")
    with open(pth_path, 'w') as f:
        # Both paths might be needed
        f.write(f"{current_dir}\n")
        f.write(f"{parent_dir}\n")
    
    print("Created .pth file successfully!")


if __name__ == "__main__":
    print("Checking if experimance package is importable...")
    
    try:
        import experimance
        print(f"✓ Successfully imported experimance")
    except ImportError:
        print("✗ Failed to import experimance directly")
        print("Creating .pth file to fix imports...")
        create_pth_file()
        
        print("\nTrying import again...")
        try:
            # Re-initialize site module to find the new .pth file
            importlib_reload = getattr(__import__('importlib', fromlist=['reload']), 'reload')
            importlib_reload(site)
            
            # Clear sys.path_importer_cache to force Python to re-evaluate paths
            sys.path_importer_cache.clear()
            
            # Try import again
            import experimance
            print(f"✓ Successfully imported experimance after creating .pth file")
        except ImportError as e:
            print(f"✗ Still failed to import: {e}")
            print("\nPlease try one of these solutions:")
            print("1. Restart your Python interpreter/terminal")
            print("2. Use 'import bootstrap' before importing experimance")
            print("3. Add this to your code:")
            print("   import sys; sys.path.insert(0, '/home/ryankelln/Documents/Projects/Art/experimance/installation/software')")
