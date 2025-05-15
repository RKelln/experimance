"""
Simple test script to verify that the environment is set up correctly.
"""

import sys
import importlib.util
import platform


def check_module(module_name, as_optional=False):
    """Check if a module can be imported and print status."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} is installed and working")
        return True
    except ImportError:
        if as_optional:
            print(f"⚠️  Optional module {module_name} is not installed")
        else:
            print(f"❌ Required module {module_name} is not installed")
        return False


def check_environment():
    """Run tests on the Python environment."""
    print(f"\nPython version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    
    print("\nChecking core dependencies...")
    check_module("numpy")
    check_module("zmq")
    check_module("pydantic")
    check_module("toml")
    check_module("dotenv")
    check_module("asyncio")
    check_module("aiohttp")
    check_module("uuid")
    
    print("\nChecking image processing dependencies...")
    check_module("PIL")
    check_module("cv2")
    realsense = check_module("pyrealsense2", as_optional=True)
    
    print("\nChecking display dependencies...")
    check_module("pyglet")
    check_module("OpenGL")
    sdl2 = check_module("sdl2", as_optional=True)
    check_module("ffmpegcv", as_optional=True)
    
    print("\nChecking audio dependencies...")
    check_module("pythonosc", as_optional=True)
    check_module("sounddevice", as_optional=True)
    
    # Try importing the common library
    print("\nChecking experimance-common...")
    try:
        import experimance_common
        print("✅ experimance-common is installed and working")
    except ImportError:
        print("❌ experimance-common is not installed or cannot be imported")
    
    # Report critical issues
    print("\nEnvironment check summary:")
    if not realsense:
        print("⚠️  pyrealsense2 is not installed - depth camera features will not work")
    if not sdl2:
        print("⚠️  sdl2 is not installed - some display features may not work")
    
    print("\nRun this script with --verbose for detailed import information")
    

if __name__ == "__main__":
    check_environment()
