#!/usr/bin/env python3
"""
List all Reolink cameras on the local network.

This script uses the comprehensive discovery system to find and list all
Reolink cameras with detailed information where available.

Usage:
    python scripts/list_cameras.py                    # Comprehensive discovery
    python scripts/list_cameras.py --fast            # Fast port scan only
    python scripts/list_cameras.py --signature       # Signature-based detection
    python scripts/list_cameras.py --known-ip <ip>   # Test specific IP first
"""

import asyncio
import argparse
import logging
import sys
import os

# Add the services/agent/src directory to Python path so we can import the discovery module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'agent', 'src'))

from agent.vision.reolink_discovery import (
    discover_reolink_cameras_comprehensive,
    discover_reolink_cameras_fast_scan, 
    discover_reolink_cameras,
    test_camera_credentials
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def list_cameras_comprehensive(known_ip=None):
    """Use comprehensive discovery to find cameras."""
    print("🎯 Comprehensive Reolink Camera Discovery")
    print("=" * 50)
    
    if known_ip:
        print(f"Testing known IP: {known_ip}")
    else:
        print("Scanning network for Reolink cameras...")
    
    cameras = await discover_reolink_cameras_comprehensive(known_ip=known_ip)
    
    if cameras:
        print(f"\n🎥 Found {len(cameras)} confirmed Reolink camera(s):")
        for i, camera in enumerate(cameras, 1):
            print(f"  {i}. {camera}")
    else:
        print("\n❌ No Reolink cameras found")
        print("\n💡 Troubleshooting tips:")
        print("   - Ensure cameras are powered on and connected to network")
        print("   - Check if cameras are on same subnet as this computer")
        print("   - Try --fast to see all HTTPS devices on network")
    
    return cameras


async def list_cameras_fast():
    """Use fast port scanning to find potential cameras."""
    print("🚀 Fast Network Scan (HTTPS Port Detection)")
    print("=" * 50)
    print("Scanning for devices with HTTPS (port 443)...")
    
    devices = await discover_reolink_cameras_fast_scan()
    
    if devices:
        print(f"\n📡 Found {len(devices)} device(s) with HTTPS:")
        for i, device in enumerate(devices, 1):
            print(f"  {i}. {device.host} - {device.model}")
        
        print(f"\n💡 Note: These are just devices with HTTPS ports open.")
        print(f"   Use default mode (comprehensive) to verify which are Reolink cameras.")
    else:
        print("\n❌ No devices with HTTPS found on network")
    
    return devices


async def list_cameras_signature():
    """Use signature-based detection to find cameras."""
    print("🔍 Signature-Based Discovery (Credential-Free)")
    print("=" * 50)
    print("Testing devices for Reolink API signatures...")
    
    cameras = await discover_reolink_cameras()
    
    if cameras:
        print(f"\n🎥 Found {len(cameras)} Reolink camera(s):")
        for i, camera in enumerate(cameras, 1):
            print(f"  {i}. {camera}")
    else:
        print("\n❌ No Reolink cameras detected via signature analysis")
    
    return cameras


async def test_credentials(cameras, user=None, password=None):
    """Test credentials on discovered cameras."""
    if not cameras or not user or not password:
        return
    
    print(f"\n🔐 Testing Credentials")
    print("=" * 30)
    
    for camera in cameras:
        print(f"Testing {camera.host}...", end=" ")
        try:
            valid = await test_camera_credentials(camera.host, user, password)
            if valid:
                print("✅ Valid")
            else:
                print("❌ Invalid")
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Discover Reolink cameras on local network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/list_cameras.py                           # Comprehensive discovery
  python scripts/list_cameras.py --fast                   # Fast port scan
  python scripts/list_cameras.py --signature              # Signature detection
  python scripts/list_cameras.py --known-ip 192.168.1.100 # Test specific IP first
  python scripts/list_cameras.py --test-creds admin pass123 # Test credentials
        """
    )
    
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Fast port scanning (HTTPS detection only)"
    )
    
    parser.add_argument(
        "--signature", 
        action="store_true", 
        help="Signature-based detection (slower but precise)"
    )
    
    parser.add_argument(
        "--known-ip", 
        help="Known IP to test first in comprehensive discovery"
    )
    
    parser.add_argument(
        "--test-creds", 
        nargs=2, 
        metavar=("USER", "PASSWORD"),
        help="Test credentials on discovered cameras"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('agent.vision.reolink_discovery').setLevel(logging.DEBUG)
    
    cameras = []
    
    try:
        if args.fast:
            cameras = await list_cameras_fast()
        elif args.signature:
            cameras = await list_cameras_signature()
        else:
            # Default: comprehensive discovery
            cameras = await list_cameras_comprehensive(args.known_ip)
        
        # Test credentials if provided
        if args.test_creds and cameras:
            user, password = args.test_creds
            await test_credentials(cameras, user, password)
        
        print(f"\n📊 Discovery Summary")
        print("=" * 20)
        print(f"Cameras found: {len(cameras)}")
        
        if cameras:
            print("Next steps:")
            if not args.test_creds:
                print("  • Test credentials: --test-creds <user> <password>")
            print("  • Configure in projects/fire/.env:")
            for camera in cameras:
                print(f"    REOLINK_HOST={camera.host}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Discovery interrupted by user")
    except Exception as e:
        print(f"\n❌ Discovery failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
