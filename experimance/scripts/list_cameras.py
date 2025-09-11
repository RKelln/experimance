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
    discover_reolink_cameras_arp_scan,
    discover_reolink_cameras_nmap,
    discover_reolink_cameras_mdns,
    test_camera_credentials
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def list_cameras_comprehensive(known_ip=None, subnet=None):
    """Use comprehensive discovery to find cameras."""
    print("🎯 Comprehensive Reolink Camera Discovery")
    print("=" * 50)
    
    if known_ip:
        print(f"Testing known IP: {known_ip}")
    else:
        print("Scanning network for Reolink cameras...")
    
    cameras = await discover_reolink_cameras_comprehensive(
        known_ip=known_ip,
        subnet=subnet
    )
    
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


async def list_cameras_arp():
    """Use ARP table discovery to find cameras."""
    print("📡 ARP Table Discovery (Fastest Method)")
    print("=" * 50)
    print("Scanning ARP table for active devices and testing for Reolink cameras...")
    
    cameras = await discover_reolink_cameras_arp_scan()
    
    if cameras:
        print(f"\n🎥 Found {len(cameras)} Reolink camera(s) via ARP discovery:")
        for i, camera in enumerate(cameras, 1):
            print(f"  {i}. {camera}")
    else:
        print("\n❌ No Reolink cameras found in ARP table")
        print("\n💡 Troubleshooting tips:")
        print("   - Ensure cameras have been accessed recently (to appear in ARP table)")
        print("   - Try pinging the camera first: ping <camera-ip>")
        print("   - Use --nmap or --comprehensive for broader discovery")
    
    return cameras


async def list_cameras_nmap():
    """Use nmap discovery to find cameras."""
    print("🗺️ Nmap-Based Discovery")
    print("=" * 50)
    print("Using nmap to find devices with HTTPS ports and testing for Reolink cameras...")
    
    cameras = await discover_reolink_cameras_nmap()
    
    if cameras:
        print(f"\n🎥 Found {len(cameras)} Reolink camera(s) via nmap:")
        for i, camera in enumerate(cameras, 1):
            print(f"  {i}. {camera}")
    else:
        print("\n❌ No Reolink cameras found via nmap")
        print("\n💡 Troubleshooting tips:")
        print("   - Ensure nmap is installed: brew install nmap (macOS) or apt install nmap (Linux)")
        print("   - Camera might be on different port or subnet")
        print("   - Try --arp or --comprehensive for other discovery methods")
    
    return cameras


async def list_cameras_mdns():
    """Use mDNS/Bonjour discovery to find cameras."""
    print("🔍 mDNS/Bonjour Discovery")
    print("=" * 50)
    print("Scanning for devices advertising camera services via mDNS...")
    
    cameras = await discover_reolink_cameras_mdns()
    
    if cameras:
        print(f"\n🎥 Found {len(cameras)} device(s) via mDNS:")
        for i, camera in enumerate(cameras, 1):
            print(f"  {i}. {camera}")
        print("\n💡 Note: mDNS results are verified by testing Reolink API endpoints")
    else:
        print("\n❌ No cameras found via mDNS discovery")
        print("\n💡 Troubleshooting tips:")
        print("   - Install zeroconf: pip install zeroconf")
        print("   - Some cameras don't advertise via mDNS")
        print("   - Try --arp or --nmap for more reliable discovery")
    
    return cameras


async def list_cameras_curl_scan(subnet=None):
    """Use curl-based network scan to find cameras."""
    print("🔧 Curl-Based Network Scan (Fallback Method)")
    print("=" * 50)
    print("Using curl to scan for Reolink cameras (bypasses Python networking issues)...")
    
    # Import the curl scan function
    from agent.vision.reolink_discovery import _discover_cameras_with_curl, _get_local_subnet
    
    # Determine subnet
    if subnet is None:
        subnet = _get_local_subnet()
        if subnet is None:
            print("❌ Could not determine local subnet")
            return []
    
    print(f"Scanning subnet: {subnet}")
    
    cameras = await _discover_cameras_with_curl(subnet)
    
    if cameras:
        print(f"\n🎥 Found {len(cameras)} Reolink camera(s) via curl scan:")
        for i, camera in enumerate(cameras, 1):
            print(f"  {i}. {camera}")
    else:
        print("\n❌ No Reolink cameras found via curl scan")
        print("\n💡 Troubleshooting tips:")
        print("   - Ensure cameras are powered on and connected to network")
        print("   - Check if cameras are on same subnet as this computer")
        print("   - Verify camera web interface is accessible manually")
    
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
  python scripts/list_cameras.py                           # Comprehensive discovery (ARP → mDNS → nmap → port scan)
  python scripts/list_cameras.py --fast                   # Fast port scan
  python scripts/list_cameras.py --signature              # Signature detection
  python scripts/list_cameras.py --arp                    # ARP table discovery (fastest)
  python scripts/list_cameras.py --nmap                   # nmap-based discovery (if available)
  python scripts/list_cameras.py --mdns                   # mDNS/Bonjour discovery
  python scripts/list_cameras.py --known-ip 192.168.1.100 # Test specific IP first
  python scripts/list_cameras.py --test-creds admin pass123 # Test credentials
  python scripts/list_cameras.py --force-curl-scan        # Force full curl-based scan
  python scripts/list_cameras.py --subnet 192.168.1.0/24  # Scan specific subnet
        """
    )
    
    parser.add_argument(
        "--arp", 
        action="store_true", 
        help="ARP table discovery (fastest, most efficient)"
    )
    
    parser.add_argument(
        "--nmap", 
        action="store_true", 
        help="nmap-based discovery (requires nmap to be installed)"
    )
    
    parser.add_argument(
        "--mdns", 
        action="store_true", 
        help="mDNS/Bonjour discovery (requires zeroconf library)"
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
        "--force-curl-scan", 
        action="store_true", 
        help="Skip port scanning and go directly to curl-based network scan"
    )
    
    parser.add_argument(
        "--subnet", 
        help="Specific subnet to scan (e.g., 192.168.1.0/24)"
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
    
    cameras = []
    
    try:
        if args.force_curl_scan:
            # Force curl-based network scan
            cameras = await list_cameras_curl_scan(args.subnet)
        elif args.arp:
            cameras = await list_cameras_arp()
        elif args.nmap:
            cameras = await list_cameras_nmap()
        elif args.mdns:
            cameras = await list_cameras_mdns()
        elif args.fast:
            cameras = await list_cameras_fast()
        elif args.signature:
            cameras = await list_cameras_signature()
        else:
            # Default: comprehensive discovery
            cameras = await list_cameras_comprehensive(args.known_ip, args.subnet)
        
        # Ensure cameras is always a list
        if cameras is None:
            cameras = []
        
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
