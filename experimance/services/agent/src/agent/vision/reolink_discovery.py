"""
Reolink camera discovery utilities for automatic camera detection on local network.

This module provides functions to discover Reolink cameras on the local subnet
by scanning common IP ranges and testing for Reolink API endpoints.
"""

import asyncio
import ipaddress
import socket
import logging
from typing import List, Optional, Dict, Any
import aiohttp
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for discovery
urllib3.disable_warnings(InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ReolinkCameraInfo:
    """Information about a discovered Reolink camera."""
    
    def __init__(self, host: str, model: str = "Unknown", name: str = "Unknown", 
                 serial: str = "Unknown", firmware: str = "Unknown"):
        self.host = host
        self.model = model
        self.name = name
        self.serial = serial
        self.firmware = firmware
    
    def __str__(self):
        return f"Reolink {self.model} '{self.name}' at {self.host} (FW: {self.firmware})"


async def discover_reolink_cameras(
    timeout: float = 1.0,  # Much faster timeout for port scanning
    max_concurrent: int = 100,  # Higher concurrency for speed
    subnet: Optional[str] = None
) -> List[ReolinkCameraInfo]:
    """
    Discover Reolink cameras on the local network using fast, credential-free detection.
    
    This method uses port scanning and HTTP signature detection - NO credentials sent.
    
    Args:
        timeout: Timeout for each IP test (seconds) - kept short for speed
        max_concurrent: Maximum concurrent connection attempts  
        subnet: Specific subnet to scan (e.g., "192.168.1.0/24"). If None, auto-detects.
        
    Returns:
        List of discovered Reolink cameras (without detailed info)
    """
    logger.info("Starting fast Reolink camera discovery (credential-free)...")
    
    # Determine subnet to scan
    if subnet is None:
        subnet = _get_local_subnet()
        if subnet is None:
            logger.warning("Could not determine local subnet for discovery")
            return []
    
    logger.info(f"Scanning subnet: {subnet}")
    
    # Generate IP addresses to test
    try:
        network = ipaddress.IPv4Network(subnet, strict=False)
        ip_addresses = [str(ip) for ip in network.hosts()]
    except Exception as e:
        logger.error(f"Invalid subnet {subnet}: {e}")
        return []
    
    # Limit to reasonable range for speed
    if len(ip_addresses) > 254:
        logger.info(f"Large subnet detected ({len(ip_addresses)} hosts), limiting to first 254")
        ip_addresses = ip_addresses[:254]
    
    # Fast concurrent scanning - NO credentials sent
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [_fast_test_reolink_camera(ip, semaphore, timeout) for ip in ip_addresses]
    
    logger.info(f"Fast-scanning {len(ip_addresses)} IP addresses for Reolink signatures...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect successful discoveries
    cameras = []
    for result in results:
        if isinstance(result, ReolinkCameraInfo):
            cameras.append(result)
            logger.info(f"Found: {result}")
    
    logger.info(f"Fast discovery complete: found {len(cameras)} Reolink camera(s)")
    return cameras


async def _fast_test_reolink_camera(
    host: str, 
    semaphore: asyncio.Semaphore, 
    timeout: float
) -> Optional[ReolinkCameraInfo]:
    """
    Fast test if an IP hosts a Reolink camera using signature detection.
    NO CREDENTIALS SENT - only checks for Reolink-specific responses.
    """
    async with semaphore:
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=client_timeout
            ) as session:
                
                # Test HTTPS first (most Reolink cameras use HTTPS)
                for protocol in ["https", "http"]:
                    
                    # Method 1: Check for Reolink-specific API endpoint signature
                    api_url = f"{protocol}://{host}/cgi-bin/api.cgi"
                    
                    try:
                        async with session.get(api_url) as response:
                            # Reolink cameras return specific responses even without auth
                            if response.status in [200, 401, 403]:
                                text = await response.text()
                                
                                # Look for Reolink signatures in response
                                reolink_signatures = [
                                    "please login first",
                                    "reolink",
                                    "cgi-bin/api.cgi",
                                    '"cmd"',
                                    '"code"',
                                    '"action"'
                                ]
                                
                                text_lower = text.lower()
                                signature_count = sum(1 for sig in reolink_signatures if sig in text_lower)
                                
                                if signature_count >= 2:  # Multiple signatures = likely Reolink
                                    logger.debug(f"Reolink signatures detected at {host} ({signature_count} matches)")
                                    
                                    return ReolinkCameraInfo(
                                        host=host,
                                        model="Unknown (detected by signature)",
                                        name=f"reolink-{host.split('.')[-1]}",  # Use last IP octet as name
                                        serial="Unknown",
                                        firmware="Unknown"
                                    )
                    
                    except Exception:
                        pass  # Try next method
                    
                    # Method 2: Check for common Reolink web interface paths
                    web_paths = ["/", "/login.html", "/index.html"]
                    
                    for path in web_paths:
                        try:
                            web_url = f"{protocol}://{host}{path}"
                            async with session.get(web_url) as response:
                                if response.status == 200:
                                    text = await response.text()
                                    
                                    # Look for Reolink branding/signatures in web interface
                                    web_signatures = [
                                        "reolink",
                                        "RLC-",
                                        "argus",
                                        "go2rtc",
                                        "nvr"
                                    ]
                                    
                                    text_lower = text.lower()
                                    if any(sig in text_lower for sig in web_signatures):
                                        logger.debug(f"Reolink web interface detected at {host}")
                                        
                                        return ReolinkCameraInfo(
                                            host=host,
                                            model="Unknown (detected by web interface)",
                                            name=f"reolink-{host.split('.')[-1]}",
                                            serial="Unknown",
                                            firmware="Unknown"
                                        )
                        
                        except Exception:
                            continue
                        
        except Exception:
            pass  # Skip this IP entirely
    
    return None


async def discover_reolink_cameras_fast_scan(
    subnet: Optional[str] = None,
    port_timeout: float = 0.5  # Very fast port scanning
) -> List[ReolinkCameraInfo]:
    """
    Ultra-fast Reolink discovery using port scanning only.
    
    This method only checks if HTTPS port 443 is open - MUCH faster than HTTP requests.
    Use this for quick scans when you just need to find camera IPs.
    """
    logger.info("Starting ultra-fast port-based Reolink discovery...")
    
    # Determine subnet
    if subnet is None:
        subnet = _get_local_subnet()
        if subnet is None:
            return []
    
    try:
        network = ipaddress.IPv4Network(subnet, strict=False)
        ip_list = list(network.hosts())
        if len(ip_list) > 254:
            ip_list = ip_list[:254]
        ip_addresses = [str(ip) for ip in ip_list]
    except Exception as e:
        logger.error(f"Invalid subnet {subnet}: {e}")
        return []
    
    logger.info(f"Port-scanning {len(ip_addresses)} IPs for HTTPS (port 443)...")
    
    # Scan for HTTPS port only (most Reolink cameras use HTTPS)
    semaphore = asyncio.Semaphore(100)
    tasks = [_check_port(ip, 443, semaphore, port_timeout) for ip in ip_addresses]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Find IPs with HTTPS open
    cameras = []
    for result in results:
        if isinstance(result, tuple) and len(result) == 3:
            ip, port, is_open = result
            if is_open:
                cameras.append(ReolinkCameraInfo(
                    host=ip,
                    model="Unknown (port scan detection)",
                    name=f"camera-{ip.split('.')[-1]}",
                    serial="Unknown",
                    firmware="Unknown"
                ))
                logger.info(f"Found device with HTTPS at {ip}")
    
    logger.info(f"Port scan complete: found {len(cameras)} device(s) with HTTPS")
    return cameras


async def discover_reolink_cameras_comprehensive(
    known_ip: Optional[str] = None,
    timeout: float = 5.0,
    port_timeout: float = 0.3,
    subnet: Optional[str] = None
) -> List[ReolinkCameraInfo]:
    """
    Comprehensive Reolink camera discovery with intelligent fallback strategy.
    
    Strategy:
    1. If known_ip provided, test that IP first (fastest)
    2. If not found or no IP provided, do fast port scan for HTTPS devices
    3. Then do signature-based verification on discovered IPs to confirm Reolink cameras
    
    Args:
        known_ip: Optional IP to test first
        timeout: Timeout for HTTP requests in signature verification
        port_timeout: Timeout for port scanning
        subnet: Network subnet to scan (auto-detected if None)
        
    Returns:
        List of confirmed Reolink cameras
    """
    logger.info("Starting comprehensive Reolink camera discovery...")
    
    cameras = []
    
    # Step 1: Test known IP if provided
    if known_ip:
        logger.info(f"🎯 Testing known IP: {known_ip}")
        
        # Quick port check first
        semaphore = asyncio.Semaphore(1)
        port_result = await _check_port(known_ip, 443, semaphore, port_timeout)
        
        if port_result[2]:  # HTTPS port is open
            logger.info(f"✅ HTTPS port open on {known_ip}, checking if it's a Reolink camera...")
            
            # Check if it's actually a Reolink camera
            signature_cameras = await discover_reolink_cameras(
                timeout=timeout, 
                subnet=f"{known_ip}/32"  # Just this one IP
            )
            
            if signature_cameras:
                logger.info(f"🎥 Confirmed: {known_ip} is a Reolink camera")
                return signature_cameras
            else:
                logger.info(f"❌ {known_ip} has HTTPS but is not a Reolink camera")
        else:
            logger.info(f"❌ {known_ip} does not have HTTPS port open")
    
    # Step 2: Fast network scan for HTTPS devices
    logger.info("🚀 Known IP not found or not provided, scanning network for HTTPS devices...")
    
    fast_scan_results = await discover_reolink_cameras_fast_scan(
        subnet=subnet,
        port_timeout=port_timeout
    )
    
    if not fast_scan_results:
        logger.info("❌ No HTTPS devices found on network")
        return []
    
    logger.info(f"📡 Found {len(fast_scan_results)} HTTPS device(s), verifying which are Reolink cameras...")
    
    # Step 3: Signature-based verification on discovered IPs
    candidate_ips = [camera.host for camera in fast_scan_results]
    
    # Create a custom subnet string for just these IPs
    # We'll check each IP individually for Reolink signatures
    semaphore = asyncio.Semaphore(10)  # Limit concurrent signature checks
    
    async def verify_reolink_signature(ip: str) -> List[ReolinkCameraInfo]:
        """Check if a single IP is a Reolink camera"""
        try:
            # Use the signature-based discovery on just this IP
            single_ip_subnet = f"{ip}/32"
            result = await discover_reolink_cameras(timeout=timeout, subnet=single_ip_subnet)
            return result
        except Exception as e:
            logger.debug(f"Signature check failed for {ip}: {e}")
            return []
    
    # Check all candidate IPs for Reolink signatures
    verification_tasks = [verify_reolink_signature(ip) for ip in candidate_ips]
    verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
    
    # Collect confirmed Reolink cameras
    confirmed_cameras = []
    for result in verification_results:
        if isinstance(result, list):
            confirmed_cameras.extend(result)
        elif isinstance(result, Exception):
            logger.debug(f"Verification task failed: {result}")
    
    if confirmed_cameras:
        logger.info(f"🎥 Discovery complete: found {len(confirmed_cameras)} confirmed Reolink camera(s)")
        for camera in confirmed_cameras:
            logger.info(f"  • {camera.host} - {camera.model} ({camera.name})")
    else:
        logger.info("❌ No Reolink cameras found after signature verification")
    
    return confirmed_cameras


async def _check_port(
    host: str, 
    port: int, 
    semaphore: asyncio.Semaphore, 
    timeout: float
) -> tuple[str, int, bool]:
    """Check if a specific port is open on a host."""
    async with semaphore:
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            writer.close()
            await writer.wait_closed()
            return (host, port, True)
        except Exception:
            return (host, port, False)


async def _test_reolink_camera(
    host: str, 
    semaphore: asyncio.Semaphore, 
    timeout: float
) -> Optional[ReolinkCameraInfo]:
    """Test if a specific IP address hosts a Reolink camera."""
    async with semaphore:
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=client_timeout
            ) as session:
                
                # Only test HTTPS (HTTP returns HTML login pages)
                # Use a dummy login to test if this is a Reolink camera
                url = f"https://{host}/cgi-bin/api.cgi?cmd=Login&token=null"
                payload = [{
                    "cmd": "Login",
                    "action": 0,
                    "param": {
                        "User": {"userName": "test", "password": "test"}  # Dummy credentials
                    }
                }]
                
                try:
                    async with session.post(
                        url,
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status == 200:
                            text = await response.text()
                            
                            # Parse response (Reolink returns JSON as text/html)
                            try:
                                import json
                                data = json.loads(text)
                                
                                if data and len(data) > 0:
                                    # Check if this looks like a Reolink response
                                    # Even with wrong credentials, Reolink will return a proper JSON error
                                    cmd = data[0].get("cmd")
                                    if cmd == "Login":
                                        # This is likely a Reolink camera!
                                        # Return basic info (can't get device details without valid credentials)
                                        return ReolinkCameraInfo(
                                            host=host,
                                            model="Unknown (authentication required)",
                                            name="Unknown (authentication required)", 
                                            serial="Unknown (authentication required)",
                                            firmware="Unknown (authentication required)"
                                        )
                            except Exception:
                                pass  # Not valid JSON or not a Reolink response
                
                except Exception:
                    pass  # Connection failed, skip this IP
                        
        except Exception:
            pass  # Skip this IP
    
    return None


def _get_local_subnet() -> Optional[str]:
    """
    Attempt to detect the local subnet by examining network interfaces.
    
    Returns:
        Subnet string like "192.168.1.0/24" or None if detection fails
    """
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        
        # Assume /24 subnet (most common for home networks)
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
        return str(network)
        
    except Exception as e:
        logger.debug(f"Could not determine local subnet: {e}")
        return None


async def find_reolink_camera_by_model(model_pattern: str) -> Optional[ReolinkCameraInfo]:
    """
    Find a specific Reolink camera by model pattern.
    
    Args:
        model_pattern: Model pattern to search for (e.g., "RLC-820A")
        
    Returns:
        First matching camera or None if not found
    """
    cameras = await discover_reolink_cameras()
    
    for camera in cameras:
        if model_pattern.lower() in camera.model.lower():
            return camera
    
    return None


async def test_camera_credentials(
    host: str, 
    user: str, 
    password: str,
    timeout: float = 5.0
) -> bool:
    """
    Test if given credentials work for a Reolink camera.
    
    Args:
        host: Camera IP address
        user: Username
        password: Password
        timeout: Connection timeout
        
    Returns:
        True if credentials are valid, False otherwise
    """
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=client_timeout
        ) as session:
            
            # Try HTTPS first
            for protocol in ["https", "http"]:
                url = f"{protocol}://{host}/cgi-bin/api.cgi?cmd=Login&token=null"
                
                payload = [{
                    "cmd": "Login",
                    "action": 0,
                    "param": {
                        "User": {"userName": user, "password": password}
                    }
                }]
                
                try:
                    async with session.post(
                        url,
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status == 200:
                            text = await response.text()
                            
                            try:
                                import json
                                data = json.loads(text)
                                
                                if (data and len(data) > 0 and 
                                    data[0].get("code") == 0 and
                                    data[0].get("value", {}).get("Token", {}).get("name")):
                                    return True
                                    
                            except Exception:
                                pass
                                
                except Exception:
                    continue  # Try next protocol
    
    except Exception:
        pass
    
    return False


# CLI interface for testing
if __name__ == "__main__":
    async def main():
        import sys
        
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "test-creds":
                # Test credentials: python reolink_discovery.py test-creds <host> <user> <password>
                if len(sys.argv) != 5:
                    print("Usage: python reolink_discovery.py test-creds <host> <user> <password>")
                    return
                
                host, user, password = sys.argv[2:5]
                result = await test_camera_credentials(host, user, password)
                print(f"Credentials {'valid' if result else 'invalid'} for {host}")
            
            elif command == "fast":
                # Fast port-based discovery
                print("🚀 Ultra-fast discovery (port scanning only)...")
                cameras = await discover_reolink_cameras_fast_scan()
                
                if cameras:
                    print(f"\n📡 Found {len(cameras)} device(s) with HTTPS:")
                    for camera in cameras:
                        print(f"  • {camera}")
                    print("\n💡 Note: These are just devices with HTTPS - test credentials to confirm they're Reolink cameras")
                else:
                    print("❌ No devices with HTTPS found")
            
            elif command == "signature":
                # Signature-based discovery (slower but no credentials)
                print("🔍 Signature-based discovery (credential-free)...")
                cameras = await discover_reolink_cameras()
                
                if cameras:
                    print(f"\n🎥 Found {len(cameras)} Reolink camera(s):")
                    for camera in cameras:
                        print(f"  • {camera}")
                else:
                    print("❌ No Reolink cameras found")
            
            elif command == "comprehensive":
                # Comprehensive discovery with optional IP
                known_ip = sys.argv[2] if len(sys.argv) > 2 else None
                
                if known_ip:
                    print(f"🎯 Comprehensive discovery starting with known IP: {known_ip}")
                else:
                    print("🔍 Comprehensive discovery (no known IP provided)")
                
                cameras = await discover_reolink_cameras_comprehensive(known_ip=known_ip)
                
                if cameras:
                    print(f"\n🎥 Found {len(cameras)} confirmed Reolink camera(s):")
                    for camera in cameras:
                        print(f"  • {camera}")
                else:
                    print("❌ No Reolink cameras found")
            
            else:
                print("Usage:")
                print("  python reolink_discovery.py fast                    # Ultra-fast port scan")
                print("  python reolink_discovery.py signature               # Signature detection")
                print("  python reolink_discovery.py comprehensive [ip]      # Smart progressive discovery")
                print("  python reolink_discovery.py test-creds <host> <user> <password>")
        
        else:
            # Default: comprehensive discovery
            print("🎯 Comprehensive Discovery (smart progressive detection)...")
            
            cameras = await discover_reolink_cameras_comprehensive()
            
            if cameras:
                print(f"\n🎥 Found {len(cameras)} confirmed Reolink camera(s):")
                for camera in cameras:
                    print(f"  • {camera}")
                
                print(f"\n💡 Use 'comprehensive <ip>' to test a specific IP first")
            else:
                print("❌ No Reolink cameras found")
                print("\n💡 Try individual methods:")
                print("  python reolink_discovery.py fast        # See all HTTPS devices")
                print("  python reolink_discovery.py signature   # Detailed signature check")
    
    # Set up logging for CLI usage
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    asyncio.run(main())
