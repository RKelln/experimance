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
    port_timeout: float = 2.0  # Increased from 0.5 to 2.0 for better reliability
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
    port_timeout: float = 2.0,  # Increased from 0.3 to 2.0 seconds
    subnet: Optional[str] = None
) -> List[ReolinkCameraInfo]:
    """
    Comprehensive Reolink camera discovery with intelligent fallback strategy.
    
    Strategy:
    1. If known_ip provided, test that IP first (fastest)
    2. If not found or no IP provided, do fast port scan for HTTPS devices
    3. Then do signature-based verification on discovered IPs to confirm Reolink cameras
    4. As a fallback, if Python networking fails, use curl to test connectivity
    
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
        
        # Skip port check for known IPs - some cameras don't respond well to raw socket connections
        # Go directly to HTTP/HTTPS signature testing
        logger.info(f"✅ Skipping port check, testing HTTP/HTTPS directly on {known_ip}")
        
        # First, try Python networking
        # Check if it's actually a Reolink camera using signature detection
        signature_cameras = await discover_reolink_cameras(
            timeout=timeout, 
            subnet=f"{known_ip}/32"  # Just this one IP
        )
        
        if signature_cameras:
            logger.info(f"🎥 Confirmed: {known_ip} is a Reolink camera")
            return signature_cameras
        else:
            logger.info(f"❌ Python networking failed for {known_ip}, trying curl fallback...")
            
            # Fallback: Use curl to test if the camera responds
            curl_result = await _test_camera_with_curl(known_ip)
            if curl_result:
                logger.info(f"🎥 Confirmed via curl: {known_ip} is a Reolink camera")
                return [ReolinkCameraInfo(
                    host=known_ip,
                    model="Unknown (detected via curl fallback)",
                    name=f"reolink-{known_ip.split('.')[-1]}",
                    serial="Unknown",
                    firmware="Unknown"
                )]
            else:
                logger.info(f"❌ {known_ip} is not a Reolink camera or not responding to any requests")
    
    # Step 2: Fast network scan for HTTPS devices
    logger.info("🚀 Known IP not found or not provided, starting intelligent network discovery...")
    
    # Step 2: Intelligent network discovery using multiple methods
    logger.info("Starting intelligent network discovery...")
    
    # Try discovery methods in order of efficiency/reliability
    discovery_methods = [
        ("ARP table", discover_reolink_cameras_arp_scan),
        ("mDNS/Bonjour", discover_reolink_cameras_mdns),
        ("nmap", discover_reolink_cameras_nmap),
    ]
    
    for method_name, method_func in discovery_methods:
        logger.info(f"Trying {method_name} discovery...")
        
        try:
            cameras = await method_func()
            
            if cameras:
                logger.info(f"🎥 Found {len(cameras)} camera(s) via {method_name}")
                
                # For mDNS results, verify they are actually Reolink cameras
                if method_name == "mDNS/Bonjour":
                    verified_cameras = []
                    for camera in cameras:
                        if await _test_camera_with_curl(camera.host):
                            verified_cameras.append(camera)
                    
                    if verified_cameras:
                        return verified_cameras
                    else:
                        logger.info(f"mDNS cameras failed Reolink verification, trying next method...")
                        continue
                else:
                    return cameras
            else:
                logger.info(f"{method_name} discovery found no cameras, trying next method...")
                
        except Exception as e:
            logger.debug(f"{method_name} discovery failed: {e}")
            continue
    
    # If all efficient methods fail, fall back to port scanning
    logger.info("🔌 All efficient methods failed, falling back to port scanning...")
    
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
        """Check if a single IP is a Reolink camera with Python + curl fallback"""
        try:
            # First try Python networking
            single_ip_subnet = f"{ip}/32"
            result = await discover_reolink_cameras(timeout=timeout, subnet=single_ip_subnet)
            if result:
                return result
            
            # If Python networking fails, try curl fallback
            logger.debug(f"Python networking failed for {ip}, trying curl fallback...")
            curl_result = await _test_camera_with_curl(ip)
            if curl_result:
                return [ReolinkCameraInfo(
                    host=ip,
                    model="Unknown (detected via curl fallback)",
                    name=f"reolink-{ip.split('.')[-1]}",
                    serial="Unknown",
                    firmware="Unknown"
                )]
            
            return []
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
        # Step 4: If no cameras found via port scanning, try curl-based full network scan
        logger.info("❌ No cameras found via port scanning, trying comprehensive curl-based scan...")
        
        # Determine subnet for full scan
        scan_subnet = subnet
        if scan_subnet is None:
            scan_subnet = _get_local_subnet()
            if scan_subnet is None:
                logger.warning("Could not determine subnet for comprehensive scan")
                return []
        
        curl_cameras = await _discover_cameras_with_curl(scan_subnet, timeout)
        if curl_cameras:
            logger.info(f"🎥 Found {len(curl_cameras)} camera(s) via comprehensive curl scan")
            confirmed_cameras.extend(curl_cameras)
        else:
            logger.info("❌ No Reolink cameras found after comprehensive scan")
    
    return confirmed_cameras


async def find_first_reolink_camera(
    known_ip: Optional[str] = None,
    timeout: float = 5.0
) -> Optional[ReolinkCameraInfo]:
    """
    Find the first available Reolink camera, optimized for speed.
    
    This function is designed for applications that just need to connect to ANY camera,
    not enumerate all cameras. It stops as soon as it finds one working camera.
    
    Strategy:
    1. If known_ip provided, test that first
    2. Try ARP table (fast, only tests active devices)
    3. Stop immediately when first camera is found
    
    Args:
        known_ip: Optional IP to test first
        timeout: Timeout for each test
        
    Returns:
        First camera found, or None if no cameras found
    """
    logger.info("🎯 Finding first available Reolink camera...")
    
    # Step 1: Test known IP if provided
    if known_ip:
        logger.info(f"🧪 Testing known IP: {known_ip}")
        try:
            # Try fast Python networking first
            result = await _fast_test_reolink_camera(known_ip, asyncio.Semaphore(1), timeout)
            if result:
                logger.info(f"✅ Found camera at known IP: {result}")
                return result
        except Exception:
            pass
        
        # Fallback to curl
        if await _test_camera_with_curl(known_ip):
            logger.info(f"✅ Found camera at known IP via curl: {known_ip}")
            return ReolinkCameraInfo(
                host=known_ip,
                model="Unknown (detected via curl)",
                name=f"reolink-{known_ip.split('.')[-1]}",
                serial="Unknown",
                firmware="Unknown"
            )
    
    # Step 2: Try ARP table scan (fast, only active devices)
    logger.info("🔍 Scanning ARP table for first available camera...")
    
    active_ips = await _get_active_ips_from_arp()
    if not active_ips:
        logger.info("❌ No active devices found in ARP table")
        return None
    
    logger.info(f"📡 Testing {len(active_ips)} active devices from ARP table...")
    
    # Test each IP until we find a camera
    for ip_info in active_ips:
        ip = ip_info['ip']
        mac = ip_info.get('mac', 'Unknown')
        
        try:
            # Try Python networking first
            result = await _fast_test_reolink_camera(ip, asyncio.Semaphore(1), timeout)
            if result:
                result.serial = f"MAC: {mac}" if mac != 'Unknown' else result.serial
                logger.info(f"✅ Found first camera via ARP scan: {result}")
                return result
        except Exception:
            pass
        
        # Fallback to curl
        if await _test_camera_with_curl(ip):
            camera = ReolinkCameraInfo(
                host=ip,
                model="Unknown (detected via ARP + curl)",
                name=f"reolink-{ip.split('.')[-1]}",
                serial=f"MAC: {mac}" if mac != 'Unknown' else "Unknown",
                firmware="Unknown"
            )
            logger.info(f"✅ Found first camera via ARP + curl: {camera}")
            return camera
    
    logger.info("❌ No Reolink cameras found")
    return None


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


async def discover_reolink_cameras_arp_scan() -> List[ReolinkCameraInfo]:
    """
    Discover Reolink cameras using ARP table analysis.
    Much faster than full network scanning - only tests devices that are actually on the network.
    """
    logger.info("Starting ARP-based Reolink camera discovery...")
    
    # Get active devices from ARP table
    active_ips = await _get_active_ips_from_arp()
    
    if not active_ips:
        logger.info("No active devices found in ARP table")
        return []
    
    logger.info(f"Found {len(active_ips)} active devices in ARP table, testing for Reolink cameras...")
    
    # Test each active IP for Reolink signatures
    semaphore = asyncio.Semaphore(20)  # Higher concurrency since we have fewer IPs
    
    async def test_active_ip(ip_info: dict) -> Optional[ReolinkCameraInfo]:
        """Test an active IP for Reolink camera."""
        ip = ip_info['ip']
        mac = ip_info.get('mac', 'Unknown')
        
        async with semaphore:
            # Try Python networking first
            try:
                result = await _fast_test_reolink_camera(ip, semaphore, 3.0)
                if result:
                    # Update with MAC address if we have it
                    result.serial = f"MAC: {mac}" if mac != 'Unknown' else result.serial
                    return result
            except Exception:
                pass
            
            # Fallback to curl if Python networking fails
            if await _test_camera_with_curl(ip):
                return ReolinkCameraInfo(
                    host=ip,
                    model="Unknown (detected via ARP + curl)",
                    name=f"reolink-{ip.split('.')[-1]}",
                    serial=f"MAC: {mac}" if mac != 'Unknown' else "Unknown",
                    firmware="Unknown"
                )
            
            return None
    
    # Test all active IPs
    tasks = [test_active_ip(ip_info) for ip_info in active_ips]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect successful discoveries
    cameras = []
    for result in results:
        if isinstance(result, ReolinkCameraInfo):
            cameras.append(result)
            logger.info(f"Found via ARP scan: {result}")
    
    logger.info(f"ARP-based discovery complete: found {len(cameras)} Reolink camera(s)")
    return cameras


async def _get_active_ips_from_arp() -> List[dict]:
    """
    Get list of active IP addresses from ARP table (works on Linux, macOS, Windows).
    Returns list of dicts with 'ip' and 'mac' keys.
    """
    import platform
    import subprocess
    
    try:
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            cmd = ["arp", "-a"]
        elif system == "linux":
            cmd = ["arp", "-a"]
        elif system == "windows":
            cmd = ["arp", "-a"]
        else:
            logger.warning(f"Unsupported platform for ARP scanning: {system}")
            return []
        
        # Run ARP command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.warning(f"ARP command failed: {stderr.decode()}")
            return []
        
        # Parse ARP output
        arp_output = stdout.decode('utf-8')
        active_devices = []
        
        for line in arp_output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Parse different ARP output formats
            ip = None
            mac = None
            
            if system == "darwin":  # macOS format: "? (192.168.1.1) at aa:bb:cc:dd:ee:ff on en0 ifscope [ethernet]"
                import re
                match = re.search(r'\(([0-9.]+)\) at ([a-fA-F0-9:]{17})', line)
                if match:
                    ip = match.group(1)
                    mac = match.group(2)
            
            elif system == "linux":  # Linux format: "? (192.168.1.1) at aa:bb:cc:dd:ee:ff [ether] on eth0"
                import re
                match = re.search(r'\(([0-9.]+)\) at ([a-fA-F0-9:]{17})', line)
                if match:
                    ip = match.group(1)
                    mac = match.group(2)
            
            elif system == "windows":  # Windows format: "  192.168.1.1           aa-bb-cc-dd-ee-ff     dynamic"
                parts = line.split()
                if len(parts) >= 2 and '.' in parts[0] and '-' in parts[1]:
                    ip = parts[0]
                    mac = parts[1].replace('-', ':')
            
            if ip and mac:
                # Filter out invalid/special addresses
                if not ip.startswith('169.254.') and ip != '127.0.0.1' and not ip.startswith('224.'):
                    active_devices.append({'ip': ip, 'mac': mac})
        
        logger.debug(f"Found {len(active_devices)} active devices in ARP table")
        return active_devices
        
    except Exception as e:
        logger.warning(f"Failed to get ARP table: {e}")
        return []


async def discover_reolink_cameras_nmap() -> List[ReolinkCameraInfo]:
    """
    Discover Reolink cameras using nmap (if available).
    Very fast and reliable for finding active devices with open ports.
    """
    logger.info("Starting nmap-based Reolink camera discovery...")
    
    try:
        # Check if nmap is available
        process = await asyncio.create_subprocess_exec(
            "nmap", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        if process.returncode != 0:
            logger.info("nmap not available, skipping nmap discovery")
            return []
        
    except FileNotFoundError:
        logger.info("nmap not installed, skipping nmap discovery")
        return []
    except Exception as e:
        logger.debug(f"nmap availability check failed: {e}")
        return []
    
    try:
        # Get local subnet
        subnet = _get_local_subnet()
        if not subnet:
            logger.warning("Could not determine subnet for nmap scan")
            return []
        
        logger.info(f"Using nmap to scan {subnet} for devices with HTTPS ports...")
        
        # Run nmap to find devices with port 443 open
        cmd = [
            "nmap", "-p", "443", "--open", 
            "-T4",  # Timing template (faster)
            "--host-timeout", "5s",
            subnet
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.warning(f"nmap scan failed: {stderr.decode()}")
            return []
        
        # Parse nmap output to find IPs with port 443 open
        nmap_output = stdout.decode('utf-8')
        active_ips = []
        
        import re
        # Look for "Nmap scan report for IP" followed by "443/tcp open"
        ip_pattern = r'Nmap scan report for ([0-9.]+)'
        port_pattern = r'443/tcp\s+open'
        
        lines = nmap_output.split('\n')
        current_ip = None
        
        for line in lines:
            line = line.strip()
            
            # Check for IP address
            ip_match = re.search(ip_pattern, line)
            if ip_match:
                current_ip = ip_match.group(1)
                continue
            
            # Check for open port 443
            if current_ip and re.search(port_pattern, line):
                active_ips.append(current_ip)
                current_ip = None
        
        logger.info(f"nmap found {len(active_ips)} device(s) with HTTPS port open")
        
        if not active_ips:
            return []
        
        # Test each IP for Reolink signatures
        semaphore = asyncio.Semaphore(20)
        
        async def test_nmap_ip(ip: str) -> Optional[ReolinkCameraInfo]:
            """Test an nmap-discovered IP for Reolink camera."""
            async with semaphore:
                # Try Python networking first
                try:
                    result = await _fast_test_reolink_camera(ip, semaphore, 3.0)
                    if result:
                        return result
                except Exception:
                    pass
                
                # Fallback to curl
                if await _test_camera_with_curl(ip):
                    return ReolinkCameraInfo(
                        host=ip,
                        model="Unknown (detected via nmap + curl)",
                        name=f"reolink-{ip.split('.')[-1]}",
                        serial="Unknown",
                        firmware="Unknown"
                    )
                
                return None
        
        # Test all nmap-discovered IPs
        tasks = [test_nmap_ip(ip) for ip in active_ips]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful discoveries
        cameras = []
        for result in results:
            if isinstance(result, ReolinkCameraInfo):
                cameras.append(result)
                logger.info(f"Found via nmap scan: {result}")
        
        logger.info(f"nmap-based discovery complete: found {len(cameras)} Reolink camera(s)")
        return cameras
        
    except Exception as e:
        logger.debug(f"nmap discovery failed: {e}")
        return []


async def discover_reolink_cameras_mdns() -> List[ReolinkCameraInfo]:
    """
    Discover Reolink cameras using mDNS/Bonjour (if available).
    Many IP cameras advertise themselves via mDNS.
    """
    logger.info("Starting mDNS-based Reolink camera discovery...")
    
    try:
        # Try to use zeroconf for mDNS discovery
        from zeroconf import ServiceBrowser, Zeroconf
        import time
        
    except ImportError:
        logger.info("zeroconf library not available, skipping mDNS discovery")
        logger.debug("Install with: pip install zeroconf")
        return []
        
        class ReolinkListener:
            def __init__(self):
                self.cameras = []
            
            def add_service(self, zeroconf, type, name):
                info = zeroconf.get_service_info(type, name)
                if info:
                    # Check if this looks like a Reolink camera
                    name_lower = name.lower()
                    if 'reolink' in name_lower or 'camera' in name_lower:
                        if info.addresses:
                            ip = str(ipaddress.ip_address(info.addresses[0]))
                            self.cameras.append(ReolinkCameraInfo(
                                host=ip,
                                model="Unknown (mDNS discovery)",
                                name=name.split('.')[0],
                                serial="Unknown",
                                firmware="Unknown"
                            ))
            
            def remove_service(self, zeroconf, type, name):
                pass
            
            def update_service(self, zeroconf, type, name):
                pass
        
        zeroconf = Zeroconf()
        listener = ReolinkListener()
        
        # Look for common camera service types
        services = [
            "_http._tcp.local.",
            "_https._tcp.local.", 
            "_rtsp._tcp.local.",
            "_camera._tcp.local.",
            "_device-info._tcp.local."
        ]
        
        browsers = []
        for service in services:
            browser = ServiceBrowser(zeroconf, service, listener)
            browsers.append(browser)
        
        # Wait a bit for discovery
        await asyncio.sleep(3)
        
        # Clean up
        for browser in browsers:
            browser.cancel()
        zeroconf.close()
        
        logger.info(f"mDNS discovery found {len(listener.cameras)} potential camera(s)")
        return listener.cameras
        
    except ImportError:
        logger.info("zeroconf library not available, skipping mDNS discovery")
        logger.debug("Install with: pip install zeroconf")
        return []
    except Exception as e:
        logger.info(f"mDNS discovery failed: {e}")
        return []


async def _discover_cameras_with_curl(subnet: str, timeout: float = 5.0, max_concurrent: int = 20) -> List[ReolinkCameraInfo]:
    """
    Comprehensive curl-based camera discovery when Python networking fails.
    Scans the entire subnet using curl to find Reolink cameras.
    """
    try:
        import ipaddress
        network = ipaddress.IPv4Network(subnet, strict=False)
        ip_addresses = [str(ip) for ip in network.hosts()]
    except Exception as e:
        logger.error(f"Invalid subnet {subnet}: {e}")
        return []
    
    # Limit to reasonable range
    if len(ip_addresses) > 254:
        logger.info(f"Large subnet detected ({len(ip_addresses)} hosts), limiting to first 254")
        ip_addresses = ip_addresses[:254]
    
    logger.info(f"Curl-scanning {len(ip_addresses)} IP addresses for Reolink cameras...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def test_ip_with_curl(ip: str) -> Optional[ReolinkCameraInfo]:
        """Test a single IP with curl."""
        async with semaphore:
            if await _test_camera_with_curl(ip):
                return ReolinkCameraInfo(
                    host=ip,
                    model="Unknown (detected via curl scan)",
                    name=f"reolink-{ip.split('.')[-1]}",
                    serial="Unknown",
                    firmware="Unknown"
                )
            return None
    
    # Test all IPs concurrently
    tasks = [test_ip_with_curl(ip) for ip in ip_addresses]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect successful discoveries
    cameras = []
    for result in results:
        if isinstance(result, ReolinkCameraInfo):
            cameras.append(result)
            logger.info(f"Found via curl scan: {result}")
    
    return cameras


async def _test_camera_with_curl(host: str) -> bool:
    """
    Fallback test using curl when Python networking fails.
    Tests if a camera is a Reolink by checking API signatures.
    """
    try:
        # Test HTTPS API endpoint first (most common)
        for protocol in ["https", "http"]:
            cmd = [
                "curl", "-k", "--connect-timeout", "5", "--max-time", "10", "-s",
                f"{protocol}://{host}/cgi-bin/api.cgi"
            ]
            
            logger.debug(f"Testing {host} with curl: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                response_text = stdout.decode('utf-8', errors='ignore').lower()
                
                # Check for Reolink signatures
                signatures = ["please login first", "reolink", '"cmd"', '"code"', '"action"']
                matches = [sig for sig in signatures if sig in response_text]
                
                logger.debug(f"Curl response for {host}: {stdout.decode('utf-8', errors='ignore')[:200]}...")
                logger.debug(f"Signature matches: {matches}")
                
                if len(matches) >= 2:
                    logger.info(f"✅ Reolink camera detected at {host} via curl ({protocol.upper()})")
                    return True
        
    except Exception as e:
        logger.debug(f"Curl test failed for {host}: {e}")
    
    return False


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
    # Try curl fallback first since Python networking is currently broken
    curl_result = await _test_camera_credentials_with_curl(host, user, password, timeout)
    if curl_result is not None:
        return curl_result
    
    # Fallback to Python HTTP (though this is currently failing)
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


async def _test_camera_credentials_with_curl(
    host: str,
    user: str, 
    password: str,
    timeout: float = 5.0
) -> Optional[bool]:
    """
    Test camera credentials using curl as fallback when Python networking fails.
    
    Returns:
        True if credentials valid, False if invalid, None if curl failed
    """
    try:
        import json
        
        # Prepare the login payload
        payload = [{
            "cmd": "Login",
            "action": 0,
            "param": {
                "User": {"userName": user, "password": password}
            }
        }]
        
        payload_json = json.dumps(payload)
        
        # Test HTTPS first, then HTTP if needed
        for protocol in ["https", "http"]:
            url = f"{protocol}://{host}/cgi-bin/api.cgi?cmd=Login&token=null"
            
            cmd = [
                "curl", "-k", "--connect-timeout", str(int(timeout)), 
                "--max-time", str(int(timeout * 2)), "-s",
                "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d", payload_json,
                url
            ]
            
            logger.debug(f"Testing credentials for {host} with curl: {user}/***** via {protocol.upper()}")
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    response_text = stdout.decode('utf-8', errors='ignore')
                    
                    try:
                        data = json.loads(response_text)
                        
                        if (data and len(data) > 0):
                            code = data[0].get("code")
                            
                            if code == 0:
                                # Check for valid token
                                token_info = data[0].get("value", {}).get("Token", {})
                                if token_info.get("name"):
                                    logger.debug(f"✅ Credentials valid for {host} via curl ({protocol.upper()})")
                                    return True
                            elif code == 1:
                                # Invalid credentials
                                logger.debug(f"❌ Invalid credentials for {host} via curl ({protocol.upper()})")
                                return False
                        
                    except json.JSONDecodeError:
                        logger.debug(f"Failed to parse JSON response from {host} via {protocol}")
                        continue
                else:
                    logger.debug(f"Curl failed for {host} via {protocol}: return code {process.returncode}")
                    
            except Exception as e:
                logger.debug(f"Curl execution failed for {host} via {protocol}: {e}")
                continue
        
        # If we get here, curl worked but no valid response
        return False
        
    except Exception as e:
        logger.debug(f"Curl credential test failed for {host}: {e}")
        return None


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
