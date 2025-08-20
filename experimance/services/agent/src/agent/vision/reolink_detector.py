"""
Simple Reolink camera audience detector for Experimance Agent Service.

This detector polls Reolink camera's AI detection for audience presence.
Designed as a complete replacement for computer vision processing.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional

import aiohttp
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ReolinkDetector:
    """
    Simple audience detector using Reolink camera AI.
    
    Polls camera's AI alarm status for person detection with hysteresis
    to avoid rapid state changes.
    """
    
    @classmethod
    async def create_with_discovery(
        cls,
        known_ip: Optional[str] = None,
        user: str = "admin",
        password: str = "admin", 
        model_pattern: Optional[str] = None,
        **kwargs
    ) -> "ReolinkDetector":
        """
        Create a ReolinkDetector with automatic comprehensive camera discovery.
        
        Uses intelligent progressive discovery:
        1. If known_ip provided, tests that first
        2. Falls back to network scan if needed  
        3. Signature verification to confirm cameras
        
        Args:
            known_ip: Optional specific IP to test first (fastest)
            user: Camera username
            password: Camera password
            model_pattern: Optional model pattern to search for (e.g., "RLC-820A")
            **kwargs: Additional arguments passed to ReolinkDetector constructor
            
        Returns:
            Configured ReolinkDetector instance
            
        Raises:
            RuntimeError: If no camera found or credentials invalid
        """
        from .reolink_discovery import discover_reolink_cameras_comprehensive, find_reolink_camera_by_model, test_camera_credentials
        
        logger.info("Discovering Reolink cameras with comprehensive discovery...")
        
        # Find cameras using comprehensive discovery
        if model_pattern:
            # If model pattern specified, use traditional discovery for now
            # TODO: Could enhance comprehensive discovery to filter by model
            camera_info = await find_reolink_camera_by_model(model_pattern)
            cameras = [camera_info] if camera_info else []
        else:
            # Use new comprehensive discovery
            cameras = await discover_reolink_cameras_comprehensive(known_ip=known_ip)
        
        if not cameras:
            if known_ip:
                raise RuntimeError(f"No Reolink cameras found (tested known IP {known_ip} and network scan)")
            else:
                raise RuntimeError("No Reolink cameras found on network")
        
        # Test credentials for each camera
        for camera in cameras:
            logger.info(f"Testing credentials for {camera.host} ({camera.model})...")
            
            if await test_camera_credentials(camera.host, user, password):
                logger.info(f"✅ Successfully connected to {camera}")
                
                # Create detector instance
                detector = cls(
                    host=camera.host,
                    user=user,
                    password=password,
                    **kwargs
                )
                return detector
            else:
                logger.warning(f"❌ Invalid credentials for {camera.host}")
        
        raise RuntimeError(f"Found {len(cameras)} camera(s) but credentials invalid for all")
    
    def __init__(
        self,
        host: str,
        user: str = "admin", 
        password: str = "admin",
        https: bool = True,
        timeout: int = 10,
        channel: int = 0,
        hysteresis_count: int = 3  # Number of consistent readings needed for state change
    ):
        self.host = host
        self.user = user
        self.password = password
        self.https = https
        self.timeout = timeout
        self.channel = channel
        self.hysteresis_count = hysteresis_count
        
        # Simple state tracking with hysteresis
        self._current_state = False  # Current stable state
        self._reading_count = 0      # Consecutive readings of same value
        self._last_reading = None    # Last raw reading from camera
        
        # Basic stats
        self._total_checks = 0
        self._connection_errors = 0
        self._state_changes = 0
        
        # HTTP session and authentication
        self._session: Optional[aiohttp.ClientSession] = None
        self._token: str = "null"  # Must be literal "null" for first login request
        self._token_expires_at: Optional[float] = None  # Unix timestamp when token expires
        
    async def start(self):
        """Initialize the detector."""
        connector = aiohttp.TCPConnector(ssl=False)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        # Login to get authentication token
        try:
            await self._login()
            logger.info(f"Simple Reolink detector ready: {self.host}")
        except Exception as e:
            logger.error(f"Failed to connect to Reolink camera: {e}")
            raise
    
    async def stop(self):
        """Clean up resources."""
        if self._session:
            # Try to logout gracefully
            try:
                await self._logout()
            except Exception:
                pass  # Ignore logout errors during cleanup
            await self._session.close()
            self._session = None
    
    async def _login(self):
        """Login to camera and get authentication token."""
        if not self._session:
            raise RuntimeError("Session not initialized")
        
        protocol = "https" if self.https else "http"
        url = f"{protocol}://{self.host}/cgi-bin/api.cgi?cmd=Login&token=null"
        
        payload = [{
            "cmd": "Login",
            "action": 0,
            "param": {
                "User": {
                    "userName": self.user,
                    "password": self.password
                }
            }
        }]
        
        logger.debug(f"Login attempt to: {url}")
        
        async with self._session.post(
            url, 
            json=payload, 
            headers={'Content-Type': 'application/json'}
        ) as response:
            response.raise_for_status()
            
            # Get response text
            text = await response.text()
            
            # IMPORTANT: Reolink cameras return JSON data with "text/html" Content-Type header
            # This is a quirk of their firmware - they send valid JSON but wrong MIME type
            try:
                import json
                data = json.loads(text)
            except Exception as e:
                logger.error(f"Failed to parse JSON: {e}, raw text: {text[:200]}...")
                # If JSON parsing fails, check if it looks like an auth error
                if 'login' in text.lower() or 'unauthorized' in text.lower():
                    raise ValueError("Authentication failed - check username/password")
                else:
                    raise ValueError(f"Unexpected response format: {text[:200]}...")
            
            if not data or data[0].get("code") != 0:
                error_detail = data[0].get('error', {}).get('detail', 'Login failed') if data else 'Empty response'
                logger.error(f"Login API error: {error_detail}")
                raise ValueError(f"Login failed: {error_detail}")
            
            # Extract token and lease time
            token_info = data[0].get("value", {}).get("Token", {})
            token = token_info.get("name")
            lease_time = token_info.get("leaseTime", 3600)  # Default 1 hour
            
            if not token:
                logger.error(f"No token in response: {data}")
                raise ValueError("Login succeeded but no token returned")
            
            self._token = token
            # Set expiry to 90% of lease time to allow for renewal before expiry
            import time
            self._token_expires_at = time.time() + (lease_time * 0.9)
            logger.info(f"Reolink authentication successful (token expires in {lease_time}s)")
    
    async def _logout(self):
        """Logout from camera and invalidate token."""
        if not self._session or self._token == "null":
            return
        
        try:
            await self._api_call("Logout", {})
        finally:
            self._token = "null"
            self._token_expires_at = None
    
    def _token_needs_renewal(self) -> bool:
        """Check if token needs renewal (approaching expiry)."""
        if self._token == "null" or self._token_expires_at is None:
            return True
        import time
        return time.time() >= self._token_expires_at
    
    async def _api_call(self, cmd: str, param: dict) -> dict:
        """Make authenticated API call to camera with automatic token renewal."""
        if not self._session:
            raise RuntimeError("Session not initialized")
        
        # Check if token needs renewal before making the call
        if self._token_needs_renewal():
            logger.info("Token expired or approaching expiry, renewing...")
            await self._login()
        
        protocol = "https" if self.https else "http"
        url = f"{protocol}://{self.host}/cgi-bin/api.cgi?cmd={cmd}&token={self._token}"
        
        payload = [{
            "cmd": cmd,
            "action": 0,
            "param": param
        }]
        
        async with self._session.post(
            url, 
            json=payload, 
            headers={'Content-Type': 'application/json'}
        ) as response:
            response.raise_for_status()
            
            # IMPORTANT: Reolink cameras return JSON data with "text/html" Content-Type header  
            # This is a quirk of their firmware - they send valid JSON but wrong MIME type
            try:
                import json
                text = await response.text()
                data = json.loads(text)
            except Exception as e:
                text = await response.text() if 'text' not in locals() else text
                logger.error(f"Failed to parse JSON: {e}, raw text: {text[:200]}...")
                # Check if it looks like an auth error (shouldn't happen with proactive renewal)
                if 'login' in text.lower() or 'unauthorized' in text.lower():
                    raise ValueError("Authentication failed - token may have expired unexpectedly")
                else:
                    raise ValueError(f"Unexpected response format: {text[:200]}...")
            
            if not data or data[0].get("code") != 0:
                error_detail = data[0].get('error', {}).get('detail', 'API call failed') if data else 'Empty response'
                raise ValueError(f"API call {cmd} failed: {error_detail}")
            
            return data[0].get("value", {})
    
    async def check_audience_present(self) -> bool:
        """
        Check if audience is present with hysteresis.
        
        Returns:
            bool: True if audience detected (stable state)
        """
        try:
            # Get raw detection from camera
            person_detected = await self._check_person_detected()
            self._total_checks += 1
            
            # Apply hysteresis
            stable_state = self._apply_hysteresis(person_detected)
            
            return stable_state
            
        except Exception as e:
            self._connection_errors += 1
            logger.warning(f"Camera check failed: {e}")
            # Return last known state on error
            return self._current_state
    
    async def _check_person_detected(self) -> bool:
        """Get person detection status from camera AI."""
        if not self._session:
            raise RuntimeError("Detector not started")
        
        # Call GetAiState API
        result = await self._api_call("GetAiState", {"channel": self.channel})
        
        # The response is flat - people info is directly in the result
        people_alarm = result.get("people", {})
        
        # Check if people detection is supported and active
        if people_alarm.get("support", 0) != 1:
            logger.warning("People detection not supported by camera")
            return False
        
        return people_alarm.get("alarm_state", 0) == 1
    
    def _apply_hysteresis(self, new_reading: bool) -> bool:
        """Apply hysteresis to avoid rapid state changes."""
        
        # Count consecutive readings of the same value
        if new_reading == self._last_reading:
            self._reading_count += 1
        else:
            self._reading_count = 1
            self._last_reading = new_reading
        
        # Change state only after enough consistent readings
        if (self._reading_count >= self.hysteresis_count and 
            new_reading != self._current_state):
            
            self._current_state = new_reading
            self._state_changes += 1
            
            state_name = "present" if new_reading else "absent" 
            logger.info(f"Audience state changed to: {state_name}")
        
        return self._current_state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simple detector statistics."""
        return {
            'current_state': self._current_state,
            'total_checks': self._total_checks,
            'connection_errors': self._connection_errors,
            'state_changes': self._state_changes,
            'hysteresis_count': self.hysteresis_count,
            'consecutive_readings': self._reading_count
        }
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics (alias for get_stats for compatibility)."""
        stats = self.get_stats()
        stats['current_state'] = 'present' if stats['current_state'] else 'absent'
        return stats
