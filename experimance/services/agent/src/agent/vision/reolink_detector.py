"""
Simple Reolink camera audience detector for Experimance Agent Service.

This detector polls Reolink camera's AI detection for audience presence.
Designed as a complete replacement for computer vision processing.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from enum import Enum

import aiohttp
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(InsecureRequestWarning)

logger = logging.getLogger(__name__)


class DetectionMode(Enum):
    """Detection modes for hybrid detector."""
    MONITORING = "monitoring"  # Using camera AI only
    ACTIVE = "active"         # Using YOLO on camera frames


class ReolinkDetector:
    """
    Smart hybrid audience detector using Reolink camera.
    
    Two modes of operation:
    1. MONITORING: Uses camera's AI detection as a lightweight trigger (low CPU)
    2. ACTIVE: Uses YOLO on camera frames for precise detection (higher CPU)
    
    Flow:
    - Starts in MONITORING mode, polling camera AI
    - Camera AI detection triggers switch to ACTIVE mode  
    - YOLO handles precise detection and "all clear" detection
    - Returns to MONITORING when YOLO confirms no people present
    
    This provides the best of both worlds: efficiency + accuracy.
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
        hysteresis_present: Optional[int] = None,  # Readings needed to confirm present, default 3
        hysteresis_absent: Optional[int] = None,   # Readings needed to confirm absent, default 3
        # Hybrid mode parameters
        hybrid_mode: bool = False,                 # Enable hybrid camera AI + YOLO detection
        yolo_config: Optional[Dict[str, Any]] = None,  # YOLO configuration dictionary
        yolo_absent_threshold: int = 5,            # YOLO "absent" readings before switching back to monitoring
        yolo_check_interval: float = 1.0          # Seconds between YOLO checks in active mode
    ):
        self.host = host
        self.user = user
        self.password = password
        self.https = https
        self.timeout = timeout
        self.channel = channel
        
        # Asymmetric hysteresis - default to symmetric if not specified
        self.hysteresis_present = hysteresis_present if hysteresis_present is not None else 3
        self.hysteresis_absent = hysteresis_absent if hysteresis_absent is not None else 3
        
        # Hybrid mode configuration
        self.hybrid_mode = hybrid_mode
        self.yolo_config = yolo_config or {}
        self.yolo_absent_threshold = yolo_absent_threshold
        self.yolo_check_interval = yolo_check_interval
        
        # Detection mode state
        self._detection_mode = DetectionMode.MONITORING
        self._yolo_detector = None
        self._yolo_absent_count = 0  # Consecutive "absent" readings from YOLO
        self._mode_switches = 0
        self._current_person_count = 0  # Current detected person count from YOLO
        
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
        if self._session is not None:
            logger.debug("Detector already started, skipping initialization")
            return
            
        connector = aiohttp.TCPConnector(ssl=False)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        # Login to get authentication token
        try:
            await self._login()
            logger.info(f"Reolink detector ready: {self.host}")
        except Exception as e:
            logger.error(f"Failed to connect to Reolink camera: {e}")
            raise
        
        # Initialize YOLO detector if hybrid mode is enabled
        if self.hybrid_mode:
            try:
                from .yolo_person_detector import YOLO11PersonDetector
                self._yolo_detector = YOLO11PersonDetector.from_dict(self.yolo_config)
                logger.info(f"✅ Hybrid mode enabled: Camera AI + YOLO detection")
            except ImportError as e:
                logger.error(f"❌ Hybrid mode failed: YOLO detector not available: {e}")
                self.hybrid_mode = False
            except Exception as e:
                logger.error(f"❌ Hybrid mode failed: Could not initialize YOLO: {e}")
                self.hybrid_mode = False
    
    async def stop(self):
        """Clean up resources."""
        if self._session:
            # Try to logout gracefully
            try:
                await self._logout()
            except Exception:
                pass  # Ignore logout errors during cleanup
            
            # Close session safely
            try:
                await self._session.close()
            except Exception:
                pass  # Ignore close errors during cleanup
            finally:
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
    
    async def _capture_frame(self):
        """Capture a frame from the camera for YOLO processing."""
        if not self._session:
            raise RuntimeError("Session not initialized")
        
        protocol = "https" if self.https else "http"
        url = f"{protocol}://{self.host}/cgi-bin/api.cgi?cmd=Snap&channel={self.channel}&token={self._token}"
        
        async with self._session.get(url) as response:
            response.raise_for_status()
            
            # Return raw image data
            image_data = await response.read()
            
            # Decode image using OpenCV
            import numpy as np
            import cv2
            
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode camera frame")
            
            return frame
    
    async def check_audience_present(self) -> bool:
        """
        Check if audience is present using hybrid detection (camera AI + YOLO).
        
        Returns:
            bool: True if audience detected (stable state)
        """
        try:
            if not self.hybrid_mode or self._yolo_detector is None:
                # Simple mode: just use camera AI with hysteresis
                person_detected = await self._check_person_detected()
                self._total_checks += 1
                
                logger.debug(f"Reolink raw detection: {person_detected} (check #{self._total_checks})")
                
                # Apply hysteresis
                stable_state = self._apply_hysteresis(person_detected)
                return stable_state
            
            else:
                # Hybrid mode: switch between camera AI and YOLO
                return await self._hybrid_detection()
                
        except Exception as e:
            self._connection_errors += 1
            logger.warning(f"Detection check failed: {e}")
            # Return last known state on error
            return self._current_state
    
    async def _hybrid_detection(self) -> bool:
        """
        Hybrid detection using camera AI as trigger and YOLO for precision.
        
        State machine:
        MONITORING -> ACTIVE: When camera AI detects person
        ACTIVE -> MONITORING: When YOLO detects no person for threshold count
        """
        self._total_checks += 1
        
        if self._detection_mode == DetectionMode.MONITORING:
            # Monitor using camera AI (lightweight)
            camera_ai_detected = await self._check_person_detected()
            
            logger.debug(f"MONITORING: Camera AI detection = {camera_ai_detected}")
            
            if camera_ai_detected:
                # Camera AI triggered - switch to ACTIVE mode
                self._detection_mode = DetectionMode.ACTIVE
                self._yolo_absent_count = 0
                self._mode_switches += 1
                
                logger.info(f"🎯 Camera AI triggered - switching to ACTIVE mode (switch #{self._mode_switches})")
                
                # Immediately run YOLO check
                return await self._yolo_detection()
            else:
                # No trigger, stay in monitoring
                return self._current_state
        
        elif self._detection_mode == DetectionMode.ACTIVE:
            # Active YOLO detection
            return await self._yolo_detection()
        
        else:
            logger.error(f"Invalid detection mode: {self._detection_mode}")
            return self._current_state
    
    async def _yolo_detection(self) -> bool:
        """Run YOLO detection on current camera frame."""
        try:
            # Capture frame from camera
            frame = await self._capture_frame()
            
            # Run YOLO detection
            people_count, max_confidence, detections = self._yolo_detector.detect_people(frame)
            
            # Store current person count
            self._current_person_count = people_count
            
            yolo_detected = people_count > 0
            
            # Only log YOLO results when interesting (detection changes or high confidence)
            if yolo_detected != self._current_state or max_confidence > 0.8:
                logger.debug(f"ACTIVE: YOLO detected {people_count} people (conf={max_confidence:.3f})")
            
            if yolo_detected:
                # YOLO detected people - reset absent counter
                self._yolo_absent_count = 0
                
                # Apply hysteresis for presence detection
                stable_state = self._apply_hysteresis(True)
                return stable_state
            
            else:
                # YOLO detected no people - increment absent counter
                self._yolo_absent_count += 1
                self._current_person_count = 0  # No people detected                logger.debug(f"ACTIVE: No people detected, absent_count={self._yolo_absent_count}/{self.yolo_absent_threshold}")
                
                if self._yolo_absent_count >= self.yolo_absent_threshold:
                    # Switch back to monitoring mode
                    self._detection_mode = DetectionMode.MONITORING
                    self._mode_switches += 1
                    
                    logger.info(f"🔄 YOLO confirmed no people - switching to MONITORING mode (switch #{self._mode_switches})")
                
                # Apply hysteresis for absence detection
                stable_state = self._apply_hysteresis(False)
                return stable_state
                
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            # On YOLO error, fall back to camera AI
            camera_ai_detected = await self._check_person_detected()
            return self._apply_hysteresis(camera_ai_detected)
    
    async def _check_person_detected(self) -> bool:
        """Get person detection status from camera AI."""
        if not self._session:
            raise RuntimeError("Detector not started")
        
        # Call GetAiState API
        result = await self._api_call("GetAiState", {"channel": self.channel})
        
        # The response is flat - people info is directly in the result
        people_alarm = result.get("people", {})
        
        # Debug logging for API response
        logger.debug(f"Reolink AI API response: people_alarm={people_alarm}")
        
        # Check if people detection is supported and active
        if people_alarm.get("support", 0) != 1:
            logger.warning("People detection not supported by camera")
            return False
        
        alarm_state = people_alarm.get("alarm_state", 0)
        detected = alarm_state == 1
        
        logger.debug(f"Reolink people detection: support={people_alarm.get('support')}, alarm_state={alarm_state}, detected={detected}")
        
        return detected
    
    def _apply_hysteresis(self, new_reading: bool) -> bool:
        """Apply asymmetric hysteresis to avoid rapid state changes."""
        
        # Count consecutive readings of the same value
        if new_reading == self._last_reading:
            self._reading_count += 1
        else:
            self._reading_count = 1
            self._last_reading = new_reading
        
        # Determine required threshold based on target state
        required_count = self.hysteresis_present if new_reading else self.hysteresis_absent
        
        # Debug logging only when approaching or reaching threshold
        if self._reading_count <= required_count or new_reading != self._current_state:
            current_state_str = "present" if self._current_state else "absent"
            new_reading_str = "present" if new_reading else "absent"
            logger.debug(f"Hysteresis: raw={new_reading_str}, stable={current_state_str}, consecutive={self._reading_count}/{required_count}")
        
        # Change state only after enough consistent readings
        if (self._reading_count >= required_count and 
            new_reading != self._current_state):
            
            old_state = self._current_state
            self._current_state = new_reading
            self._state_changes += 1
            
            state_name = "present" if new_reading else "absent" 
            logger.info(f"Audience state changed to: {state_name} (after {self._reading_count} consecutive {new_reading_str} readings)")
        
        return self._current_state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics including hybrid mode info."""
        stats = {
            'current_state': self._current_state,
            'total_checks': self._total_checks,
            'connection_errors': self._connection_errors,
            'state_changes': self._state_changes,
            'hysteresis_present': self.hysteresis_present,
            'hysteresis_absent': self.hysteresis_absent,
            'consecutive_readings': self._reading_count,
            'last_reading': self._last_reading,
        }
        
        # Add hybrid mode stats
        if self.hybrid_mode:
            stats.update({
                'hybrid_mode': True,
                'detection_mode': self._detection_mode.value,
                'mode_switches': self._mode_switches,
                'yolo_absent_count': self._yolo_absent_count,
                'yolo_absent_threshold': self.yolo_absent_threshold,
                'yolo_available': self._yolo_detector is not None,
                'current_person_count': self._current_person_count
            })
            
            # Add YOLO stats if available
            if self._yolo_detector:
                yolo_stats = self._yolo_detector.get_statistics()
                stats['yolo_stats'] = yolo_stats
        else:
            stats['hybrid_mode'] = False
            
        return stats
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics (alias for get_stats for compatibility)."""
        stats = self.get_stats()
        stats['current_state'] = 'present' if stats['current_state'] else 'absent'
        return stats
