"""
Reolink Camera Presence Detection and Control

This module provides a Python client for interacting with Reolink IP cameras via their HTTP API.
It supports presence detection (person, vehicle, pet) and basic camera control functions.

Tested with:
- Reolink RLC-820A (firmware v3.1.0.2368_23062508)

Features:
- Real-time presence detection (person, vehicle, pet via AI)
- Camera control (stealth mode, LED control, IR lights)
- API exploration and status checking
- SSL certificate handling for self-signed certs

Usage Examples:
    # Basic presence monitoring
    python reolink_presence.py --host 192.168.1.100 --user admin --password mypass
    
    # Turn camera to stealth mode (disable LEDs/IR)
    python reolink_presence.py --host 192.168.1.100 --user admin --password mypass --camera-off
    
    # Control individual features
    python reolink_presence.py --host 192.168.1.100 --user admin --password mypass --power-led off
    python reolink_presence.py --host 192.168.1.100 --user admin --password mypass --ir-lights off
    
    # Explore available API commands
    python reolink_presence.py --host 192.168.1.100 --user admin --password mypass --explore

Requirements:
- requests
- urllib3

Camera Setup:
1. Enable AI detection in camera settings
2. Create a user with appropriate permissions (or use admin)
3. Note the camera's IP address on your network

Security Notes:
- Create a dedicated user with minimal required permissions
- Use HTTPS when possible (default)
- SSL warnings are disabled for self-signed certificates (common on IP cameras)

Author: Generated for Experimance project
License: Same as parent project
"""

import time
import json
import logging
import warnings
from typing import Dict, Any, Optional
import requests
import urllib3

# Disable SSL warnings for self-signed certificates (common on IP cameras)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger("reolink")
logging.basicConfig(level=logging.INFO)

class ReolinkClient:
    """
    Client for interacting with Reolink IP cameras via HTTP API.
    
    Supports presence detection, camera control, and status monitoring.
    Handles authentication, SSL certificates, and API command formatting.
    
    Example:
        client = ReolinkClient("192.168.1.100", "admin", "password")
        client.login()
        try:
            presence = client.presence_summary()
            print(presence)  # {"person_present": True, "vehicle_present": False, ...}
        finally:
            client.logout()
    
    Attributes:
        scheme (str): HTTP or HTTPS
        base (str): Base API URL
        user (str): Username for authentication
        password (str): Password for authentication  
        timeout (int): Request timeout in seconds
        s (requests.Session): HTTP session for connection reuse
        verify (bool): Whether to verify SSL certificates
        token (str): Authentication token from camera
    """
    def __init__(self, host: str, user: str, password: str, https: bool = True, timeout=5):
        """
        Initialize Reolink camera client.
        
        Args:
            host: Camera IP address or hostname (e.g. '192.168.1.50')
            user: Camera username (recommend creating dedicated user vs admin)
            password: Camera password
            https: Use HTTPS (True) or HTTP (False). HTTPS recommended.
            timeout: Request timeout in seconds
            
        Note:
            Most Reolink cameras use self-signed SSL certificates, so SSL
            verification is disabled by default for HTTPS connections.
        """
        self.scheme = "https" if https else "http"
        self.base = f"{self.scheme}://{host}/cgi-bin/api.cgi"
        self.user = user
        self.password = password
        self.timeout = timeout
        self.s = requests.Session()
        # Many Reolink cams use a self-signed cert; skip verification if needed
        self.verify = False if https else True
        self.token = "null"  # must be literal "null" for the first Login request

    def _api(self, cmd: str, param: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send API command to camera and return response.
        
        Handles Reolink's specific API format, authentication, and error handling.
        Automatically retries once on authentication errors.
        
        Args:
            cmd: API command name (e.g. 'GetAiState', 'SetIrLights')
            param: Optional parameters dict for the command
            
        Returns:
            Response data from camera (the 'value' field of response)
            
        Raises:
            RuntimeError: If command fails or returns error code
            requests.RequestException: If HTTP request fails
            
        Note:
            Reolink API expects commands in array format with specific structure.
        """
        payload = [{
            "cmd": cmd,
            "action": 0,
            "param": (param or {})
        }]
        url = f"{self.base}?cmd={cmd}&token={self.token}"
        r = self.s.post(url, json=payload, timeout=self.timeout, verify=self.verify)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Unexpected response shape: {data}")
        obj = data[0]
        if obj.get("code", -1) != 0:
            # token expired -> try relogin once
            if obj.get("error", {}).get("rspCode") in (1, 401, 403):
                log.info("Token probably expired; re‑logging in…")
                self.login()
                url = f"{self.base}?cmd={cmd}&token={self.token}"
                r = self.s.post(url, json=payload, timeout=self.timeout, verify=self.verify)
                r.raise_for_status()
                data = r.json()
                obj = data[0]
                if obj.get("code", -1) != 0:
                    raise RuntimeError(f"{cmd} error after relogin: {obj}")
            else:
                raise RuntimeError(f"{cmd} error: {obj}")
        return obj.get("value", {})

    def login(self):
        """
        Authenticate with camera and acquire session token.
        
        Must be called before using other API methods.
        Token is stored and used for subsequent requests.
        
        Raises:
            RuntimeError: If login fails or no token returned
        """
        val = self._api("Login", {
            "User": {"userName": self.user, "password": self.password}
        })
        tok = val.get("Token", {}).get("name")
        if not tok:
            raise RuntimeError("Login succeeded but no token returned")
        self.token = tok
        log.info("Logged in, token acquired")

    def logout(self):
        """
        End camera session and invalidate token.
        
        Should be called when done to properly clean up session.
        Safe to call multiple times or if not logged in.
        """
        try:
            self._api("Logout")
        except Exception:
            pass
        self.token = "null"

    def get_ai_state(self, channel: int = 0) -> Dict[str, Any]:
        """
        Get current AI detection state from camera.
        
        Args:
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            Dict containing AI state for each detection type:
            {
                "channel": 0,
                "people": {"alarm_state": 0/1, "support": 1},
                "vehicle": {"alarm_state": 0/1, "support": 1}, 
                "dog_cat": {"alarm_state": 0/1, "support": 1},
                "face": {"alarm_state": 0/1, "support": 0/1}
            }
            
        Note:
            - alarm_state: 1 if detected, 0 if not detected
            - support: 1 if feature supported, 0 if not supported by hardware
        """
        return self._api("GetAiState", {"channel": channel})

    def presence_summary(self, channel: int = 0) -> Dict[str, bool]:
        """
        Get simplified presence detection summary.
        
        Args:
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            Dict with boolean flags for each supported detection type:
            {
                "person_present": True/False,
                "vehicle_present": True/False, 
                "pet_present": True/False
            }
            
        Note:
            Only includes detection types supported by the camera hardware.
            Unsupported features will always return False.
        """
        ai = self.get_ai_state(channel)
        def on(k):  # helper maps alarm_state -> bool safely
            v = ai.get(k, {})
            return bool(v.get("alarm_state", 0)) if v.get("support", 0) == 1 else False
        return {
            "person_present":  on("people"),
            "vehicle_present": on("vehicle"),
            "pet_present":     on("dog_cat"),
            # add other classes if your firmware exposes them
        }

    def get_recording_state(self, channel: int = 0) -> Dict[str, Any]:
        """Get current recording configuration"""
        return self._api("GetRec", {"channel": channel})

    def set_recording(self, enabled: bool, channel: int = 0) -> bool:
        """Enable/disable recording"""
        try:
            current = self.get_recording_state(channel)
            if "schedule" in current:
                # Update the recording schedule
                schedule = current["schedule"]
                schedule["enable"] = 1 if enabled else 0
                self._api("SetRec", {"Rec": schedule})
                return True
            return False
        except Exception as e:
            log.error(f"Failed to set recording: {e}")
            return False

    def get_motion_detection_state(self, channel: int = 0) -> Dict[str, Any]:
        """Get motion detection configuration"""
        try:
            return self._api("GetMdState", {"channel": channel})
        except:
            # Fallback for different firmware versions
            return self._api("GetAlarm", {"Alarm": {"channel": channel, "type": "md"}})

    def set_motion_detection(self, enabled: bool, channel: int = 0) -> bool:
        """Enable/disable motion detection"""
        try:
            current = self.get_motion_detection_state(channel)
            if "enable" in current:
                current["enable"] = 1 if enabled else 0
                self._api("SetMdState", current)
                return True
            elif "Alarm" in current and "enable" in current["Alarm"]:
                alarm_config = current["Alarm"]
                alarm_config["enable"] = 1 if enabled else 0
                self._api("SetAlarm", {"Alarm": alarm_config})
                return True
            return False
        except Exception as e:
            log.error(f"Failed to set motion detection: {e}")
            return False

    def set_ir_lights(self, mode: str = "Auto", channel: int = 0) -> bool:
        """
        Control camera IR lights.
        
        Args:
            mode: IR light mode - "Auto", "On", or "Off"
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            True if command succeeded, False otherwise
            
        Note:
            - "Auto": IR lights turn on automatically in low light
            - "On": IR lights always on  
            - "Off": IR lights always off (no night vision)
        """
        try:
            result = self._api("SetIrLights", {"IrLights": {"state": mode, "channel": channel}})
            return result.get("rspCode") == 200
        except Exception as e:
            log.error(f"Failed to set IR lights: {e}")
            return False

    def get_ir_lights(self, channel: int = 0) -> Dict[str, Any]:
        """
        Get current IR lights state.
        
        Args:
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            Dict containing IR lights configuration:
            {"IrLights": {"state": "Auto"/"On"/"Off"}}
        """
        return self._api("GetIrLights", {"channel": channel})

    def set_power_led(self, enabled: bool, channel: int = 0) -> bool:
        """
        Control camera power LED.
        
        Args:
            enabled: True to turn LED on, False to turn off
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            True if command succeeded, False otherwise
            
        Note:
            Turning off power LED makes camera appear "inactive" while
            still functioning normally (stealth mode).
        """
        try:
            state = "On" if enabled else "Off"
            result = self._api("SetPowerLed", {"PowerLed": {"state": state, "channel": channel}})
            return result.get("rspCode") == 200
        except Exception as e:
            log.error(f"Failed to set power LED: {e}")
            return False

    def get_power_led(self, channel: int = 0) -> Dict[str, Any]:
        """
        Get current power LED state.
        
        Args:
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            Dict containing power LED state:
            {"PowerLed": {"state": "On"/"Off", "channel": 0}}
        """
        return self._api("GetPowerLed", {"channel": channel})

    def camera_stealth_mode(self, enabled: bool, channel: int = 0) -> Dict[str, bool]:
        """
        Enable or disable camera stealth mode.
        
        Args:
            enabled: True for stealth mode (LEDs off), False for normal mode
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            Dict showing success/failure of each operation:
            {
                "power_led_off": True,
                "ir_lights_off": True
            }
            
        Note:
            Stealth mode makes camera appear inactive by turning off all LEDs
            while maintaining full functionality (recording, AI detection, etc).
        """
        if enabled:
            results = {
                "power_led_off": self.set_power_led(False, channel),
                "ir_lights_off": self.set_ir_lights("Off", channel)
            }
        else:
            results = {
                "power_led_on": self.set_power_led(True, channel),
                "ir_lights_auto": self.set_ir_lights("Auto", channel)
            }
        return results

    def camera_off(self, channel: int = 0) -> Dict[str, bool]:
        """
        Turn camera to "off" state using stealth mode.
        
        Args:
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            Dict showing success/failure of each operation
            
        Note:
            This is the best available "off" mode for RLC-820A cameras.
            Camera continues functioning but appears inactive.
            For true power-off, use a smart switch or physical control.
        """
        return self.camera_stealth_mode(True, channel)

    def camera_on(self, channel: int = 0) -> Dict[str, bool]:
        """
        Turn camera to normal operation mode.
        
        Args:
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            Dict showing success/failure of each operation
            
        Note:
            Restores camera to normal visible operation with LEDs on
            and IR lights in automatic mode.
        """
        return self.camera_stealth_mode(False, channel)

    def get_camera_status(self, channel: int = 0) -> Dict[str, Any]:
        """
        Get comprehensive camera status and capabilities.
        
        Args:
            channel: Camera channel (0 for single-channel cameras)
            
        Returns:
            Dict containing:
            - supported_features: List of working API commands
            - errors: List of unsupported/failed commands  
            - Additional data from successful commands
            
        Note:
            Useful for debugging and understanding camera capabilities.
        """
        status: Dict[str, Any] = {"supported_features": [], "errors": []}
        
        # Try different methods to get camera state
        methods_to_try = [
            ("recording_state", lambda: self._api("GetRec", {"channel": channel})),
            ("motion_detection", lambda: self._api("GetMdState", {"channel": channel})),
            ("alarm_config", lambda: self._api("GetAlarm", {"Alarm": {"channel": channel, "type": "md"}})),
            ("device_info", lambda: self._api("GetDevInfo")),
            ("general_info", lambda: self._api("GetGeneral")),
            ("network_info", lambda: self._api("GetNetCfg")),
        ]
        
        for name, method in methods_to_try:
            try:
                result = method()
                status[name] = result
                status["supported_features"].append(name)
            except Exception as e:
                status["errors"].append(f"{name}: {str(e)}")
        
        return status

    def explore_available_commands(self) -> Dict[str, Any]:
        """
        Test common Reolink API commands to see what this camera supports.
        
        Returns:
            Dict containing:
            - supported: List of working commands
            - unsupported: List of commands that failed
            - details: Full response/error for each command
            
        Note:
            Useful for discovering capabilities of different camera models.
            Different firmware versions may support different commands.
        """
        common_commands = [
            "GetDevInfo", "GetGeneral", "GetTime", "GetUser", "GetNetCfg",
            "GetWifi", "GetRec", "SetRec", "GetMdState", "SetMdState", 
            "GetAlarm", "SetAlarm", "GetAiState", "GetPtzPreset", "SetPtzPreset",
            "GetIrLights", "SetIrLights", "GetImage", "SetImage", "GetOsd", "SetOsd",
            "GetPowerLed", "SetPowerLed", "GetStatusLed", "SetStatusLed"
        ]
        
        results = {"supported": [], "unsupported": [], "details": {}}
        
        for cmd in common_commands:
            try:
                # Try with minimal parameters
                if cmd == "GetAlarm":
                    result = self._api(cmd, {"Alarm": {"channel": 0, "type": "md"}})
                elif cmd in ["GetMdState", "GetRec"]:
                    result = self._api(cmd, {"channel": 0})
                else:
                    result = self._api(cmd)
                
                results["supported"].append(cmd)
                results["details"][cmd] = result
            except Exception as e:
                results["unsupported"].append(cmd)
                results["details"][cmd] = str(e)
        
        return results

if __name__ == "__main__":
    import argparse
    import sys
    
    # Command line argument parser with comprehensive help
    p = argparse.ArgumentParser(
        description="Reolink Camera Presence Detection and Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic presence monitoring (continuous)
  %(prog)s --host 192.168.1.100 --user admin --password mypass
  
  # Turn camera to stealth mode (appears off but still functions)  
  %(prog)s --host 192.168.1.100 --user admin --password mypass --camera-off
  
  # Turn camera back to normal mode
  %(prog)s --host 192.168.1.100 --user admin --password mypass --camera-on
  
  # Control individual features
  %(prog)s --host 192.168.1.100 --user admin --password mypass --power-led off
  %(prog)s --host 192.168.1.100 --user admin --password mypass --ir-lights off
  
  # Check what your camera supports
  %(prog)s --host 192.168.1.100 --user admin --password mypass --explore
  
  # Debug mode with raw AI data
  %(prog)s --host 192.168.1.100 --user admin --password mypass --debug

Camera Models Tested:
  - RLC-820A (firmware v3.1.0.2368_23062508)
  
Security Notes:
  - Create a dedicated camera user instead of using admin
  - Use HTTPS when possible (enabled by default)
  - Camera uses self-signed SSL cert (warnings disabled)
        """
    )
    # Connection settings
    p.add_argument("--host", required=True, 
                   help="Camera IP address or hostname (e.g. 192.168.1.100)")
    p.add_argument("--user", required=True,
                   help="Camera username (recommend creating dedicated user)")
    p.add_argument("--password", required=True,
                   help="Camera password")
    p.add_argument("--channel", type=int, default=0,
                   help="Camera channel number (default: 0 for single-channel cameras)")
    p.add_argument("--http", action="store_true",
                   help="Use HTTP instead of HTTPS (not recommended)")
    
    # Monitoring settings  
    p.add_argument("--interval", type=float, default=2.0,
                   help="Polling interval in seconds for continuous monitoring (default: 2.0)")
    p.add_argument("--debug", action="store_true",
                   help="Show raw AI state data along with presence summary")
    
    # Camera control options
    p.add_argument("--camera-on", action="store_true",
                   help="Turn camera to normal mode (enable LEDs and IR)")
    p.add_argument("--camera-off", action="store_true", 
                   help="Turn camera to stealth mode (disable LEDs and IR)")
    p.add_argument("--ir-lights", choices=["auto", "on", "off"],
                   help="Set IR lights mode: auto (default), on (always), or off (disabled)")
    p.add_argument("--power-led", choices=["on", "off"],
                   help="Turn power LED on or off")
    
    # Information commands
    p.add_argument("--status", action="store_true",
                   help="Show camera status and supported features, then exit")
    p.add_argument("--explore", action="store_true",
                   help="Test all common API commands to see what camera supports")
    
    args = p.parse_args()

    client = ReolinkClient(args.host, args.user, args.password, https=not args.http)
    client.login()
    
    try:
        # Handle camera control commands
        if args.camera_on:
            print("Turning camera on...")
            result = client.camera_on(args.channel)
            print(json.dumps(result, indent=2))
            sys.exit(0)
        
        if args.camera_off:
            print("Turning camera off (stealth mode)...")
            result = client.camera_off(args.channel)
            print(json.dumps(result, indent=2))
            sys.exit(0)
            
        if args.ir_lights:
            mode = args.ir_lights.title()  # Convert to "Auto", "On", "Off"
            print(f"Setting IR lights to {mode}...")
            success = client.set_ir_lights(mode, args.channel)
            print(f"IR lights set to {mode}: {success}")
            sys.exit(0)
            
        if args.power_led:
            enabled = args.power_led == "on"
            print(f"Setting power LED to {args.power_led}...")
            success = client.set_power_led(enabled, args.channel)
            print(f"Power LED {'enabled' if enabled else 'disabled'}: {success}")
            sys.exit(0)
        
        if args.status:
            print("Camera status:")
            status = client.get_camera_status(args.channel)
            print(json.dumps(status, indent=2))
            sys.exit(0)
            
        if args.explore:
            print("Exploring available API commands...")
            commands = client.explore_available_commands()
            print(json.dumps(commands, indent=2))
            sys.exit(0)
        
        # Default behavior - continuous presence monitoring
        while True:
            if args.debug:
                ai_state = client.get_ai_state(channel=args.channel)
                print(f"Raw AI state: {json.dumps(ai_state, indent=2)}")
            summary = client.presence_summary(channel=args.channel)
            print(json.dumps(summary))
            time.sleep(args.interval)
    finally:
        client.logout()


"""
API REFERENCE FOR DEVELOPERS

ReolinkClient Methods:

Core Methods:
    login() - Authenticate with camera (required first)
    logout() - End session (recommended cleanup)
    
Presence Detection:
    get_ai_state(channel=0) - Get raw AI detection data
    presence_summary(channel=0) - Get simplified boolean presence flags
    
Camera Control (RLC-820A supported):
    set_ir_lights(mode, channel=0) - Control IR lights ("Auto", "On", "Off")
    get_ir_lights(channel=0) - Get current IR lights state
    set_power_led(enabled, channel=0) - Control power LED (True/False)  
    get_power_led(channel=0) - Get current power LED state
    camera_stealth_mode(enabled, channel=0) - Enable/disable stealth mode
    camera_off(channel=0) - Turn to stealth mode (best "off" available)
    camera_on(channel=0) - Turn to normal mode
    
Information/Debug:
    get_camera_status(channel=0) - Get comprehensive status
    explore_available_commands() - Test all common API commands
    
Legacy Methods (may not work on all models):
    get_recording_state(channel=0) - Get recording config (if supported)
    set_recording(enabled, channel=0) - Control recording (if supported)
    get_motion_detection_state(channel=0) - Get motion detection config
    set_motion_detection(enabled, channel=0) - Control motion detection

Response Formats:

presence_summary() returns:
{
    "person_present": bool,
    "vehicle_present": bool, 
    "pet_present": bool
}

get_ai_state() returns:
{
    "channel": int,
    "people": {"alarm_state": 0/1, "support": 0/1},
    "vehicle": {"alarm_state": 0/1, "support": 0/1},
    "dog_cat": {"alarm_state": 0/1, "support": 0/1},
    "face": {"alarm_state": 0/1, "support": 0/1}
}

camera_off()/camera_on() returns:
{
    "power_led_off": bool,
    "ir_lights_off": bool
}

Error Handling:
- Methods return False/empty dict on failure
- _api() method raises RuntimeError on API errors
- Use try/except blocks around API calls
- Check logs for detailed error messages

Example Integration:

    from reolink_presence import ReolinkClient
    
    client = ReolinkClient("192.168.1.100", "user", "pass")
    client.login()
    
    try:
        # Check for presence
        presence = client.presence_summary()
        if presence["person_present"]:
            print("Person detected!")
            
        # Control camera
        client.camera_off()  # Stealth mode
        time.sleep(10)
        client.camera_on()   # Normal mode
        
    finally:
        client.logout()

Notes:
- Different camera models support different features
- Use explore_available_commands() to check capabilities
- RLC-820A doesn't support recording/motion detection control via API
- Stealth mode is the best "off" option (LEDs off, camera still functions)
- Always call login() before other methods
- Always call logout() when done for proper cleanup
"""
