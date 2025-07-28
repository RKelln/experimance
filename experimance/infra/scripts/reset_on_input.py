#!/usr/bin/env python3
"""
Generic keyboard/controller listener for restarting Experimance exhibit.
Listens for any key press from any keyboard/controller and runs a full reset
including audio device refresh.

SAFETY FEATURES:
- Escape sequences: Ctrl+Alt+E or Escape key disables the listener for 5 minutes
- SSH detection: Only works on local console sessions
- Physical device priority: Prefers external USB devices over built-in keyboards
"""

import subprocess
import sys
import time
import logging
import os
from pathlib import Path
import signal
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/restart_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global state for safety features
class SafetyState:
    def __init__(self):
        self.escape_mode = False
        self.escape_until = 0
        self.escape_timeout_minutes = 5.0  # Escape mode lasts 5 minutes
        self.ctrl_pressed = False
        self.alt_pressed = False
        
    def is_escaped(self):
        """Check if we're in escape mode."""
        current_time = time.time()
        
        # Check escape mode timeout
        if self.escape_mode and current_time > self.escape_until:
            logger.info("🔓 Escape mode expired, re-enabling listener")
            self.escape_mode = False
            self.escape_until = 0
        
        if self.escape_mode:
            remaining = int((self.escape_until - current_time) / 60)
            return True, f"Escape mode active ({remaining} minutes remaining)"
        
        return False, ""
    
    def trigger_escape(self):
        """Activate escape mode for admin access."""
        self.escape_mode = True
        self.escape_until = time.time() + (self.escape_timeout_minutes * 60)
        logger.warning(f"🚨 ESCAPE MODE ACTIVATED - Listener disabled for {self.escape_timeout_minutes} minutes")
        logger.warning("🔓 Safe to use keyboard for system administration")
    
    def update_activity(self):
        """Update last activity timestamp (kept for compatibility)."""
        pass  # No longer needed but kept in case other code calls it

safety = SafetyState()

def check_ssh_session():
    """Check if we're in an SSH session - if so, don't intercept keys."""
    try:
        # Check if SSH_CLIENT or SSH_CONNECTION environment variables exist
        if os.environ.get('SSH_CLIENT') or os.environ.get('SSH_CONNECTION'):
            return True
        
        # Check if parent process chain includes sshd
        try:
            result = subprocess.run(['ps', '-o', 'comm=', '-p', str(os.getppid())], 
                                  capture_output=True, text=True)
            if 'sshd' in result.stdout:
                return True
        except:
            pass
        
        return False
    except Exception:
        return False

def find_usb_keyboard():
    """Find any USB keyboard or input device."""
    try:
        result = subprocess.run(['ls', '/dev/input/'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Could not list input devices")
            return None
        
        devices = [f"/dev/input/{d}" for d in result.stdout.split() if d.startswith('event')]
        logger.info(f"Scanning {len(devices)} input devices for keyboards/controllers...")
        
        # First, try to find the specific HID 1189:8890 device (your USB controller)
        for device in devices:
            try:
                info_result = subprocess.run(
                    ['udevadm', 'info', '--name=' + device], 
                    capture_output=True, text=True
                )
                
                if '1189:8890' in info_result.stdout:
                    logger.info(f"✅ Found preferred USB controller (HID 1189:8890): {device}")
                    return device
            except Exception:
                continue
        
        # If not found, look for any physical keyboard device (exclude virtual ones)
        logger.info("Looking for physical keyboard devices (excluding virtual/internal keyboards)...")
        for device in devices:
            try:
                info_result = subprocess.run(
                    ['udevadm', 'info', '--name=' + device], 
                    capture_output=True, text=True
                )
                
                # Debug logging
                device_name = "Unknown"
                for line in info_result.stdout.split('\n'):
                    if 'ID_MODEL=' in line:
                        device_name = line.split('=')[1] if '=' in line else "Unknown"
                        break
                logger.debug(f"Checking device {device}: {device_name}")
                
                # Look for keyboard indicators (exclude virtual/software keyboards)
                has_keyboard_indicator = any(indicator in info_result.stdout for indicator in [
                    'ID_INPUT_KEYBOARD=1',
                    'ID_INPUT_KEY=1', 
                    'keyboard',
                    'Keyboard'
                ])
                
                is_virtual = any(virtual in info_result.stdout.lower() for virtual in [
                    'virtual',
                    'atkbd',  # AT keyboard (often virtual)
                    'software',
                    'at translated set 2 keyboard'  # Common internal laptop keyboard
                ])
                
                if has_keyboard_indicator and not is_virtual:
                    logger.info(f"✅ Found physical keyboard device: {device} ({device_name})")
                    return device
                elif has_keyboard_indicator and is_virtual:
                    logger.debug(f"⏭️  Skipping virtual keyboard: {device} ({device_name})")
                    
            except Exception:
                continue
        
        # Fallback: try any HID device
        for device in devices:
            try:
                info_result = subprocess.run(
                    ['udevadm', 'info', '--name=' + device], 
                    capture_output=True, text=True
                )
                
                if 'HID' in info_result.stdout:
                    logger.info(f"✅ Found HID device (fallback): {device}")
                    return device
            except Exception:
                continue
        
        logger.error("❌ Could not find any suitable input device")
        logger.info("💡 Available devices:")
        for device in devices[:10]:  # Show first 10 devices
            try:
                info = subprocess.run(['udevadm', 'info', '--name=' + device], capture_output=True, text=True)
                device_name = "Unknown"
                for line in info.stdout.split('\n'):
                    if 'DEVNAME=' in line:
                        device_name = line.split('=')[1] if '=' in line else "Unknown"
                        break
                logger.info(f"  {device}: {device_name}")
            except:
                continue
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding input device: {e}")
        return None

def restart_experimance():
    """Restart all Experimance services using the reset script (includes audio refresh)."""
    logger.info("🔄 Restarting Experimance exhibit with full reset (including audio)...")
    
    try:
        # Get the path to the reset script
        script_dir = Path(__file__).parent.parent / "infra" / "scripts"
        reset_script = script_dir / "reset.sh"
        
        if not reset_script.exists():
            logger.error(f"Reset script not found at: {reset_script}")
            # Fallback to basic restart
            return restart_experimance_basic()
        
        # Run the reset script with the experimance project
        reset_cmd = ['sudo', str(reset_script), '--project', 'experimance']
        logger.info(f"Running: {' '.join(reset_cmd)}")
        
        result = subprocess.run(reset_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"Reset script failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            logger.error(f"STDOUT: {result.stdout}")
            return False
        else:
            logger.info("✅ Reset script completed successfully")
            logger.info(f"Reset output: {result.stdout}")
            return True
            
    except subprocess.TimeoutExpired:
        logger.error("Reset script timed out after 60 seconds")
        return False
    except Exception as e:
        logger.error(f"Error running reset script: {e}")
        # Fallback to basic restart
        return restart_experimance_basic()

def restart_experimance_basic():
    """Fallback restart function using basic systemctl commands."""
    logger.info("🔄 Using basic restart (systemctl only)...")
    
    try:
        # Stop all experimance services first
        stop_cmd = ['sudo', 'systemctl', 'stop', 'experimance@experimance.target']
        result = subprocess.run(stop_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Stop command failed: {result.stderr}")
        else:
            logger.info("✅ Stopped experimance services")
        
        # Wait a moment
        time.sleep(2)
        
        # Start all experimance services
        start_cmd = ['sudo', 'systemctl', 'start', 'experimance@experimance.target']
        result = subprocess.run(start_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Start command failed: {result.stderr}")
            return False
        else:
            logger.info("✅ Started experimance services")
            return True
            
    except Exception as e:
        logger.error(f"Error restarting services: {e}")
        return False

def listen_for_input(device_path):
    """Listen for input from any keyboard/controller."""
    logger.info(f"👂 Listening for input on {device_path}")
    logger.info("🎮 Press any key on any keyboard/controller to reset Experimance...")
    logger.info("🚨 SAFETY: Press Ctrl+Alt+E or Escape key to disable listener for admin access")
    
    last_restart = 0
    cooldown = 5  # 5 second cooldown between restarts
    
    def signal_handler(signum, frame):
        logger.info("🛑 Received termination signal, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Use evtest to monitor the device
        process = subprocess.Popen(
            ['sudo', 'evtest', device_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        assert process.stdout is not None, "Failed to start evtest process"
        
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
            
            # Check safety state before processing any input
            escaped, reason = safety.is_escaped()
            if escaped:
                logger.info(f"🔒 Input ignored: {reason}")
                # Check every 30 seconds if we should re-enable
                time.sleep(30)
                continue
            
            # Track modifier keys for escape sequence
            if 'EV_KEY' in line:
                if 'KEY_LEFTCTRL' in line or 'KEY_RIGHTCTRL' in line:
                    safety.ctrl_pressed = 'value 1' in line
                elif 'KEY_LEFTALT' in line or 'KEY_RIGHTALT' in line:
                    safety.alt_pressed = 'value 1' in line
                elif 'KEY_E' in line and 'value 1' in line:
                    # Check for Ctrl+Alt+E escape sequence
                    if safety.ctrl_pressed and safety.alt_pressed:
                        safety.trigger_escape()
                        continue
                elif 'KEY_ESC' in line and 'value 1' in line:
                    # Escape key pressed - activate escape mode
                    logger.info("🔑 Escape key pressed")
                    safety.trigger_escape()
                    continue
            
            # Look for any key press events (value 1 = key press, not release)
            # This will work with any keyboard or controller
            if 'EV_KEY' in line and 'value 1' in line:
                current_time = time.time()
                safety.update_activity()
                
                # Skip modifier-only presses
                if any(modifier in line for modifier in ['CTRL', 'ALT', 'SHIFT', 'KEY_LEFTMETA', 'KEY_RIGHTMETA']):
                    continue
                
                if current_time - last_restart > cooldown:
                    logger.info(f"🎹 Key press detected: {line}")
                    logger.info("🔄 Gallery staff triggered exhibit reset")
                    success = restart_experimance()
                    if success:
                        logger.info("🎉 Full reset completed successfully!")
                    else:
                        logger.error("❌ Reset failed!")
                    last_restart = current_time
                else:
                    logger.info(f"⏳ Cooldown active ({cooldown - (current_time - last_restart):.1f}s remaining)")
                    
    except KeyboardInterrupt:
        logger.info("🛑 Stopping controller listener...")
    except Exception as e:
        logger.error(f"Error listening for input: {e}")
    finally:
        if 'process' in locals():
            process.terminate()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Experimance restart controller with safety features')
    parser.add_argument('--bypass-ssh-check', action='store_true', 
                       help='Bypass SSH session check (for testing only)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: shorter timeouts and more verbose logging')
    args = parser.parse_args()
    
    if args.test_mode:
        logger.info("🧪 TEST MODE ENABLED")
        safety.escape_timeout_minutes = 1.0  # 1 minute for testing
    
    logger.info("🎮 Starting Experimance restart controller (generic keyboard support)...")
    
    # Safety check: Don't run if we're in an SSH session (unless bypassed for testing)
    if not args.bypass_ssh_check and check_ssh_session():
        logger.error("❌ SSH session detected - refusing to start for security")
        logger.error("💡 This service only works on local console to prevent remote interference")
        logger.error("💡 Use --bypass-ssh-check flag only for testing")
        sys.exit(1)
    
    if args.bypass_ssh_check:
        logger.warning("⚠️  SSH check bypassed - this should only be used for testing!")
    else:
        logger.info("✅ Local console detected - safe to start")
    
    # Find any keyboard/controller device
    device = find_usb_keyboard()
    if not device:
        logger.error("❌ Could not find any suitable keyboard/controller device")
        logger.info("💡 Try running 'ls /dev/input/' and 'sudo evtest' to identify your device")
        sys.exit(1)
    
    logger.info(f"📱 Using device: {device}")
    logger.info("🔒 SAFETY FEATURES ACTIVE:")
    logger.info(f"  • Escape sequences: Ctrl+Alt+E or Escape key (disables for {safety.escape_timeout_minutes} minutes)")
    logger.info("  • SSH protection: Only works on local console")
    logger.info("  • Physical device priority: Prefers external USB devices")
    
    # Listen for input
    listen_for_input(device)

if __name__ == '__main__':
    main()
