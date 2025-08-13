#!/usr/bin/env python3
"""
Generic keyboard/controller listener for restarting Experimance exhibit.
Listens for any key press from any keyboard/controller and runs a full reset
including audio device refresh. Also listens for power button to safely
shutdown the exhibit before system power off.

SAFETY FEATURES:
- Escape sequences: Ctrl+Alt+E or Escape key disables the listener for 5 minutes
- SSH detection: Only works on local console sessions
- Physical device priority: Prefers external USB devices over built-in keyboards

FUNCTIONALITY:
- Any key press: Triggers full exhibit reset (restart all services)
- Power button: Triggers safe shutdown (stops services, cleans up, then allows power off)
- Escape sequences: Temporarily disables listener for system administration
"""

import subprocess
import sys
import time
import logging
import os
from pathlib import Path
import signal
import argparse
import threading
from queue import Queue

# Set up logging
log_dir = Path('/var/log/experimance')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'input.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
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

def find_input_devices():
    """Find both keyboard/controller devices AND the power button device."""
    devices = {}
    
    try:
        result = subprocess.run(['ls', '/dev/input/'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Could not list input devices")
            return devices
        
        all_devices = [f"/dev/input/{d}" for d in result.stdout.split() if d.startswith('event')]
        logger.info(f"Scanning {len(all_devices)} input devices...")
        
        for device in all_devices:
            try:
                # Get udevadm info first
                udev_result = subprocess.run(
                    ['udevadm', 'info', '--name=' + device], 
                    capture_output=True, text=True
                )
                
                logger.debug(f"Device {device} udev info: {udev_result.stdout[:200]}...")
                
                # Check if device is power button using ACPI device path
                if 'PNP0C0C' in udev_result.stdout or 'Power Button' in udev_result.stdout:
                    devices['power'] = device
                    logger.info(f"✅ Found Power Button: {device}")
                    continue
                
                # Check for our preferred HID controller (only take the first one)
                if '1189:8890' in udev_result.stdout and 'controller' not in devices:
                    devices['controller'] = device
                    logger.info(f"✅ Found HID Controller: {device}")
                    continue
                
                # Check for keyboards (but exclude power button and already found devices)
                if device not in devices.values():
                    has_keyboard = any(indicator in udev_result.stdout for indicator in [
                        'ID_INPUT_KEYBOARD=1',
                        'ID_INPUT_KEY=1'
                    ])
                    
                    is_virtual = any(virtual in udev_result.stdout.lower() for virtual in [
                        'virtual', 'atkbd', 'software', 'at translated set 2 keyboard'
                    ])
                    
                    # Skip power button and system devices
                    is_system = any(system in udev_result.stdout for system in [
                        'PNP0C0C', 'LNXPWRBN', 'Power Button'
                    ])
                    
                    if has_keyboard and not is_virtual and not is_system and 'keyboard' not in devices:
                        device_name = "Unknown"
                        for line in udev_result.stdout.split('\n'):
                            if 'ID_MODEL=' in line:
                                device_name = line.split('=')[1] if '=' in line else "Unknown"
                                break
                        devices['keyboard'] = device
                        logger.info(f"✅ Found Physical Keyboard: {device} ({device_name})")
                    elif has_keyboard and is_virtual and not is_system and 'virtual_keyboard' not in devices:
                        # Fallback to virtual keyboard if no physical found
                        devices['virtual_keyboard'] = device
                        logger.info(f"📍 Found Virtual Keyboard (fallback): {device}")
                    
            except Exception as e:
                logger.debug(f"Error checking {device}: {e}")
                continue
        
        return devices
        
    except Exception as e:
        logger.error(f"Error finding devices: {e}")
        return devices

def shutdown_experimance():
    """Shutdown Experimance safely using the shutdown script."""
    logger.info("🔌 Power button pressed - initiating safe shutdown...")
    
    try:
        # Get the path to the shutdown script
        script_dir = Path(__file__).parent
        shutdown_script = script_dir / "shutdown.sh"
        
        if not shutdown_script.exists():
            logger.error(f"Shutdown script not found at: {shutdown_script}")
            return False
        
        # Run the shutdown script with the experimance project
        shutdown_cmd = [str(shutdown_script), '--project', 'experimance']
        logger.info(f"Running: {' '.join(shutdown_cmd)}")
        
        result = subprocess.run(shutdown_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.error(f"Shutdown script failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            logger.error(f"STDOUT: {result.stdout}")
            return False
        else:
            logger.info("✅ Shutdown script completed successfully")
            logger.info(f"Shutdown output: {result.stdout}")
            return True
            
    except subprocess.TimeoutExpired:
        logger.error("Shutdown script timed out after 30 seconds")
        return False
    except Exception as e:
        logger.error(f"Error running shutdown script: {e}")
        return False

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
        reset_cmd = [str(reset_script), '--project', 'experimance']
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

def listen_for_input(devices):
    """Listen for input from multiple devices (keyboards, controllers, and power button)."""
    logger.info("👂 Listening for input from multiple devices:")
    for device_type, device_path in devices.items():
        logger.info(f"  • {device_type}: {device_path}")
    
    logger.info("🎮 Press any key on any keyboard/controller to reset Experimance...")
    logger.info("🔌 Press power button to safely shutdown Experimance (always active)...")
    logger.info("🚨 SAFETY: Press Ctrl+Alt+E or Escape key to disable listener for admin access")
    
    last_restart = 0
    cooldown = 5  # 5 second cooldown between restarts
    
    # Store processes and threads for cleanup
    processes = []
    threads = []
    shutdown_flag = threading.Event()
    
    def signal_handler(signum, frame):
        logger.info("🛑 Received termination signal, shutting down...")
        shutdown_flag.set()
        
        # Terminate all evtest processes
        for process in processes:
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    process.wait(timeout=2)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create event queue for thread communication
    event_queue = Queue()
    
    def monitor_device_improved(device_path, event_queue, device_type):
        """Monitor a single device and put events in the queue."""
        process = None
        try:
            logger.info(f"🎧 Starting monitor for {device_type}: {device_path}")
            process = subprocess.Popen(
                ['sudo', 'evtest', device_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            processes.append(process)  # Add to cleanup list
            
            if process.stdout is None:
                logger.error(f"Failed to start evtest for {device_path}")
                return
            
            while not shutdown_flag.is_set():
                try:
                    line = process.stdout.readline()
                    if not line:  # EOF
                        break
                    line = line.strip()
                    if line and 'EV_KEY' in line:
                        if not shutdown_flag.is_set():
                            event_queue.put((device_type, device_path, line))
                except:
                    break
                    
        except Exception as e:
            logger.error(f"Error monitoring {device_path}: {e}")
        finally:
            if process:
                try:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=2)
                except:
                    try:
                        process.kill()
                    except:
                        pass
    
    # Start monitoring threads for each device
    for device_type, device_path in devices.items():
        thread = threading.Thread(
            target=monitor_device_improved, 
            args=(device_path, event_queue, device_type),
            name=f"Monitor-{device_type}"
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    try:
        while not shutdown_flag.is_set():
            try:
                # Wait for events from any device
                device_type, device_path, line = event_queue.get(timeout=1)
                
                logger.info(f"📨 Event from {device_type} ({device_path}): {line}")
                
                # Handle power button FIRST - always active for safety
                if 'KEY_POWER' in line and 'value 1' in line:
                    logger.info("🔌 Power button detected")
                    safety.trigger_escape()  # Activate escape mode to prevent further input processing
                    current_time = time.time()
                    if current_time - last_restart > cooldown:
                        logger.info("🔄 Power button triggered safe shutdown")
                        success = shutdown_experimance()
                        if success:
                            logger.info("🎉 Safe shutdown completed successfully!")
                            logger.info("💤 System will now power off...")
                        else:
                            logger.error("❌ Shutdown failed!")
                        last_restart = current_time
                    else:
                        logger.info(f"⏳ Cooldown active ({cooldown - (current_time - last_restart):.1f}s remaining)")
                    continue
                
                # Check safety state before processing other input
                escaped, reason = safety.is_escaped()
                if escaped:
                    logger.debug(f"🔒 Input ignored: {reason}")
                    continue
                
                # Track modifier keys for escape sequence
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
                
                # Handle other key presses (reset functionality)
                # Only accept keyboard keys (KEY_*), not mouse buttons (BTN_*)
                if 'value 1' in line and 'KEY_' in line:
                    current_time = time.time()
                    safety.update_activity()
                    
                    # Skip modifier-only presses
                    if any(modifier in line for modifier in ['CTRL', 'ALT', 'SHIFT', 'KEY_LEFTMETA', 'KEY_RIGHTMETA']):
                        continue
                    
                    if current_time - last_restart > cooldown:
                        logger.info(f"🎹 Key press detected from {device_type}: {line}")
                        logger.info("🔄 Gallery staff triggered exhibit reset")
                        success = restart_experimance()
                        if success:
                            logger.info("🎉 Full reset completed successfully!")
                        else:
                            logger.error("❌ Reset failed!")
                        last_restart = current_time
                    else:
                        logger.info(f"⏳ Cooldown active ({cooldown - (current_time - last_restart):.1f}s remaining)")
                        
            except:
                # Queue timeout or other error - continue loop if not shutting down
                if shutdown_flag.is_set():
                    break
                continue
                
    except KeyboardInterrupt:
        logger.info("🛑 Stopping controller listener...")
        shutdown_flag.set()
    except Exception as e:
        logger.error(f"Error in main event loop: {e}")
        shutdown_flag.set()
    finally:
        logger.info("🧹 Cleaning up monitoring threads...")
        shutdown_flag.set()
        
        # Clean up processes
        for process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=2)
            except:
                try:
                    process.kill()
                except:
                    pass

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
        # Enable debug logging in test mode
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🎮 Starting Experimance input controller (keyboard/controller + power button)...")
    
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
    
    # Find input devices (keyboards, controllers, power button)
    devices = find_input_devices()
    if not devices:
        logger.error("❌ Could not find any suitable input devices")
        logger.info("💡 Try running 'ls /dev/input/' and 'sudo evtest' to identify your devices")
        sys.exit(1)
    
    logger.info(f"📱 Found {len(devices)} input devices:")
    for device_type, device_path in devices.items():
        logger.info(f"  • {device_type}: {device_path}")
    
    logger.info("🔒 SAFETY FEATURES ACTIVE:")
    logger.info(f"  • Escape sequences: Ctrl+Alt+E or Escape key (disables for {safety.escape_timeout_minutes} minutes)")
    logger.info("  • SSH protection: Only works on local console")
    logger.info("  • Physical device priority: Prefers external USB devices")
    logger.info("🎮 INPUT HANDLING:")
    logger.info("  • Any key press: Full exhibit reset (restart services)")
    logger.info("  • Power button: Safe shutdown (stop services, then power off)")
    if 'power' not in devices:
        logger.warning("⚠️  Power button not found - only keyboard/controller reset will work")
        logger.info("💡 NOTE: Power button may require system configuration to generate KEY_POWER events")
        logger.info("         If power button doesn't work, check /etc/systemd/logind.conf HandlePowerKey setting")
    
    # Listen for input from all devices
    listen_for_input(devices)

if __name__ == '__main__':
    main()
