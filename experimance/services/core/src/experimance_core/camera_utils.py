"""
Camera utility functions for Experimance Core Service.

This module provides utility functions for camera diagnostics, reset, and recovery operations.
"""
# mypy: disable-error-code="attr-defined"

import logging
import subprocess
import time
from typing import Dict, Any, Optional

import psutil
import pyrealsense2 as rs  # type: ignore

logger = logging.getLogger(__name__)


def get_camera_diagnostics() -> Dict[str, Any]:
    """
    Get comprehensive camera diagnostics.
    
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        'devices': [],
        'processes': [],
        'usb_devices': [],
        'realsense_info': {}
    }
    
    try:
        # RealSense device enumeration
        ctx = rs.context() # type: ignore
        devices = ctx.query_devices()
        
        for i, dev in enumerate(devices):
            device_info = {
                'index': i,
                'name': dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else 'Unknown', # type: ignore
                'serial': dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else 'Unknown', # type: ignore
                'firmware': dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else 'Unknown', # type: ignore
                'product_id': dev.get_info(rs.camera_info.product_id) if dev.supports(rs.camera_info.product_id) else 'Unknown',  # type: ignore
                'usb_type': dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else 'Unknown', # type: ignore
                'sensors': []
            }
            
            # Get sensor information
            for sensor in dev.query_sensors():
                sensor_info = {
                    'name': sensor.get_info(rs.camera_info.name) if sensor.supports(rs.camera_info.name) else 'Unknown', # type: ignore
                    'profiles': []
                }
                
                try:
                    profiles = sensor.get_stream_profiles()
                    for profile in profiles[:5]:  # Limit to first 5 profiles
                        if profile.is_video_stream_profile():
                            vp = profile.as_video_stream_profile()
                            sensor_info['profiles'].append({
                                'stream': str(vp.stream_type()),
                                'format': str(vp.format()),
                                'width': vp.width(),
                                'height': vp.height(),
                                'fps': vp.fps()
                            })
                except Exception as e:
                    sensor_info['error'] = str(e)
                
                device_info['sensors'].append(sensor_info)
                
            diagnostics['devices'].append(device_info)
            
        diagnostics['realsense_info'] = {
            'device_count': len(devices),
            'context_created': True
        }
        
    except Exception as e:
        diagnostics['realsense_info'] = {
            'error': str(e),
            'context_created': False
        }
    
    # Check for processes using camera
    try:
        camera_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if process might be using camera
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                name = proc.info['name'].lower()
                
                if any(keyword in name or keyword in cmdline.lower() for keyword in 
                       ['realsense', 'camera', 'opencv', 'gstreamer', 'v4l2', 'experimance']):
                    camera_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        diagnostics['processes'] = camera_processes
        
    except Exception as e:
        diagnostics['processes'] = [{'error': str(e)}]
    
    # Check USB devices
    try:
        usb_devices = []
        lsusb_output = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if lsusb_output.returncode == 0:
            for line in lsusb_output.stdout.split('\n'):
                if 'intel' in line.lower() or '8086' in line or 'realsense' in line.lower():
                    usb_devices.append(line.strip())
        diagnostics['usb_devices'] = usb_devices
        
    except Exception as e:
        diagnostics['usb_devices'] = [f'Error: {e}']
    
    return diagnostics


def kill_camera_processes() -> bool:
    """
    Kill processes that might be holding camera resources.
    
    Returns:
        True if any processes were killed, False otherwise
    """
    killed = False
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                name = proc.info['name'].lower()
                
                # Be conservative - only kill obvious camera processes
                if any(keyword in name for keyword in ['realsense-viewer', 'intel-realsense']):
                    logger.info(f"Killing camera process: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.terminate()
                    killed = True
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        if killed:
            time.sleep(2)  # Wait for processes to terminate
            
    except Exception as e:
        logger.error(f"Error killing camera processes: {e}")
        
    return killed


def usb_reset_device(vendor_id: str = "8086", product_id: Optional[str] = None) -> bool:
    """
    Attempt to reset USB device by vendor/product ID.
    
    Args:
        vendor_id: USB vendor ID (default: Intel)
        product_id: USB product ID (optional)
        
    Returns:
        True if reset was attempted, False otherwise
    """
    try:
        # Find USB device
        lsusb_output = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if lsusb_output.returncode != 0:
            logger.warning("lsusb command failed")
            return False
            
        usb_device = None
        for line in lsusb_output.stdout.split('\n'):
            if vendor_id in line:
                if product_id is None or product_id in line:
                    # Extract bus and device numbers
                    parts = line.split()
                    if len(parts) >= 4:
                        bus = parts[1]
                        device = parts[3].rstrip(':')
                        usb_device = f"/dev/bus/usb/{bus}/{device}"
                        break
        
        if not usb_device:
            logger.warning(f"USB device with vendor ID {vendor_id} not found")
            return False
            
        # Attempt USB reset using python usb library if available
        try:
            import usb.core
            import usb.util
            
            devices = usb.core.find(find_all=True, idVendor=int(vendor_id, 16))
            if devices:
                for device in devices:
                    if product_id is None or device.idProduct == int(product_id, 16): # type: ignore
                        logger.info(f"Resetting USB device: vendor={vendor_id}, product={device.idProduct:04x}") # type: ignore
                        device.reset() # type: ignore
                        return True
            return False  
        except ImportError:
            logger.warning("pyusb not available for USB reset")
        except Exception as e:
            logger.warning(f"USB reset via pyusb failed: {e}")
            
        # Fallback: try to unbind/rebind driver
        try:
            # This is more complex and requires root privileges
            logger.info("USB reset attempted but requires additional privileges")
            return False
            
        except Exception as e:
            logger.warning(f"USB driver reset failed: {e}")
            
    except Exception as e:
        logger.error(f"USB reset error: {e}")
        
    return False


def reset_realsense_camera(aggressive: bool = False) -> bool:
    """
    Reset the RealSense camera hardware with multiple strategies.
    
    Args:
        aggressive: If True, use more aggressive reset strategies
    
    Returns:
        True if reset was successful, False otherwise
    """
    logger.info(f"Starting camera reset (aggressive={aggressive})")
    
    # Step 1: Get diagnostics before reset
    diagnostics = get_camera_diagnostics()
    logger.info(f"Found {len(diagnostics['devices'])} RealSense devices")
    
    if len(diagnostics['devices']) == 0:
        logger.warning('No RealSense devices found for reset')
        return False
    
    # Step 2: Kill potentially interfering processes
    if aggressive:
        logger.info("Killing potentially interfering processes...")
        kill_camera_processes()
    
    # Step 3: Hardware reset
    success = False
    try:
        ctx = rs.context() # type: ignore
        devices = ctx.query_devices()
        
        for i, dev in enumerate(devices):
            device_name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else f'Device {i}' # type: ignore
            logger.info(f'Resetting device: {device_name}')
            
            try:
                dev.hardware_reset()
                logger.info(f'Hardware reset successful for {device_name}')
                success = True
                
                # Wait longer for device to reinitialize
                time.sleep(3 if not aggressive else 5)
                
            except Exception as e:
                logger.warning(f'Hardware reset failed for {device_name}: {e}')
                
        if not success:
            logger.error('All hardware resets failed')
            
    except Exception as e:
        logger.error(f'Camera enumeration failed during reset: {e}')
        
    # Step 4: USB reset (if aggressive and hardware reset failed)
    if aggressive and not success:
        logger.info("Attempting USB reset...")
        usb_success = usb_reset_device()
        if usb_success:
            logger.info("USB reset completed")
            time.sleep(5)  # Wait longer after USB reset
            success = True
        else:
            logger.warning("USB reset failed or not available")
    
    # Step 5: Verify reset by checking device availability
    if success:
        logger.info("Verifying reset by checking device availability...")
        time.sleep(2)  # Additional wait
        
        try:
            ctx = rs.context() # type: ignore
            devices = ctx.query_devices()
            if len(devices) > 0:
                logger.info(f"Reset verification successful: {len(devices)} devices available")
                return True
            else:
                logger.warning("Reset verification failed: no devices found")
                return False
                
        except Exception as e:
            logger.warning(f"Reset verification failed: {e}")
            return False
    
    logger.error("Camera reset failed")
    return False
