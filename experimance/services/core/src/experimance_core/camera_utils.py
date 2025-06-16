"""
Camera utility functions for Experimance Core Service.

This module provides utility functions for camera diagnostics, reset, and recovery operations.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

import psutil
import pyrealsense2 as rs  # type: ignore

logger = logging.getLogger(__name__)


async def get_camera_diagnostics_async() -> Dict[str, Any]:
    """
    Get comprehensive camera diagnostics (async version).
    
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
        # Run RealSense enumeration in executor to avoid blocking
        def _enumerate_devices():
            try:
                ctx = rs.context() # type: ignore
                devices = ctx.query_devices()
                device_list = []
                
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
                    device_list.append(device_info)
                
                return {
                    'devices': device_list,
                    'device_count': len(devices),
                    'context_created': True
                }
            except Exception as e:
                return {
                    'devices': [],
                    'device_count': 0,
                    'context_created': False,
                    'error': str(e)
                }
        
        # Run in executor with timeout
        device_info = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _enumerate_devices),
            timeout=5.0
        )
        diagnostics.update(device_info)
        
    except asyncio.TimeoutError:
        logger.warning("Camera diagnostics timed out")
        diagnostics['realsense_info'] = {
            'error': 'Timeout during device enumeration',
            'context_created': False
        }
    except Exception as e:
        diagnostics['realsense_info'] = {
            'error': str(e),
            'context_created': False
        }
    
    # Check for processes using camera (async)
    try:
        def _get_camera_processes():
            camera_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
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
            return camera_processes
        
        diagnostics['processes'] = await asyncio.get_event_loop().run_in_executor(None, _get_camera_processes)
        
    except Exception as e:
        diagnostics['processes'] = [{'error': str(e)}]
    
    # Check USB devices (async)
    try:
        proc = await asyncio.create_subprocess_exec(
            'lsusb',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        
        if proc.returncode == 0:
            usb_devices = []
            for line in stdout.decode().split('\n'):
                if 'intel' in line.lower() or '8086' in line or 'realsense' in line.lower():
                    usb_devices.append(line.strip())
            diagnostics['usb_devices'] = usb_devices
        else:
            diagnostics['usb_devices'] = [f'lsusb failed: {stderr.decode()}']
            
    except asyncio.TimeoutError:
        diagnostics['usb_devices'] = ['lsusb timed out']
    except Exception as e:
        diagnostics['usb_devices'] = [f'Error: {e}']
    
    return diagnostics


async def kill_camera_processes_async() -> bool:
    """
    Kill processes that might be holding camera resources (async version).
    
    Returns:
        True if any processes were killed, False otherwise
    """
    def _kill_processes():
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
            return killed
        except Exception as e:
            logger.error(f"Error killing camera processes: {e}")
            return False
    
    killed = await asyncio.get_event_loop().run_in_executor(None, _kill_processes)
    
    if killed:
        await asyncio.sleep(2)  # Wait for processes to terminate
        
    return killed


async def usb_reset_device_async(vendor_id: str = "8086", product_id: Optional[str] = None) -> bool:
    """
    Attempt to reset USB device by vendor/product ID (async version).
    
    Args:
        vendor_id: USB vendor ID (default: Intel)
        product_id: USB product ID (optional)
        
    Returns:
        True if reset was attempted, False otherwise
    """
    try:
        # Find USB device using subprocess
        proc = await asyncio.create_subprocess_exec(
            'lsusb',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        
        if proc.returncode != 0:
            logger.warning("lsusb command failed")
            return False
            
        usb_device = None
        for line in stdout.decode().split('\n'):
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
        def _usb_reset():
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
                return False
            except Exception as e:
                logger.warning(f"USB reset via pyusb failed: {e}")
                return False
        
        return await asyncio.get_event_loop().run_in_executor(None, _usb_reset)
            
    except asyncio.TimeoutError:
        logger.warning("USB device lookup timed out")
        return False
    except Exception as e:
        logger.error(f"USB reset error: {e}")
        return False


async def reset_realsense_camera_async(aggressive: bool = False) -> bool:
    """
    Reset the RealSense camera hardware with multiple strategies (async version).
    
    Args:
        aggressive: If True, use more aggressive reset strategies
    
    Returns:
        True if reset was successful, False otherwise
    """
    logger.info(f"Starting camera reset (aggressive={aggressive})")
    
    try:
        # Step 1: Get diagnostics before reset (with timeout)
        try:
            diagnostics = await asyncio.wait_for(get_camera_diagnostics_async(), timeout=10.0)
            logger.info(f"Found {len(diagnostics['devices'])} RealSense devices")
            
            if len(diagnostics['devices']) == 0:
                logger.warning('No RealSense devices found for reset')
                return False
        except asyncio.TimeoutError:
            logger.warning("Diagnostics timed out, proceeding with reset anyway")
        
        # Step 2: Kill potentially interfering processes
        if aggressive:
            logger.info("Killing potentially interfering processes...")
            try:
                await asyncio.wait_for(kill_camera_processes_async(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Process killing timed out")
        
        # Step 3: Hardware reset (with timeout and cancellation support)
        success = False
        
        def _hardware_reset():
            try:
                ctx = rs.context() # type: ignore
                devices = ctx.query_devices()
                reset_success = False
                
                for i, dev in enumerate(devices):
                    device_name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else f'Device {i}' # type: ignore
                    logger.info(f'Resetting device: {device_name}')
                    
                    try:
                        dev.hardware_reset()
                        logger.info(f'Hardware reset successful for {device_name}')
                        reset_success = True
                        
                    except Exception as e:
                        logger.warning(f'Hardware reset failed for {device_name}: {e}')
                        
                return reset_success
                
            except Exception as e:
                logger.error(f'Camera enumeration failed during reset: {e}')
                return False
        
        try:
            success = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _hardware_reset),
                timeout=15.0
            )
            
            if success:
                # Wait for device to reinitialize (but allow cancellation)
                await asyncio.sleep(3 if not aggressive else 5)
            else:
                logger.error('All hardware resets failed')
                
        except asyncio.TimeoutError:
            logger.error("Hardware reset timed out")
            success = False
        
        # Step 4: USB reset (if aggressive and hardware reset failed)
        if aggressive and not success:
            logger.info("Attempting USB reset...")
            try:
                usb_success = await asyncio.wait_for(usb_reset_device_async(), timeout=10.0)
                if usb_success:
                    logger.info("USB reset completed")
                    await asyncio.sleep(5)  # Wait longer after USB reset
                    success = True
                else:
                    logger.warning("USB reset failed or not available")
            except asyncio.TimeoutError:
                logger.warning("USB reset timed out")
        
        # Step 5: Verify reset by checking device availability
        if success:
            logger.info("Verifying reset by checking device availability...")
            await asyncio.sleep(2)  # Additional wait
            
            def _verify_reset():
                try:
                    ctx = rs.context() # type: ignore
                    devices = ctx.query_devices()
                    return len(devices) > 0
                except Exception as e:
                    logger.warning(f"Reset verification failed: {e}")
                    return False
            
            try:
                verified = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, _verify_reset),
                    timeout=5.0
                )
                
                if verified:
                    logger.info("Reset verification successful")
                    return True
                else:
                    logger.warning("Reset verification failed: no devices found")
                    return False
                    
            except asyncio.TimeoutError:
                logger.warning("Reset verification timed out")
                return False
        
        logger.error("Camera reset failed")
        return False
        
    except asyncio.CancelledError:
        logger.info("Camera reset was cancelled")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during camera reset: {e}")
        return False


# Keep synchronous versions for backward compatibility
def get_camera_diagnostics() -> Dict[str, Any]:
    """Synchronous version - use async version when possible."""
    try:
        return asyncio.run(get_camera_diagnostics_async())
    except Exception as e:
        logger.error(f"Error in sync diagnostics: {e}")
        return {'devices': [], 'error': str(e)}


def kill_camera_processes() -> bool:
    """Synchronous version - use async version when possible."""
    try:
        return asyncio.run(kill_camera_processes_async())
    except Exception as e:
        logger.error(f"Error in sync process killing: {e}")
        return False


def usb_reset_device(vendor_id: str = "8086", product_id: Optional[str] = None) -> bool:
    """Synchronous version - use async version when possible."""
    try:
        return asyncio.run(usb_reset_device_async(vendor_id, product_id))
    except Exception as e:
        logger.error(f"Error in sync USB reset: {e}")
        return False


def reset_realsense_camera(aggressive: bool = False) -> bool:
    """Synchronous version - use async version when possible."""
    try:
        return asyncio.run(reset_realsense_camera_async(aggressive))
    except Exception as e:
        logger.error(f"Error in sync camera reset: {e}")
        return False