"""
Mock audience detector for testing fire_agent service without a camera.

This detector can be controlled manually via keyboard input in the terminal
or through simple file-based commands.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MockAudienceDetector:
    """
    Mock audience detector that can be controlled manually for testing.
    
    Control methods:
    1. Keyboard: Press 'p' for presence, 'a' for absence, 'q' to quit
    2. File-based: Create/delete files in /tmp/mock_detector/
       - Create 'present' file to signal presence
       - Create 'absent' file to signal absence  
       - Create 'count_N' file to set person count to N
    """
    
    def __init__(
        self,
        control_method: str = "keyboard",  # "keyboard" or "file"
        control_dir: str = "/tmp/mock_detector",
        initial_state: bool = False,
        initial_count: int = 0
    ):
        """
        Initialize mock detector.
        
        Args:
            control_method: "keyboard" or "file" 
            control_dir: Directory for file-based control
            initial_state: Initial presence state
            initial_count: Initial person count
        """
        self.control_method = control_method
        self.control_dir = Path(control_dir)
        
        # State
        self._current_state = initial_state
        self._current_person_count = initial_count
        
        # Stats tracking
        self._total_checks = 0
        self._state_changes = 0
        self._connection_errors = 0
        self._last_reading = initial_state
        self._reading_count = 0
        
        # Control
        self._running = False
        self._control_task = None
        
        logger.info(f"Mock detector initialized - control: {control_method}, "
                   f"initial state: {initial_state}, count: {initial_count}")
    
    async def start(self):
        """Start the mock detector."""
        self._running = True
        
        # Setup control directory for file-based control
        if self.control_method == "file":
            self.control_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean up any existing control files
            for file in self.control_dir.glob("*"):
                file.unlink(missing_ok=True)
                
            logger.info(f"File-based control enabled in: {self.control_dir}")
            logger.info("Control files:")
            logger.info("  - Create 'present' file to signal presence")
            logger.info("  - Create 'absent' file to signal absence") 
            logger.info("  - Create 'count_N' file to set person count to N")
            
        elif self.control_method == "keyboard":
            logger.info("Keyboard control enabled:")
            logger.info("  - Press 'p' + Enter for presence")
            logger.info("  - Press 'a' + Enter for absence")
            logger.info("  - Press 'c' + number + Enter to set person count")
            logger.info("  - Press 'q' + Enter to quit")
            
            # Start keyboard input handler
            self._control_task = asyncio.create_task(self._keyboard_control_loop())
        
        logger.info("Mock audience detector started")
    
    async def stop(self):
        """Stop the mock detector."""
        self._running = False
        
        if self._control_task:
            self._control_task.cancel()
            try:
                await self._control_task
            except asyncio.CancelledError:
                pass
            
        # Clean up control directory
        if self.control_method == "file" and self.control_dir.exists():
            for file in self.control_dir.glob("*"):
                file.unlink(missing_ok=True)
            
        logger.info("Mock audience detector stopped")
    
    async def check_audience_present(self) -> bool:
        """
        Check if audience is present.
        
        Returns:
            bool: True if audience detected
        """
        if not self._running:
            return False
            
        self._total_checks += 1
        
        # Check for control updates
        if self.control_method == "file":
            await self._check_file_controls()
        
        # Track reading changes
        if self._current_state != self._last_reading:
            self._state_changes += 1
            self._reading_count = 1
        else:
            self._reading_count += 1
            
        self._last_reading = self._current_state
        
        logger.debug(f"Mock detection check #{self._total_checks}: "
                    f"state={self._current_state}, count={self._current_person_count}")
        
        return self._current_state
    
    async def _check_file_controls(self):
        """Check for file-based control commands."""
        if not self.control_dir.exists():
            return
            
        try:
            for file in self.control_dir.iterdir():
                if file.is_file():
                    filename = file.name
                    
                    if filename == "present":
                        if not self._current_state:
                            logger.info("File control: Setting presence to True")
                            self._current_state = True
                            if self._current_person_count == 0:
                                self._current_person_count = 1
                        file.unlink()
                        
                    elif filename == "absent":
                        if self._current_state:
                            logger.info("File control: Setting presence to False")
                            self._current_state = False
                            self._current_person_count = 0
                        file.unlink()
                        
                    elif filename.startswith("count_"):
                        try:
                            count_str = filename.split("_", 1)[1]
                            count = int(count_str)
                            self._current_person_count = max(0, count)
                            self._current_state = count > 0
                            logger.info(f"File control: Setting person count to {count}")
                        except (ValueError, IndexError):
                            logger.warning(f"Invalid count file: {filename}")
                        file.unlink()
                        
        except Exception as e:
            logger.error(f"Error checking file controls: {e}")
    
    async def _keyboard_control_loop(self):
        """Handle keyboard input for manual control."""
        try:
            while self._running:
                try:
                    # Use asyncio to check for input without blocking
                    # Note: This is a simple implementation - in production you'd want
                    # a more sophisticated async input handler
                    await asyncio.sleep(0.1)
                    
                    # Check if stdin has data (non-blocking)
                    import sys
                    import select
                    
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline().strip().lower()
                        
                        if line == 'p':
                            logger.info("Keyboard control: Setting presence to True")
                            self._current_state = True
                            if self._current_person_count == 0:
                                self._current_person_count = 1
                                
                        elif line == 'a':
                            logger.info("Keyboard control: Setting presence to False")
                            self._current_state = False
                            self._current_person_count = 0
                            
                        elif line.startswith('c'):
                            try:
                                count_str = line[1:].strip()
                                if count_str:
                                    count = int(count_str)
                                    self._current_person_count = max(0, count)
                                    self._current_state = count > 0
                                    logger.info(f"Keyboard control: Setting person count to {count}")
                            except ValueError:
                                logger.warning(f"Invalid count: {line}")
                                
                        elif line == 'q':
                            logger.info("Keyboard control: Quit requested")
                            break
                            
                        elif line == 'h' or line == 'help':
                            logger.info("Commands: 'p' (present), 'a' (absent), 'c<N>' (count), 'q' (quit), 'h' (help)")
                            
                except Exception as e:
                    logger.error(f"Error in keyboard control: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Keyboard control loop error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics matching ReolinkDetector interface."""
        return {
            'current_state': self._current_state,
            'total_checks': self._total_checks,
            'connection_errors': self._connection_errors,
            'state_changes': self._state_changes,
            'hysteresis_present': 1,  # Mock values
            'hysteresis_absent': 1,
            'consecutive_readings': self._reading_count,
            'last_reading': self._last_reading,
            'hybrid_mode': False,
            'detection_mode': 'mock',
            'mode_switches': 0,
            'yolo_absent_count': 0,
            'yolo_absent_threshold': 0,
            'yolo_available': False,
            'current_person_count': self._current_person_count
        }
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics (alias for get_stats for compatibility)."""
        stats = self.get_stats()
        stats['current_state'] = 'present' if stats['current_state'] else 'absent'
        return stats


class FileControlHelper:
    """Helper class to control MockAudienceDetector from another process."""
    
    def __init__(self, control_dir: str = "/tmp/mock_detector"):
        self.control_dir = Path(control_dir)
        
    def set_present(self):
        """Signal presence."""
        self.control_dir.mkdir(parents=True, exist_ok=True)
        (self.control_dir / "present").touch()
        print("Set presence: True")
        
    def set_absent(self):
        """Signal absence."""
        self.control_dir.mkdir(parents=True, exist_ok=True)
        (self.control_dir / "absent").touch()
        print("Set presence: False")
        
    def set_count(self, count: int):
        """Set person count."""
        self.control_dir.mkdir(parents=True, exist_ok=True)
        (self.control_dir / f"count_{count}").touch()
        print(f"Set person count: {count}")


if __name__ == "__main__":
    """Simple CLI for controlling mock detector."""
    import sys
    
    helper = FileControlHelper()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python mock_detector.py present")
        print("  python mock_detector.py absent") 
        print("  python mock_detector.py count <number>")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    
    if command == "present":
        helper.set_present()
    elif command == "absent":
        helper.set_absent()
    elif command == "count" and len(sys.argv) > 2:
        try:
            count = int(sys.argv[2])
            helper.set_count(count)
        except ValueError:
            print("Error: count must be a number")
            sys.exit(1)
    else:
        print("Unknown command:", command)
        sys.exit(1)
