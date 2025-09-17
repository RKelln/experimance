#!/usr/bin/env python3
"""
Matter Device Scheduler for Experimance

Schedules Matter device control commands using cron-like syntax.
Supports smart plugs, lights, and other Matter devices via chip-tool.
"""

import asyncio
import logging
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Union
import argparse
import signal

import tomllib
from croniter import croniter

# Add experimance_common to path for logging
sys.path.append(str(Path(__file__).parent.parent / "libs" / "common" / "src"))

try:
    from experimance_common.logger import setup_logging
    from experimance_common.config import load_config
except ImportError:
    # Fallback if experimance_common not available
    def setup_logging(level="INFO", **kwargs):
        logging.basicConfig(level=getattr(logging, level), 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def load_config(*args, **kwargs):
        return {}


logger = logging.getLogger(__name__)


class MatterDeviceScheduler:
    """Schedules and executes Matter device commands using chip-tool."""
    
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.config = {}
        self.running = False
        self.tasks = []
        
    def load_config(self) -> bool:
        """Load configuration from TOML file."""
        try:
            with open(self.config_file, 'rb') as f:
                self.config = tomllib.load(f)
            logger.info(f"Loaded configuration from {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    async def execute_command(self, device_id: int, command: str, endpoint: int = 1) -> bool:
        """Execute a chip-tool command for a Matter device."""
        try:
            # Build chip-tool command
            if command in ['on', 'off', 'toggle']:
                cmd = ['chip-tool', 'onoff', command, str(device_id), str(endpoint)]
            else:
                logger.error(f"Unsupported command: {command}")
                return False
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Command successful: {command} on device {device_id}")
                return True
            else:
                logger.error(f"Command failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return False
    
    def get_next_schedule_time(self, cron_expr: str) -> Optional[datetime]:
        """Get the next scheduled time for a cron expression."""
        try:
            cron = croniter(cron_expr, datetime.now())
            return cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Invalid cron expression '{cron_expr}': {e}")
            return None
    
    async def schedule_task(self, schedule_config: Dict):
        """Schedule a single task based on configuration."""
        name = schedule_config.get('name', 'unnamed')
        cron_expr = schedule_config.get('schedule')
        device_id = schedule_config.get('device_id')
        command = schedule_config.get('command')
        endpoint = schedule_config.get('endpoint', 1)
        enabled = schedule_config.get('enabled', True)
        
        if not enabled:
            logger.info(f"Schedule '{name}' is disabled, skipping")
            return
        
        if not all([cron_expr, device_id, command]):
            logger.error(f"Invalid schedule config for '{name}': missing required fields")
            return
        
        logger.info(f"Setting up schedule '{name}': {cron_expr}")
        
        while self.running:
            try:
                next_time = self.get_next_schedule_time(cron_expr)
                if not next_time:
                    break
                
                # Wait until scheduled time
                wait_seconds = (next_time - datetime.now()).total_seconds()
                if wait_seconds > 0:
                    logger.debug(f"Schedule '{name}' waiting {wait_seconds:.1f}s until {next_time}")
                    await asyncio.sleep(wait_seconds)
                
                if not self.running:
                    break
                
                # Execute the command
                logger.info(f"Executing scheduled task '{name}'")
                await self.execute_command(device_id, command, endpoint)
                
            except asyncio.CancelledError:
                logger.info(f"Schedule '{name}' cancelled")
                break
            except Exception as e:
                logger.error(f"Error in schedule '{name}': {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def run_sequence(self, sequence_config: Dict):
        """Run a sequence of commands with delays."""
        name = sequence_config.get('name', 'unnamed_sequence')
        steps = sequence_config.get('steps', [])
        enabled = sequence_config.get('enabled', True)
        
        if not enabled:
            logger.info(f"Sequence '{name}' is disabled, skipping")
            return
        
        logger.info(f"Starting sequence '{name}' with {len(steps)} steps")
        
        for i, step in enumerate(steps):
            try:
                device_id = step.get('device_id')
                command = step.get('command')
                endpoint = step.get('endpoint', 1)
                delay = step.get('delay', 0)
                
                if device_id and command:
                    logger.info(f"Sequence '{name}' step {i+1}: {command} on device {device_id}")
                    await self.execute_command(device_id, command, endpoint)
                
                if delay > 0 and i < len(steps) - 1:  # Don't delay after last step
                    logger.debug(f"Sequence '{name}' waiting {delay}s")
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error in sequence '{name}' step {i+1}: {e}")
    
    async def start(self):
        """Start the scheduler."""
        if not self.load_config():
            return
        
        self.running = True
        logger.info("Matter device scheduler starting")
        
        try:
            # Start scheduled tasks
            schedules = self.config.get('schedules', [])
            for schedule_config in schedules:
                task = asyncio.create_task(self.schedule_task(schedule_config))
                self.tasks.append(task)
            
            # Run one-time sequences if specified
            sequences = self.config.get('sequences', [])
            for sequence_config in sequences:
                if sequence_config.get('run_once', False):
                    task = asyncio.create_task(self.run_sequence(sequence_config))
                    self.tasks.append(task)
            
            logger.info(f"Started {len(self.tasks)} scheduled tasks")
            
            # Wait for all tasks or termination
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            else:
                logger.warning("No schedules configured, scheduler idle")
                while self.running:
                    await asyncio.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping scheduler...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Scheduler stopped")


def signal_handler(signum, frame, scheduler):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    asyncio.create_task(scheduler.stop())


async def main():
    parser = argparse.ArgumentParser(description="Matter Device Scheduler")
    parser.add_argument('config', type=Path, help='Configuration file path')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')
    parser.add_argument('--test', action='store_true',
                       help='Test configuration and exit')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, service_name="matter_scheduler")
    
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    scheduler = MatterDeviceScheduler(args.config)
    
    if args.test:
        # Test configuration
        if scheduler.load_config():
            logger.info("Configuration is valid")
            print("Configuration test passed!")
            return 0
        else:
            logger.error("Configuration test failed")
            return 1
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(
            sig, lambda s=sig: asyncio.create_task(scheduler.stop())
        )
    
    try:
        await scheduler.start()
        return 0
    except Exception as e:
        logger.error(f"Scheduler failed: {e}")
        return 1


if __name__ == '__main__':
    exit(asyncio.run(main()))