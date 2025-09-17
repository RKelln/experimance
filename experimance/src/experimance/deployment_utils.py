"""Deployment configuration parser for distributed log access.

This module parses deployment.toml files to discover which services run on which machines,
enabling the timeline CLI to automatically fetch logs from the appropriate remote locations.
"""
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import toml

logger = logging.getLogger(__name__)

@dataclass
class MachineConfig:
    """Configuration for a machine in the deployment."""
    hostname: str
    services: List[str]
    platform: str
    mode: str
    user: str
    ssh_hostname: Optional[str] = None  # Override hostname for SSH connections
    
    def get_ssh_hostname(self) -> str:
        """Get the hostname to use for SSH connections."""
        return self.ssh_hostname or self.hostname

@dataclass
class ServiceConfig:
    """Configuration for a service override."""
    module_name: Optional[str] = None

@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    project: str
    version: str
    machines: Dict[str, MachineConfig]
    services: Dict[str, ServiceConfig]

@dataclass
class LogLocation:
    """Location of log directories for a specific log type."""
    machine_id: str
    hostname: str
    user: str
    remote_path: str
    local_cache_path: Optional[Path] = None

class DeploymentParser:
    """Parser for deployment.toml files."""
    
    def __init__(self, deployment_file: Path):
        """Initialize with deployment file path."""
        self.deployment_file = deployment_file
        self.config: Optional[DeploymentConfig] = None
        
    def parse(self) -> DeploymentConfig:
        """Parse the deployment.toml file."""
        if not self.deployment_file.exists():
            raise FileNotFoundError(f"Deployment file not found: {self.deployment_file}")
        
        try:
            with open(self.deployment_file, 'r') as f:
                data = toml.load(f)
            
            # Parse deployment metadata
            deployment_data = data.get('deployment', {})
            project = deployment_data.get('project', 'unknown')
            version = deployment_data.get('version', '1.0.0')
            
            # Parse machines
            machines_data = data.get('machines', {})
            machines = {}
            
            for machine_id, machine_data in machines_data.items():
                machines[machine_id] = MachineConfig(
                    hostname=machine_data.get('hostname', machine_id),
                    services=machine_data.get('services', []),
                    platform=machine_data.get('platform', 'linux'),
                    mode=machine_data.get('mode', 'prod'),
                    user=machine_data.get('user', 'experimance'),
                    ssh_hostname=machine_data.get('ssh_hostname')
                )
            
            # Parse service overrides
            services_data = data.get('services', {})
            services = {}
            
            for service_name, service_data in services_data.items():
                services[service_name] = ServiceConfig(
                    module_name=service_data.get('module_name')
                )
            
            self.config = DeploymentConfig(
                project=project,
                version=version,
                machines=machines,
                services=services
            )
            
            logger.info(f"Parsed deployment config for project '{project}' with {len(machines)} machines")
            return self.config
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse deployment file {self.deployment_file}: {e}")
    
    def get_log_locations(self, base_log_dir: str = "/var/log/experimance") -> Dict[str, LogLocation]:
        """
        Determine where transcript and prompt logs are located based on services.
        
        Returns a dict mapping log type ('transcripts', 'prompts') to LogLocation.
        """
        if not self.config:
            raise RuntimeError("Must call parse() first")
        
        locations = {}
        
        # Find which machine runs the agent service (transcripts)
        for machine_id, machine in self.config.machines.items():
            if 'agent' in machine.services:
                # Determine the actual log directory based on deployment mode
                log_dir = self._get_machine_log_dir(machine, base_log_dir)
                locations['transcripts'] = LogLocation(
                    machine_id=machine_id,
                    hostname=machine.get_ssh_hostname(),
                    user=machine.user,
                    remote_path=f"{log_dir}/transcripts"
                )
                break
        
        # Find which machine runs the core service (prompts)  
        for machine_id, machine in self.config.machines.items():
            if 'core' in machine.services:
                # Determine the actual log directory based on deployment mode
                log_dir = self._get_machine_log_dir(machine, base_log_dir)
                locations['prompts'] = LogLocation(
                    machine_id=machine_id,
                    hostname=machine.get_ssh_hostname(),
                    user=machine.user,
                    remote_path=f"{log_dir}/prompts"
                )
                break
        
        logger.info(f"Found log locations: {list(locations.keys())}")
        for log_type, location in locations.items():
            logger.debug(f"  {log_type}: {location.user}@{location.hostname}:{location.remote_path}")
        
        return locations
    
    def _get_machine_log_dir(self, machine: MachineConfig, base_log_dir: str) -> str:
        """
        Determine the actual log directory for a machine based on its configuration.
        
        Returns the first candidate that will be checked. The actual discovery happens
        in sync_remote_logs which checks all possible locations.
        """
        # Always return production path as the primary candidate
        # RemoteLogAccess will check multiple locations automatically
        return base_log_dir

class RemoteLogAccess:
    """Handles access to remote log directories via SSH."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize with optional cache directory."""
        self.cache_dir = cache_dir or Path.home() / ".experimance" / "log_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using cache directory: {self.cache_dir}")
    
    def sync_remote_logs(self, location: LogLocation, patterns: Optional[List[str]] = None, project_name: str = "default") -> Path:
        """
        Sync remote log files to local cache via rsync over SSH.
        
        Automatically discovers the correct log directory by checking multiple possible locations:
        1. /var/log/experimance/transcripts (or prompts) - production location
        2. ~/experimance/logs/transcripts (or prompts) - development location
        3. ~/logs/transcripts (or prompts) - alternative development location
        
        Args:
            location: Remote log location configuration
            patterns: Optional list of file patterns to sync (e.g., ['*.jsonl'])
            project_name: Project name for cache organization
        
        Returns:
            Path to local cache directory containing synced files
        """
        if patterns is None:
            patterns = ['*.jsonl']
        
        # Create local cache path for this project/machine/location
        cache_path = self.cache_dir / project_name / location.machine_id / Path(location.remote_path).name
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Update location with cache path
        location.local_cache_path = cache_path
        
        logger.info(f"Discovering log location on {location.user}@{location.hostname}")
        logger.debug(f"  Patterns: {patterns}")
        logger.debug(f"  Cache: {cache_path}")
        
        try:
            # Test SSH connectivity first
            if not self._test_ssh_connection(location):
                logger.warning(f"SSH connection to {location.hostname} failed - using cached data only")
                return cache_path
            
            # Get the log type from the remote path (transcripts or prompts)
            log_type = Path(location.remote_path).name
            
            # Define candidate directories to check (in priority order)
            candidates = [
                f"/var/log/experimance/{log_type}",  # Production
                f"/home/{location.user}/experimance/logs/{log_type}",  # Dev - full path
                f"experimance/logs/{log_type}",  # Dev - relative to home
                f"logs/{log_type}",  # Dev - project root
            ]
            
            # Find the first directory that has files
            found_location = None
            for candidate in candidates:
                if self._check_remote_directory(location, candidate, patterns):
                    found_location = candidate
                    logger.info(f"Found logs at {location.hostname}:{candidate}")
                    break
            
            if not found_location:
                logger.warning(f"No log directories found on {location.hostname} for patterns {patterns}")
                return cache_path
            
            # Update the location to use the discovered path
            original_path = location.remote_path
            location.remote_path = found_location
            
            # Sync each pattern from the discovered location
            total_synced = 0
            for pattern in patterns:
                remote_pattern = f"{location.user}@{location.hostname}:{found_location}/{pattern}"
                
                # Use rsync to sync files
                cmd = [
                    'rsync',
                    '-avz',  # archive, verbose, compress
                    '--timeout=10',  # 10 second timeout
                    remote_pattern,
                    str(cache_path) + '/'
                ]
                
                logger.debug(f"Running: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    synced_files = [f for f in cache_path.glob(pattern) if f.is_file()]
                    pattern_count = len(synced_files)
                    total_synced += pattern_count
                    logger.debug(f"  Synced {pattern_count} files matching {pattern}")
                else:
                    if "No such file or directory" in result.stderr:
                        logger.debug(f"No {pattern} files found in {found_location}")
                    elif "no matches found" in result.stderr:
                        logger.debug(f"No {pattern} files found in {found_location}")
                    else:
                        logger.warning(f"rsync failed for pattern {pattern}: {result.stderr}")
            
            # Restore original path for logging
            location.remote_path = original_path
            
            # List final cached files
            all_files = list(cache_path.glob('*.jsonl'))
            logger.info(f"Synced {total_synced} files to cache (total: {len(all_files)} files)")
            
            return cache_path
            
        except subprocess.TimeoutExpired:
            logger.warning(f"rsync timeout to {location.hostname} - using cached data")
            return cache_path
        except Exception as e:
            logger.error(f"Failed to sync logs from {location.hostname}: {e}")
            return cache_path
    
    def _test_ssh_connection(self, location: LogLocation, timeout: int = 5) -> bool:
        """Test SSH connectivity to a remote machine."""
        try:
            cmd = [
                'ssh',
                '-o', 'ConnectTimeout=5',
                '-o', 'BatchMode=yes',  # No password prompts
                f"{location.user}@{location.hostname}",
                'echo "connected"'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0 and 'connected' in result.stdout:
                logger.debug(f"SSH connection to {location.hostname} successful")
                return True
            else:
                logger.debug(f"SSH connection to {location.hostname} failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.debug(f"SSH test to {location.hostname} failed: {e}")
            return False
    
    def _check_remote_directory(self, location: LogLocation, remote_path: str, patterns: List[str]) -> bool:
        """
        Check if a remote directory exists and contains files matching any of the patterns.
        
        Args:
            location: LogLocation with connection details
            remote_path: Remote directory path to check
            patterns: List of file patterns to look for
        
        Returns:
            True if directory exists and contains matching files, False otherwise
        """
        try:
            # Construct a command that checks for files matching any pattern
            pattern_checks = []
            for pattern in patterns:
                pattern_checks.append(f"ls {remote_path}/{pattern} 2>/dev/null")
            
            # Use || to check multiple patterns, exit with success if any pattern matches
            check_cmd = " || ".join(pattern_checks)
            full_cmd = f"({check_cmd}) | head -1"  # Just need to know if at least one file exists
            
            cmd = [
                'ssh',
                '-o', 'ConnectTimeout=5',
                '-o', 'BatchMode=yes',
                f"{location.user}@{location.hostname}",
                full_cmd
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # If we got any output, there are files matching the patterns
            has_files = result.returncode == 0 and result.stdout.strip()
            
            if has_files:
                logger.debug(f"Found files in {location.hostname}:{remote_path}")
            else:
                logger.debug(f"No matching files in {location.hostname}:{remote_path}")
            
            return bool(has_files)
            
        except Exception as e:
            logger.debug(f"Error checking remote directory {remote_path}: {e}")
            return False
    
    def list_remote_files(self, location: LogLocation, pattern: str = "*.jsonl") -> List[str]:
        """List files matching pattern on remote machine."""
        try:
            cmd = [
                'ssh',
                '-o', 'ConnectTimeout=5',
                f"{location.user}@{location.hostname}",
                f"ls {location.remote_path}/{pattern} 2>/dev/null || true"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                return files
            else:
                logger.warning(f"Failed to list remote files: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing remote files: {e}")
            return []

def find_deployment_file(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find deployment.toml file by searching up the directory tree.
    Respects PROJECT_ENV to look for project-specific deployment files first.
    
    Args:
        start_path: Starting directory (defaults to current working directory)
    
    Returns:
        Path to deployment.toml file or None if not found
    """
    if start_path is None:
        start_path = Path.cwd()
    
    # Search up the directory tree
    current = start_path.resolve()
    
    while current != current.parent:
        # Check for deployment.toml in current directory
        deployment_file = current / "deployment.toml"
        if deployment_file.exists():
            logger.debug(f"Found deployment.toml at {deployment_file}")
            return deployment_file
        
        # Check for deployment.toml in projects subdirectories
        projects_dir = current / "projects"
        if projects_dir.exists():
            # First try the current project (from PROJECT_ENV)
            project_env = os.environ.get("PROJECT_ENV")
            if project_env:
                project_deployment = projects_dir / project_env / "deployment.toml"
                if project_deployment.exists():
                    logger.debug(f"Found project-specific deployment.toml at {project_deployment}")
                    return project_deployment
            
            # Fall back to any deployment.toml in projects directories
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    deployment_file = project_dir / "deployment.toml"
                    if deployment_file.exists():
                        logger.debug(f"Found deployment.toml at {deployment_file}")
                        return deployment_file
        
        current = current.parent
    
    logger.debug("No deployment.toml file found")
    return None