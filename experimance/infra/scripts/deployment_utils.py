#!/usr/bin/env python3
"""
Deployment utilities for multi-machine Experimance deployments.

This module provides functions to read deployment configuration and determine
which services should run on the current machine.
"""

import os
import sys
import socket
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Optional, Any

import tomllib


def get_current_hostname() -> str:
    """Get the current machine's hostname."""
    return socket.gethostname()


def get_current_platform() -> str:
    """Get the current platform (linux/macos)."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system


def find_deployment_config(project: str, repo_dir: Path) -> Optional[Path]:
    """Find the deployment configuration file for a project."""
    config_path = repo_dir / "projects" / project / "deployment.toml"
    if config_path.exists():
        return config_path
    return None


def load_deployment_config(config_path: Path) -> Dict[str, Any]:
    """Load deployment configuration from TOML file."""
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Error loading deployment config from {config_path}: {e}", file=sys.stderr)
        sys.exit(1)


def get_services_for_machine(
    config: Dict[str, Any], 
    hostname_override: Optional[str] = None
) -> List[str]:
    """
    Determine which services should run on the current machine.
    
    Args:
        config: Deployment configuration dictionary
        hostname_override: Optional hostname override for testing/multi-machine deployment
        
    Returns:
        List of service names that should run on this machine
    """
    current_hostname = hostname_override or get_current_hostname()
    current_platform = get_current_platform()
    
    machines = config.get("machines", {})
    
    # Try exact hostname match first
    for machine_id, machine_config in machines.items():
        if machine_config.get("hostname") == current_hostname:
            return machine_config.get("services", [])
    
    # Fall back to platform match
    for machine_id, machine_config in machines.items():
        if machine_config.get("platform") == current_platform:
            return machine_config.get("services", [])
    
    # If no match found, return empty list
    print(f"Warning: No machine configuration found for hostname '{current_hostname}' or platform '{current_platform}'", file=sys.stderr)
    return []


def get_machine_mode(
    config: Dict[str, Any], 
    hostname_override: Optional[str] = None
) -> str:
    """Get the deployment mode (dev/prod) for the current machine."""
    current_hostname = hostname_override or get_current_hostname()
    current_platform = get_current_platform()
    
    machines = config.get("machines", {})
    
    # Try exact hostname match first
    for machine_id, machine_config in machines.items():
        if machine_config.get("hostname") == current_hostname:
            return machine_config.get("mode", "prod")
    
    # Fall back to platform match
    for machine_id, machine_config in machines.items():
        if machine_config.get("platform") == current_platform:
            return machine_config.get("mode", "prod")
    
    # Default to prod
    return "prod"


def get_machine_user(
    config: Dict[str, Any], 
    hostname_override: Optional[str] = None
) -> Optional[str]:
    """Get the system user for the current machine from deployment config."""
    current_hostname = hostname_override or get_current_hostname()
    current_platform = get_current_platform()
    
    machines = config.get("machines", {})
    
    # Try exact hostname match first
    for machine_id, machine_config in machines.items():
        if machine_config.get("hostname") == current_hostname:
            return machine_config.get("user")
    
    # Fall back to platform match
    for machine_id, machine_config in machines.items():
        if machine_config.get("platform") == current_platform:
            return machine_config.get("user")
    
    # No user specified in config
    return None


def get_service_module_name(
    config: Dict[str, Any], 
    service_name: str, 
    project: str
) -> str:
    """
    Get the module name for a service from deployment config.
    
    Args:
        config: Deployment configuration dictionary
        service_name: Name of the service (e.g., 'core', 'agent')
        project: Project name (e.g., 'fire')
        
    Returns:
        Module name to use for the service
    """
    services_config = config.get("services", {})
    service_config = services_config.get(service_name, {})
    
    # Check if there's a custom module name
    module_name = service_config.get("module_name")
    if module_name:
        return module_name
    
    # Fall back to default naming convention
    if service_name in ["core", "agent"]:
        return f"{project}_{service_name}"
    else:
        return f"experimance_{service_name}"


def get_services_with_modules(
    config: Dict[str, Any], 
    project: str,
    hostname_override: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Get services for current machine with their module names.
    
    Returns:
        List of dicts with 'service' and 'module' keys
    """
    services = get_services_for_machine(config, hostname_override)
    
    result = []
    for service in services:
        module_name = get_service_module_name(config, service, project)
        result.append({
            'service': service,
            'module': module_name
        })
    
    return result


def print_deployment_info(config: Dict[str, Any], hostname_override: Optional[str] = None) -> None:
    """Print deployment information for debugging."""
    current_hostname = hostname_override or get_current_hostname()
    current_platform = get_current_platform()
    
    services = get_services_for_machine(config, hostname_override)
    mode = get_machine_mode(config, hostname_override)
    user = get_machine_user(config, hostname_override)
    
    print(f"Deployment Information:")
    print(f"  Current hostname: {current_hostname}")
    print(f"  Current platform: {current_platform}")
    print(f"  Deployment mode: {mode}")
    print(f"  System user: {user or 'Not specified'}")
    print(f"  Services for this machine: {', '.join(services) if services else 'None'}")
    
    machines = config.get("machines", {})
    print(f"\nAvailable machine configurations:")
    for machine_id, machine_config in machines.items():
        hostname = machine_config.get("hostname", "unknown")
        platform = machine_config.get("platform", "unknown")
        machine_services = machine_config.get("services", [])
        machine_mode = machine_config.get("mode", "prod")
        machine_user = machine_config.get("user", "not specified")
        print(f"  {machine_id}: {hostname} ({platform}, {machine_mode}, user: {machine_user}) -> {', '.join(machine_services)}")


def main():
    """Command-line interface for deployment utilities."""
    if len(sys.argv) < 3:
        print("Usage: deployment_utils.py <project> <action> [hostname_override]")
        print("Actions: services, mode, user, modules, services-with-modules, info")
        sys.exit(1)
    
    project = sys.argv[1]
    action = sys.argv[2]
    hostname_override = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Find repo directory
    script_dir = Path(__file__).parent
    repo_dir = script_dir.parent.parent
    
    # Load deployment config
    config_path = find_deployment_config(project, repo_dir)
    if not config_path:
        if action in ["services", "modules", "services-with-modules"]:
            # Fall back to get_project_services.py for compatibility
            fallback_script = script_dir / "get_project_services.py"
            if fallback_script.exists():
                if action == "services":
                    import subprocess
                    result = subprocess.run([sys.executable, str(fallback_script), project], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        # Convert systemd format back to service names
                        systemd_services = result.stdout.strip().split('\n')
                        services = [s.split('@')[0] for s in systemd_services if '@' in s]
                        print(" ".join(services))
                        sys.exit(0)
                elif action == "modules":
                    # For modules without deployment config, use default naming
                    try:
                        from experimance_common.config import get_project_services
                        services = get_project_services(project)
                        for service in services:
                            if service in ["core", "agent"]:
                                print(f"{project}_{service}")
                            else:
                                print(f"experimance_{service}")
                        sys.exit(0)
                    except ImportError:
                        pass
        
        print(f"No deployment configuration found for project '{project}'")
        sys.exit(1)
    
    config = load_deployment_config(config_path)
    
    if action == "services":
        services = get_services_for_machine(config, hostname_override)
        print(" ".join(services))
    elif action == "mode":
        mode = get_machine_mode(config, hostname_override)
        print(mode)
    elif action == "user":
        user = get_machine_user(config, hostname_override)
        print(user or "")
    elif action == "modules":
        services = get_services_for_machine(config, hostname_override)
        modules = []
        for service in services:
            module = get_service_module_name(config, service, project)
            modules.append(module)
        print(" ".join(modules))
    elif action == "services-with-modules":
        services_with_modules = get_services_with_modules(config, project, hostname_override)
        for item in services_with_modules:
            print(f"{item['service']}:{item['module']}")
    elif action == "info":
        print_deployment_info(config, hostname_override)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
