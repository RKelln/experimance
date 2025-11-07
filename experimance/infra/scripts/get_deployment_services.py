#!/usr/bin/env python3
"""
Get deployment services for a project based on deployment configuration.

This script determines which services should run on the current machine
by reading the deployment.toml configuration file.

This replaces and consolidates the functionality of get_project_services.py
for projects that have deployment configurations.
"""

import os
import sys
from pathlib import Path

# Add the infra/scripts directory to Python path to import deployment_utils
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from deployment_utils import (
    find_deployment_config,
    load_deployment_config,
    get_services_for_machine,
    get_current_hostname
)


def main():
    if len(sys.argv) < 2:
        print("Usage: get_deployment_services.py <project> [hostname_override]", file=sys.stderr)
        sys.exit(1)
    
    project = sys.argv[1]
    hostname_override = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Find repo directory
    repo_dir = script_dir.parent.parent
    
    # Check for deployment configuration
    config_path = find_deployment_config(project, repo_dir)
    if not config_path:
        # No deployment config found, fall back to get_project_services.py
        # This maintains compatibility with single-machine deployments
        fallback_script = script_dir / "get_project_services.py"
        if fallback_script.exists():
            # Execute the fallback script
            import subprocess
            result = subprocess.run([sys.executable, str(fallback_script), project], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
                sys.exit(0)
            else:
                print(f"Fallback script failed: {result.stderr}", file=sys.stderr)
                sys.exit(1)
        else:
            # No fallback available, try importing experimance_common as last resort
            try:
                from experimance_common.config import get_project_services
                services = get_project_services(project)
                
                # Convert to systemd format for compatibility
                systemd_services = []
                for service in services:
                    systemd_services.append(f"{service}@{project}")
                
                # Always include health service if not already present
                health_service = f"health@{project}"
                if health_service not in systemd_services:
                    systemd_services.append(health_service)
                
                # Remove duplicates and sort for consistent output
                systemd_services = sorted(list(set(systemd_services)))
                
                # Output one service per line for easy bash parsing
                for service in systemd_services:
                    print(service)
                sys.exit(0)
                    
            except ImportError:
                # Last resort fallback
                default_services = ["core", "display", "audio", "agent", "image_server", "health"]
                for service in default_services:
                    print(f"{service}@{project}")
                sys.exit(0)
    
    # Load deployment configuration
    config = load_deployment_config(config_path)
    
    # Get services for this machine
    services = get_services_for_machine(config, hostname_override)
    
    # Convert to systemd format (service@project) for compatibility with deploy.sh
    systemd_services = []
    for service in services:
        systemd_services.append(f"{service}@{project}")
    
    # Remove duplicates and sort for consistent output
    systemd_services = sorted(list(set(systemd_services)))
    
    # Output one service per line (expected by deploy.sh)
    for service in systemd_services:
        print(service)


if __name__ == "__main__":
    main()
