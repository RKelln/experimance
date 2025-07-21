#!/usr/bin/env python3
"""
Helper script to get the list of services for a project.
Used by deploy.sh to dynamically detect which services to manage.

Usage: uv run python infra/scripts/get_project_services.py [project_name]
"""

import sys
import os
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_project_services.py [project_name]", file=sys.stderr)
        sys.exit(1)
    
    project_name = sys.argv[1]
    
    try:
        # Import here to avoid import issues when not in proper environment
        from experimance_common.config import get_project_services
        
        services = get_project_services(project_name)
        # Convert to standardized service_type@project format
        systemd_services = []
        for service in services:
            # Use service type directly with @project suffix
            systemd_services.append(f"{service}@{project_name}")
        
        # Always include health service if not already present
        health_service = f"health@{project_name}"
        if health_service not in systemd_services:
            systemd_services.append(health_service)
        
        # Remove duplicates and sort for consistent output
        systemd_services = sorted(list(set(systemd_services)))
        
        # Output one service per line for easy bash parsing
        for service in systemd_services:
            print(service)
            
    except Exception as e:
        print(f"Error getting services for project {project_name}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
