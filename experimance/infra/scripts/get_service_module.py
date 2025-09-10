#!/usr/bin/env python3
"""
Get the module name for a specific service from deployment configuration.
Usage: get_service_module.py <project> <service_type> [config_path]
"""

import sys
from pathlib import Path

# Add the infra/scripts directory to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from deployment_utils import load_deployment_config, get_service_module_name, find_deployment_config


def main():
    if len(sys.argv) < 3:
        print("Usage: get_service_module.py <project> <service_type> [config_path]", file=sys.stderr)
        sys.exit(1)
    
    project = sys.argv[1]
    service_type = sys.argv[2]
    
    # Get config path
    if len(sys.argv) > 3:
        config_path = Path(sys.argv[3])
    else:
        # Try to find it
        repo_dir = script_dir.parent.parent
        config_path = find_deployment_config(project, repo_dir)
    
    if not config_path or not config_path.exists():
        # No deployment config, use default naming
        if service_type in ["core", "agent"]:
            print(f"{project}_{service_type}")
        else:
            print(f"experimance_{service_type}")
        sys.exit(0)
    
    # Load config and get module name
    config = load_deployment_config(config_path)
    module_name = get_service_module_name(config, service_type, project)
    print(module_name)


if __name__ == "__main__":
    main()
