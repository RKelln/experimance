#!/usr/bin/env python3
"""
Interactive script to create a new project configuration.

This script helps set up a new project by:
1. Creating a project directory under projects/
2. Copying service configuration files from existing projects or service defaults
3. Creating project-specific constants.py and schemas.py files
4. Setting up environment variables

Usage:
    uv run python scripts/create_new_project.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import tempfile

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "libs" / "common" / "src"))

# Available services in the codebase
AVAILABLE_SERVICES = [
    "core",
    "display", 
    "audio",
    "agent",
    "image_server"
]

# Service display names and descriptions
SERVICE_INFO = {
    "core": "Core orchestration service (state machine, camera, etc.)",
    "display": "Display service for sand table visualization",
    "audio": "Audio service for sound generation and SuperCollider integration",
    "agent": "AI agent service for intelligent interactions",
    "image_server": "Image generation service (AI-generated satellite imagery)"
}

def get_existing_projects() -> List[str]:
    """Get list of existing project directories."""
    projects_dir = PROJECT_ROOT / "projects"
    if not projects_dir.exists():
        return []
    return [d.name for d in projects_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

def get_project_name() -> str:
    """Interactively get the project name."""
    existing_projects = get_existing_projects()
    
    print("🎨 Experimance Project Setup")
    print("=" * 40)
    
    if existing_projects:
        print(f"Existing projects: {', '.join(existing_projects)}")
    else:
        print("No existing projects found.")
    
    while True:
        project_name = input("\n📝 Enter new project name (lowercase, alphanumeric): ").strip().lower()
        
        if not project_name:
            print("❌ Project name cannot be empty.")
            continue
            
        if not project_name.replace('_', '').replace('-', '').isalnum():
            print("❌ Project name must contain only letters, numbers, hyphens, and underscores.")
            continue
            
        if project_name in existing_projects:
            print(f"❌ Project '{project_name}' already exists.")
            continue
            
        return project_name

def select_services() -> List[str]:
    """Interactively select which services to include."""
    print("\n🔧 Service Selection")
    print("=" * 20)
    print("Available services:")
    
    for i, service in enumerate(AVAILABLE_SERVICES, 1):
        print(f"  {i}. {service} - {SERVICE_INFO[service]}")
    
    print("\nSelect services to include in your project:")
    print("  - Enter numbers separated by commas (e.g., '1,2,5')")
    print("  - Enter 'all' to include all services")
    print("  - Press Enter to include core service only")
    
    while True:
        selection = input("\n🎯 Your selection: ").strip()
        
        if not selection:
            return ["core"]
            
        if selection.lower() == "all":
            return AVAILABLE_SERVICES.copy()
            
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            selected_services = []
            
            for idx in indices:
                if 1 <= idx <= len(AVAILABLE_SERVICES):
                    service = AVAILABLE_SERVICES[idx - 1]
                    if service not in selected_services:
                        selected_services.append(service)
                else:
                    print(f"❌ Invalid selection: {idx}. Must be between 1 and {len(AVAILABLE_SERVICES)}.")
                    break
            else:
                if selected_services:
                    return selected_services
                    
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")

def get_source_project() -> Optional[str]:
    """Ask which existing project to copy configs from, or use service defaults."""
    existing_projects = get_existing_projects()
    
    print("\n📋 Configuration Source")
    print("=" * 25)
    print("Copy service configurations from:")
    
    options = []
    option_num = 1
    
    # Add existing projects as options
    for project in existing_projects:
        print(f"  {option_num}. {project} (existing project)")
        options.append(project)
        option_num += 1
    
    # Add service defaults option
    print(f"  {option_num}. Service defaults (from services/*/config.toml)")
    options.append("__service_defaults__")
    option_num += 1
    
    # Add minimal config option
    print(f"  {option_num}. Create minimal configs (when no defaults available)")
    options.append(None)
    
    while True:
        selection = input(f"\n🎯 Your selection (1-{len(options)}): ").strip()
        
        if not selection:
            continue
            
        try:
            idx = int(selection)
            if 1 <= idx <= len(options):
                selected = options[idx - 1]
                if selected == "__service_defaults__":
                    return "__service_defaults__"
                else:
                    return selected
            else:
                print(f"❌ Invalid selection. Must be between 1 and {len(options)}.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def copy_service_config(service: str, project_name: str, source_project: Optional[str]) -> bool:
    """Copy a service config file to the new project."""
    target_path = PROJECT_ROOT / "projects" / project_name / f"{service}.toml"
    
    if source_project == "__service_defaults__":
        # Copy from service default only
        service_default_path = PROJECT_ROOT / "services" / service / "config.toml"
        if service_default_path.exists():
            shutil.copy2(service_default_path, target_path)
            print(f"  ✅ Copied {service}.toml from service default")
            return True
        else:
            # Create minimal config if no service default exists
            minimal_config = f'''# {service.title()} service configuration for {project_name}
# This is a minimal configuration - please customize as needed

service_name = "{project_name}-{service}"

# TODO: Add {service}-specific configuration here
# See services/{service}/config.py for available options
'''
            target_path.write_text(minimal_config)
            print(f"  ⚠️  Created minimal {service}.toml (no service default found)")
            return True
    
    elif source_project:
        # Copy from existing project
        source_path = PROJECT_ROOT / "projects" / source_project / f"{service}.toml"
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            print(f"  ✅ Copied {service}.toml from {source_project}")
            return True
        else:
            print(f"  ⚠️  {service}.toml not found in {source_project}, trying service default...")
            # Fall back to service default
            service_default_path = PROJECT_ROOT / "services" / service / "config.toml"
            if service_default_path.exists():
                shutil.copy2(service_default_path, target_path)
                print(f"  ✅ Copied {service}.toml from service default (fallback)")
                return True
    
    # Create minimal config as last resort
    minimal_config = f'''# {service.title()} service configuration for {project_name}
# This is a minimal configuration - please customize as needed

service_name = "{project_name}-{service}"

# TODO: Add {service}-specific configuration here
# See services/{service}/config.py for available options
'''
    target_path.write_text(minimal_config)
    print(f"  ⚠️  Created minimal {service}.toml (no default found)")
    return True

def create_project_constants(project_name: str, source_project: Optional[str]) -> None:
    """Create project-specific constants.py file."""
    constants_path = PROJECT_ROOT / "projects" / project_name / "constants.py"
    
    if source_project:
        source_constants = PROJECT_ROOT / "projects" / source_project / "constants.py"
        if source_constants.exists():
            shutil.copy2(source_constants, constants_path)
            print(f"  ✅ Copied constants.py from {source_project}")
            return
    
    # Create minimal constants file
    constants_content = f'''"""
Project-specific constants for {project_name}.

This file can extend or override constants from experimance_common.constants_base.
Any constants defined here will be available in experimance_common.constants.
"""

# Project-specific constants go here
# Example:
# PROJECT_SPECIFIC_TIMEOUT = 30.0
# CUSTOM_ENDPOINT = "https://api.{project_name}.example.com"

# Add any project-specific constants here
'''
    
    constants_path.write_text(constants_content)
    print(f"  ✅ Created minimal constants.py")

def create_project_schemas(project_name: str, source_project: Optional[str]) -> None:
    """Create project-specific schemas.py file."""
    schemas_path = PROJECT_ROOT / "projects" / project_name / "schemas.py"
    
    if source_project:
        source_schemas = PROJECT_ROOT / "projects" / source_project / "schemas.py"
        if source_schemas.exists():
            shutil.copy2(source_schemas, schemas_path)
            print(f"  ✅ Copied schemas.py from {source_project}")
            return
    
    # Create minimal schemas file based on experimance template
    schemas_content = f'''"""
Project-specific schemas for {project_name}.

This file extends base schemas from experimance_common.schemas_base.
"""

from experimance_common.schemas_base import (
    StringComparableEnum,
    SpaceTimeUpdate as _BaseSpaceTimeUpdate,
    RenderRequest as _BaseRenderRequest,
    ImageReady as _BaseImageReady,
    DisplayMedia as _BaseDisplayMedia,
)

# Define project-specific enums
class Era(StringComparableEnum):
    """Time periods for {project_name}."""
    # TODO: Define your project's time periods/eras
    ANCIENT = "ancient"
    MODERN = "modern"
    FUTURE = "future"

class Biome(StringComparableEnum):
    """Environmental biomes for {project_name}."""
    # TODO: Define your project's biomes/environments
    FOREST = "forest"
    DESERT = "desert"
    OCEAN = "ocean"
    URBAN = "urban"

# Extend base message classes with project-specific fields
class SpaceTimeUpdate(_BaseSpaceTimeUpdate):
    """Space-time update with project-specific era and biome."""
    era: Era
    biome: Biome

class RenderRequest(_BaseRenderRequest):
    """Render request with project-specific era and biome."""
    era: Era
    biome: Biome

class ImageReady(_BaseImageReady):
    """Image ready message with project-specific era and biome."""
    era: Era
    biome: Biome

class DisplayMedia(_BaseDisplayMedia):
    """Display media with project-specific era and biome."""
    era: Era
    biome: Biome

# TODO: Add any other project-specific payload classes here
# Example:
# class SuggestBiomePayload(BaseModel):
#     suggested_biome: Biome
#     confidence: float
'''
    
    schemas_path.write_text(schemas_content)
    print(f"  ✅ Created schemas.py template")

def create_project_env(project_name: str) -> None:
    """Create project-specific .env file."""
    env_path = PROJECT_ROOT / "projects" / project_name / ".env"
    
    env_content = f'''# Environment variables for {project_name} project

# Set the project environment
PROJECT_ENV={project_name}

# TODO: Add project-specific environment variables here
# Example:
# API_KEY=your_api_key_here
# DEBUG=true
# MAX_WORKERS=4
'''
    
    env_path.write_text(env_content)
    print(f"  ✅ Created .env file")

def create_type_stubs(project_name: str) -> None:
    """Create type stub file for the project."""
    stub_path = PROJECT_ROOT / "projects" / project_name / "schemas.pyi"
    
    stub_content = f'''"""
Type stubs for {project_name} project schemas.
"""

from experimance_common.schemas_base import (
    StringComparableEnum,
    SpaceTimeUpdate as _BaseSpaceTimeUpdate,
    RenderRequest as _BaseRenderRequest,
    ImageReady as _BaseImageReady,
    DisplayMedia as _BaseDisplayMedia,
)

class Era(StringComparableEnum): ...
class Biome(StringComparableEnum): ...

class SpaceTimeUpdate(_BaseSpaceTimeUpdate):
    era: Era
    biome: Biome

class RenderRequest(_BaseRenderRequest):
    era: Era
    biome: Biome

class ImageReady(_BaseImageReady):
    era: Era
    biome: Biome

class DisplayMedia(_BaseDisplayMedia):
    era: Era
    biome: Biome
'''
    
    stub_path.write_text(stub_content)
    print(f"  ✅ Created schemas.pyi type stub")

def main():
    """Main function to create a new project."""
    try:
        # Get project details
        project_name = get_project_name()
        selected_services = select_services()
        source_project = get_source_project()
        
        # Confirm setup
        print(f"\n📋 Project Setup Summary")
        print("=" * 25)
        print(f"Project name: {project_name}")
        print(f"Services: {', '.join(selected_services)}")
        if source_project == "__service_defaults__":
            print("Copy configs from: service defaults")
        elif source_project:
            print(f"Copy configs from: {source_project}")
        else:
            print("Copy configs from: create minimal configs")
        
        confirm = input("\n✅ Proceed with setup? (y/N): ").strip().lower()
        if confirm not in ('y', 'yes'):
            print("❌ Setup cancelled.")
            return
        
        # Create project directory
        project_dir = PROJECT_ROOT / "projects" / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n🎯 Creating project: {project_name}")
        print(f"📁 Project directory: {project_dir}")
        
        # Copy service configurations
        print("\n📋 Setting up service configurations:")
        for service in selected_services:
            copy_service_config(service, project_name, source_project)
        
        # Create project files
        print("\n📄 Creating project files:")
        create_project_constants(project_name, source_project)
        create_project_schemas(project_name, source_project)
        create_project_env(project_name)
        create_type_stubs(project_name)
        
        # Update type stubs
        print("\n🔧 Updating type stubs...")
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 
                str(PROJECT_ROOT / "scripts" / "update_pyi_stubs.py")
            ], cwd=PROJECT_ROOT, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("  ✅ Type stubs updated successfully")
            else:
                print(f"  ⚠️  Type stub update had issues: {result.stderr}")
        except Exception as e:
            print(f"  ⚠️  Could not update type stubs: {e}")
        
        # Success message
        print(f"\n🎉 Project '{project_name}' created successfully!")
        print("\n📋 Next steps:")
        print(f"1. Review and customize config files in projects/{project_name}/")
        print(f"2. Update schemas.py with your project's specific Era and Biome values")
        print(f"3. Add project-specific constants to constants.py if needed")
        print(f"4. Set PROJECT_ENV={project_name} in your environment or use:")
        print(f"   PROJECT_ENV={project_name} uv run -m experimance_core")
        print(f"5. Start developing your project-specific services!")
        
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
