"""
Project utilities for Experimance multi-project support.

This module provides centralized logic for detecting which project should be used
based on environment variables and project files, as well as utilities for
managing project settings.
"""
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_FILE_NAME = ".project"

def detect_project_name(projects_dir: Optional[Path] = None) -> str:
    """Detect the current project name using the standard precedence rules.
    
    Precedence (highest to lowest):
    1. PROJECT_ENV environment variable (if already set)
    2. projects/.project file content
    3. Default to "experimance"
    
    Args:
        projects_dir: Path to projects directory (defaults to PROJECT_SPECIFIC_DIR)
        
    Returns:
        Project name string
    """
    # Environment variable takes highest precedence
    if "PROJECT_ENV" in os.environ:
        return os.environ["PROJECT_ENV"]
    
    # Try to read from .project file
    if projects_dir is None:
        # Use PROJECT_SPECIFIC_DIR to find the correct projects directory regardless of cwd
        from experimance_common.constants_base import PROJECT_SPECIFIC_DIR
        projects_dir = PROJECT_SPECIFIC_DIR

    project_file = projects_dir / PROJECT_FILE_NAME

    if project_file.exists():
        try:
            project_name = project_file.read_text().strip()
            if project_name:
                return project_name
        except Exception:
            # Silently fall through to default if file can't be read
            pass
    
    # Default fallback
    return "experimance"


def ensure_project_env_set(projects_dir: Optional[Path] = None) -> str:
    """Ensure PROJECT_ENV is set in the environment, detecting from file if needed.
    
    This function should be called once during application startup to ensure
    PROJECT_ENV is available for all subsequent code that needs it.
    
    Args:
        projects_dir: Path to projects directory (defaults to PROJECT_SPECIFIC_DIR/projects)

    Returns:
        The project name that was set
    """
    if "PROJECT_ENV" not in os.environ:
        project_name = detect_project_name(projects_dir)
        os.environ["PROJECT_ENV"] = project_name
        return project_name
    else:
        return os.environ["PROJECT_ENV"]


def cli_main() -> None:
    """Command-line interface for setting the current project.
    
    This function can be used as an entry point in pyproject.toml scripts.
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Set the current project for Experimance services"
    )
    parser.add_argument(
        "project_name",
        help="Project name to set (e.g., 'experimance', 'fire')"
    )
    parser.add_argument(
        "--projects-dir",
        type=Path,
        help="Path to projects directory (default: PROJECT_SPECIFIC_DIR)"
    )
    
    args = parser.parse_args()
    
    # Default to PROJECT_SPECIFIC_DIR if not specified
    if args.projects_dir is None:
        from experimance_common.constants_base import PROJECT_SPECIFIC_DIR
        args.projects_dir = PROJECT_SPECIFIC_DIR

    # Validate project exists
    project_dir = args.projects_dir / args.project_name
    if not project_dir.exists():
        print(f"Error: Project directory '{project_dir}' does not exist")
        print(f"Available projects:")
        for p in args.projects_dir.iterdir():
            if p.is_dir() and not p.name.startswith('.'):
                print(f"  - {p.name}")
        sys.exit(1)
    
    # Write .project file
    project_file = args.projects_dir / PROJECT_FILE_NAME
    try:
        project_file.write_text(args.project_name + "\n")
        print(f"Set current project to: {args.project_name}")
        print(f"Project file: {project_file}")
    except Exception as e:
        print(f"Error writing project file: {e}")
        sys.exit(1)


def set_project(project_name: str, projects_dir: Optional[Path] = None) -> None:
    """Set the current project by writing to .project file.
    
    Args:
        project_name: Name of the project to set
        projects_dir: Path to projects directory (defaults to PROJECT_SPECIFIC_DIR)

    Raises:
        SystemExit: If project directory doesn't exist or file cannot be written
    """
    if projects_dir is None:
        from experimance_common.constants_base import PROJECT_SPECIFIC_DIR
        projects_dir = PROJECT_SPECIFIC_DIR

    # Validate project exists
    project_dir = projects_dir / project_name
    if not project_dir.exists():
        print(f"Error: Project directory '{project_dir}' does not exist")
        print(f"Available projects:")
        for p in projects_dir.iterdir():
            if p.is_dir() and not p.name.startswith('.'):
                print(f"  - {p.name}")
        sys.exit(1)
    
    # Write .project file
    project_file = projects_dir / PROJECT_FILE_NAME
    try:
        project_file.write_text(project_name + "\n")
        print(f"Set current project to: {project_name}")
        print(f"Project file: {project_file}")
    except Exception as e:
        print(f"Error writing project file: {e}")
        sys.exit(1)
