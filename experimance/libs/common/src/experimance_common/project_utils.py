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
        Project name string (may include variant as "project/variant")
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
            project_spec = project_file.read_text().strip()
            if project_spec:
                return project_spec
        except Exception:
            # Silently fall through to default if file can't be read
            pass
    
    # Default fallback
    return "experimance"


def detect_variant_name(project_spec: Optional[str] = None, projects_dir: Optional[Path] = None) -> Optional[str]:
    """Extract variant name from project specification.
    
    Project specifications can optionally include a variant using "/" syntax:
    - "experimance" → project="experimance", variant=None
    - "experimance/nochat_demo" → project="experimance", variant="nochat_demo"
    
    Args:
        project_spec: Full project specification (e.g., "experimance" or "experimance/nochat_demo").
                     If None, detects from environment/file.
        projects_dir: Path to projects directory (defaults to PROJECT_SPECIFIC_DIR)
        
    Returns:
        Variant name if specified, None otherwise
    """
    if project_spec is None:
        project_spec = detect_project_name(projects_dir)
    
    if "/" in project_spec:
        parts = project_spec.split("/", 1)
        return parts[1] if len(parts) > 1 else None
    
    return None


def get_base_project_name(project_spec: Optional[str] = None, projects_dir: Optional[Path] = None) -> str:
    """Extract base project name from project specification.
    
    Strips the variant suffix if present. For example:
    - "experimance" → "experimance"
    - "experimance/nochat_demo" → "experimance"
    
    Args:
        project_spec: Full project specification (e.g., "experimance" or "experimance/nochat_demo").
                     If None, detects from environment/file.
        projects_dir: Path to projects directory (defaults to PROJECT_SPECIFIC_DIR)
        
    Returns:
        Base project name
    """
    if project_spec is None:
        project_spec = detect_project_name(projects_dir)
    
    if "/" in project_spec:
        return project_spec.split("/", 1)[0]
    
    return project_spec


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


def load_project_dotenv(override: bool = True) -> None:
    """Load project .env file(s) without clobbering PROJECT_ENV.

    Loads base project .env first, then variant .env on top (if active).
    PROJECT_ENV is always restored after loading so .env files can never
    override the variant spec set by .project / set_project.py.

    Call this instead of manually save/restore-ing PROJECT_ENV around
    load_dotenv() calls in constants.py, schemas.py, and __init__.py.
    """
    from dotenv import load_dotenv
    from experimance_common.constants_base import PROJECT_ROOT

    project_spec = os.environ.get("PROJECT_ENV", "experimance")
    base_project = get_base_project_name(project_spec)
    variant = detect_variant_name(project_spec)
    saved = project_spec  # always restore after loading

    base_env = PROJECT_ROOT / f"projects/{base_project}/.env"
    if base_env.exists():
        load_dotenv(base_env, override=override)

    if variant:
        variant_env = PROJECT_ROOT / f"projects/{base_project}/{variant}/.env"
        if variant_env.exists():
            load_dotenv(variant_env, override=override)

    os.environ["PROJECT_ENV"] = saved


def cli_main() -> None:
    """Command-line interface for setting the current project.
    
    This function can be used as an entry point in pyproject.toml scripts.
    
    Supports both simple project names and project/variant specifications:
    - `set_project.py experimance` — set base project
    - `set_project.py experimance/nochat_demo` — set project with variant
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Set the current project for Experimance services"
    )
    parser.add_argument(
        "project_spec",
        help="Project to set: 'project' (e.g., 'experimance') or 'project/variant' (e.g., 'experimance/nochat_demo')"
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

    # Extract base project name and optional variant
    base_project = get_base_project_name(args.project_spec)
    variant = detect_variant_name(args.project_spec)
    
    # Validate base project exists
    project_dir = args.projects_dir / base_project
    if not project_dir.exists():
        print(f"Error: Project '{base_project}' not found at '{project_dir}'")
        print(f"Available projects:")
        for p in args.projects_dir.iterdir():
            if p.is_dir() and not p.name.startswith('.'):
                print(f"  - {p.name}")
                # List variants if any
                for v in p.iterdir():
                    if v.is_dir() and not v.name.startswith('.'):
                        print(f"    - {p.name}/{v.name}")
        sys.exit(1)
    
    # Validate variant exists if specified
    if variant:
        variant_dir = project_dir / variant
        if not variant_dir.exists():
            print(f"Error: Variant '{variant}' not found in project '{base_project}'")
            print(f"  Expected path: {variant_dir}")
            print(f"Available variants in {base_project}:")
            for v in project_dir.iterdir():
                if v.is_dir() and not v.name.startswith('.'):
                    print(f"  - {base_project}/{v.name}")
            sys.exit(1)
    
    # Write .project file
    project_file = args.projects_dir / PROJECT_FILE_NAME
    try:
        project_file.write_text(args.project_spec + "\n")
        if variant:
            print(f"Set current project to: {base_project} (variant: {variant})")
        else:
            print(f"Set current project to: {base_project}")
        print(f"Project file: {project_file}")
    except Exception as e:
        print(f"Error writing project file: {e}")
        sys.exit(1)


def set_project(project_spec: str, projects_dir: Optional[Path] = None) -> None:
    """Set the current project by writing to .project file.
    
    Supports both simple project names and project/variant specifications:
    - `set_project("experimance")` — set base project
    - `set_project("experimance/nochat_demo")` — set project with variant
    
    Args:
        project_spec: Project specification (e.g., "experimance" or "experimance/nochat_demo")
        projects_dir: Path to projects directory (defaults to PROJECT_SPECIFIC_DIR)

    Raises:
        SystemExit: If project/variant directory doesn't exist or file cannot be written
    """
    if projects_dir is None:
        from experimance_common.constants_base import PROJECT_SPECIFIC_DIR
        projects_dir = PROJECT_SPECIFIC_DIR

    # Extract base project name and optional variant
    base_project = get_base_project_name(project_spec)
    variant = detect_variant_name(project_spec)
    
    # Validate base project exists
    project_dir = projects_dir / base_project
    if not project_dir.exists():
        print(f"Error: Project '{base_project}' not found at '{project_dir}'")
        print(f"Available projects:")
        for p in projects_dir.iterdir():
            if p.is_dir() and not p.name.startswith('.'):
                print(f"  - {p.name}")
                # List variants if any
                for v in p.iterdir():
                    if v.is_dir() and not v.name.startswith('.'):
                        print(f"    - {p.name}/{v.name}")
        sys.exit(1)
    
    # Validate variant exists if specified
    if variant:
        variant_dir = project_dir / variant
        if not variant_dir.exists():
            print(f"Error: Variant '{variant}' not found in project '{base_project}'")
            print(f"  Expected path: {variant_dir}")
            print(f"Available variants in {base_project}:")
            for v in project_dir.iterdir():
                if v.is_dir() and not v.name.startswith('.'):
                    print(f"  - {base_project}/{v.name}")
            sys.exit(1)
    
    # Write .project file
    project_file = projects_dir / PROJECT_FILE_NAME
    try:
        project_file.write_text(project_spec + "\n")
        if variant:
            print(f"Set current project to: {base_project} (variant: {variant})")
        else:
            print(f"Set current project to: {base_project}")
        print(f"Project file: {project_file}")
    except Exception as e:
        print(f"Error writing project file: {e}")
        sys.exit(1)
