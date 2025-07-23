#!/usr/bin/env python3
"""
Script to automatically update .pyi stub files for experimance_common.

This script generates type stub files for the dynamically loaded modules:
- experimance_common/schemas.pyi
- experimance_common/constants.pyi

The script introspects the base modules and project-specific extensions
to generate comprehensive type stubs that work with static type checkers.
"""

import ast
import os
import sys
import argparse
import tempfile
import difflib
from pathlib import Path
from typing import List, Set, Dict, Tuple, Any
import importlib.util
import inspect

# Add the project root to Python path so we can import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "libs" / "common" / "src"))

def get_project_dirs() -> List[str]:
    """Get list of available project directories."""
    projects_dir = PROJECT_ROOT / "projects"
    if not projects_dir.exists():
        return []
    return sorted([d.name for d in projects_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

def extract_classes_from_ast(file_path: Path) -> List[str]:
    """Extract class names from a Python file using AST parsing."""
    if not file_path.exists():
        return []
    
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return classes
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []

def extract_constants_from_ast(file_path: Path) -> List[str]:
    """Extract top-level constants from a Python file using AST parsing."""
    if not file_path.exists():
        return []
    
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        constants = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(target.id)
        
        return constants
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []

def get_constants_base_exports() -> List[str]:
    """Get exported constants from constants_base.py __all__ plus important missing ones."""
    constants_base_path = PROJECT_ROOT / "libs" / "common" / "src" / "experimance_common" / "constants_base.py"
    
    try:
        with open(constants_base_path, 'r') as f:
            tree = ast.parse(f.read())
        
        # Get __all__ list
        all_constants = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            all_constants = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
                            break
        
        # Add important constants that might not be in __all__ but should be available
        important_missing = ["PROJECT_SPECIFIC_DIR", "DATA_DIR"]
        for const in important_missing:
            if const not in all_constants:
                all_constants.append(const)
        
        return all_constants
        
    except Exception as e:
        print(f"Warning: Could not parse constants_base.py: {e}")
        return []

def get_schemas_base_classes() -> List[str]:
    """Get class names from schemas_base.py."""
    schemas_base_path = PROJECT_ROOT / "libs" / "common" / "src" / "experimance_common" / "schemas_base.py"
    return extract_classes_from_ast(schemas_base_path)

def get_project_specific_classes(project: str, module_name: str) -> List[str]:
    """Get class names from a project-specific module."""
    project_file = PROJECT_ROOT / "projects" / project / f"{module_name}.py"
    return extract_classes_from_ast(project_file)

def get_project_specific_constants(project: str) -> List[str]:
    """Get constants from a project-specific constants.py file."""
    project_file = PROJECT_ROOT / "projects" / project / "constants.py"
    return extract_constants_from_ast(project_file)

def get_common_schemas_across_projects() -> List[str]:
    """Get schemas that are defined in ALL projects."""
    projects = get_project_dirs()
    if not projects:
        return []
    
    # Get schemas from each project
    project_schemas = {}
    for project in projects:
        project_schemas[project] = set(get_project_specific_classes(project, "schemas"))
    
    # Find intersection of all project schemas
    if project_schemas:
        common_schemas = set.intersection(*project_schemas.values())
        return sorted(list(common_schemas))
    
    return []

def generate_schemas_pyi() -> str:
    """Generate the content for schemas.pyi."""
    base_classes = get_schemas_base_classes()
    projects = get_project_dirs()
    common_schemas = get_common_schemas_across_projects()
    
    # Classes that are typically NOT extended by projects (from current schemas.pyi)
    non_extended_classes = {
        "StringComparableEnum", "MessageSchema", "MessageBase", "TransitionStyle",
        "DisplayContentType", "DisplayTransitionType", "MessageType", "ContentType",
        "ImageSource", "IdleStatus", "TransitionReady", "LoopReady", 
        "PresenceStatus",
        "AudiencePresent", "SpeechDetected", # agent messages
        "AgentControlEvent", "DisplayText", "RemoveText", "TransitionRequest", "LoopRequest"
    }
    
    content = '''"""
Static-analysis stub for experimance_common.schemas.

This stub provides type information for the dynamically loaded schemas.
At runtime, the actual module loads project-specific extensions based on
the PROJECT_ENV environment variable and makes them available in this namespace.

For static type checking, this file conditionally imports the appropriate
project-specific types based on the PROJECT_ENV environment variable.
"""

import os
from typing import TYPE_CHECKING

# Re-export base schemas that are NOT extended by projects
from experimance_common.schemas_base import (
    # Base classes
    StringComparableEnum,
    MessageSchema,
    MessageBase,
    
    # Enums that are project-independent
    TransitionStyle,
    DisplayContentType,
    DisplayTransitionType,
    MessageType,
    ContentType,
    
    # Message types that are NOT extended by projects
'''
    
    # Add other non-extended classes
    for cls in sorted(non_extended_classes - {"StringComparableEnum", "MessageSchema", "MessageBase", 
                                            "TransitionStyle", "DisplayContentType", "DisplayTransitionType", 
                                            "MessageType", "ContentType"}):
        if cls in base_classes:
            content += f"    {cls},\n"
    
    content += ''')

# Conditionally import project-specific types based on PROJECT_ENV
if TYPE_CHECKING:
    _PROJECT_ENV = os.getenv("PROJECT_ENV", "experimance")
    
'''
    
    # Add conditional imports for each project
    for i, project in enumerate(projects):
        project_classes = get_project_specific_classes(project, "schemas")
        
        if i == 0:
            content += f'    if _PROJECT_ENV == "{project}":\n'
        else:
            content += f'    elif _PROJECT_ENV == "{project}":\n'
        
        if project_classes:
            content += f"        from projects.{project}.schemas import (\n"
            for cls in sorted(project_classes):
                content += f"            {cls},\n"
            content += "        )\n"
        else:
            content += f"        # No project-specific schemas for {project}\n"
    
    # Add fallback
    content += '''    else:
        # Fallback for unknown projects - use base types and create minimal stubs
        from experimance_common.schemas_base import (
            SpaceTimeUpdate,
            RenderRequest,
            ImageReady,
            DisplayMedia,
        )
        
        # Create minimal project-specific enums for unknown projects
        class Era(StringComparableEnum):
            """Fallback Era enum for unknown projects."""
            ...
        
        class Biome(StringComparableEnum):
            """Fallback Biome enum for unknown projects."""
            ...

__all__: list[str] = [
    # Base classes
    "StringComparableEnum",
    "MessageSchema", 
    "MessageBase",
    
    # Enums
    "TransitionStyle",
    "DisplayContentType",
    "DisplayTransitionType", 
    "MessageType",
    "ContentType",
    
    # Message types that are NOT extended by projects
'''
    
    # Add __all__ entries for non-extended classes
    for cls in sorted(non_extended_classes - {"StringComparableEnum", "MessageSchema", "MessageBase", 
                                            "TransitionStyle", "DisplayContentType", "DisplayTransitionType", 
                                            "MessageType", "ContentType"}):
        if cls in base_classes:
            content += f'    "{cls}",\n'
    
    content += '''    
    # Common project-specific types (available in all projects)
'''
    
    # Add common schemas that exist in all projects
    for schema in common_schemas:
        content += f'    "{schema}",  # Extended by all projects\n'
    
    content += '''    
    # Note: Project-specific types like Era, Emotion, RequestBiome, etc.
    # are not included here since they're not universal across all projects.
    # They are still available for import when the appropriate PROJECT_ENV is set.
]'''
    
    return content

def generate_constants_pyi() -> str:
    """Generate the content for constants.pyi."""
    base_constants = get_constants_base_exports()
    projects = get_project_dirs()
    
    content = '''"""
Static-analysis stub for experimance_common.constants.

This stub provides type information for the dynamically loaded constants.
At runtime, the actual module loads project-specific extensions based on
the PROJECT_ENV environment variable and makes them available in this namespace.

For static type checking, this file conditionally imports the appropriate
project-specific constants based on the PROJECT_ENV environment variable.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

# Re-export base constants that are NOT extended by projects
from experimance_common.constants_base import (
'''
    
    # Group constants by category for better organization
    constant_groups = {
        "# Project structure constants": ["PROJECT_ROOT", "PROJECT_SPECIFIC_DIR"],
        "# Config helpers": ["get_project_config_path"],
        "# Port configurations": ["DEFAULT_PORTS"],
        "# Timeout settings": ["DEFAULT_TIMEOUT", "DEFAULT_RETRY_ATTEMPTS", 
                              "DEFAULT_RETRY_DELAY", "DEFAULT_RECV_TIMEOUT", "TICK"],
        "# Service types": ["SERVICE_TYPES"],
        "# Image transport configuration": ["IMAGE_TRANSPORT_MODES", "DEFAULT_IMAGE_TRANSPORT_MODE", 
                                          "IMAGE_TRANSPORT_SIZE_THRESHOLD"],
        "# Temporary file settings": ["TEMP_FILE_PREFIX", "TEMP_FILE_SUFFIX", "TEMP_FILE_CLEANUP_AGE",
                                    "TEMP_FILE_CLEANUP_INTERVAL", "DEFAULT_TEMP_DIR"],
        "# URI and URL constants": ["FILE_URI_PREFIX", "DATA_URL_PREFIX", "BASE64_PNG_PREFIX"],
        "# ZMQ address patterns": ["ZMQ_TCP_BIND_PREFIX", "ZMQ_TCP_CONNECT_PREFIX"],
        "# Data directory": ["DATA_DIR"],
        "# Media directories (relative paths)": ["MEDIA_DIR", "IMAGES_DIR", "GENERATED_IMAGES_DIR",
                                               "MOCK_IMAGES_DIR", "AUDIO_DIR", "VIDEOS_DIR"],
        "# Media directories (absolute paths)": ["MEDIA_DIR_ABS", "IMAGES_DIR_ABS", "GENERATED_IMAGES_DIR_ABS",
                                               "MOCK_IMAGES_DIR_ABS", "AUDIO_DIR_ABS", "VIDEOS_DIR_ABS"],
        "# Services directories": ["SERVICES_DIR", "CORE_SERVICE_DIR", "AUDIO_SERVICE_DIR",
                                 "IMAGE_SERVER_SERVICE_DIR", "AGENT_SERVICE_DIR", "DISPLAY_SERVICE_DIR"]
    }
    
    for group_comment, constants in constant_groups.items():
        content += f"    {group_comment}\n"
        for const in constants:
            if const in base_constants:
                content += f"    {const},\n"
        content += "\n"
    
    content += ''')

# PROJECT constant that's set dynamically
PROJECT: str

# Conditionally import project-specific constants based on PROJECT_ENV
if TYPE_CHECKING:
    _PROJECT_ENV = os.getenv("PROJECT_ENV", "experimance")
    
'''
    
    # Add conditional imports for each project
    for i, project in enumerate(projects):
        project_constants = get_project_specific_constants(project)
        
        if i == 0:
            content += f'    if _PROJECT_ENV == "{project}":\n'
        else:
            content += f'    elif _PROJECT_ENV == "{project}":\n'
        
        if project_constants:
            content += f"        # Import {project}-specific constants if they exist\n"
            for const in sorted(project_constants):
                content += f"        # {const}\n"
        else:
            content += f"        # Import {project}-specific constants if they exist\n"
            content += f"        # For now, the {project} constants.py file is mostly empty,\n"
            content += f"        # but this allows for future project-specific constants\n"
        content += "        pass\n"
    
    # Add fallback
    content += '''    else:
        # Fallback for unknown projects
        pass

__all__: list[str] = [
    # Dynamic project constant
    "PROJECT",
    
'''
    
    # Add __all__ entries grouped by category
    for group_comment, constants in constant_groups.items():
        content += f"    {group_comment}\n"
        for const in constants:
            if const in base_constants:
                content += f'    "{const}",\n'
        content += "\n"
    
    content += ']'
    
    return content

def write_file_if_changed(file_path: Path, content: str) -> bool:
    """Write content to file only if it has changed. Returns True if file was updated."""
    if file_path.exists():
        with open(file_path, 'r') as f:
            existing_content = f.read()
        if existing_content == content:
            return False
    
    with open(file_path, 'w') as f:
        f.write(content)
    return True

def show_diff(file_path: Path, new_content: str) -> bool:
    """Show diff between existing file and new content. Returns True if there are differences."""
    if not file_path.exists():
        print(f"\n--- {file_path} (NEW FILE)")
        print(f"+++ {file_path}")
        print("@@ -0,0 +1,{} @@".format(len(new_content.splitlines())))
        for line in new_content.splitlines():
            print(f"+{line}")
        return True
    
    with open(file_path, 'r') as f:
        existing_content = f.read()
    
    if existing_content == new_content:
        return False
    
    existing_lines = existing_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        existing_lines, 
        new_lines, 
        fromfile=f"{file_path} (current)",
        tofile=f"{file_path} (generated)",
        lineterm=""
    )
    
    print(f"\n--- Diff for {file_path} ---")
    for line in diff:
        print(line)
    
    return True

def write_to_temp_file(file_path: Path, content: str) -> Path:
    """Write content to a temporary file and return the path."""
    suffix = file_path.suffix
    prefix = f"{file_path.stem}_generated_"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, prefix=prefix, delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    return temp_path

def ask_yes_no(question: str) -> bool:
    """Ask a yes/no question and return the result."""
    while True:
        answer = input(f"{question} (y/n): ").lower().strip()
        if answer in ('y', 'yes'):
            return True
        elif answer in ('n', 'no'):
            return False
        else:
            print("Please answer 'y' or 'n'.")

def main():
    """Main function to update .pyi files."""
    parser = argparse.ArgumentParser(
        description="Update .pyi stub files for experimance_common",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Update files normally
  %(prog)s --dry-run          # Show generated content without writing
  %(prog)s --diff             # Show diffs and ask before writing
  %(prog)s --dry-run --temp   # Save to temp files and show paths
        """
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show generated content on stdout without writing to files'
    )
    parser.add_argument(
        '--diff', 
        action='store_true',
        help='Show diffs and ask for confirmation before writing'
    )
    parser.add_argument(
        '--temp', 
        action='store_true',
        help='Save to temporary files (useful with --dry-run)'
    )
    
    args = parser.parse_args()
    
    common_src = PROJECT_ROOT / "libs" / "common" / "src" / "experimance_common"
    
    # Generate content
    schemas_pyi_path = common_src / "schemas.pyi"
    schemas_content = generate_schemas_pyi()
    
    constants_pyi_path = common_src / "constants.pyi"
    constants_content = generate_constants_pyi()
    
    files_to_process = [
        (schemas_pyi_path, schemas_content, "schemas.pyi"),
        (constants_pyi_path, constants_content, "constants.pyi")
    ]
    
    if args.dry_run:
        print("=== DRY RUN MODE ===")
        
        if args.temp:
            print("Saving generated content to temporary files:")
            for file_path, content, name in files_to_process:
                temp_path = write_to_temp_file(file_path, content)
                print(f"  {name}: {temp_path}")
        else:
            for file_path, content, name in files_to_process:
                print(f"\n{'='*60}")
                print(f"Generated content for {name}:")
                print('='*60)
                print(content)
        
        print("\n=== END DRY RUN ===")
        return
    
    if args.diff:
        print("=== DIFF MODE ===")
        
        # Show all diffs first
        changes_found = False
        for file_path, content, name in files_to_process:
            if show_diff(file_path, content):
                changes_found = True
            else:
                print(f"\nNo changes needed for {file_path}")
        
        if not changes_found:
            print("\nNo changes detected in any files.")
            return
        
        # Ask for confirmation
        print(f"\n{'='*60}")
        if not ask_yes_no("Apply these changes?"):
            print("Aborted.")
            return
    
    # Write files
    updated_files = []
    for file_path, content, name in files_to_process:
        if args.diff or write_file_if_changed(file_path, content):
            # In diff mode, always write since we already checked for changes
            if args.diff:
                with open(file_path, 'w') as f:
                    f.write(content)
            updated_files.append(file_path)
            print(f"Updated {file_path}")
        else:
            print(f"No changes needed for {file_path}")
    
    if updated_files:
        print(f"\nSuccessfully updated {len(updated_files)} file(s).")
    else:
        print("\nAll files are already up to date.")
    
    print("Done!")

if __name__ == "__main__":
    main()
