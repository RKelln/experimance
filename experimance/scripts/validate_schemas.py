#!/usr/bin/env python3
"""
Schema validation script for Experimance project.

This script validates that Python schemas in experimance_common.schemas 
match the corresponding JSON configuration files.
"""

import json
import sys
from pathlib import Path
from typing import List, Set

# Add the libs/common/src to the path so we can import schemas
sys.path.insert(0, str(Path(__file__).parent.parent / "libs" / "common" / "src"))

from experimance_common.schemas import Era, Biome


def load_json_config(file_path: Path) -> dict:
    """Load and parse a JSON configuration file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Remove JavaScript-style comments for JSON parsing
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                # Remove lines that start with // 
                if line.strip().startswith('//'):
                    continue
                # Remove inline comments
                if '//' in line:
                    line = line[:line.index('//')]
                cleaned_lines.append(line)
            cleaned_content = '\n'.join(cleaned_lines)
            return json.loads(cleaned_content)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def validate_enum_against_json(enum_cls, json_values: List[str], name: str) -> bool:
    """Validate that a Python enum matches JSON array values."""
    python_values = set(item.value for item in enum_cls)
    json_values_set = set(json_values)
    
    print(f"\nValidating {name}:")
    print(f"  Python enum has {len(python_values)} values")
    print(f"  JSON config has {len(json_values_set)} values")
    
    # Check for missing values in Python enum
    missing_in_python = json_values_set - python_values
    if missing_in_python:
        print(f"  ❌ Missing in Python enum: {missing_in_python}")
        return False
    
    # Check for extra values in Python enum
    extra_in_python = python_values - json_values_set
    if extra_in_python:
        print(f"  ❌ Extra in Python enum: {extra_in_python}")
        return False
    
    print(f"  ✅ {name} schemas match perfectly")
    return True


def main():
    """Main validation function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Path to the general configuration
    general_config_path = project_root / "data" / "experimance_config.json"
    
    print("Experimance Schema Validation")
    print("=" * 40)
    
    if not general_config_path.exists():
        print(f"❌ General config file not found: {general_config_path}")
        return False
    
    # Load general configuration
    general_config = load_json_config(general_config_path)
    if not general_config:
        print("❌ Failed to load general configuration")
        return False
    
    print(f"✅ Loaded general config from: {general_config_path}")
    
    # Validate Era enum
    eras_valid = True
    if 'eras' in general_config:
        eras_valid = validate_enum_against_json(Era, general_config['eras'], "Era")
    else:
        print("❌ 'eras' not found in general config")
        eras_valid = False
    
    # Validate Biome enum
    biomes_valid = True
    if 'biomes' in general_config:
        biomes_valid = validate_enum_against_json(Biome, general_config['biomes'], "Biome")
    else:
        print("❌ 'biomes' not found in general config")
        biomes_valid = False
    
    # Summary
    print("\nValidation Summary:")
    print("=" * 20)
    
    all_valid = eras_valid and biomes_valid
    
    if all_valid:
        print("✅ All schema validations passed!")
        return True
    else:
        print("❌ Some schema validations failed!")
        print("\nTo fix schema mismatches:")
        print("1. Update the Python enums in libs/common/src/experimance_common/schemas.py")
        print("2. Or update the JSON arrays in data/experimance_config.json")
        print("3. Re-run this validation script")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
