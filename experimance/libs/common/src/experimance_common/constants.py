import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# CRITICAL: Ensure PROJECT_ENV is set BEFORE loading any constants
# This must happen before any other imports or environment variable checks

# First, set a reasonable default
os.environ.setdefault("PROJECT_ENV", "experimance")

# Load project (and variant) .env — PROJECT_ENV is preserved across all loads
from experimance_common.project_utils import load_project_dotenv, get_base_project_name
load_project_dotenv()

logger = logging.getLogger(__name__)

# Import all base constants and make them available in this module
import experimance_common.constants_base as _base_constants
from experimance_common.constants_base import PROJECT_SPECIFIC_DIR

# Make all public symbols from base constants available in this module
for _name in dir(_base_constants):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_base_constants, _name)

PROJECT = get_base_project_name(os.getenv("PROJECT_ENV", "experimance"))
override_file = PROJECT_SPECIFIC_DIR / PROJECT / "constants.py"

if override_file.exists():     
    try: # 2. project deltas - execute in current module namespace
        with open(override_file, 'r') as f:
            project_code = f.read()
        
        # Execute the project-specific code in the current module's globals
        # This ensures constants are created in the experimance_common.constants namespace
        exec(project_code, globals())
        
        logger.debug(f"Successfully loaded project-specific constants from {override_file}")
    except Exception as e:
        logger.warning(f"Failed to load constants from {override_file}: {e}")