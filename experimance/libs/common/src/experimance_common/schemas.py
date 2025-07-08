import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# CRITICAL: Ensure PROJECT_ENV is set BEFORE loading any schemas
# This must happen before any other imports or environment variable checks

# First, set a reasonable default
os.environ.setdefault("PROJECT_ENV", "experimance")

# Then load project-specific .env file which can override the default
from experimance_common.constants_base import PROJECT_ROOT
proj_env = PROJECT_ROOT / f"projects/{os.environ['PROJECT_ENV']}/.env"
if proj_env.exists():
    load_dotenv(proj_env, override=True)
else:
    pass  # No project-specific .env file found

logger = logging.getLogger(__name__)

# Import all base schemas and make them available in this module
import experimance_common.schemas_base as _base_schemas
from experimance_common.constants_base import PROJECT_SPECIFIC_DIR

# Make all public symbols from base schemas available in this module
for _name in dir(_base_schemas):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_base_schemas, _name)

PROJECT = os.getenv("PROJECT_ENV", "experimance")
override_file = PROJECT_SPECIFIC_DIR / PROJECT / "schemas.py"

if override_file.exists():     
    try: # 2. project deltas - execute in current module namespace
        with open(override_file, 'r') as f:
            project_code = f.read()
        
        # Execute the project-specific code in the current module's globals
        # This ensures classes are created in the experimance_common.schemas namespace
        exec(project_code, globals())
        
        logger.debug(f"Successfully loaded project-specific schemas from {override_file}")
    except Exception as e:
        logger.warning(f"Failed to load schemas from {override_file}: {e}")
        import traceback
        traceback.print_exc()


# CRITICAL: Override MessageBase.from_dict to use project-specific classes
# This ensures that ZMQ deserialization creates extended schema objects instead of base ones
def _project_from_dict(cls, data):
    """
    Override of MessageBase.from_dict that uses project-specific schema classes.
    This ensures ZMQ message deserialization creates extended schema objects.
    """
    from typing import Dict, Any, Union
    
    # Get the message type
    message_type = data.get("type")
    if not message_type:
        return data
    
    # Map message types to the current module's classes (which are project-specific)
    current_module = globals()
    
    # Look for a class in the current module that has this type
    for name, obj in current_module.items():
        if (hasattr(obj, '__bases__') and 
            hasattr(obj, 'model_fields') and 
            'type' in obj.model_fields):
            
            field_info = obj.model_fields['type']
            if hasattr(field_info, 'default') and field_info.default == message_type:
                try:
                    return obj(**data)
                except Exception as e:
                    logger.warning(f"Failed to create {name} from data: {e}")
                    return data
    
    # Fallback: return original data if no matching class found
    return data

# Apply the override to MessageBase (imported from base schemas)
globals()['MessageBase'].from_dict = classmethod(_project_from_dict)