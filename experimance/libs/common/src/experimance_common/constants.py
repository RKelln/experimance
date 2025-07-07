import logging
# 1. shared defaults
from experimance_common.constants_base import *     
import importlib.util, os, sys

from experimance_common.constants_base import PROJECT_SPECIFIC_DIR

PROJECT = os.getenv("PROJECT_ENV", "experimance")
override_file = PROJECT_SPECIFIC_DIR / PROJECT / "constants.py"

if override_file.exists():     
    try: # 2. project deltas
        spec = importlib.util.spec_from_file_location(f"{PROJECT}_constants", override_file)
        if spec is not None and spec.loader is not None:
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            # Only inject uppercase variables (constants)
            globals().update({k: v for k, v in mod.__dict__.items() if k.isupper()})
        else:
            logging.warning(f"Could not create module spec or loader for {override_file}")
    except Exception as e:
        logging.warning(f"Failed to load constants from {override_file}: {e}")