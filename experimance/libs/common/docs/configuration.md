# Configuration System Reference

This document covers `BaseConfig`, `BaseServiceConfig`, TOML loading, environment variable overrides, and CLI integration.

## Overview

Configuration is backed by [Pydantic v2](https://docs.pydantic.dev/) with TOML file loading on top. The key classes are:

| Class | Module | Purpose |
|-------|--------|---------|
| `BaseConfig` | `config.py` | Root config base; provides `from_overrides()` |
| `BaseServiceConfig` | `config.py` | Adds `service_name` field; use for all services |
| `load_config_with_overrides` | `config.py` | Low-level loader; usually called via `from_overrides()` |

---

## BaseConfig and BaseServiceConfig

```python
from experimance_common.config import BaseServiceConfig
from pydantic import Field

class MyServiceConfig(BaseServiceConfig):
    # Override default service name
    service_name: str = "my-service"

    # Service-specific fields with defaults
    work_interval: float = Field(default=1.0, ge=0.001,
                                  description="Seconds between work cycles")
    max_retries: int = Field(default=3, ge=0)
    debug_mode: bool = False
```

`BaseConfig` model settings:

- `validate_assignment = True` — validates on every attribute set
- `extra = "forbid"` — unknown keys in config files raise `ConfigError`
- `str_strip_whitespace = True` — strips leading/trailing whitespace from strings

---

## Loading configuration

### from_overrides (recommended)

```python
# Defaults only
config = MyServiceConfig.from_overrides()

# Override specific keys
config = MyServiceConfig.from_overrides(
    override_config={"work_interval": 0.5, "debug_mode": True}
)

# From a TOML file, with optional programmatic overrides
config = MyServiceConfig.from_overrides(
    config_file="config.toml",
    override_config={"work_interval": 0.5}
)

# With CLI args (argparse.Namespace)
config = MyServiceConfig.from_overrides(
    config_file="config.toml",
    args=parsed_args
)
```

### Priority order (highest to lowest)

1. CLI args (`args` parameter)
2. `override_config` dictionary
3. TOML `config_file`
4. Model field defaults

### Config file resolution

`get_project_config_path(service_name, fallback_dir)` resolves config paths:

1. `projects/<PROJECT_ENV>/<service_name>.toml` (if `PROJECT_ENV` env var is set)
2. `<fallback_dir>/config.toml`
3. `projects/<PROJECT_ENV>/<service_name>.toml` (even if it doesn't exist yet)

```python
from experimance_common.constants_base import get_project_config_path, CORE_SERVICE_DIR

config_path = get_project_config_path("core", CORE_SERVICE_DIR)
```

---

## TOML config file format

```toml
# services/my_service/config.toml
[service]
service_name = "my-service"
work_interval = 2.0
max_retries = 5

[zmq.publisher]
address = "tcp://*"
port = 5555

[zmq.subscriber]
address = "tcp://localhost"
port = 5555
topics = ["heartbeat", "status"]
```

Sections map directly to nested Pydantic model fields. An empty section (`[]`) clears all defaults; remove sections you don't need rather than leaving them empty.

---

## Environment variable overrides

Environment variables are applied on top of the TOML file. The naming convention is:

```
<SERVICE_PREFIX>_<FIELD_NAME>
```

Example — if your config has `service_name = "experimance-core"`, prefix is `EXPERIMANCE`:

```bash
EXPERIMANCE_WORK_INTERVAL=0.5     # → work_interval = 0.5
ZMQ_PUBLISHER_PORT=5560           # → zmq.publisher.port = 5560
CAMERA_FPS=15                     # → camera.fps = 15
```

Type coercion is automatic (strings are converted to int/float/bool as needed).

---

## CLI integration

Use `create_simple_main` in `__main__.py` for a standard entry point:

```python
# src/my_service/__main__.py
from experimance_common.cli import create_simple_main
from .my_service import MyService
from .config import MyServiceConfig

if __name__ == "__main__":
    create_simple_main(MyService, MyServiceConfig)()
```

This generates CLI arguments automatically from `MyServiceConfig` fields:

```
$ uv run -m my_service --help
$ uv run -m my_service --config config.toml
$ uv run -m my_service --work-interval 0.5
$ uv run -m my_service --log-level DEBUG
```

Pydantic field names map to `--kebab-case` flags. Nested fields use dotted notation internally but `--section-field` on the CLI.

---

## Path resolution helpers

```python
from experimance_common.config import resolve_path, load_file_content

# Resolve a relative path with a hint about which service directory
abs_path = resolve_path("prompts/system.txt", hint="core")
# → <CORE_SERVICE_DIR>/prompts/system.txt

# Load file content with path resolution
prompt = load_file_content("prompts/system.txt", hint="project")
# → contents of projects/<PROJECT_ENV>/prompts/system.txt
```

Valid `hint` values:

| Hint | Resolves to |
|------|------------|
| `"core"` | `CORE_SERVICE_DIR` |
| `"agent"` | `AGENT_SERVICE_DIR` |
| `"display"` | `DISPLAY_SERVICE_DIR` |
| `"audio_service"` | `AUDIO_SERVICE_DIR` |
| `"image_server"` | `IMAGE_SERVER_SERVICE_DIR` |
| `"project"` | `projects/<PROJECT_ENV>/` |
| `"data"` | `DATA_DIR` |
| `"audio_dir"` | `AUDIO_DIR_ABS` |
| `"images_dir"` | `IMAGES_DIR_ABS` |
| absolute path string | as-is |

---

## Multi-project support

`PROJECT_ENV` selects the active project profile:

```bash
export PROJECT_ENV=experimance   # default
export PROJECT_ENV=fire          # alternate project
```

On import, `experimance_common` auto-detects `PROJECT_ENV`:
1. Reads `PROJECT_ENV` from the environment or a `.project` file at the repo root
2. Loads `projects/<PROJECT_ENV>/.env` if it exists (values override the base `.env`)

```python
from experimance_common.project_utils import ensure_project_env_set

ensure_project_env_set()           # called automatically by __init__.py
os.environ["PROJECT_ENV"]          # "experimance" (or whatever was detected)
```

---

## Errors

| Exception | Cause |
|-----------|-------|
| `ConfigError` | File not found, parse error, unknown config key |
| `pydantic.ValidationError` | Value fails field validator (wrong type, out of range, etc.) |

```python
from experimance_common.config import ConfigError

try:
    config = MyServiceConfig.from_overrides(config_file="missing.toml")
except ConfigError as e:
    print(f"Config problem: {e}")
```
