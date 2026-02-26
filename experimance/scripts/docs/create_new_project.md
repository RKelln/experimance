# create_new_project.py

Interactive wizard for setting up a new Experimance project. Creates the project directory structure, copies service config files, and generates type stubs.

See `scripts/create_new_project.py`.

## Quick Start

```bash
uv run python scripts/create_new_project.py
```

The script is fully interactive — no flags required.

## What It Creates

For a project named `my_art_project`, the wizard creates:

```
projects/my_art_project/
├── .env                   # Project-specific environment variables
├── constants.py           # Project constants (overrides base constants)
├── schemas.py             # Project schemas (extends base schemas)
├── core.toml              # (if core service selected)
├── display.toml           # (if display service selected)
├── audio.toml             # (if audio service selected)
├── agent.toml             # (if agent service selected)
└── image_server.toml      # (if image_server service selected)
```

Type stubs are also regenerated (`libs/common/src/experimance_common/schemas.pyi` and `constants.pyi`) so IDEs get proper autocomplete for the new project's types.

## Interactive Steps

1. **Project name** — enter a new unique name (existing projects are listed for reference)
2. **Service selection** — pick which services to include (`1,2,5` or `all`)
3. **Config source** — for each service, choose to copy from an existing project or use service defaults
4. **Confirmation** — review the plan before files are written

## Available Services

| Key | Service | Description |
|---|---|---|
| `core` | Core | State machine, camera, orchestration |
| `display` | Display | Sand table visualization |
| `audio` | Audio | Sound generation + SuperCollider integration |
| `agent` | Agent | AI agent for visitor interactions |
| `image_server` | Image Server | AI-generated satellite imagery |

## After Running

1. Review and customize the generated config files in `projects/<name>/`
2. Edit `constants.py` and `schemas.py` to add project-specific values
3. Run services with `PROJECT_ENV=<name> ./scripts/dev all`

## Validation

The script:
- Prevents overwriting existing project names
- Validates the project name format
- Creates minimal placeholder files if no source config exists for a service
