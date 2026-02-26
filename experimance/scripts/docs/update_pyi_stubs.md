# update_pyi_stubs.py

Regenerates `.pyi` type stub files for `experimance_common` so that static type checkers and IDEs understand the dynamically loaded project-specific `schemas` and `constants` modules.

Run this after adding new classes to any project's `schemas.py` or `constants.py`, or after creating a new project with `create_new_project.py`.

See `scripts/update_pyi_stubs.py`.

## Quick Start

```bash
# Regenerate stubs in place
uv run python scripts/update_pyi_stubs.py

# Preview changes without writing
uv run python scripts/update_pyi_stubs.py --dry-run

# Show a diff of what would change
uv run python scripts/update_pyi_stubs.py --diff
```

## What It Generates

| Output file | Content |
|---|---|
| `libs/common/src/experimance_common/schemas.pyi` | Union types for all `Era`, `Biome`, etc. classes across all projects |
| `libs/common/src/experimance_common/constants.pyi` | Type stubs for project-specific constants |

The script introspects the base modules and all projects under `projects/*/` to build comprehensive stubs that cover every project variant.

## When to Run

- After running `create_new_project.py` (it runs this automatically)
- After adding a new class to any `projects/<name>/schemas.py`
- After adding a new constant to any `projects/<name>/constants.py`
- After modifying the base schemas or constants in `libs/common/`

## Options

| Flag | Description |
|---|---|
| `--dry-run` | Print what would be written without modifying files |
| `--diff` | Show a unified diff for each file that would change |
