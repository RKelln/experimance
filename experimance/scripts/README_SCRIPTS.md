# Scripts Directory Guidelines

This directory contains development and maintenance oriented scripts that are NOT packaged as end-user console commands.

## Categories

- Provisioning / deployment helpers (dev only)
- One-off diagnostics and tuning tools (e.g. `tune_detector.py`)
- Data generation helpers (unless promoted to a packaged CLI)
- Experimental utilities under iteration

## Promoted CLI Tools

Tools intended for regular operational use are moved into the main `experimance` package and exposed as console scripts via `pyproject.toml`.

Current promoted tools:

- `transcripts`  (transcript browsing & streaming)
- `vastai`       (Vast.ai instance management)

You can run them with:

```bash
uv run transcripts list
uv run vastai list
```

## Migration Pattern

To promote a script:
1. Move or refactor logic into `src/experimance/<tool>_cli.py` (or a subpackage like `experimance.tools`).
2. Add an entry in `[project.scripts]` in `pyproject.toml`.
3. Keep a thin wrapper here (optional) or remove the old script once users switch.
4. Update documentation (README sections referencing the script path).

## Deprecation Note

The legacy `scripts/vastai_cli.py` remains for now but new usage should prefer:

```bash
uv run vastai <command>
```

This improves portability for production environments where only the installed package is present.
