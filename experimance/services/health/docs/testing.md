# Testing the Health Service

## Environment

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- The `experimance-common` package must be installed (it is a workspace dependency)

## Run the Tests

```bash
# From the health service directory
cd services/health
uv run pytest
```

```bash
# With verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_health_service.py -v
```

## Dev Dependencies

Declared in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
    "pytest-mock",
]
```

Install them with:

```bash
uv sync --extra dev
```

## Manual Smoke Test

Start the service pointing at a local health directory:

```bash
uv run -m experimance_health --log-level DEBUG
```

The service resolves to `dev_health_dir = logs/health` automatically in development. Drop a
test health file there to observe monitoring:

```bash
mkdir -p logs/health
cat > logs/health/core.json <<'EOF'
{
    "service_name": "experimance_core",
    "overall_status": "healthy",
    "checks": [],
    "last_updated": "2025-01-01T00:00:00",
    "last_check": "2025-01-01T00:00:00",
    "uptime": 0,
    "restart_count": 0,
    "error_count": 0
}
EOF
```

Then update `last_check` to the current time to keep it fresh, or leave it stale to trigger an
`ERROR` notification (after `startup_grace_period` expires).

## Link Validation

Verify all internal markdown links in the service docs:

```bash
uv run python scripts/check_docs_links.py
```

Run from the repository root.
