# experimance_health

Health monitoring service for the Experimance installation.

Reads JSON health status files written by each service, detects stale or missing data, and sends
notifications when services become unhealthy.

```bash
# Run in development
uv run -m experimance_health

# Run with debug logging
uv run -m experimance_health --log-level DEBUG
```

For full documentation see [../../docs/architecture.md](../../docs/architecture.md).
