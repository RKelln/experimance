# Experimance Health Service

Monitors the health of all other services in the Experimance installation by reading their JSON
health status files, detecting stale or missing data, and sending notifications when something
goes wrong.

## Quick Start

```bash
# Run in development (reads from logs/health/)
uv run -m experimance_health

# Run with debug logging
uv run -m experimance_health --log-level DEBUG

# Run with a custom config file
uv run -m experimance_health --config /path/to/config.toml
```

### As a systemd service (production)

```bash
systemctl start health@experimance
systemctl status health@experimance
journalctl -u health@experimance -f
```

## Overview

The health service:

1. Reads `{service_type}.json` files written by each service into `health_dir`
2. Detects stale files (age > `service_timeout`) and marks the service `ERROR`
3. Detects missing files and marks the service `UNKNOWN`
4. Sends notifications on status changes via log file, ntfy, and/or webhook
5. Collects and logs system-wide health statistics every 10 seconds
6. Cleans up health files older than `max_health_file_age`

## Environment

- **OS:** Linux (systemd for production deployment)
- **Python:** 3.11+
- **Required services:** None — the health service is intentionally self-contained
- **Health file directory:** `logs/health/` (dev) or `/var/cache/experimance/health/` (production)

## Configuration

The service reads `services/health/config.toml` by default. Key settings:

```toml
check_interval          = 30.0   # seconds between health checks
startup_grace_period    = 10.0   # seconds before alerts fire after startup
service_timeout         = 120.0  # seconds before a stale file becomes ERROR
notification_cooldown   = 300.0  # seconds between repeated same-status alerts
notification_level      = "warning"   # "error" | "warning" | "info"

expected_services = [
    "experimance_core",
    "experimance_display",
    "experimance_agent",
    "experimance_audio",
    "image_server",
]

# Optional push notifications
# ntfy_topic  = "my-topic"
# ntfy_server = "ntfy.sh"

# Optional webhook
# webhook_url = "https://hooks.example.com/endpoint"
```

See [docs/configuration.md](docs/configuration.md) for all keys and environment-aware path
resolution.

## Health File Format

Each service writes a JSON file like this:

```json
{
    "service_name": "experimance_core",
    "overall_status": "healthy",
    "checks": [
        {
            "name": "periodic_check",
            "status": "healthy",
            "message": "Service is responsive",
            "timestamp": "2025-11-18T14:30:00.123456",
            "metadata": {}
        }
    ],
    "last_updated": "2025-11-18T14:30:00.123456",
    "last_check": "2025-11-18T14:30:00.123456",
    "uptime": 1234.5,
    "restart_count": 0,
    "error_count": 0
}
```

See [docs/health-file-format.md](docs/health-file-format.md) for the full schema.

## Additional Docs

| Document | Description |
|---|---|
| [docs/index.md](docs/index.md) | All service docs at a glance |
| [docs/architecture.md](docs/architecture.md) | Monitoring loops, status flow, and internal design |
| [docs/configuration.md](docs/configuration.md) | All config keys, defaults, environment overrides |
| [docs/health-file-format.md](docs/health-file-format.md) | JSON schema for health status files |
| [docs/notifications.md](docs/notifications.md) | Channels, filtering, buffering, cooldown |
| [docs/testing.md](docs/testing.md) | Running tests and manual smoke testing |
| [docs/roadmap.md](docs/roadmap.md) | Near-term goals and known gaps |

See also the system-wide health overview at [/docs/health_system.md](/docs/health_system.md).
