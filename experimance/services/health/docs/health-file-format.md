# Health File Format

Each Experimance service writes a JSON file to the health directory to report its current status.
The health service reads these files to determine overall system health.

## File Location

```
Development:  logs/health/{service_type}.json
Production:   /var/cache/experimance/health/{service_type}.json
```

## Schema

```json
{
    "service_name": "core",
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

### Top-Level Fields

| Field | Type | Description |
|---|---|---|
| `service_name` | string | Identifier matching the expected service name in the health service config |
| `overall_status` | string | Worst status across all checks: `healthy`, `warning`, `error`, `fatal`, or `unknown` |
| `checks` | array | Individual health check results (see below) |
| `last_updated` | ISO 8601 datetime | When the file was last written |
| `last_check` | ISO 8601 datetime | Timestamp the health service uses to detect stale data |
| `uptime` | float | Seconds since the service started |
| `restart_count` | integer | Number of times the service has restarted |
| `error_count` | integer | Cumulative error count |

### Check Object Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Unique name for this health check |
| `status` | string | `healthy`, `warning`, `error`, `fatal`, or `unknown` |
| `message` | string | Human-readable description of the check result |
| `timestamp` | ISO 8601 datetime | When this check was recorded |
| `metadata` | object | Arbitrary key-value pairs for additional context |

## Staleness Detection

The health service compares the current time against `last_check`. If the age exceeds
`service_timeout` (default 120 s), the service is considered stale:

- During the startup grace period: stale → `UNKNOWN`
- After the grace period: stale → `ERROR`

## Writing Health Files (for Service Authors)

Services that extend `BaseService` get health reporting automatically via `HealthReporter`. No
manual file writing is needed. See [/docs/health_system.md](/docs/health_system.md) for details on
adding custom health checks.
