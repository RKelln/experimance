# Health Service Architecture

## Overview

The health service is an async Python service (`HealthService`) that extends `BaseService` from
`experimance_common`. It runs three concurrent loops:

| Loop | Interval | Purpose |
|---|---|---|
| `_service_health_monitoring_loop` | `check_interval` (default 30 s) | Read health files, detect stale data, send notifications |
| `_stats_collection_loop` | 10 s | Aggregate per-service stats and log a summary on change |
| `_cleanup_loop` | `cleanup_interval` (default 1 h) | Delete health files older than `max_health_file_age` |

## Health File Monitoring

Each monitored service writes a JSON health file to `health_dir/{service_type}.json`. The health
service reads these files on every check cycle.

```
health_dir/
├── core.json
├── display.json
├── agent.json
├── audio.json
└── image_server.json
```

A health file is considered **stale** when the `last_check` timestamp inside it is older than
`service_timeout` (default 120 s). Stale files produce an `ERROR` status (or `UNKNOWN` during the
startup grace period).

A missing file produces `UNKNOWN` status.

## Startup Grace Period

For `startup_grace_period` seconds (default 60 s in code, 10 s in `config.toml`) after the health
service starts, stale files are reported as `UNKNOWN` instead of `ERROR`. This prevents false alerts
during system boot when other services are still starting up.

After the grace period expires, the health service sends an initial system-wide notification
summarising the state of all monitored services.

## Status Flow

```
Health file absent           -> UNKNOWN
Health file present, fresh   -> status read from file (HEALTHY / WARNING / ERROR / FATAL)
Health file present, stale   -> ERROR  (UNKNOWN during grace period)
JSON parse error             -> UNKNOWN
```

`HealthStatus` values (from `experimance_common.health`):

| Value | Meaning |
|---|---|
| `HEALTHY` | Service is operating normally |
| `WARNING` | Issues present but service is still functional |
| `ERROR` | Significant issues affecting functionality |
| `FATAL` | Service is non-functional; intervention required |
| `UNKNOWN` | Status cannot be determined (missing or unreadable file) |

## Notification Flow

```
_check_all_services()
  -> _check_service_health()   # per service
  -> _send_notifications()     # batches status changes
       -> _should_notify()     # applies level filter + cooldown
       -> handler.send_notification()
```

On shutdown, `_flush_notification_buffers()` is called before the service exits to ensure no
buffered notifications are lost.

## Files Touched

| File | Role |
|---|---|
| `src/experimance_health/health_service.py` | `HealthService` class, all async loops and notification logic |
| `src/experimance_health/config.py` | `HealthServiceConfig` Pydantic model |
| `src/experimance_health/__main__.py` | CLI entry point via `create_simple_main` |
| `src/experimance_health/__init__.py` | Package exports |
| `config.toml` | Default runtime configuration |
